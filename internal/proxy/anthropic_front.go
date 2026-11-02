package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"go.uber.org/zap"
)

// Anthropic-compatible front API (Phase 1: routing foundation).
//
// This file implements the second public front door of the gateway: an
// Anthropic Messages API surface (POST /v1/messages, POST /v1/messages/count_tokens)
// that lets Anthropic-protocol clients — official Anthropic SDKs, Claude Code CLI,
// Cursor, Cline — talk to every backend provider the gateway already routes.
//
// Phase 1 scope (this file):
//   - exact route recognition and dispatch from Handler.Handle
//   - Anthropic-shaped error envelopes ({"type":"error","error":{...}})
//   - zero-allocation request validation via buger/jsonparser (no encoding/json
//     struct unmarshaling, no anthropic-sdk-go on the hot path)
//
// Phase 2 scope (hook points marked below):
//   - Anthropic → provider request translation (system/messages/tools blocks)
//   - reuse of the existing credential acquisition + retry/failover loop
//     (executeWithRetry), telemetry (pipeline.Emit) and streaming bridge
//     (StreamProxy + per-provider transmuxers), exactly like
//     /v1/chat/completions does today.
//
// Until the Phase 2 bridge lands, valid requests receive a well-formed
// Anthropic api_error envelope instead of an OpenAI-shaped error or a 404,
// so SDK clients always surface a clean, typed failure.

// Canonical request paths of the Anthropic-compatible front API.
// Handler.Handle dispatches on exact matches; the auth and rate-limit
// middleware recognize the family via the "/v1/messages" prefix.
const (
	anthropicMessagesPath    = "/v1/messages"
	anthropicCountTokensPath = "/v1/messages/count_tokens"
)

// isAnthropicFrontPath reports whether the request path belongs to the
// Anthropic-compatible front API rather than the OpenAI-compatible pipeline.
func isAnthropicFrontPath(path string) bool {
	p := path
	for strings.HasPrefix(p, "/v1") {
		p = strings.TrimPrefix(p, "/v1")
	}
	return p == "/messages" || p == "/messages/count_tokens"
}

// AnthropicErrorResponse writes an Anthropic-formatted error envelope:
//
//	{"type":"error","error":{"type":"<errType>","message":"<message>"}}
//
// errType is one of the Anthropic error types: invalid_request_error,
// authentication_error, permission_error, not_found_error, rate_limit_error,
// api_error, overloaded_error.
func AnthropicErrorResponse(c *gin.Context, status int, errType, message string) {
	c.AbortWithStatusJSON(status, gin.H{
		"type": "error",
		"error": gin.H{
			"type":    errType,
			"message": message,
		},
	})
}

// validateAnthropicBody performs the shared, zero-allocation structural
// validation of an Anthropic request body (both /v1/messages and
// /v1/messages/count_tokens).
//
// Returns the model identifier and the stream flag on success. On failure an
// Anthropic error envelope has already been written and ok=false is returned.
func validateAnthropicBody(c *gin.Context, h *Handler, bodyBytes []byte, requireMaxTokens bool) (model string, isStream bool, ok bool) {
	if len(bytes.TrimSpace(bodyBytes)) == 0 {
		AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "request body is empty")
		return "", false, false
	}

	// model is required and must be a JSON string.
	modelBytes, valueType, _, err := jsonparser.Get(bodyBytes, "model")
	if err != nil || len(modelBytes) == 0 || valueType != jsonparser.String {
		AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "missing or invalid 'model' field: expected a non-empty string")
		return "", false, false
	}
	model = string(modelBytes)

	// messages is required and must be a non-empty array of message objects.
	msgValue, msgType, _, err := jsonparser.Get(bodyBytes, "messages")
	if err != nil || msgType != jsonparser.Array {
		AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "missing or invalid 'messages' field: expected an array of message objects")
		return "", false, false
	}
	messageCount := 0
	if _, arrErr := jsonparser.ArrayEach(msgValue, func(_ []byte, _ jsonparser.ValueType, _ int, _ error) {
		messageCount++
	}); arrErr != nil || messageCount == 0 {
		AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "invalid 'messages' field: array cannot be empty")
		return "", false, false
	}

	// max_tokens is required by the Messages API spec on /v1/messages
	// (count_tokens does not accept it).
	if requireMaxTokens {
		maxTokens, err := jsonparser.GetInt(bodyBytes, "max_tokens")
		if err != nil {
			AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "missing or invalid 'max_tokens' field: a positive integer is required")
			return "", false, false
		}
		if maxTokens <= 0 {
			AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", "'max_tokens' must be a positive integer")
			return "", false, false
		}
	}

	// Stream flag: absent means non-streaming, mirroring both the Anthropic
	// and OpenAI APIs. jsonparser.GetBoolean never allocates.
	isStream, _ = jsonparser.GetBoolean(bodyBytes, "stream")

	return model, isStream, true
}

// handleAnthropicMessages is the Phase-2 entry point for POST /v1/messages.
//
// It owns structural validation, pool resolution, request preparation and
// observability for the Anthropic Messages API. The body is handed in already
// buffered by Handler.Handle (pooled scratch buffer + defensive copy), so this
// method performs no I/O on the request body.
//
// Flow:
//  1. Validate the Anthropic body (model/messages/max_tokens).
//  2. Resolve the routing pool exactly like Proxy.Handle (cache → prefix).
//  3. Pre-translate the body into the canonical OpenAI Chat Chat Completions
//     JSON once for validation (skipped when the pool only serves
//     Anthropic-protocol credentials — the zero-copy path). The translation
//     happens lazily inside forwardRequest when a non-native credential
//     appears later (cross-provider fallback).
//  4. Rewrite the request path to /v1/chat/completions so every provider path
//     transformer and request builder behaves exactly as the OpenAI front door.
//  5. Enter the shared credential acquisition + retry/failover loop
//     (executeWithRetry). forwardRequest picks per credential: native
//     Anthropic-protocol providers get PrepareNativeAnthropicPayload
//     (zero-copy), everything else gets the translated OpenAI payload and the
//     full existing provider-specific request pipeline.
func (h *Handler) handleAnthropicMessages(c *gin.Context, bodyBytes []byte) {
	requestStart := time.Now()
	model, isStream, ok := validateAnthropicBody(c, h, bodyBytes, true)
	if !ok {
		return
	}

	requestID, _ := c.Get("request_id")
	tenantName, _ := c.Get("tenant_name")
	h.logger.Info("anthropic messages request received",
		zap.String("request_id", fmt.Sprintf("%v", requestID)),
		zap.String("tenant", fmt.Sprintf("%v", tenantName)),
		zap.String("model", model),
		zap.Bool("stream", isStream),
		zap.Int("body_bytes", len(bodyBytes)),
		zap.String("client_ip", c.ClientIP()),
	)

	// Step 1: resolve the routing pool — identical lookup order to Proxy.Handle
	// (exact cache key first, then wildcard/prefix matching for model families).
	poolVal, found := h.cache.Get(cache.PoolKey(model))
	if !found {
		poolVal, found = h.findPoolByPrefix(model)
	}
	if !found {
		h.logger.Warn("anthropic request rejected: no routing pool configured",
			zap.String("request_id", fmt.Sprintf("%v", requestID)),
			zap.String("model", model),
			zap.String("tenant", fmt.Sprintf("%v", tenantName)),
		)
		AnthropicErrorResponse(c, http.StatusNotFound, "not_found_error",
			"model not found or no routing pool configured: "+model)
		return
	}
	pool, _ := poolVal.(*credentials.BalancedChannelPool)

	// Step 2: strip the routing prefix so upstreams see the clean model id
	// (e.g. "anthropic/claude-sonnet-4" -> "claude-sonnet-4").
	cleanModel := stripAnthropicRoutingModel(model, pool.ModelPattern)

	// Step 3: telemetry prompt extraction (system + last user message).
	promptText := ExtractAnthropicPromptText(bodyBytes)

	// Step 4: per-provider routing flags — same prefix detection as Proxy.Handle.
	isNvidia := strings.HasPrefix(model, "nvidia/")
	isOneMinAI := strings.HasPrefix(model, "1min/")
	isCloudflare := strings.HasPrefix(model, "cloudflare/")
	isSarvam := strings.HasPrefix(model, "sarvam/")
	isPuter := strings.HasPrefix(model, "puter/")
	isAgentRouter := strings.HasPrefix(model, "agentrouter/")
	isZenMux := strings.HasPrefix(model, "zenmux/")
	isJiekou := strings.HasPrefix(model, "jiekou/")

	// Step 5: pre-translate once when the pool can serve non-native
	// credentials — this validates the translation up front and avoids
	// re-translating on every retry attempt. All-native pools take the
	// zero-copy path and translate lazily inside forwardRequest if a
	// cross-provider fallback ever lands on a non-native provider.
	var translatedBody []byte
	if !poolAllNativeAnthropic(pool) {
		translated, terr := TranslateAnthropicToOpenAI(bodyBytes, cleanModel, anthropicFallbackMaxTokens)
		if terr != nil {
			h.logger.Warn("anthropic request translation failed",
				zap.String("request_id", fmt.Sprintf("%v", requestID)),
				zap.String("model", model),
				zap.Error(terr),
			)
			AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error",
				"failed to translate Anthropic request: "+terr.Error())
			return
		}
		// NVIDIA reasoning parameter injection — mirrored from Proxy.Handle.
		if isNvidia && supportsNvidiaReasoning(cleanModel) {
			translated = injectNvidiaParams(translated, translated, h.logger)
		}
		translatedBody = translated
	}

	// Step 6: rewrite the request path so every provider path transformer and
	// the provider-specific request builders inside forwardRequest behave
	// exactly as they do for the OpenAI front door:
	//   - native providers map it back to /v1/messages (anthropicPath,
	//     agentrouterPath) or /v1/messages?beta=true;
	//   - OpenAI-compatible providers receive /v1/chat/completions;
	//   - Gemini/Ollama/Cloudflare/1min.ai get their native endpoints.
	// The dispatch above already happened, so this cannot affect routing.
	c.Request.URL.Path = "/v1/chat/completions"

	body := translatedBody
	if body == nil {
		body = bodyBytes // all-native pool: keep the Anthropic body for now
	}

	estimatedInputTokens, _ := EstimateAnthropicInputTokens(bodyBytes)

	pctx := &proxyContext{
		model:                cleanModel,
		isStream:             isStream,
		isNvidia:             isNvidia,
		isOneMinAI:           isOneMinAI,
		isCloudflare:         isCloudflare,
		isSarvam:             isSarvam,
		isPuter:              isPuter,
		isAgentRouter:        isAgentRouter,
		isZenMux:             isZenMux,
		isJiekou:             isJiekou,
		requestedModel:       model,
		rawBody:              body,
		body:                 body,
		promptText:           promptText,
		anthropicFront:       true,
		anthropicRawBody:     bodyBytes,
		anthropicTranslated:  translatedBody,
		anthropicInputTokens: estimatedInputTokens,
		pool:                 pool,
	}

	if tenantID, exists := c.Get("tenant_id"); exists {
		pctx.tenantID = tenantID.(string)
	}

	// Step 7: enter the shared credential acquisition + retry/failover loop.
	maxAttempts := int(pool.TotalCount)
	if maxAttempts < 1 {
		maxAttempts = 1
	}
	if maxAttempts > 100 {
		maxAttempts = 100
	}
	h.executeWithRetry(c, pctx, requestStart, maxAttempts)
}

// anthropicFallbackMaxTokens is the generation budget used by the OpenAI
// translation when an Anthropic request carries no usable max_tokens. It
// mirrors the documented fallback of quota.SlidingWindow.EstimateMaxTokens.
const anthropicFallbackMaxTokens = int64(4096)

// handleAnthropicCountTokens implements POST /v1/messages/count_tokens.
//
// Anthropic SDKs and Claude Code CLI call this endpoint with the same body
// shape as /v1/messages (model, system, messages, tools, thinking) to estimate
// input token usage before sending a request. It responds with:
//
//	{"input_tokens": <estimated_tokens>}
func (h *Handler) handleAnthropicCountTokens(c *gin.Context, bodyBytes []byte) {
	model, _, ok := validateAnthropicBody(c, h, bodyBytes, false)
	if !ok {
		return
	}

	requestID, _ := c.Get("request_id")
	tenantName, _ := c.Get("tenant_name")
	h.logger.Info("anthropic count_tokens request received",
		zap.String("request_id", fmt.Sprintf("%v", requestID)),
		zap.String("tenant", fmt.Sprintf("%v", tenantName)),
		zap.String("model", model),
		zap.Int("body_bytes", len(bodyBytes)),
		zap.String("client_ip", c.ClientIP()),
	)

	tokens, err := EstimateAnthropicInputTokens(bodyBytes)
	if err != nil {
		h.logger.Warn("anthropic count_tokens estimation failed",
			zap.String("request_id", fmt.Sprintf("%v", requestID)),
			zap.String("model", model),
			zap.Error(err),
		)
		AnthropicErrorResponse(c, http.StatusBadRequest, "invalid_request_error", err.Error())
		return
	}

	c.JSON(http.StatusOK, gin.H{"input_tokens": tokens})
}

// EstimateAnthropicInputTokens estimates the input token count of an Anthropic
// Messages request body with a deterministic, tokenizer-independent heuristic:
//
//   - every text string (system blocks, message content, tool_result payloads,
//     tool names/descriptions/schemas) contributes (chars+3)/4 tokens — the
//     classic ~4-characters-per-token approximation;
//   - each message contributes 4 structural overhead tokens;
//   - each tool definition contributes 12 overhead tokens;
//   - each image/document block contributes a flat 1200 tokens (matching
//     Anthropic's documented ~1600px image budget);
//   - the result is floored at 1 token for any request with messages.
//
// The walk uses buger/jsonparser directly on the raw body — no Go structs, no
// reflection, a single pass over the payload.
func EstimateAnthropicInputTokens(body []byte) (int, error) {
	if len(bytes.TrimSpace(body)) == 0 {
		return 0, errors.New("request body is empty")
	}
	if !json.Valid(body) {
		return 0, errors.New("request body is not valid JSON")
	}
	msgVal, msgType, _, msgErr := jsonparser.Get(body, "messages")
	if msgErr != nil || msgType != jsonparser.Array {
		return 0, errors.New("missing or invalid 'messages' field")
	}

	tokens := 0

	// System prompt: string form or array of text blocks.
	sysVal, sysType, _, sysErr := jsonparser.Get(body, "system")
	if sysErr == nil && len(sysVal) > 0 {
		switch sysType {
		case jsonparser.String:
			if s, err := jsonparser.GetString(body, "system"); err == nil {
				tokens += anthropicTextTokenEstimate(s)
			}
		case jsonparser.Array:
			jsonparser.ArrayEach(sysVal, func(block []byte, dataType jsonparser.ValueType, _ int, _ error) {
				if dataType != jsonparser.Object {
					return
				}
				if t, _ := jsonparser.GetString(block, "type"); t == "text" {
					if txt, _ := jsonparser.GetString(block, "text"); txt != "" {
						tokens += anthropicTextTokenEstimate(txt)
					}
				}
			})
		}
	}

	// Messages: per-message overhead + content.
	jsonparser.ArrayEach(msgVal, func(msg []byte, dataType jsonparser.ValueType, _ int, _ error) {
		if dataType != jsonparser.Object {
			return
		}
		tokens += 4 // per-message structural overhead
		contentVal, contentType, _, contentErr := jsonparser.Get(msg, "content")
		if contentErr != nil {
			return
		}
		switch contentType {
		case jsonparser.String:
			if s, err := jsonparser.GetString(msg, "content"); err == nil {
				tokens += anthropicTextTokenEstimate(s)
			}
		case jsonparser.Array:
			jsonparser.ArrayEach(contentVal, func(block []byte, blockType jsonparser.ValueType, _ int, _ error) {
				if blockType != jsonparser.Object {
					return
				}
				switch bt, _ := jsonparser.GetString(block, "type"); bt {
				case "text":
					if txt, _ := jsonparser.GetString(block, "text"); txt != "" {
						tokens += anthropicTextTokenEstimate(txt)
					}
				case "image", "document":
					tokens += 1200 // flat per-block estimate
				case "tool_use":
					if name, _ := jsonparser.GetString(block, "name"); name != "" {
						tokens += anthropicTextTokenEstimate(name)
					}
					if inputRaw, inputType, _, inputErr := jsonparser.Get(block, "input"); inputErr == nil &&
						(inputType == jsonparser.Object || inputType == jsonparser.Array) {
						tokens += anthropicTextTokenEstimate(string(inputRaw))
					}
				case "tool_result":
					tokens += anthropicTextTokenEstimate(extractAnthropicToolResultText(block))
				}
			})
		}
	})

	// Tool definitions: overhead + name + description + schema.
	toolsVal, toolsType, _, toolsErr := jsonparser.Get(body, "tools")
	if toolsErr == nil && toolsType == jsonparser.Array {
		jsonparser.ArrayEach(toolsVal, func(tool []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			tokens += 12 // per-tool definition overhead
			if name, _ := jsonparser.GetString(tool, "name"); name != "" {
				tokens += anthropicTextTokenEstimate(name)
			}
			if desc, _ := jsonparser.GetString(tool, "description"); desc != "" {
				tokens += anthropicTextTokenEstimate(desc)
			}
			if schemaVal, schemaType, _, schemaErr := jsonparser.Get(tool, "input_schema"); schemaErr == nil && schemaType == jsonparser.Object {
				tokens += anthropicTextTokenEstimate(string(schemaVal))
			}
		})
	}

	if tokens < 1 {
		tokens = 1 // Anthropic guarantees at least 1 token for non-empty messages
	}
	return tokens, nil
}

// anthropicTextTokenEstimate applies the ~4-characters-per-token heuristic:
// (charCount + 3) / 4, counting Unicode code points (so CJK text, which is
// roughly one token per character, is not undercounted).
func anthropicTextTokenEstimate(s string) int {
	if s == "" {
		return 0
	}
	return (utf8.RuneCountInString(s) + 3) / 4
}

// poolAllNativeAnthropic reports whether the pool serves only Anthropic-protocol
// credentials, in which case the OpenAI pre-translation can be skipped entirely
// (the zero-copy native path).
func poolAllNativeAnthropic(pool *credentials.BalancedChannelPool) bool {
	if pool == nil || len(pool.Credentials) == 0 {
		return false
	}
	for _, cred := range pool.Credentials {
		if !isNativeAnthropicProvider(cred.Provider) {
			return false
		}
	}
	return true
}

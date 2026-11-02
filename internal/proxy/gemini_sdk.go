package proxy

import (
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"time"

	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/transmux"
	"go.uber.org/zap"
	"google.golang.org/genai"
	"google.golang.org/grpc/codes"
	"google.golang.org/grpc/status"
)

// ─── Client Factory ───────────────────────────────────────────────────────────

// newGeminiSDKClient creates a transient genai.Client for a single API key.
// NewClient performs no network I/O, so calling it inside executeWithRetry is safe.
func newGeminiSDKClient(ctx context.Context, apiKey string) (*genai.Client, error) {
	return genai.NewClient(ctx, &genai.ClientConfig{
		APIKey:  apiKey,
		Backend: genai.BackendGeminiAPI,
	})
}

// ─── Entry Point ──────────────────────────────────────────────────────────────

// executeGeminiWithSDK replaces the raw HTTP transport for cred.Provider == "gemini".
// It re-parses the original OpenAI body, builds SDK-typed Content structs, calls the
// SDK, and translates the response back to OpenAI format.
//
// Returns (statusCode, errBody, err) to match forwardRequest's conventions:
//   - statusCode: HTTP status to report to the retry loop
//   - errBody:    response bytes (OpenAI JSON); for 2xx stream requests this is
//     a telemetry summary blob, not the actual HTTP body (already written to c.Writer)
//   - err:        non-nil only for panics; transport errors are encoded in statusCode
func (h *Handler) executeGeminiWithSDK(
	c *gin.Context,
	pctx *proxyContext,
	apiKey string,
	openAIBody []byte,
) (statusCode int, errBody []byte, err error) {
	ctx := c.Request.Context()

	client, clientErr := newGeminiSDKClient(ctx, apiKey)
	if clientErr != nil {
		return http.StatusInternalServerError, nil,
			fmt.Errorf("gemini sdk client init: %w", clientErr)
	}

	// Re-parse the original OpenAI body to build typed SDK Content objects.
	// pctx.body holds the body after Handle() stripped the "gemini/" prefix from the
	// model field; the message content is unmodified.
	var req openAIRequest
	if jsonErr := json.Unmarshal(openAIBody, &req); jsonErr != nil {
		return http.StatusBadRequest, nil,
			fmt.Errorf("failed to parse openai body for sdk: %w", jsonErr)
	}

	contents, sysInstruction, config, buildErr := buildSDKRequest(req, pctx.model)
	if buildErr != nil {
		return http.StatusBadRequest, nil, buildErr
	}
	if sysInstruction != nil {
		config.SystemInstruction = sysInstruction
	}

	if pctx.isStream {
		return h.geminiSDKStream(c, client, pctx.model, contents, config, pctx)
	}
	return h.geminiSDKNonStream(c, client, pctx.model, contents, config, pctx)
}

// ─── Request Builder ──────────────────────────────────────────────────────────

// buildSDKRequest maps an openAIRequest to SDK-typed Content slices and a
// GenerateContentConfig, re-using the same helper functions that
// transpileOpenAIToGemini uses (extractTextContent, coerceToJSONObject, etc.).
func buildSDKRequest(req openAIRequest, model string) (
	contents []*genai.Content,
	systemInstruction *genai.Content,
	config *genai.GenerateContentConfig,
	err error,
) {
	injectThoughtSig := isGemini3Model(model)
	// Maps tool_call_id → function name for resolving tool result names.
	toolCallNames := make(map[string]string)

	// ── System instruction ────────────────────────────────────────────────────
	var sysParts []*genai.Part
	for _, msg := range req.Messages {
		if msg.Role == "system" || msg.Role == "developer" {
			if text := extractTextContent(msg.Content); text != "" {
				sysParts = append(sysParts, &genai.Part{Text: text})
			}
		}
	}
	if len(sysParts) > 0 {
		systemInstruction = &genai.Content{Parts: sysParts}
	}

	// ── Conversation turns ────────────────────────────────────────────────────
	var prevRole string
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "developer":
			continue // already handled above

		case "user":
			parts, pErr := buildSDKPartsFromContent(msg.Content)
			if pErr != nil {
				return nil, nil, nil, pErr
			}
			// Merge consecutive user turns (Gemini rejects same-role adjacency)
			if prevRole == "user" && len(contents) > 0 {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, parts...)
			} else {
				contents = append(contents, &genai.Content{Role: genai.RoleUser, Parts: parts})
			}
			prevRole = "user"

		case "assistant":
			for _, tc := range msg.ToolCalls {
				if tc.ID != "" && tc.Function.Name != "" {
					toolCallNames[tc.ID] = tc.Function.Name
				}
			}
			parts, pErr := buildSDKAssistantParts(msg, injectThoughtSig)
			if pErr != nil {
				return nil, nil, nil, pErr
			}
			if len(parts) > 0 {
				contents = append(contents, &genai.Content{Role: genai.RoleModel, Parts: parts})
				prevRole = "model"
			}

		case "tool":
			funcName := msg.Name
			if funcName == "" {
				if name, ok := toolCallNames[msg.ToolCallID]; ok {
					funcName = name
				} else {
					funcName = "function_result"
				}
			}
			// Gemini requires functionResponse.response to be a JSON object.
			resultJSON := coerceToJSONObject(msg.Content)
			var responseMap map[string]any
			if unmarshalErr := json.Unmarshal(resultJSON, &responseMap); unmarshalErr != nil || responseMap == nil {
				responseMap = map[string]any{"output": string(resultJSON)}
			}
			part := &genai.Part{
				FunctionResponse: &genai.FunctionResponse{
					Name:     funcName,
					Response: responseMap,
				},
			}
			// Group consecutive tool results into a single user turn.
			if prevRole == "tool" && len(contents) > 0 {
				contents[len(contents)-1].Parts = append(contents[len(contents)-1].Parts, part)
			} else {
				contents = append(contents, &genai.Content{
					Role:  genai.RoleUser,
					Parts: []*genai.Part{part},
				})
			}
			prevRole = "tool"
		}
	}

	// ── GenerateContentConfig ─────────────────────────────────────────────────
	config = &genai.GenerateContentConfig{
		SafetySettings: sdkDefaultSafetySettings(),
	}

	if req.Temperature != nil {
		v := float32(*req.Temperature)
		config.Temperature = &v
	}
	if req.TopP != nil {
		v := float32(*req.TopP)
		config.TopP = &v
	}
	// max_completion_tokens takes precedence (newer OpenAI spec)
	if req.MaxCompletionTokens != nil {
		config.MaxOutputTokens = int32(*req.MaxCompletionTokens)
	} else if req.MaxTokens != nil {
		config.MaxOutputTokens = int32(*req.MaxTokens)
	}

	// Stop sequences (string or array)
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var stopStr string
		var stopArr []string
		if jsonErr := json.Unmarshal(req.Stop, &stopStr); jsonErr == nil {
			config.StopSequences = []string{stopStr}
		} else if jsonErr2 := json.Unmarshal(req.Stop, &stopArr); jsonErr2 == nil {
			config.StopSequences = stopArr
		}
	}

	// JSON mode / structured outputs
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object":
			config.ResponseMIMEType = "application/json"
		case "json_schema":
			config.ResponseMIMEType = "application/json"
			if schemaRaw, _, _, sErr := jsonparser.Get(req.ResponseFormat.JSONSchema, "schema"); sErr == nil && len(schemaRaw) > 0 {
				config.ResponseJsonSchema = sanitizeGeminiToolSchema(schemaRaw)
			} else {
				config.ResponseJsonSchema = sanitizeGeminiToolSchema(req.ResponseFormat.JSONSchema)
			}
		}
	}

	// Tools — convert to SDK FunctionDeclarations
	if len(req.Tools) > 0 {
		var decls []*genai.FunctionDeclaration
		for _, t := range req.Tools {
			if t.Type != "function" {
				continue
			}
			decl := &genai.FunctionDeclaration{
				Name:        t.Function.Name,
				Description: t.Function.Description,
			}
			if len(t.Function.Parameters) > 0 {
				// Use ParametersJsonSchema — accepts raw JSON and avoids
				// genai.Schema struct mapping complexity.
				decl.ParametersJsonSchema = sanitizeGeminiToolSchema(t.Function.Parameters)
			}
			decls = append(decls, decl)
		}
		if len(decls) > 0 {
			config.Tools = []*genai.Tool{{FunctionDeclarations: decls}}
		}
		// tool_choice → ToolConfig
		if len(req.ToolChoice) > 0 {
			choiceStr := strings.Trim(string(req.ToolChoice), `"`)
			var mode genai.FunctionCallingConfigMode
			var allowedFns []string
			switch choiceStr {
			case "none":
				mode = genai.FunctionCallingConfigModeNone
			case "required", "any":
				mode = genai.FunctionCallingConfigModeAny
			case "auto":
				mode = genai.FunctionCallingConfigModeAuto
			default:
				if name, fnErr := jsonparser.GetString(req.ToolChoice, "function", "name"); fnErr == nil && name != "" {
					mode = genai.FunctionCallingConfigModeAny
					allowedFns = []string{name}
				} else {
					mode = genai.FunctionCallingConfigModeAuto
				}
			}
			config.ToolConfig = &genai.ToolConfig{
				FunctionCallingConfig: &genai.FunctionCallingConfig{
					Mode:                 mode,
					AllowedFunctionNames: allowedFns,
				},
			}
		}
	}

	// Thinking / reasoning budget
	if req.ThinkingBudget > 0 {
		budget := int32(req.ThinkingBudget)
		config.ThinkingConfig = &genai.ThinkingConfig{
			ThinkingBudget:  &budget,
			IncludeThoughts: true,
		}
	} else if req.ReasoningEffort != "" {
		budgetInt := geminiReasoningBudget(req.ReasoningEffort)
		if budgetInt != 0 {
			budget := int32(budgetInt)
			config.ThinkingConfig = &genai.ThinkingConfig{
				ThinkingBudget:  &budget,
				IncludeThoughts: true,
			}
		}
	}

	return contents, systemInstruction, config, nil
}

// buildSDKPartsFromContent converts an OpenAI message content (string or []contentPart)
// into a slice of *genai.Part, handling text and image_url parts.
func buildSDKPartsFromContent(raw json.RawMessage) ([]*genai.Part, error) {
	if len(raw) == 0 {
		return []*genai.Part{{Text: ""}}, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []*genai.Part{{Text: s}}, nil
	}
	var parts []openAIContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return []*genai.Part{{Text: string(raw)}}, nil
	}
	var gemParts []*genai.Part
	for _, p := range parts {
		switch p.Type {
		case "text":
			gemParts = append(gemParts, &genai.Part{Text: p.Text})
		case "image_url":
			if p.ImageURL == nil {
				continue
			}
			gp, err := convertImageURLToSDK(p.ImageURL.URL)
			if err != nil {
				return nil, err
			}
			gemParts = append(gemParts, gp)
		}
	}
	if len(gemParts) == 0 {
		gemParts = []*genai.Part{{Text: ""}}
	}
	return gemParts, nil
}

// convertImageURLToSDK converts an OpenAI image_url value into a *genai.Part.
// data: URIs → InlineData (raw binary bytes in genai.Blob.Data, which json.Marshal base64-encodes).
// http(s):// URLs → fetched in-memory and converted to InlineData.
func convertImageURLToSDK(url string) (*genai.Part, error) {
	if strings.HasPrefix(url, "data:") {
		mimeType, rawBytes, _, err := ExtractBase64FromDataURI(url)
		if err != nil {
			return nil, err
		}
		return &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: mimeType,
				Data:     rawBytes,
			},
		}, nil
	}

	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		mimeType, _, dataURI, err := FetchAndEncodeImage(context.Background(), url)
		if err != nil {
			return nil, fmt.Errorf("failed to fetch remote image for gemini sdk: %w", err)
		}
		_, rawBytes, _, err := ExtractBase64FromDataURI(dataURI)
		if err != nil {
			return nil, fmt.Errorf("failed to decode fetched image for gemini sdk: %w", err)
		}
		return &genai.Part{
			InlineData: &genai.Blob{
				MIMEType: mimeType,
				Data:     rawBytes,
			},
		}, nil
	}

	return nil, fmt.Errorf("unsupported image URL format: must be data URI or http(s) URL")
}

// buildSDKAssistantParts converts an OpenAI assistant message (text content and/or
// tool_calls) into a slice of *genai.Part.
//
// When injectThoughtSig is true (Gemini 3 models), every FunctionCall part receives
// the thought_signature bypass sentinel. ThoughtSignature lives on the Part itself
// as []byte, so we convert the sentinel string to bytes.
func buildSDKAssistantParts(msg openAIMessage, injectThoughtSig bool) ([]*genai.Part, error) {
	var parts []*genai.Part
	text := extractTextContent(msg.Content)
	if text != "" {
		parts = append(parts, &genai.Part{Text: text})
	}
	for _, tc := range msg.ToolCalls {
		if tc.Type != "function" {
			continue
		}
		// Gemini requires functionCall.args to be a JSON object.
		argsRaw := coerceToJSONObject(json.RawMessage(tc.Function.Arguments))
		var argsMap map[string]any
		if jsonErr := json.Unmarshal(argsRaw, &argsMap); jsonErr != nil || argsMap == nil {
			argsMap = map[string]any{}
		}
		part := &genai.Part{
			FunctionCall: &genai.FunctionCall{
				Name: tc.Function.Name,
				Args: argsMap,
			},
		}
		// Inject the bypass sentinel on the Part for Gemini 3.
		// ThoughtSignature is []byte on genai.Part (field added in SDK for Gemini 3+).
		if injectThoughtSig {
			part.ThoughtSignature = []byte(geminiThoughtSignatureBypass)
		}
		parts = append(parts, part)
	}
	return parts, nil
}

// sdkDefaultSafetySettings returns the SDK equivalent of defaultGeminiSafetySettings.
func sdkDefaultSafetySettings() []*genai.SafetySetting {
	return []*genai.SafetySetting{
		{Category: genai.HarmCategoryHateSpeech, Threshold: genai.HarmBlockThresholdBlockNone},
		{Category: genai.HarmCategoryDangerousContent, Threshold: genai.HarmBlockThresholdBlockNone},
		{Category: genai.HarmCategorySexuallyExplicit, Threshold: genai.HarmBlockThresholdBlockNone},
		{Category: genai.HarmCategoryHarassment, Threshold: genai.HarmBlockThresholdBlockNone},
	}
}

// ─── Non-Streaming Response ───────────────────────────────────────────────────

func (h *Handler) geminiSDKNonStream(
	c *gin.Context,
	client *genai.Client,
	model string,
	contents []*genai.Content,
	config *genai.GenerateContentConfig,
	pctx *proxyContext,
) (int, []byte, error) {
	resp, err := client.Models.GenerateContent(c.Request.Context(), model, contents, config)
	if err != nil {
		status, body, _ := mapSDKError(err)
		return status, body, nil
	}

	// In-Flight Content Validation: Reject responses with empty candidates, nil text, and no function calls
	hasContent := false
	if resp != nil && len(resp.Candidates) > 0 && resp.Candidates[0].Content != nil {
		for _, part := range resp.Candidates[0].Content.Parts {
			if part.Text != "" || part.FunctionCall != nil {
				hasContent = true
				break
			}
		}
	}
	if !hasContent {
		h.logger.Warn("gemini sdk non-stream empty content response caught, triggering failover",
			zap.String("model", pctx.model),
			zap.Int("credential_id", pctx.credential.Credential.ID),
		)
		// Return 502 Bad Gateway to let executeWithRetry rotate keys
		return http.StatusBadGateway, []byte(`{"error":{"message":"empty or filtered response from upstream","type":"api_error"}}`), nil
	}

	translated, trErr := translateSDKResponse(resp, pctx.requestedModel)
	if trErr != nil {
		h.logger.Warn("gemini sdk non-stream response translation failed",
			zap.String("model", pctx.model),
			zap.Error(trErr),
		)
		return http.StatusInternalServerError, nil, trErr
	}

	translated = h.normalizeNonStreamingReasoning(translated)
	translated = h.rewriteResponseModel(translated, pctx.requestedModel)

	c.Writer.Header().Set("Content-Type", "application/json")
	c.Writer.Header().Set("X-Gateway-Provider", "gemini")
	c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
	c.Writer.WriteHeader(http.StatusOK)
	c.Writer.Write(translated) //nolint:errcheck
	return http.StatusOK, translated, nil
}

// translateSDKResponse converts a *genai.GenerateContentResponse into an OpenAI
// chat completion JSON body. It is the SDK-typed equivalent of translateGeminiResponse
// in gemini.go, but reads from strongly-typed struct fields.
func translateSDKResponse(resp *genai.GenerateContentResponse, requestedModel string) ([]byte, error) {
	if resp == nil || len(resp.Candidates) == 0 {
		return nil, fmt.Errorf("empty gemini sdk response")
	}
	cand := resp.Candidates[0]
	finishReason := mapGeminiFinishReasonNS(string(cand.FinishReason))

	var textContent strings.Builder
	var reasoningContent strings.Builder
	var toolCalls []openAIToolCallResp

	if cand.Content != nil {
		for _, part := range cand.Content.Parts {
			if part.FunctionCall != nil {
				argsJSON, _ := json.Marshal(part.FunctionCall.Args)
				toolCalls = append(toolCalls, openAIToolCallResp{
					ID:   fmt.Sprintf("call_%d", len(toolCalls)),
					Type: "function",
					Function: openAIFuncCallResp{
						Name:      part.FunctionCall.Name,
						Arguments: string(argsJSON),
					},
				})
				continue
			}
			if part.Text != "" {
				if part.Thought {
					reasoningContent.WriteString(part.Text)
				} else {
					textContent.WriteString(part.Text)
				}
			}
		}
	}

	// Token usage
	var promptTokens, candidateTokens, thoughtsTokens int64
	if resp.UsageMetadata != nil {
		promptTokens = int64(resp.UsageMetadata.PromptTokenCount)
		candidateTokens = int64(resp.UsageMetadata.CandidatesTokenCount)
		thoughtsTokens = int64(resp.UsageMetadata.ThoughtsTokenCount)
	}
	totalTokens := promptTokens + candidateTokens + thoughtsTokens

	type oaMessage struct {
		Role             string               `json:"role"`
		Content          string               `json:"content"`
		ReasoningContent string               `json:"reasoning_content,omitempty"`
		ToolCalls        []openAIToolCallResp `json:"tool_calls,omitempty"`
	}
	type oaChoice struct {
		Index        int       `json:"index"`
		Message      oaMessage `json:"message"`
		FinishReason string    `json:"finish_reason"`
	}
	type oaUsage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	}
	type oaCompletion struct {
		ID      string     `json:"id"`
		Object  string     `json:"object"`
		Created int64      `json:"created"`
		Model   string     `json:"model"`
		Choices []oaChoice `json:"choices"`
		Usage   oaUsage    `json:"usage"`
	}

	msg := oaMessage{
		Role:    "assistant",
		Content: textContent.String(),
	}
	if reasoningContent.Len() > 0 {
		msg.ReasoningContent = reasoningContent.String()
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		if finishReason == "stop" {
			finishReason = "tool_calls"
		}
	}

	result := oaCompletion{
		ID:      "chatcmpl-gate",
		Object:  "chat.completion",
		Created: time.Now().Unix(),
		Model:   requestedModel,
		Choices: []oaChoice{{Index: 0, Message: msg, FinishReason: finishReason}},
		Usage: oaUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: candidateTokens + thoughtsTokens,
			TotalTokens:      totalTokens,
		},
	}
	return json.Marshal(result)
}

// ─── Streaming Response ───────────────────────────────────────────────────────

func (h *Handler) geminiSDKStream(
	c *gin.Context,
	client *genai.Client,
	model string,
	contents []*genai.Content,
	config *genai.GenerateContentConfig,
	pctx *proxyContext,
) (int, []byte, error) {
	flusher, hasFlusher := c.Writer.(http.Flusher)

	var responseBuilder strings.Builder
	var tokenEstimate int
	// Re-use the existing GeminiTransmuxer to translate SDK chunks to SSE format.
	// Serializing each SDK chunk to Gemini REST JSON and then passing through
	// TranslateChunk reuses all existing multi-part/tool-call/reasoning logic.
	tmx := transmux.NewGeminiTransmuxer()
	defer tmx.Close()

	var anthropicEnc *transmux.AnthropicStreamEncoder
	if pctx.anthropicFront {
		msgID := "msg_gate_" + strconv.FormatInt(time.Now().UnixNano(), 36)
		anthropicEnc = transmux.NewAnthropicStreamEncoder(msgID, pctx.requestedModel, pctx.anthropicInputTokens)
	}

	maxFailoverAttempts := 5
	attempt := 0
	emittedText := ""
	currentClient := client

	headersWritten := false
	writeHeaders := func() {
		if headersWritten {
			return
		}
		c.Writer.Header().Set("Content-Type", "text/event-stream")
		c.Writer.Header().Set("Cache-Control", "no-cache")
		c.Writer.Header().Set("Connection", "keep-alive")
		c.Writer.Header().Set("Transfer-Encoding", "chunked")
		c.Writer.Header().Set("X-Accel-Buffering", "no")
		c.Writer.Header().Set("X-Gateway-Provider", "gemini")
		c.Writer.Header().Set("X-Gateway-Model-Pattern", pctx.model)
		c.Writer.WriteHeader(http.StatusOK)
		headersWritten = true
	}

	hasEmittedValidData := false

	for attempt < maxFailoverAttempts {
		attempt++

		// Prepare contents for this run. Gemini requires the prompt history to alternate
		// and strictly end on a USER turn. We append the partial generated output as a MODEL turn,
		// followed by a USER turn steering continuation.
		var runContents []*genai.Content
		if emittedText != "" {
			runContents = make([]*genai.Content, len(contents))
			copy(runContents, contents)

			// Edge Case Protection: If the last message is already a model turn, merge the text
			if len(runContents) > 0 && (runContents[len(runContents)-1].Role == genai.RoleModel || runContents[len(runContents)-1].Role == "model") {
				originalLast := runContents[len(runContents)-1]
				clonedLast := &genai.Content{
					Role: originalLast.Role,
				}
				if len(originalLast.Parts) > 0 {
					clonedLast.Parts = make([]*genai.Part, len(originalLast.Parts))
					copy(clonedLast.Parts, originalLast.Parts)
					// Merge emittedText into the last part
					lastPartIdx := len(clonedLast.Parts) - 1
					origPart := clonedLast.Parts[lastPartIdx]
					clonedLast.Parts[lastPartIdx] = &genai.Part{
						Text:             origPart.Text + emittedText,
						InlineData:       origPart.InlineData,
						FunctionCall:     origPart.FunctionCall,
						FunctionResponse: origPart.FunctionResponse,
						FileData:         origPart.FileData,
					}
				} else {
					clonedLast.Parts = []*genai.Part{{Text: emittedText}}
				}
				runContents[len(runContents)-1] = clonedLast
			} else {
				// Otherwise, append a clean model turn
				runContents = append(runContents, &genai.Content{
					Role:  genai.RoleModel,
					Parts: []*genai.Part{{Text: emittedText}},
				})
			}

			// Append the required user instruction turn to force continuation
			runContents = append(runContents, &genai.Content{
				Role: genai.RoleUser,
				Parts: []*genai.Part{
					{
						Text: "[SYSTEM CONTEXT: Your previous output was cut off mid-generation due to a network dropout. Continue generating the response precisely from the last character above. Do NOT output any introductory remarks, do NOT repeat any code, and do NOT wrap code inside markdown blocks. Resume the code output directly.]",
					},
				},
			})
		} else {
			runContents = contents
		}

		// Initialize lookahead sanitizer only for failover retry attempts
		sanitizer := NewStreamSanitizer(emittedText != "")

		stream := currentClient.Models.GenerateContentStream(c.Request.Context(), model, runContents, config)

		var streamErr error
		for chunk, err := range stream {
			if err != nil {
				streamErr = err
				break
			}

			// Pre-emptive safety/filter block check using stringified FinishReason from SDK
			if len(chunk.Candidates) > 0 {
				reasonRaw := strings.ToUpper(string(chunk.Candidates[0].FinishReason))
				if reasonRaw != "" && reasonRaw != "STOP" && reasonRaw != "MAX_TOKENS" && reasonRaw != "STOP_SEQUENCE" {
					if !hasEmittedValidData {
						streamErr = fmt.Errorf("upstream generation dropped early by content filter: %s", reasonRaw)
						break
					}
				}
			}

			chunkJSON, marshalErr := json.Marshal(chunk)
			if marshalErr != nil {
				h.logger.Debug("gemini sdk chunk marshal error", zap.Error(marshalErr))
				continue
			}

			translated, tmxErr := tmx.TranslateChunk(chunkJSON)
			if tmxErr != nil || len(translated) == 0 {
				continue
			}

			// Pre-emptive safety/filter block interceptor
			if finishReason, fErr := jsonparser.GetString(translated, "choices", "[0]", "finish_reason"); fErr == nil {
				if (finishReason == "content_filter" || finishReason == "error") && !hasEmittedValidData {
					streamErr = fmt.Errorf("upstream generation intercepted by content filter block")
					break
				}
			}
			if finishReason, fErr := jsonparser.GetString(translated, "choices", "[0]", "delta", "finish_reason"); fErr == nil {
				if (finishReason == "content_filter" || finishReason == "error") && !hasEmittedValidData {
					streamErr = fmt.Errorf("upstream generation intercepted by content filter block")
					break
				}
			}

			// Rewrite the model field to match the requested model name.
			if pctx.requestedModel != "" {
				if _, mErr := jsonparser.GetString(translated, "model"); mErr == nil {
					if updated, sErr := jsonparser.Set(translated, []byte(`"`+pctx.requestedModel+`"`), "model"); sErr == nil {
						translated = updated
					}
				}
			}

			// Accumulate content and apply leading formatting cleanup if sanitization is active
			var cleanText string
			if content, cErr := jsonparser.GetString(translated, "choices", "[0]", "delta", "content"); cErr == nil {
				cleanText = sanitizer.Sanitize(content)
				if cleanText != content {
					if updated, sErr := jsonparser.Set(translated, []byte(`"`+cleanText+`"`), "choices", "[0]", "delta", "content"); sErr == nil {
						translated = updated
					}
				}
				if cleanText != "" {
					responseBuilder.WriteString(cleanText)
					emittedText += cleanText
					tokenEstimate++
					hasEmittedValidData = true
				}
			} else if reasoning, rErr := jsonparser.GetString(translated, "choices", "[0]", "delta", "reasoning_content"); rErr == nil {
				cleanText = sanitizer.Sanitize(reasoning)
				if cleanText != reasoning {
					if updated, sErr := jsonparser.Set(translated, []byte(`"`+cleanText+`"`), "choices", "[0]", "delta", "reasoning_content"); sErr == nil {
						translated = updated
					}
				}
				if cleanText != "" {
					responseBuilder.WriteString(cleanText)
					emittedText += cleanText
					tokenEstimate++
					hasEmittedValidData = true
				}
			} else {
				cleanText = "PASSTHROUGH" // Non-text deltas (metadata, tool calls) bypass buffering
				// Mark as valid data if this chunk contains function/tool calls
				if _, _, _, tErr := jsonparser.Get(translated, "choices", "[0]", "delta", "tool_calls"); tErr == nil {
					hasEmittedValidData = true
				}
			}

			// If sanitizer is still buffering leading characters to look for markdown blocks, skip outputting
			if cleanText == "" {
				continue
			}

			writeHeaders()

			if pctx.anthropicFront && anthropicEnc != nil {
				anthBytes := anthropicEnc.EncodeOpenAIChunk(translated)
				if len(anthBytes) > 0 {
					c.Writer.Write(anthBytes) //nolint:errcheck
					if hasFlusher {
						flusher.Flush()
					}
				}
			} else {
				c.Writer.Write([]byte("data: ")) //nolint:errcheck
				c.Writer.Write(translated)       //nolint:errcheck
				c.Writer.Write([]byte("\n\n"))   //nolint:errcheck
				if hasFlusher {
					flusher.Flush()
				}
			}
		}

		// Catch success exit with empty generated body
		if streamErr == nil && !hasEmittedValidData && sanitizer.buffer == "" {
			streamErr = fmt.Errorf("empty streaming body received from upstream")
		}

		if streamErr == nil {
			// Stream completed successfully!
			if pctx.anthropicFront && anthropicEnc != nil {
				finalBytes := anthropicEnc.Finalize()
				if len(finalBytes) > 0 {
					c.Writer.Write(finalBytes) //nolint:errcheck
					if hasFlusher {
						flusher.Flush()
					}
				}
			} else {
				// Flush any remaining buffered text from the sanitizer (e.g. short streams)
				if flushedText := sanitizer.Flush(); flushedText != "" {
					responseBuilder.WriteString(flushedText)
					emittedText += flushedText
					tokenEstimate++
					hasEmittedValidData = true

					writeHeaders()

					modelName := pctx.requestedModel
					if modelName == "" {
						modelName = pctx.model
					}
					fakeChunk := fmt.Sprintf(`{"id":"chatcmpl-gate","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"content":%q}}],"model":%q}`, flushedText, modelName)
					c.Writer.Write([]byte("data: "))  //nolint:errcheck
					c.Writer.Write([]byte(fakeChunk)) //nolint:errcheck
					c.Writer.Write([]byte("\n\n"))    //nolint:errcheck
					if hasFlusher {
						flusher.Flush()
					}
				}
			}
			break
		}

		// Mid-stream error encountered (e.g. 429, safety filter, connection close, empty output)
		h.logger.Warn("mid-stream failure caught, initiating key failover protocol",
			zap.String("model", pctx.model),
			zap.Int("credential_id", pctx.credential.Credential.ID),
			zap.Int("emitted_len", len(emittedText)),
			zap.Error(streamErr),
		)

		// Track cluster-wide failover spike
		if h.AlertManager != nil {
			_ = h.AlertManager.TrackFailover(c.Request.Context(), pctx.credential.Credential.ID, pctx.model, streamErr.Error())
		}

		// Cooldown the failed key locally
		cooldownDuration := 30 * time.Second
		pctx.pool.PenalizeToken(pctx.credential.Index, cooldownDuration)
		h.broadcaster.PublishPenalize(pctx.pool.ModelPattern, pctx.credential.Credential.ID, pctx.credential.Index, time.Now().Add(cooldownDuration))

		// Isolate the token in Redis to notify other cluster nodes
		rdb := h.broadcaster.RDB()
		if rdb != nil {
			cooldownKey := fmt.Sprintf("gate:key:%d:cooldown", pctx.credential.Credential.ID)
			rdb.Set(c.Request.Context(), cooldownKey, "rate_limited_backoff", cooldownDuration)
		}

		// Acquire next available token from pool, skipping those that are rate limited in Redis
		var nextCred *credentials.AcquireResult
		for i := 0; i < int(pctx.pool.TotalCount); i++ {
			cand := pctx.pool.AcquireActiveToken()
			if cand == nil {
				cand = pctx.pool.AcquireLeastPenalizedToken()
			}
			if cand == nil {
				break
			}
			if h.isKeyRateLimitedInRedis(c.Request.Context(), cand.Credential) {
				pctx.pool.PenalizeToken(cand.Index, 5*time.Second)
				continue
			}
			nextCred = cand
			break
		}

		if nextCred == nil {
			h.logger.Error("failover failed: all pool tokens exhausted", zap.String("model", pctx.model))
			break
		}

		pctx.credential = nextCred

		// Setup new SDK client with target key
		newClient, clientErr := newGeminiSDKClient(c.Request.Context(), nextCred.Credential.APIKey)
		if clientErr != nil {
			h.logger.Error("failover client init failed", zap.Error(clientErr))
			break
		}
		currentClient = newClient

		// Small delay to allow connections to settle before resuming
		time.Sleep(200 * time.Millisecond)
	}

	if !headersWritten {
		// All attempts failed without writing any headers or content
		type oaErr struct {
			Message string `json:"message"`
			Type    string `json:"type"`
		}
		type oaEnvelope struct {
			Error oaErr `json:"error"`
		}
		body, _ := json.Marshal(oaEnvelope{Error: oaErr{Message: "Gemini pool exhausted without generating valid content", Type: "api_error"}})
		return http.StatusBadGateway, body, nil
	}

	if !pctx.anthropicFront {
		c.Writer.Write([]byte("data: [DONE]\n\n")) //nolint:errcheck
		if hasFlusher {
			flusher.Flush()
		}
	}

	type streamResult struct {
		Text   string `json:"text"`
		Tokens int    `json:"tokens"`
	}
	resJSON, _ := json.Marshal(streamResult{Text: responseBuilder.String(), Tokens: tokenEstimate})
	return http.StatusOK, resJSON, nil
}

// ─── Error Mapping ────────────────────────────────────────────────────────────

// mapSDKError translates a genai SDK error into (httpStatus, OpenAI error body, nil).
// The SDK wraps gRPC status strings in the error message. The mapping mirrors
// normalizeGeminiError in gemini.go so the retry loop sees consistent status codes.
func mapSDKError(err error) (int, []byte, error) {
	if err == nil {
		return http.StatusOK, nil, nil
	}

	var httpStatus int
	var oaType, oaCode string

	s, ok := status.FromError(err)
	if ok {
		switch s.Code() {
		case codes.Unauthenticated:
			httpStatus, oaType, oaCode = http.StatusUnauthorized, "authentication_error", "invalid_api_key"
		case codes.PermissionDenied:
			httpStatus, oaType, oaCode = http.StatusForbidden, "permission_error", "permission_denied"
		case codes.NotFound:
			httpStatus, oaType, oaCode = http.StatusNotFound, "invalid_request_error", "model_not_found"
		case codes.ResourceExhausted:
			msg := s.Message()
			if strings.Contains(strings.ToLower(msg), "quota") {
				oaCode = "quota_exceeded"
			} else {
				oaCode = "rate_limit_exceeded"
			}
			httpStatus, oaType = http.StatusTooManyRequests, "rate_limit_error"
		case codes.InvalidArgument:
			httpStatus, oaType, oaCode = http.StatusBadRequest, "invalid_request_error", "invalid_request"
		case codes.Unavailable:
			httpStatus, oaType, oaCode = http.StatusServiceUnavailable, "api_error", "service_unavailable"
		case codes.DeadlineExceeded:
			httpStatus, oaType, oaCode = http.StatusGatewayTimeout, "api_error", "timeout"
		case codes.Internal:
			httpStatus, oaType, oaCode = http.StatusBadGateway, "api_error", "internal_error"
		default:
			httpStatus, oaType, oaCode = http.StatusBadGateway, "api_error", "upstream_error"
		}
	} else {
		// Fallback to substring matching on raw error message (handles errors.New / tests)
		msg := err.Error()
		upperMsg := strings.ToUpper(msg)
		switch {
		case strings.Contains(upperMsg, "UNAUTHENTICATED") || strings.Contains(upperMsg, "401"):
			httpStatus, oaType, oaCode = http.StatusUnauthorized, "authentication_error", "invalid_api_key"
		case strings.Contains(upperMsg, "PERMISSION") || strings.Contains(upperMsg, "403"):
			httpStatus, oaType, oaCode = http.StatusForbidden, "permission_error", "permission_denied"
		case strings.Contains(upperMsg, "NOT_FOUND") || strings.Contains(upperMsg, "NOTFOUND") || strings.Contains(upperMsg, "404"):
			httpStatus, oaType, oaCode = http.StatusNotFound, "invalid_request_error", "model_not_found"
		case strings.Contains(upperMsg, "EXHAUSTED") || strings.Contains(upperMsg, "429") || strings.Contains(upperMsg, "LIMIT"):
			if strings.Contains(strings.ToLower(msg), "quota") {
				oaCode = "quota_exceeded"
			} else {
				oaCode = "rate_limit_exceeded"
			}
			httpStatus, oaType = http.StatusTooManyRequests, "rate_limit_error"
		case strings.Contains(upperMsg, "INVALID") || strings.Contains(upperMsg, "400"):
			httpStatus, oaType, oaCode = http.StatusBadRequest, "invalid_request_error", "invalid_request"
		case strings.Contains(upperMsg, "UNAVAILABLE") || strings.Contains(upperMsg, "503"):
			httpStatus, oaType, oaCode = http.StatusServiceUnavailable, "api_error", "service_unavailable"
		case strings.Contains(upperMsg, "TIMEOUT") || strings.Contains(upperMsg, "DEADLINE"):
			httpStatus, oaType, oaCode = http.StatusGatewayTimeout, "api_error", "timeout"
		case strings.Contains(upperMsg, "INTERNAL") || strings.Contains(upperMsg, "500"):
			httpStatus, oaType, oaCode = http.StatusBadGateway, "api_error", "internal_error"
		default:
			httpStatus, oaType, oaCode = http.StatusBadGateway, "api_error", "upstream_error"
		}
	}

	type oaErr struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	}
	type oaEnvelope struct {
		Error oaErr `json:"error"`
	}
	body, _ := json.Marshal(oaEnvelope{Error: oaErr{Message: err.Error(), Type: oaType, Code: oaCode}})
	return httpStatus, body, nil
}

// isKeyRateLimitedInRedis checks both:
// 1. If the key has an active Redis cooldown key ("gate:key:<id>:cooldown")
// 2. If the request count in the current minute window exceeds 15 RPM
func (h *Handler) isKeyRateLimitedInRedis(ctx context.Context, cred *credentials.RuntimeCredential) bool {
	if h.broadcaster == nil {
		return false // Redis not configured — fail open
	}
	rdb := h.broadcaster.RDB()
	if rdb == nil {
		return false // Redis not configured — fail open
	}

	// 1. Check active cooldown
	cooldownKey := fmt.Sprintf("gate:key:%d:cooldown", cred.ID)
	exists, err := rdb.Exists(ctx, cooldownKey).Result()
	if err == nil && exists > 0 {
		return true // Key is cooling down
	}

	// 2. Check RPM limit (15 requests per minute)
	now := time.Now().Unix()
	rpmKey := fmt.Sprintf("gate:key:%d:rpm:%d", cred.ID, now/60)
	countStr, err := rdb.Get(ctx, rpmKey).Result()
	if err == nil {
		if count, cErr := strconv.Atoi(countStr); cErr == nil && count >= 15 {
			return true // Throttled
		}
	}

	return false
}

// incrementRedisRPM increments the request counter in the current minute sliding window.
func (h *Handler) incrementRedisRPM(ctx context.Context, cred *credentials.RuntimeCredential) {
	if h.broadcaster == nil {
		return
	}
	rdb := h.broadcaster.RDB()
	if rdb == nil {
		return
	}

	now := time.Now().Unix()
	rpmKey := fmt.Sprintf("gate:key:%d:rpm:%d", cred.ID, now/60)

	pipe := rdb.Pipeline()
	pipe.Incr(ctx, rpmKey)
	pipe.Expire(ctx, rpmKey, 90*time.Second)
	_, _ = pipe.Exec(ctx)
}

// StreamSanitizer filters out markdown formatting blocks introduced by model restarts.
type StreamSanitizer struct {
	isActive       bool
	checkedLeading bool
	buffer         string
}

// NewStreamSanitizer creates a new StreamSanitizer instance.
func NewStreamSanitizer(isActive bool) *StreamSanitizer {
	return &StreamSanitizer{
		isActive: isActive,
	}
}

// Sanitize checks for leading markdown code block fences and strips them
// to keep the output stream clean during failover resumption.
func (s *StreamSanitizer) Sanitize(text string) string {
	if !s.isActive {
		return text
	}
	if s.checkedLeading {
		return text
	}

	s.buffer += text

	// Wait until we have enough characters to evaluate formatting blocks safely
	if len(s.buffer) < 15 && !strings.Contains(s.buffer, "\n") {
		return ""
	}

	s.checkedLeading = true
	return stripLeadingFence(s.buffer)
}

// Flush releases any remaining buffered text when the stream completes.
func (s *StreamSanitizer) Flush() string {
	if s.checkedLeading {
		return ""
	}
	s.checkedLeading = true
	return stripLeadingFence(s.buffer)
}

// stripLeadingFence strips any leading markdown code block fence from the text.
func stripLeadingFence(cleanText string) string {
	trimmed := strings.TrimSpace(cleanText)
	if !strings.HasPrefix(trimmed, "```") {
		return cleanText
	}

	if idx := strings.Index(cleanText, "\n"); idx != -1 {
		return cleanText[idx+1:]
	}
	if idx := strings.Index(cleanText, " "); idx != -1 {
		return cleanText[idx+1:]
	}

	// Fallback for cases without newlines/spaces (e.g. short/completed streams)
	runes := []rune(cleanText)
	startIdx := 3
	for startIdx < len(runes) {
		r := runes[startIdx]
		if (r >= 'a' && r <= 'z') || (r >= 'A' && r <= 'Z') {
			startIdx++
		} else {
			break
		}
	}
	if startIdx < len(runes) {
		r := runes[startIdx]
		if r == '\n' || r == '\r' || r == ' ' || r == '_' || r == '-' {
			startIdx++
		}
	}
	return string(runes[startIdx:])
}

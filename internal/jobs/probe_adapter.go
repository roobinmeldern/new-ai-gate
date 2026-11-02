package jobs

import (
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
)

// ModelProbeRequest encapsulates the targeted HTTP endpoint, method, headers, and body for model probing.
type ModelProbeRequest struct {
	Method  string
	URL     string
	Headers map[string]string
	Body    []byte
}

// CleanUpstreamModel strips the routing pool provider prefix (e.g. nvidia/, cloudflare/, ollama/, 1min/)
// from a model pattern to produce the exact model identifier expected by the upstream provider.
func CleanUpstreamModel(modelPattern, providerID string) string {
	model := modelPattern
	provider := strings.ToLower(providerID)

	// If model starts with provider + "/", strip it (e.g. nvidia/google/gemma-2-2b-it -> google/gemma-2-2b-it)
	if strings.HasPrefix(strings.ToLower(model), provider+"/") {
		return model[len(provider)+1:]
	}

	// Handle specific provider prefix aliases
	switch provider {
	case "nvidia":
		model = strings.TrimPrefix(model, "nvidia/")
	case "ollama":
		model = strings.TrimPrefix(model, "ollama/")
	case "cloudflare":
		model = strings.TrimPrefix(model, "cloudflare/")
	case "huggingface":
		model = strings.TrimPrefix(model, "huggingface/")
	case "openrouter":
		model = strings.TrimPrefix(model, "openrouter/")
	case "deepinfra":
		model = strings.TrimPrefix(model, "deepinfra/")
	case "1minai", "1min":
		model = strings.TrimPrefix(model, "1min/")
	case "freemodel", "freemodel-cc":
		model = strings.TrimPrefix(model, "freemodel/")
		model = strings.TrimPrefix(model, "freemodel-cc/")
	case "gemini":
		model = strings.TrimPrefix(model, "gemini/")
	case "jiekou":
		model = strings.TrimPrefix(model, "jiekou/")
	case "anthropic":
		model = strings.TrimPrefix(model, "anthropic/")
	case "agentrouter":
		model = strings.TrimPrefix(model, "agentrouter/")
	}

	return model
}

// BuildAdaptiveProbeRequest constructs a capability- and provider-aware HTTP probe request.
// It routes each probe to the correct endpoint and formats the body according to the
// provider's requirements, eliminating false-positive 400/404/422 errors.
func BuildAdaptiveProbeRequest(baseURL, apiKey, providerID, modelPattern string, capabilities map[string]bool) (*ModelProbeRequest, error) {
	cleanBaseURL := strings.TrimRight(baseURL, "/")
	provider := strings.ToLower(providerID)
	modelLower := strings.ToLower(modelPattern)

	targetModel := CleanUpstreamModel(modelPattern, providerID)

	headers := map[string]string{
		"Content-Type":  "application/json",
		"Authorization": "Bearer " + apiKey,
		"User-Agent":    "CleverAIGate-HealthProbe/1.0",
	}

	// 0. ANTHROPIC & AGENTROUTER NATIVE ADAPTER
	// Anthropic and AgentRouter native endpoints expect POST to /v1/messages with
	// x-api-key and anthropic-version: 2023-06-01 headers (not Authorization: Bearer).
	if provider == "anthropic" || provider == "agentrouter" ||
		strings.Contains(cleanBaseURL, "anthropic.com") ||
		strings.Contains(cleanBaseURL, "agentrouter.org") {
		reqURL := cleanBaseURL + "/v1/messages"
		if strings.HasSuffix(cleanBaseURL, "/v1") {
			reqURL = cleanBaseURL + "/messages"
		}
		antHeaders := map[string]string{
			"Content-Type":      "application/json",
			"x-api-key":         apiKey,
			"anthropic-version": "2023-06-01",
			"User-Agent":        "CleverAIGate-HealthProbe/1.0",
		}
		bodyMap := map[string]interface{}{
			"model":      targetModel,
			"max_tokens": 1,
			"messages": []map[string]interface{}{
				{"role": "user", "content": "ping"},
			},
		}
		jsonBody, err := json.Marshal(bodyMap)
		if err != nil {
			return nil, fmt.Errorf("marshal anthropic probe: %w", err)
		}
		return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: antHeaders, Body: jsonBody}, nil
	}

	// 1. 1MIN.AI SPECIAL ADAPTER
	// 1min.ai does not expose /v1/chat/completions — all requests must go to
	// /api/chat-with-ai with the API-KEY header (not Authorization: Bearer).
	if provider == "1minai" || provider == "1min" || strings.Contains(cleanBaseURL, "1min.ai") {
		reqURL := "https://api.1min.ai/api/chat-with-ai"
		oneMinHeaders := map[string]string{
			"Content-Type": "application/json",
			"API-KEY":      apiKey,
			"User-Agent":   "CleverAIGate-HealthProbe/1.0",
		}
		bodyMap := map[string]interface{}{
			"type":  "UNIFY_CHAT_WITH_AI",
			"model": "gpt-4o-mini",
			"promptObject": map[string]interface{}{
				"prompt": "hi",
			},
		}
		jsonBody, err := json.Marshal(bodyMap)
		if err != nil {
			return nil, fmt.Errorf("marshal 1minai probe: %w", err)
		}
		return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: oneMinHeaders, Body: jsonBody}, nil
	}

	// 1b. JIEKOU.AI SPECIAL ADAPTER
	// Jiekou proxies Moonshot/Kimi, GPT-5/o1/o3/sol reasoning models, DeepSeek,
	// and other providers behind an OpenAI-compatible API. Key model families:
	//
	//   Reasoning/Beta (gpt-5*, *-sol, o1*, o3*, o4*):
	//     temperature=1, top_p=1, n=1, presence_penalty=0, frequency_penalty=0
	//     Uses max_completion_tokens (not max_tokens).
	//
	//   Kimi/Moonshot (moonshotai/*, kimi-*):
	//     temperature ∈ [0.0, 1.0], safe default 0.7.
	//
	//   Standard models (deepseek, llama, qwen, …):
	//     Standard safe probe defaults.
	if provider == "jiekou" || strings.Contains(cleanBaseURL, "jiekou.ai") || strings.Contains(cleanBaseURL, "jiekou.cloud") {
		reqURL := cleanBaseURL + "/v1/chat/completions"
		if strings.HasSuffix(cleanBaseURL, "/v1") {
			reqURL = cleanBaseURL + "/chat/completions"
		}
		// Strip any surviving "jiekou/" routing prefix from the model name.
		cleanModel := targetModel
		if strings.HasPrefix(cleanModel, "jiekou/") {
			cleanModel = cleanModel[len("jiekou/"):]
		}

		var bodyMap map[string]interface{}

		if isFixedParamReasoningModel(cleanModel) {
			// Reasoning / Beta GPT models: enforce fixed parameters exactly as
			// the upstream requires. Sending any other value causes HTTP 400:
			// "beta-limitations, temperature, top_p and n are fixed at 1,
			//  presence_penalty and frequency_penalty are fixed at 0"
			bodyMap = map[string]interface{}{
				"model": cleanModel,
				"messages": []map[string]interface{}{
					{"role": "user", "content": "hi"},
				},
				"temperature":        float64(1),
				"top_p":              float64(1),
				"n":                  float64(1),
				"presence_penalty":   float64(0),
				"frequency_penalty":  float64(0),
				"max_completion_tokens": 1,
			}
		} else if strings.Contains(strings.ToLower(cleanModel), "kimi") ||
			strings.Contains(strings.ToLower(cleanModel), "moonshot") {
			// Kimi/Moonshot: temperature must be in [0.0, 1.0].
			bodyMap = map[string]interface{}{
				"model": cleanModel,
				"messages": []map[string]interface{}{
					{"role": "user", "content": "hi"},
				},
				"max_tokens":  1,
				"temperature": 0.7,
			}
		} else {
			// Standard OpenAI-compatible models through Jiekou.
			bodyMap = map[string]interface{}{
				"model": cleanModel,
				"messages": []map[string]interface{}{
					{"role": "user", "content": "hi"},
				},
				"max_tokens":  1,
				"temperature": 0.1,
			}
		}

		jsonBody, err := json.Marshal(bodyMap)
		if err != nil {
			return nil, fmt.Errorf("marshal jiekou probe: %w", err)
		}
		return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: headers, Body: jsonBody}, nil
	}

	// 2. EMBEDDING MODELS
	if capabilities != nil && capabilities["embedding"] || strings.Contains(modelLower, "embed") || strings.Contains(modelLower, "bge-") {
		reqURL := cleanBaseURL + "/v1/embeddings"
		if strings.HasSuffix(cleanBaseURL, "/v1") {
			reqURL = cleanBaseURL + "/embeddings"
		}
		bodyMap := map[string]interface{}{
			"model": targetModel,
			"input": "health check",
		}
		jsonBody, err := json.Marshal(bodyMap)
		if err != nil {
			return nil, fmt.Errorf("marshal embedding probe: %w", err)
		}
		return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: headers, Body: jsonBody}, nil
	}

	// 3. IMAGE GENERATION MODELS
	if capabilities != nil && capabilities["image_generation"] || strings.Contains(modelLower, "flux") || strings.Contains(modelLower, "dall-e") || strings.Contains(modelLower, "sdxl") || strings.Contains(modelLower, "ideogram") {
		reqURL := cleanBaseURL + "/v1/images/generations"
		if strings.HasSuffix(cleanBaseURL, "/v1") {
			reqURL = cleanBaseURL + "/images/generations"
		}
		bodyMap := map[string]interface{}{
			"model":  targetModel,
			"prompt": "A simple white square",
			"n":      1,
			"size":   "256x256",
		}
		jsonBody, err := json.Marshal(bodyMap)
		if err != nil {
			return nil, fmt.Errorf("marshal image probe: %w", err)
		}
		return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: headers, Body: jsonBody}, nil
	}

	// 4. AUDIO / TTS MODELS
	if capabilities != nil && (capabilities["audio"] || capabilities["speech"] || capabilities["tts"]) || strings.Contains(modelLower, "tts") || strings.Contains(modelLower, "whisper") {
		reqURL := cleanBaseURL + "/v1/audio/speech"
		if strings.HasSuffix(cleanBaseURL, "/v1") {
			reqURL = cleanBaseURL + "/audio/speech"
		}
		bodyMap := map[string]interface{}{
			"model": targetModel,
			"input": "ping",
			"voice": "alloy",
		}
		jsonBody, err := json.Marshal(bodyMap)
		if err != nil {
			return nil, fmt.Errorf("marshal audio probe: %w", err)
		}
		return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: headers, Body: jsonBody}, nil
	}

	// 5. CHAT / REASONING / VISION / PARSE / GENERAL MODELS
	reqURL := cleanBaseURL + "/v1/chat/completions"
	if strings.HasSuffix(cleanBaseURL, "/v1") {
		reqURL = cleanBaseURL + "/chat/completions"
	}

	// Cloudflare Workers AI custom format
	if provider == "cloudflare" || strings.HasPrefix(cleanBaseURL, "cloudflare:") {
		accountID := strings.TrimPrefix(cleanBaseURL, "cloudflare:")
		reqURL = fmt.Sprintf("https://api.cloudflare.com/client/v4/accounts/%s/ai/v1/chat/completions", accountID)
	}

	var messages []map[string]interface{}

	// Special handling for parsing models requiring multimodal input structure
	if strings.Contains(modelLower, "parse") || strings.Contains(modelLower, "nemoretriever-parse") {
		messages = []map[string]interface{}{
			{
				"role": "user",
				"content": []map[string]interface{}{
					{"type": "text", "text": "ping"},
				},
			},
		}
	} else {
		messages = []map[string]interface{}{
			{"role": "user", "content": "ping"},
		}
	}

	bodyMap := map[string]interface{}{
		"model":      targetModel,
		"messages":   messages,
		"max_tokens": 5,
	}

	// Provider-specific parameter adjustments
	if provider == "nvidia" || strings.Contains(cleanBaseURL, "nvidia.com") {
		// Enforce positive temperature for NVIDIA NIM to avoid HTTP 422
		bodyMap["temperature"] = 0.7
	} else {
		bodyMap["temperature"] = 0.1
	}

	jsonBody, err := json.Marshal(bodyMap)
	if err != nil {
		return nil, fmt.Errorf("marshal chat probe: %w", err)
	}

	return &ModelProbeRequest{Method: http.MethodPost, URL: reqURL, Headers: headers, Body: jsonBody}, nil
}

// IsQuotaOrBalanceError returns true ONLY for genuine auth/quota/balance failures.
// HTTP 400 (Bad Request), 404 (Not Found), 422 (Unprocessable Entity), 504 (Timeout),
// and connection errors are structural probe errors that indicate our payload or
// endpoint is wrong — they do NOT mean the API key is bad or out of credits.
//
// Only these indicate the credential itself is the problem:
//   - HTTP 401 Unauthorized      — key invalid or revoked
//   - HTTP 402 Payment Required  — account needs payment
//   - HTTP 403 Forbidden         — key blocked or permissions revoked
//   - HTTP 429 Too Many Requests — quota exhausted
//   - Body keywords              — "insufficient balance", "out of credits", etc.
func IsQuotaOrBalanceError(statusCode int, bodySnippet string) bool {
	switch statusCode {
	case http.StatusUnauthorized,    // 401
		http.StatusPaymentRequired,  // 402
		http.StatusForbidden,        // 403
		http.StatusTooManyRequests:  // 429
		return true
	}

	// Body-level quota/balance keywords — provider-agnostic
	bodyLower := strings.ToLower(bodySnippet)
	quotaKeywords := []string{
		"insufficient_quota",
		"insufficient balance",
		"credit_balance_too_low",
		"account_deactivated",
		"quota exceeded",
		"out of credits",
		"billing",
		"usage limit exceeded",
		"payment required",
		"account suspended",
		"credits have been exhausted",
		"tier is insufficient",
	}
	for _, kw := range quotaKeywords {
		if strings.Contains(bodyLower, kw) {
			return true
		}
	}
	return false
}

// isFixedParamReasoningModel returns true when a model enforces hard-coded generation
// parameters (temperature=1, top_p=1, n=1, presence_penalty=0, frequency_penalty=0).
//
// This is a package-local mirror of proxy.IsFixedParamReasoningModel. Both must be
// kept in sync. Duplication avoids a circular import between jobs ↔ proxy packages.
//
// Affected families (as documented by OpenAI and Jiekou upstream error messages):
//   - gpt-5*       (gpt-5.6-sol, gpt-5o, gpt-5-mini, …)
//   - *-sol        (gpt-5.6-sol, gpt-4.5-sol, …)
//   - o1*          (o1, o1-mini, o1-preview, o1-pro, …)
//   - o3*          (o3, o3-mini, o3-pro, …)
//   - o4*          (o4-mini, …)
//   - *reasoning*  (any future reasoning variant)
func isFixedParamReasoningModel(modelName string) bool {
	lower := strings.ToLower(modelName)
	// Strip any surviving gateway prefix before matching.
	if idx := strings.LastIndex(lower, "/"); idx >= 0 {
		lower = lower[idx+1:]
	}
	return strings.HasPrefix(lower, "gpt-5") ||
		strings.HasSuffix(lower, "-sol") ||
		strings.HasPrefix(lower, "o1") ||
		strings.HasPrefix(lower, "o3") ||
		strings.HasPrefix(lower, "o4") ||
		strings.Contains(lower, "reasoning")
}

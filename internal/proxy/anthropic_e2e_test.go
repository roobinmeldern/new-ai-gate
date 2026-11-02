package proxy

import (
	"bufio"
	"encoding/json"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync/atomic"
	"testing"
	"time"

	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/config"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"github.com/skadraneshghn/clever-ai-gate/internal/middleware"
	"go.uber.org/zap"
)

// setupE2ETestEnv initializes an in-memory cache, sync manager, handler, and Gin engine
// configured with the full middleware pipeline matching production.
func setupE2ETestEnv(t *testing.T) (*Handler, *cache.Store, *credentials.SyncManager, *gin.Engine) {
	t.Helper()
	gin.SetMode(gin.TestMode)
	logger := zap.NewNop()

	cfg := &config.Config{
		CacheMaxSizeMB:      10,
		CacheNumCounters:    100,
		DefaultRateLimitRPM: 10000,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache store: %v", err)
	}
	t.Cleanup(func() { cacheStore.Close() })

	sm := credentials.NewSyncManager(nil, cacheStore, nil, logger)
	t.Cleanup(func() { sm.Stop() })

	transport := &http.Transport{
		MaxIdleConns:        100,
		IdleConnTimeout:     90 * time.Second,
		DisableCompression: true,
	}
	httpClient := &http.Client{Transport: transport, Timeout: 15 * time.Second}

	h := NewHandler(httpClient, cacheStore, nil, logger, nil, nil, nil)
	h.SetSyncManager(sm)

	engine := gin.New()
	engine.Use(middleware.ProxyAuth(cacheStore, "admin-test-key"))

	// Register proxy routes mirroring internal/router/engine.go
	engine.POST("/v1/*proxyPath", h.Handle)
	engine.GET("/v1/models", h.ListModels)
	engine.GET("/v1/models/*model", h.GetModel)

	return h, cacheStore, sm, engine
}

// ── Test 1: Native Anthropic Passthrough (Stream & Non-Stream) ────────────────
func TestAnthropicE2E_NativePassthrough_StreamAndNonStream(t *testing.T) {
	var (
		receivedPaths       []string
		receivedModels      []string
		receivedAPIKeys     []string
		receivedVersions    []string
		receivedBetaHeaders []string
	)

	// Mock native Anthropic upstream server
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		bodyBytes, _ := io.ReadAll(r.Body)
		receivedPaths = append(receivedPaths, r.URL.Path)
		receivedAPIKeys = append(receivedAPIKeys, r.Header.Get("x-api-key"))
		receivedVersions = append(receivedVersions, r.Header.Get("anthropic-version"))
		receivedBetaHeaders = append(receivedBetaHeaders, r.Header.Get("anthropic-beta"))

		model, _ := jsonparser.GetString(bodyBytes, "model")
		receivedModels = append(receivedModels, model)
		isStream, _ := jsonparser.GetBoolean(bodyBytes, "stream")

		if isStream {
			w.Header().Set("Content-Type", "text/event-stream")
			w.Header().Set("Cache-Control", "no-cache")
			w.WriteHeader(http.StatusOK)

			flusher, ok := w.(http.Flusher)
			if !ok {
				t.Fatalf("expected flusher")
			}

			events := []string{
				"event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_e2e_1\",\"type\":\"message\",\"role\":\"assistant\",\"content\":[],\"model\":\"claude-3-5-sonnet\",\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n",
				"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n",
				"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"text_delta\",\"text\":\"Hello from native Anthropic!\"}}\n\n",
				"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n",
				"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"end_turn\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":5}}\n\n",
				"event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n",
			}
			for _, ev := range events {
				_, _ = w.Write([]byte(ev))
				flusher.Flush()
			}
			return
		}

		// Non-stream response
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		respJSON := `{
			"id": "msg_e2e_nonstream",
			"type": "message",
			"role": "assistant",
			"model": "claude-3-5-sonnet",
			"content": [{"type": "text", "text": "Hello non-streaming Anthropic!"}],
			"stop_reason": "end_turn",
			"usage": {"input_tokens": 10, "output_tokens": 5}
		}`
		_, _ = w.Write([]byte(respJSON))
	}))
	defer upstreamServer.Close()

	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	// Configure routing pool for anthropic/claude-3-5-sonnet
	cred := &credentials.RuntimeCredential{
		ID:       1,
		Provider: "anthropic",
		APIKey:   "sk-ant-upstream-secret",
		BaseURL:  upstreamServer.URL,
		Weight:   1,
	}
	pool := credentials.NewBalancedPool("anthropic/claude-3-5-sonnet", "round-robin", []*credentials.RuntimeCredential{cred}, nil)
	cacheStore.Set(cache.PoolKey("anthropic/claude-3-5-sonnet"), pool, 1)
	cacheStore.Wait()
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"anthropic/claude-3-5-sonnet": pool,
	})

	client := gwServer.Client()

	// 1. Streaming Request
	streamPayload := `{
		"model": "anthropic/claude-3-5-sonnet",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hi"}],
		"stream": true,
		"cache_control": {"type": "ephemeral"}
	}`
	req1, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(streamPayload))
	req1.Header.Set("Content-Type", "application/json")
	req1.Header.Set("x-api-key", "admin-test-key")
	req1.Header.Set("anthropic-version", "2023-06-01")

	resp1, err := client.Do(req1)
	if err != nil {
		t.Fatalf("streaming request failed: %v", err)
	}
	defer resp1.Body.Close()

	if resp1.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp1.Body)
		t.Fatalf("expected 200 OK for stream, got %d: %s", resp1.StatusCode, string(body))
	}

	streamBody, _ := io.ReadAll(resp1.Body)
	streamOut := string(streamBody)
	if !strings.Contains(streamOut, "event: message_start") || !strings.Contains(streamOut, "Hello from native Anthropic!") {
		t.Errorf("unexpected streaming body: %s", streamOut)
	}

	// 2. Non-Streaming Request
	nonStreamPayload := `{
		"model": "anthropic/claude-3-5-sonnet",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hi"}],
		"stream": false
	}`
	req2, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(nonStreamPayload))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("x-api-key", "admin-test-key")
	req2.Header.Set("anthropic-version", "2023-06-01")

	resp2, err := client.Do(req2)
	if err != nil {
		t.Fatalf("non-streaming request failed: %v", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp2.Body)
		t.Fatalf("expected 200 OK for non-stream, got %d: %s", resp2.StatusCode, string(body))
	}

	var antResp map[string]interface{}
	if err := json.NewDecoder(resp2.Body).Decode(&antResp); err != nil {
		t.Fatalf("failed to decode non-stream response: %v", err)
	}
	if antResp["id"] != "msg_e2e_nonstream" {
		t.Errorf("expected msg_e2e_nonstream, got %v", antResp["id"])
	}

	// Verify assertions on upstream calls
	if len(receivedModels) != 2 {
		t.Fatalf("expected 2 upstream calls, got %d", len(receivedModels))
	}
	// Prefix must be cleanly stripped
	for i, m := range receivedModels {
		if m != "claude-3-5-sonnet" {
			t.Errorf("call %d: expected stripped model claude-3-5-sonnet, got %s", i, m)
		}
	}
	// Header forwarding
	for i, k := range receivedAPIKeys {
		if k != "sk-ant-upstream-secret" {
			t.Errorf("call %d: expected upstream key sk-ant-upstream-secret, got %s", i, k)
		}
	}
	for i, v := range receivedVersions {
		if v != "2023-06-01" {
			t.Errorf("call %d: expected anthropic-version 2023-06-01, got %s", i, v)
		}
	}
	// Automatic prompt caching beta header injection on request 1 (contains cache_control)
	if !strings.Contains(receivedBetaHeaders[0], "prompt-caching-2024-07-31") {
		t.Errorf("call 0: expected prompt-caching-2024-07-31 beta header, got %s", receivedBetaHeaders[0])
	}
}

// ── Test 2: Cross-Provider OpenAI Upstream (Thinking + Tools + Cache Accounting)
func TestAnthropicE2E_CrossProvider_OpenAIUpstream_ThinkingAndTools(t *testing.T) {
	var (
		receivedPath string
		receivedBody []byte
	)

	// Mock OpenAI upstream server
	upstreamServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedPath = r.URL.Path
		receivedBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher, ok := w.(http.Flusher)
		if !ok {
			t.Fatalf("expected flusher")
		}

		// Stream <think> tag in first chunk, then </think> + text, tool_calls, and usage
		chunks := []string{
			"data: {\"id\":\"chatcmpl-think-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"<think>Analyzing weather request\"}}]}\n\n",
			"data: {\"id\":\"chatcmpl-think-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"content\":\"</think> The forecast is sunny.\"}}]}\n\n",
			"data: {\"id\":\"chatcmpl-think-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_weather_1\",\"type\":\"function\",\"function\":{\"name\":\"get_weather\",\"arguments\":\"{\\\"location\\\":\\\"Paris\\\"}\"}}]}}]}\n\n",
			"data: {\"id\":\"chatcmpl-think-1\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}],\"usage\":{\"prompt_tokens\":50,\"completion_tokens\":25,\"prompt_tokens_details\":{\"cached_tokens\":30}}}\n\n",
			"data: [DONE]\n\n",
		}
		for _, c := range chunks {
			_, _ = w.Write([]byte(c))
			flusher.Flush()
		}
	}))
	defer upstreamServer.Close()

	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	cred := &credentials.RuntimeCredential{
		ID:       2,
		Provider: "openai",
		APIKey:   "sk-openai-key",
		BaseURL:  upstreamServer.URL,
		Weight:   1,
	}
	pool := credentials.NewBalancedPool("openai/gpt-4o", "round-robin", []*credentials.RuntimeCredential{cred}, nil)
	cacheStore.Set(cache.PoolKey("openai/gpt-4o"), pool, 1)
	cacheStore.Wait()
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"openai/gpt-4o": pool,
	})

	anthropicPayload := `{
		"model": "openai/gpt-4o",
		"max_tokens": 1024,
		"system": [{"type": "text", "text": "You are a weather bot."}],
		"messages": [{"role": "user", "content": "Weather in Paris?"}],
		"tools": [
			{
				"name": "get_weather",
				"description": "Get current weather for location",
				"input_schema": {
					"type": "object",
					"properties": {
						"location": {"type": "string"}
					},
					"required": ["location"]
				}
			}
		],
		"thinking": {"type": "enabled", "budget_tokens": 8000},
		"stream": true
	}`

	req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(anthropicPayload))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "admin-test-key")
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := gwServer.Client().Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected 200 OK, got %d: %s", resp.StatusCode, string(b))
	}

	// 1. Verify inbound translation to OpenAI format
	if receivedPath != "/v1/chat/completions" {
		t.Errorf("expected translated path /v1/chat/completions, got %s", receivedPath)
	}
	toolName, _ := jsonparser.GetString(receivedBody, "tools", "[0]", "function", "name")
	if toolName != "get_weather" {
		t.Errorf("expected translated tool get_weather, got %s", toolName)
	}

	// 2. Read SSE events and verify exact Anthropic event sequence
	scanner := bufio.NewScanner(resp.Body)
	var eventsReceived []string
	var currentEvent string
	var lastUsageJSON string

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
			eventsReceived = append(eventsReceived, currentEvent)
		} else if strings.HasPrefix(line, "data: ") {
			data := strings.TrimPrefix(line, "data: ")
			if currentEvent == "message_delta" {
				lastUsageJSON = data
			}
		}
	}

	expectedEvents := []string{
		"message_start",
		"content_block_start", // thinking
		"content_block_delta", // thinking_delta
		"content_block_delta", // signature_delta
		"content_block_stop",
		"content_block_start", // text
		"content_block_delta", // text_delta
		"content_block_stop",
		"content_block_start", // tool_use
		"content_block_delta", // input_json_delta
		"content_block_stop",
		"message_delta", // stop_reason: tool_use
		"message_stop",
	}

	if len(eventsReceived) != len(expectedEvents) {
		t.Fatalf("event count mismatch:\nExpected: %v\nGot:      %v", expectedEvents, eventsReceived)
	}

	for i, exp := range expectedEvents {
		if eventsReceived[i] != exp {
			t.Errorf("event %d: expected %s, got %s", i, exp, eventsReceived[i])
		}
	}

	// Verify cached tokens accounting in message_delta usage
	cachedTokens, err := jsonparser.GetInt([]byte(lastUsageJSON), "usage", "cache_read_input_tokens")
	if err != nil || cachedTokens != 30 {
		t.Errorf("expected cache_read_input_tokens: 30 in message_delta usage, got %d (err: %v)", cachedTokens, err)
	}
}

// ── Test 3: Cross-Provider Failover (Anthropic 429 -> OpenAI 200) ────────────
func TestAnthropicE2E_CrossProviderFailover_AnthropicToOpenAI(t *testing.T) {
	setRetryRoundSettings(t, 1, 10*time.Millisecond)

	var (
		anthropicAttempts int32
		openaiAttempts    int32
	)

	// Mock Anthropic upstream: returns 429 Too Many Requests
	mockAnthropic := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&anthropicAttempts, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusTooManyRequests)
		_, _ = w.Write([]byte(`{"type":"error","error":{"type":"rate_limit_error","message":"Anthropic overloaded"}}`))
	}))
	defer mockAnthropic.Close()

	// Mock OpenAI upstream: returns 200 OK with chat completion JSON
	mockOpenAI := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		atomic.AddInt32(&openaiAttempts, 1)
		w.Header().Set("Content-Type", "application/json")
		w.WriteHeader(http.StatusOK)
		respJSON := `{
			"id": "chatcmpl-failover-ok",
			"object": "chat.completion",
			"choices": [
				{
					"index": 0,
					"message": {
						"role": "assistant",
						"content": "Successfully recovered via OpenAI fallback!"
					},
					"finish_reason": "stop"
				}
			],
			"usage": {
				"prompt_tokens": 15,
				"completion_tokens": 8,
				"total_tokens": 23,
				"prompt_tokens_details": {
					"cached_tokens": 5
				}
			}
		}`
		_, _ = w.Write([]byte(respJSON))
	}))
	defer mockOpenAI.Close()

	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	// Configure pool with Credential 1 (Anthropic) and Credential 2 (OpenAI)
	cred1 := &credentials.RuntimeCredential{
		ID:       101,
		Provider: "anthropic",
		APIKey:   "sk-ant-primary",
		BaseURL:  mockAnthropic.URL,
		Weight:   1,
	}
	cred2 := &credentials.RuntimeCredential{
		ID:       102,
		Provider: "openai",
		APIKey:   "sk-oa-fallback",
		BaseURL:  mockOpenAI.URL,
		Weight:   1,
	}

	pool := credentials.NewBalancedPool("claude-failover-pool", "round-robin", []*credentials.RuntimeCredential{cred1, cred2}, nil)
	pool.Cursor = 1 // Start cursor so cred1 (index 0) is attempted first
	cacheStore.Set(cache.PoolKey("claude-failover-pool"), pool, 1)
	cacheStore.Wait()
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"claude-failover-pool": pool,
	})

	payload := `{
		"model": "claude-failover-pool",
		"max_tokens": 512,
		"messages": [{"role": "user", "content": "Help me"}]
	}`
	req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(payload))
	req.Header.Set("Content-Type", "application/json")
	req.Header.Set("x-api-key", "admin-test-key")
	req.Header.Set("anthropic-version", "2023-06-01")

	resp, err := gwServer.Client().Do(req)
	if err != nil {
		t.Fatalf("request failed: %v", err)
	}
	defer resp.Body.Close()

	if resp.StatusCode != http.StatusOK {
		body, _ := io.ReadAll(resp.Body)
		t.Fatalf("expected failover to 200 OK, got %d: %s", resp.StatusCode, string(body))
	}

	// Verify both upstreams were invoked
	if atomic.LoadInt32(&anthropicAttempts) != 1 {
		t.Errorf("expected 1 Anthropic attempt, got %d", atomic.LoadInt32(&anthropicAttempts))
	}
	if atomic.LoadInt32(&openaiAttempts) != 1 {
		t.Errorf("expected 1 OpenAI attempt, got %d", atomic.LoadInt32(&openaiAttempts))
	}

	// Verify response was translated back into an Anthropic Message
	var msg map[string]interface{}
	if err := json.NewDecoder(resp.Body).Decode(&msg); err != nil {
		t.Fatalf("failed to decode response: %v", err)
	}

	if msg["type"] != "message" || msg["role"] != "assistant" {
		t.Errorf("expected Anthropic Message shape, got: %+v", msg)
	}
	contentArr, ok := msg["content"].([]interface{})
	if !ok || len(contentArr) == 0 {
		t.Fatalf("expected non-empty content array, got %+v", msg["content"])
	}
	firstBlock := contentArr[0].(map[string]interface{})
	if firstBlock["text"] != "Successfully recovered via OpenAI fallback!" {
		t.Errorf("unexpected content text: %v", firstBlock["text"])
	}

	// Verify cached tokens usage mapping
	usageObj, ok := msg["usage"].(map[string]interface{})
	if !ok {
		t.Fatalf("expected usage object")
	}
	if usageObj["cache_read_input_tokens"] != float64(5) {
		t.Errorf("expected cache_read_input_tokens: 5, got %v", usageObj["cache_read_input_tokens"])
	}
}

// ── Test 4: Dual-Protocol Models Discovery & Count Tokens ────────────────────
func TestAnthropicE2E_ModelsAndCountTokens(t *testing.T) {
	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	activeModels := []credentials.ActiveModel{
		{
			Pattern:      "claude-3-7-sonnet",
			Capabilities: map[string]bool{"thinking": true},
		},
		{
			Pattern:      "openai/gpt-4o",
			Capabilities: map[string]bool{"vision": true},
		},
	}
	cacheStore.Set("system:active_models", activeModels, 1000)
	cacheStore.Wait()

	pool1 := credentials.NewBalancedPool("claude-3-7-sonnet", "round-robin", []*credentials.RuntimeCredential{}, nil)
	pool2 := credentials.NewBalancedPool("openai/gpt-4o", "round-robin", []*credentials.RuntimeCredential{}, nil)
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"claude-3-7-sonnet": pool1,
		"openai/gpt-4o":     pool2,
	})

	client := gwServer.Client()

	// 1. GET /v1/models (ListModels)
	req1, _ := http.NewRequest("GET", gwServer.URL+"/v1/models", nil)
	req1.Header.Set("x-api-key", "admin-test-key")
	req1.Header.Set("anthropic-version", "2023-06-01")

	resp1, err := client.Do(req1)
	if err != nil {
		t.Fatalf("GET /v1/models failed: %v", err)
	}
	defer resp1.Body.Close()

	if resp1.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", resp1.StatusCode)
	}

	var listResp ModelListResponse
	if err := json.NewDecoder(resp1.Body).Decode(&listResp); err != nil {
		t.Fatalf("failed to decode ModelListResponse: %v", err)
	}

	if listResp.Object != "list" {
		t.Errorf("expected object: list, got %s", listResp.Object)
	}
	if listResp.HasMore != false {
		t.Errorf("expected has_more: false")
	}
	if listResp.FirstID != "claude-3-7-sonnet" || listResp.LastID != "openai/gpt-4o" {
		t.Errorf("unexpected pagination ids: first=%s, last=%s", listResp.FirstID, listResp.LastID)
	}
	if len(listResp.Data) != 2 {
		t.Fatalf("expected 2 models, got %d", len(listResp.Data))
	}
	if listResp.Data[0].Type != "model" || listResp.Data[0].DisplayName != "claude-3-7-sonnet" || listResp.Data[0].CreatedAt == "" {
		t.Errorf("missing Anthropic model fields: %+v", listResp.Data[0])
	}

	// 2. GET /v1/models/claude-3-7-sonnet (Single Model Discovery)
	req2, _ := http.NewRequest("GET", gwServer.URL+"/v1/models/claude-3-7-sonnet", nil)
	req2.Header.Set("x-api-key", "admin-test-key")
	req2.Header.Set("anthropic-version", "2023-06-01")

	resp2, err := client.Do(req2)
	if err != nil {
		t.Fatalf("GET /v1/models/:model failed: %v", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		t.Fatalf("expected 200 OK, got %d", resp2.StatusCode)
	}
	var singleModel ModelDetail
	if err := json.NewDecoder(resp2.Body).Decode(&singleModel); err != nil {
		t.Fatalf("failed to decode single model: %v", err)
	}
	if singleModel.ID != "claude-3-7-sonnet" || singleModel.Type != "model" || singleModel.Capabilities["thinking"] != true {
		t.Errorf("unexpected single model detail: %+v", singleModel)
	}

	// 3. GET /v1/models/unknown-model (Anthropic 404 envelope)
	req3, _ := http.NewRequest("GET", gwServer.URL+"/v1/models/non-existent-model", nil)
	req3.Header.Set("x-api-key", "admin-test-key")
	req3.Header.Set("anthropic-version", "2023-06-01")

	resp3, err := client.Do(req3)
	if err != nil {
		t.Fatalf("GET /v1/models/unknown failed: %v", err)
	}
	defer resp3.Body.Close()

	if resp3.StatusCode != http.StatusNotFound {
		t.Fatalf("expected 404 Not Found, got %d", resp3.StatusCode)
	}
	var errEnvelope map[string]interface{}
	if err := json.NewDecoder(resp3.Body).Decode(&errEnvelope); err != nil {
		t.Fatalf("failed to decode 404 error envelope: %v", err)
	}
	if errEnvelope["type"] != "error" {
		t.Errorf("expected type: error, got %v", errEnvelope["type"])
	}
	errSub, ok := errEnvelope["error"].(map[string]interface{})
	if !ok || errSub["type"] != "not_found_error" {
		t.Errorf("expected not_found_error type, got %+v", errSub)
	}

	// 4. POST /v1/messages/count_tokens
	countTokensPayload := `{
		"model": "claude-3-7-sonnet",
		"messages": [
			{"role": "user", "content": "How many tokens is this sentence?"}
		]
	}`
	req4, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages/count_tokens", strings.NewReader(countTokensPayload))
	req4.Header.Set("Content-Type", "application/json")
	req4.Header.Set("x-api-key", "admin-test-key")
	req4.Header.Set("anthropic-version", "2023-06-01")

	resp4, err := client.Do(req4)
	if err != nil {
		t.Fatalf("POST count_tokens failed: %v", err)
	}
	defer resp4.Body.Close()

	if resp4.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp4.Body)
		t.Fatalf("expected 200 OK for count_tokens, got %d: %s", resp4.StatusCode, string(b))
	}
	var countResp map[string]interface{}
	if err := json.NewDecoder(resp4.Body).Decode(&countResp); err != nil {
		t.Fatalf("failed to decode count_tokens response: %v", err)
	}
	inputTokens, ok := countResp["input_tokens"].(float64)
	if !ok || inputTokens <= 0 {
		t.Errorf("expected positive input_tokens, got %v", countResp["input_tokens"])
	}
}

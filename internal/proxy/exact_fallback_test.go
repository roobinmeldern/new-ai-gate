package proxy

import (
	"bytes"
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
	"go.uber.org/zap"
)

func TestExactModelCrossProviderFallback_Success(t *testing.T) {
	// Pin single-round behavior: this test verifies the cross-provider switch
	// mechanics. The multi-round retry policy (retry every credential of a
	// pool before switching providers) has dedicated tests in retry_rounds_test.go.
	setRetryRoundSettings(t, 1, 10*time.Millisecond)

	gin.SetMode(gin.TestMode)
	logger := zap.NewNop()
	cfg := &config.Config{
		CacheMaxSizeMB:   10,
		CacheNumCounters: 100,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cacheStore.Close()

	// 1. Primary pool: nvidia/meta/llama-3.3-70b-instruct (will return 503)
	nvidiaCred := &credentials.RuntimeCredential{
		ID:       101,
		Provider: "nvidia",
		APIKey:   "nv-key-1",
		BaseURL:  "https://integrate.api.nvidia.com",
		Weight:   1,
	}
	nvidiaPool := credentials.NewBalancedPool("nvidia/meta/llama-3.3-70b-instruct", "round-robin", []*credentials.RuntimeCredential{nvidiaCred}, nil)

	// 2. Exact fallback pool: openrouter/meta-llama/llama-3.3-70b-instruct (will return 200 OK)
	orCred := &credentials.RuntimeCredential{
		ID:       202,
		Provider: "openrouter",
		APIKey:   "or-key-1",
		BaseURL:  "https://openrouter.ai/api",
		Weight:   1,
	}
	orPool := credentials.NewBalancedPool("openrouter/meta-llama/llama-3.3-70b-instruct", "round-robin", []*credentials.RuntimeCredential{orCred}, nil)

	// 3. Unrelated / cheaper model pool: puter/gpt-4o-mini (must NOT be touched)
	puterCred := &credentials.RuntimeCredential{
		ID:       303,
		Provider: "puter",
		APIKey:   "put-key-1",
		BaseURL:  "https://api.puter.com",
		Weight:   1,
	}
	puterPool := credentials.NewBalancedPool("puter/gpt-4o-mini", "round-robin", []*credentials.RuntimeCredential{puterCred}, nil)

	// Register in cache
	cacheStore.Set(cache.PoolKey("nvidia/meta/llama-3.3-70b-instruct"), nvidiaPool, 1)
	cacheStore.Set(cache.PoolKey("openrouter/meta-llama/llama-3.3-70b-instruct"), orPool, 1)
	cacheStore.Set(cache.PoolKey("puter/gpt-4o-mini"), puterPool, 1)
	cacheStore.Wait()

	// SyncManager setup
	sm := credentials.NewSyncManager(nil, cacheStore, nil, logger)
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"nvidia/meta/llama-3.3-70b-instruct":           nvidiaPool,
		"openrouter/meta-llama/llama-3.3-70b-instruct": orPool,
		"puter/gpt-4o-mini":                            puterPool,
	})

	var calledProviders []string
	var forwardedModels []string

	mockClient := &http.Client{
		Transport: &mockRoundTripper{
			roundTripFunc: func(req *http.Request) (*http.Response, error) {
				bodyBytes, _ := io.ReadAll(req.Body)
				req.Body = io.NopCloser(bytes.NewReader(bodyBytes))

				modelInBody, _ := jsonparser.GetString(bodyBytes, "model")
				forwardedModels = append(forwardedModels, modelInBody)

				if strings.Contains(req.URL.Host, "nvidia.com") {
					calledProviders = append(calledProviders, "nvidia")
					resp := &http.Response{
						StatusCode: http.StatusServiceUnavailable,
						Header:     make(http.Header),
						Body:       io.NopCloser(strings.NewReader(`{"error": {"message": "NVIDIA NIM overloaded", "type": "server_error"}}`)),
					}
					resp.Header.Set("Content-Type", "application/json")
					return resp, nil
				}

				if strings.Contains(req.URL.Host, "openrouter.ai") {
					calledProviders = append(calledProviders, "openrouter")
					resp := &http.Response{
						StatusCode: http.StatusOK,
						Header:     make(http.Header),
						Body:       io.NopCloser(strings.NewReader(`{"id": "or-resp-1", "choices": [{"message": {"role": "assistant", "content": "Hello from OpenRouter fallback"}}]}`)),
					}
					resp.Header.Set("Content-Type", "application/json")
					return resp, nil
				}

				if strings.Contains(req.URL.Host, "puter.com") {
					calledProviders = append(calledProviders, "puter")
				}

				resp := &http.Response{
					StatusCode: http.StatusInternalServerError,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"error": "unexpected"}`)),
				}
				return resp, nil
			},
		},
	}

	h := NewHandler(mockClient, cacheStore, nil, logger, nil, nil, nil)
	h.SetSyncManager(sm)

	router := gin.New()
	router.POST("/v1/chat/completions", h.Handle)

	reqPayload := `{"model": "nvidia/meta/llama-3.3-70b-instruct", "messages": [{"role": "user", "content": "test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqPayload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	if w.Code != http.StatusOK {
		t.Fatalf("expected HTTP 200 OK from exact fallback, got %d: %s", w.Code, w.Body.String())
	}

	if !strings.Contains(w.Body.String(), "Hello from OpenRouter fallback") {
		t.Fatalf("expected OpenRouter body in response, got %s", w.Body.String())
	}

	// Verify NVIDIA was tried first, then OpenRouter was tried
	if len(calledProviders) != 2 || calledProviders[0] != "nvidia" || calledProviders[1] != "openrouter" {
		t.Errorf("expected [nvidia, openrouter], got %v", calledProviders)
	}

	// Verify Puter was NEVER called
	for _, p := range calledProviders {
		if p == "puter" {
			t.Errorf("puter was called as an arbitrary fallback!")
		}
	}

	// Verify upstream model passed to OpenRouter was stripped of nvidia/ routing prefix
	if len(forwardedModels) >= 2 {
		if forwardedModels[1] != "meta-llama/llama-3.3-70b-instruct" {
			t.Errorf("expected OpenRouter model in body to be meta-llama/llama-3.3-70b-instruct, got %q", forwardedModels[1])
		}
	}

	// Verify nvidia key was not disabled for 24h
	now := time.Now().UnixNano()
	cooldownUntil := atomic.LoadInt64(&nvidiaCred.CooldownUntil)
	if cooldownUntil > now {
		cooldownDuration := time.Duration(cooldownUntil - now)
		if cooldownDuration > 30*time.Second {
			t.Errorf("nvidia key was penalized with excessive cooldown (%v) instead of transient cooldown!", cooldownDuration)
		}
	}
}

func TestExactModelFallback_NoArbitraryDowngrade_ThrowsError(t *testing.T) {
	// Pin single-round behavior: this test verifies that no downgrade happens
	// when no exact-model fallback exists. The multi-round retry policy has
	// dedicated tests in retry_rounds_test.go.
	setRetryRoundSettings(t, 1, 10*time.Millisecond)

	gin.SetMode(gin.TestMode)
	logger := zap.NewNop()
	cfg := &config.Config{
		CacheMaxSizeMB:   10,
		CacheNumCounters: 100,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cacheStore.Close()

	// 1. Primary pool: nvidia/meta/llama-3.3-70b-instruct (fails with 503)
	nvidiaCred := &credentials.RuntimeCredential{
		ID:       101,
		Provider: "nvidia",
		APIKey:   "nv-key-1",
		BaseURL:  "https://integrate.api.nvidia.com",
		Weight:   1,
	}
	nvidiaPool := credentials.NewBalancedPool("nvidia/meta/llama-3.3-70b-instruct", "round-robin", []*credentials.RuntimeCredential{nvidiaCred}, nil)

	// 2. Different version pool: openrouter/meta-llama/llama-3.1-70b-instruct (NOT exact match)
	orCred := &credentials.RuntimeCredential{
		ID:       202,
		Provider: "openrouter",
		APIKey:   "or-key-1",
		BaseURL:  "https://openrouter.ai/api",
		Weight:   1,
	}
	differentVersionPool := credentials.NewBalancedPool("openrouter/meta-llama/llama-3.1-70b-instruct", "round-robin", []*credentials.RuntimeCredential{orCred}, nil)

	// 3. Arbitrary cheaper pool: puter/gpt-4o-mini (NOT exact match)
	puterCred := &credentials.RuntimeCredential{
		ID:       303,
		Provider: "puter",
		APIKey:   "put-key-1",
		BaseURL:  "https://api.puter.com",
		Weight:   1,
	}
	puterPool := credentials.NewBalancedPool("puter/gpt-4o-mini", "round-robin", []*credentials.RuntimeCredential{puterCred}, nil)

	cacheStore.Set(cache.PoolKey("nvidia/meta/llama-3.3-70b-instruct"), nvidiaPool, 1)
	cacheStore.Set(cache.PoolKey("openrouter/meta-llama/llama-3.1-70b-instruct"), differentVersionPool, 1)
	cacheStore.Set(cache.PoolKey("puter/gpt-4o-mini"), puterPool, 1)
	cacheStore.Wait()

	sm := credentials.NewSyncManager(nil, cacheStore, nil, logger)
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"nvidia/meta/llama-3.3-70b-instruct":           nvidiaPool,
		"openrouter/meta-llama/llama-3.1-70b-instruct": differentVersionPool,
		"puter/gpt-4o-mini":                            puterPool,
	})

	calledProviders := 0

	mockClient := &http.Client{
		Transport: &mockRoundTripper{
			roundTripFunc: func(req *http.Request) (*http.Response, error) {
				calledProviders++
				resp := &http.Response{
					StatusCode: http.StatusServiceUnavailable,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"error": {"message": "NVIDIA model unavailable", "type": "server_error"}}`)),
				}
				resp.Header.Set("Content-Type", "application/json")
				return resp, nil
			},
		},
	}

	h := NewHandler(mockClient, cacheStore, nil, logger, nil, nil, nil)
	h.SetSyncManager(sm)

	router := gin.New()
	router.POST("/v1/chat/completions", h.Handle)

	reqPayload := `{"model": "nvidia/meta/llama-3.3-70b-instruct", "messages": [{"role": "user", "content": "test"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqPayload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	t.Logf("code=%d body=%s", w.Code, w.Body.String())

	// Since there is NO exact same model (only 3.1 or gpt-4o-mini), the gateway must throw the error!
	if w.Code == http.StatusOK {
		t.Fatalf("expected error response when no exact model exists, got HTTP 200: %s", w.Body.String())
	}

	// Must be an OpenAI error response format
	if !strings.Contains(w.Body.String(), `"error"`) {
		t.Errorf("expected OpenAI error response, got: %s", w.Body.String())
	}

	// Should only have called the 1 nvidia provider, never downgraded to openrouter 3.1 or puter
	if calledProviders != 1 {
		t.Errorf("expected exactly 1 provider attempt (nvidia), got %d", calledProviders)
	}

	// The error must be meaningful: name the requested model, the pool that was
	// tried, and how many rounds it consumed.
	if !strings.Contains(w.Body.String(), "nvidia/meta/llama-3.3-70b-instruct (1 round(s))") {
		t.Errorf("expected meaningful exhaustion summary listing the pool and round count, got: %s", w.Body.String())
	}
}

func TestNoLongDeactivationOnDepletedOrAuthError(t *testing.T) {
	gin.SetMode(gin.TestMode)
	logger := zap.NewNop()
	cfg := &config.Config{
		CacheMaxSizeMB:   10,
		CacheNumCounters: 100,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cacheStore.Close()

	cred := &credentials.RuntimeCredential{
		ID:       555,
		Provider: "custom",
		APIKey:   "test-key",
		BaseURL:  "https://custom-upstream.ai/v1",
		Weight:   1,
	}
	pool := credentials.NewBalancedPool("custom/test-model", "round-robin", []*credentials.RuntimeCredential{cred}, nil)
	cacheStore.Set(cache.PoolKey("custom/test-model"), pool, 1)
	cacheStore.Wait()

	mockClient := &http.Client{
		Transport: &mockRoundTripper{
			roundTripFunc: func(req *http.Request) (*http.Response, error) {
				resp := &http.Response{
					StatusCode: http.StatusPaymentRequired,
					Header:     make(http.Header),
					Body:       io.NopCloser(strings.NewReader(`{"error": {"message": "insufficient_balance: not_enough_balance", "type": "insufficient_quota"}}`)),
				}
				resp.Header.Set("Content-Type", "application/json")
				return resp, nil
			},
		},
	}

	h := NewHandler(mockClient, cacheStore, nil, logger, nil, nil, nil)
	router := gin.New()
	router.POST("/v1/chat/completions", h.Handle)

	reqPayload := `{"model": "custom/test-model", "messages": [{"role": "user", "content": "hi"}]}`
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(reqPayload))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()

	router.ServeHTTP(w, req)

	t.Logf("code=%d body=%s", w.Code, w.Body.String())

	// Check CooldownUntil
	now := time.Now().UnixNano()
	cooldownUntil := atomic.LoadInt64(&cred.CooldownUntil)
	if cooldownUntil <= now {
		t.Errorf("expected key to be cooling down, but CooldownUntil <= now")
	}

	duration := time.Duration(cooldownUntil - now)
	if duration > 30*time.Second {
		t.Fatalf("key was deactivated with long cooldown (%v), expected <= 15s transient cooldown", duration)
	}
}

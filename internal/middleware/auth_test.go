package middleware

import (
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/config"
	"github.com/skadraneshghn/clever-ai-gate/internal/database"
	"go.uber.org/zap"
)

func TestProxyAuth_MasterAdminKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	logger := zap.NewNop()
	cfg := &config.Config{
		CacheMaxSizeMB:   10,
		CacheNumCounters: 1000,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cacheStore.Close()

	adminKey := "test-master-admin-key-12345"
	mw := ProxyAuth(cacheStore, adminKey)

	r := gin.New()
	r.Use(mw)
	r.GET("/v1/models", func(c *gin.Context) {
		tenantID, _ := c.Get("tenant_id")
		tenantName, _ := c.Get("tenant_name")
		c.JSON(http.StatusOK, gin.H{
			"status":      "ok",
			"tenant_id":   tenantID,
			"tenant_name": tenantName,
		})
	})

	// 1. Missing Authorization header -> 401
	req := httptest.NewRequest("GET", "/v1/models", nil)
	w := httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for missing auth, got %d", w.Code)
	}

	// 2. Admin key -> 200 OK with tenant_id "admin"
	req = httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer "+adminKey)
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for master admin key, got %d", w.Code)
	}

	// 3. Invalid key -> 401
	req = httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer invalid-key-xyz")
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusUnauthorized {
		t.Fatalf("expected 401 for invalid key, got %d", w.Code)
	}

	// 4. Valid tenant key in cache -> 200 OK
	cacheStore.Set(cache.TenantKey("tenant-key-abc"), &database.TenantRow{
		ID:           "tenant-uuid-1",
		Name:         "Acme Corp",
		IsActive:     true,
		RateLimitRPM: 60,
		TokenBalance: 5000,
	}, 100)
	cacheStore.Wait()

	req = httptest.NewRequest("GET", "/v1/models", nil)
	req.Header.Set("Authorization", "Bearer tenant-key-abc")
	w = httptest.NewRecorder()
	r.ServeHTTP(w, req)
	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for valid tenant key, got %d", w.Code)
	}
}

// TestProxyAuth_AnthropicXAPIKey verifies x-api-key authentication on
// /v1/messages: Anthropic SDKs and Claude Code/Cursor/Cline send the key in
// the x-api-key header instead of Authorization: Bearer. All error responses
// on this path must use the Anthropic envelope format.
func TestProxyAuth_AnthropicXAPIKey(t *testing.T) {
	gin.SetMode(gin.TestMode)

	logger := zap.NewNop()
	cfg := &config.Config{
		CacheMaxSizeMB:   10,
		CacheNumCounters: 1000,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cacheStore.Close()

	adminKey := "test-master-admin-key-12345"
	mw := ProxyAuth(cacheStore, adminKey)

	r := gin.New()
	r.Use(mw)
	r.POST("/v1/messages", func(c *gin.Context) {
		tenantID, _ := c.Get("tenant_id")
		c.JSON(http.StatusOK, gin.H{"status": "ok", "tenant_id": tenantID})
	})

	cacheStore.Set(cache.TenantKey("anthropic-valid-key"), &database.TenantRow{
		ID:           "tenant-uuid-2",
		Name:         "Anthropic Tenant",
		IsActive:     true,
		RateLimitRPM: 60,
		TokenBalance: 5000,
	}, 100)
	cacheStore.Set(cache.TenantKey("anthropic-suspended-key"), &database.TenantRow{
		ID:           "tenant-uuid-3",
		Name:         "Suspended Tenant",
		IsActive:     false,
		RateLimitRPM: 60,
		TokenBalance: 5000,
	}, 100)
	cacheStore.Wait()

	tests := []struct {
		name        string
		headers     map[string]string
		wantStatus  int
		wantErrType string
		wantMessage string
	}{
		{
			name:       "admin key via x-api-key",
			headers:    map[string]string{"x-api-key": adminKey},
			wantStatus: http.StatusOK,
		},
		{
			name:       "valid tenant via x-api-key",
			headers:    map[string]string{"x-api-key": "anthropic-valid-key"},
			wantStatus: http.StatusOK,
		},
		{
			name:       "valid tenant via Bearer (OpenAI flow unchanged)",
			headers:    map[string]string{"Authorization": "Bearer anthropic-valid-key"},
			wantStatus: http.StatusOK,
		},
		{
			name:        "invalid key via x-api-key",
			headers:     map[string]string{"x-api-key": "invalid-key-xyz"},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "authentication_error",
			wantMessage: "invalid API key",
		},
		{
			name:        "suspended tenant via x-api-key",
			headers:     map[string]string{"x-api-key": "anthropic-suspended-key"},
			wantStatus:  http.StatusForbidden,
			wantErrType: "permission_error",
			wantMessage: "account is suspended",
		},
		{
			name:        "missing credentials",
			headers:     map[string]string{},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "authentication_error",
			wantMessage: "missing Authorization header",
		},
		{
			name:        "malformed Authorization wins over x-api-key",
			headers:     map[string]string{"Authorization": "Token abc", "x-api-key": adminKey},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "authentication_error",
			wantMessage: "invalid Authorization format, expected 'Bearer <key>'",
		},
		{
			name:        "empty Bearer key",
			headers:     map[string]string{"Authorization": "Bearer   "},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "authentication_error",
			wantMessage: "empty API key",
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(http.MethodPost, "/v1/messages", nil)
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)

			if w.Code != tc.wantStatus {
				t.Fatalf("expected status %d, got %d (body: %s)", tc.wantStatus, w.Code, w.Body.String())
			}
			if tc.wantStatus == http.StatusOK {
				return
			}

			body := w.Body.String()
			// encoding/json escapes '<'/'>' as \u003c/\u003e by default
			// (SetEscapeHTML). Unescape so messages containing e.g. '<key>'
			// can be matched literally.
			body = strings.ReplaceAll(body, `\u003c`, "<")
			body = strings.ReplaceAll(body, `\u003e`, ">")
			for _, want := range []string{
				`"type":"error"`,
				`"type":"` + tc.wantErrType + `"`,
				`"message":"` + tc.wantMessage + `"`,
			} {
				if !strings.Contains(body, want) {
					t.Fatalf("expected Anthropic envelope containing %s, got %s", want, body)
				}
			}
		})
	}
}

// TestProxyAuth_OpenAIPathsKeepFlatErrors verifies that OpenAI-compatible
// paths keep their original flat {"error": ...} envelope, while a request
// carrying an explicit anthropic-version header is treated as an Anthropic
// client regardless of path (it gets the nested envelope).
func TestProxyAuth_OpenAIPathsKeepFlatErrors(t *testing.T) {
	gin.SetMode(gin.TestMode)

	logger := zap.NewNop()
	cfg := &config.Config{
		CacheMaxSizeMB:   10,
		CacheNumCounters: 1000,
	}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	defer cacheStore.Close()

	adminKey := "test-master-admin-key-12345"
	mw := ProxyAuth(cacheStore, adminKey)

	r := gin.New()
	r.Use(mw)
	okHandler := func(c *gin.Context) {
		c.JSON(http.StatusOK, gin.H{"status": "ok"})
	}
	r.GET("/v1/models", okHandler)
	r.POST("/v1/chat/completions", okHandler)

	cacheStore.Set(cache.TenantKey("flat-valid-key"), &database.TenantRow{
		ID:           "tenant-uuid-4",
		Name:         "OpenAI Tenant",
		IsActive:     true,
		RateLimitRPM: 60,
		TokenBalance: 5000,
	}, 100)
	cacheStore.Wait()

	tests := []struct {
		name        string
		method      string
		path        string
		headers     map[string]string
		wantStatus  int
		wantErrType string // empty: flat {"error":...}; otherwise Anthropic envelope
		wantMessage string
	}{
		{
			name:        "x-api-key alone on chat/completions keeps flat error",
			method:      http.MethodPost,
			path:        "/v1/chat/completions",
			headers:     map[string]string{"x-api-key": "invalid-key-xyz"},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "",
			wantMessage: "invalid API key",
		},
		{
			name:        "no credentials on models keeps flat error",
			method:      http.MethodGet,
			path:        "/v1/models",
			headers:     map[string]string{},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "",
			wantMessage: "missing Authorization header",
		},
		{
			name:        "suspended tenant on chat/completions keeps flat error",
			method:      http.MethodPost,
			path:        "/v1/chat/completions",
			headers:     map[string]string{"Authorization": "Bearer invalid-key-xyz"},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "",
			wantMessage: "invalid API key",
		},
		{
			name:   "anthropic-version header triggers Anthropic envelope on OpenAI path",
			method: http.MethodGet,
			path:   "/v1/models",
			headers: map[string]string{
				"Authorization":     "Bearer invalid-key-xyz",
				"anthropic-version": "2023-06-01",
			},
			wantStatus:  http.StatusUnauthorized,
			wantErrType: "authentication_error",
			wantMessage: "invalid API key",
		},
		{
			name:       "Bearer still authenticates on chat/completions",
			method:     http.MethodPost,
			path:       "/v1/chat/completions",
			headers:    map[string]string{"Authorization": "Bearer flat-valid-key"},
			wantStatus: http.StatusOK,
		},
	}

	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			req := httptest.NewRequest(tc.method, tc.path, nil)
			for k, v := range tc.headers {
				req.Header.Set(k, v)
			}
			w := httptest.NewRecorder()
			r.ServeHTTP(w, req)

			if w.Code != tc.wantStatus {
				t.Fatalf("expected status %d, got %d (body: %s)", tc.wantStatus, w.Code, w.Body.String())
			}
			if tc.wantStatus == http.StatusOK {
				return
			}

			body := w.Body.String()
			// Unescape HTML entities (\u003c/\u003e) so literal messages
			// containing '<key>' can be matched.
			body = strings.ReplaceAll(body, `\u003c`, "<")
			body = strings.ReplaceAll(body, `\u003e`, ">")
			if tc.wantErrType == "" {
				if !strings.Contains(body, `"error":"`+tc.wantMessage+`"`) {
					t.Fatalf("expected flat error body %q, got %s", tc.wantMessage, body)
				}
				if strings.Contains(body, `"type":"error"`) {
					t.Fatalf("OpenAI path must keep the flat error format, got %s", body)
				}
				return
			}
			for _, want := range []string{
				`"type":"error"`,
				`"type":"` + tc.wantErrType + `"`,
				`"message":"` + tc.wantMessage + `"`,
			} {
				if !strings.Contains(body, want) {
					t.Fatalf("expected Anthropic envelope containing %s, got %s", want, body)
				}
			}
		})
	}
}

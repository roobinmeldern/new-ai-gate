package proxy

import (
	"encoding/json"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/config"
	"go.uber.org/zap"
)

// End-to-end Phase-1 dispatch tests for the Anthropic-compatible front API.
//
// These tests drive Handler.Handle exactly the way the engine's proxy group
// does: the middleware chain (CORS, auth, rate limit) is covered by
// internal/middleware tests; here we verify that a request with a live
// (unread) body on /v1/messages and /v1/messages/count_tokens is dispatched
// to the Anthropic front door and answered with Anthropic-shaped errors,
// while the OpenAI pipeline on /v1/chat/completions stays untouched.

// anthropicErrorEnvelope mirrors the wire format written by
// AnthropicErrorResponse: {"type":"error","error":{"type":..,"message":..}}.
type anthropicErrorEnvelope struct {
	Type  string `json:"type"`
	Error struct {
		Type    string `json:"type"`
		Message string `json:"message"`
	} `json:"error"`
}

// newAnthropicFrontTestHandler builds a minimal but real Handler for the
// Anthropic dispatch path. The Anthropic branch returns before any cache,
// pool or upstream access, so no mock upstream is needed; the cache store is
// still constructed to mirror the setup used by handler_large_payload_test.go.
func newAnthropicFrontTestHandler(t *testing.T) *Handler {
	t.Helper()
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
	t.Cleanup(func() { cacheStore.Close() })
	return NewHandler(nil, cacheStore, nil, logger, nil, nil, nil)
}

// runFrontRequest posts body to path through Handler.Handle with a live
// request body (the middleware chain consumes c.Request.Body, so Handle must
// buffer it itself — this exercises that contract).
func runFrontRequest(h *Handler, path, body string) *httptest.ResponseRecorder {
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = httptest.NewRequest(http.MethodPost, path, strings.NewReader(body))
	c.Request.Header.Set("Content-Type", "application/json")
	h.Handle(c)
	return w
}

func decodeAnthropicEnvelope(t *testing.T, raw string) anthropicErrorEnvelope {
	t.Helper()
	var env anthropicErrorEnvelope
	if err := json.Unmarshal([]byte(raw), &env); err != nil {
		t.Fatalf("response is not valid JSON: %v\nbody: %s", err, raw)
	}
	return env
}

func TestHandle_AnthropicMessagesDispatchNoPool(t *testing.T) {
	h := newAnthropicFrontTestHandler(t)
	w := runFrontRequest(h, "/v1/messages",
		`{"model":"claude-3-5-sonnet","max_tokens":1024,"messages":[{"role":"user","content":"Hello"}]}`)

	if w.Code != http.StatusNotFound {
		t.Fatalf("expected 404 for valid /v1/messages request with no configured pool, got %d: %s", w.Code, w.Body.String())
	}
	env := decodeAnthropicEnvelope(t, w.Body.String())
	if env.Type != "error" || env.Error.Type != "not_found_error" || env.Error.Message == "" {
		t.Fatalf("expected Anthropic not_found_error envelope, got: %s", w.Body.String())
	}
}

func TestHandle_AnthropicMessagesValidationErrors(t *testing.T) {
	h := newAnthropicFrontTestHandler(t)

	cases := []struct {
		name string
		body string
	}{
		{"missing model", `{"max_tokens":10,"messages":[{"role":"user","content":"hi"}]}`},
		{"missing max_tokens", `{"model":"claude-3-5-sonnet","messages":[{"role":"user","content":"hi"}]}`},
		{"zero max_tokens", `{"model":"claude-3-5-sonnet","max_tokens":0,"messages":[{"role":"user","content":"hi"}]}`},
		{"empty messages", `{"model":"claude-3-5-sonnet","max_tokens":1,"messages":[]}`},
		{"empty body", ``},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			w := runFrontRequest(h, "/v1/messages", tc.body)
			if w.Code != http.StatusBadRequest {
				t.Fatalf("expected 400, got %d: %s", w.Code, w.Body.String())
			}
			env := decodeAnthropicEnvelope(t, w.Body.String())
			if env.Type != "error" || env.Error.Type != "invalid_request_error" {
				t.Fatalf("expected invalid_request_error envelope, got: %s", w.Body.String())
			}
		})
	}
}

func TestHandle_AnthropicCountTokensDispatch(t *testing.T) {
	h := newAnthropicFrontTestHandler(t)
	// count_tokens must not require max_tokens (per the Messages API spec,
	// only /v1/messages does) and must accept a streaming flag silently.
	w := runFrontRequest(h, "/v1/messages/count_tokens",
		`{"model":"claude-3-5-sonnet","stream":true,"messages":[{"role":"user","content":"Hello"}]}`)

	if w.Code != http.StatusOK {
		t.Fatalf("expected 200 for valid count_tokens request, got %d: %s", w.Code, w.Body.String())
	}
	var res map[string]int
	if err := json.Unmarshal(w.Body.Bytes(), &res); err != nil {
		t.Fatalf("expected valid JSON response, got: %s", w.Body.String())
	}
	if res["input_tokens"] <= 0 {
		t.Fatalf("expected positive input_tokens, got %d", res["input_tokens"])
	}
}

func TestHandle_OpenAIPathNotCapturedByAnthropicFront(t *testing.T) {
	h := newAnthropicFrontTestHandler(t)
	// An OpenAI-protocol request must never receive the Anthropic 501
	// envelope: the dispatch must leave the OpenAI pipeline fully in charge.
	w := runFrontRequest(h, "/v1/chat/completions",
		`{"model":"definitely-not-cached-model-xyz","messages":[{"role":"user","content":"hi"}],"stream":false}`)

	var generic map[string]any
	if err := json.Unmarshal(w.Body.Bytes(), &generic); err != nil {
		t.Fatalf("response is not valid JSON: %v\nbody: %s", err, w.Body.String())
	}
	// Anthropic envelope shape is {"type":"error","error":{...}} — the OpenAI
	// pipeline must keep its flat {"error": ...} shape with no "type" key.
	if _, hasType := generic["type"]; hasType {
		t.Fatalf("OpenAI path must not receive the Anthropic envelope: %s", w.Body.String())
	}
	if w.Code == http.StatusNotImplemented {
		t.Fatalf("OpenAI path must not return the Anthropic 501: %s", w.Body.String())
	}
	if _, hasErr := generic["error"]; !hasErr {
		t.Fatalf("expected the flat OpenAI error object, got: %s", w.Body.String())
	}
}

package proxy

import (
	"encoding/json"
	"testing"
)

// ── IsFixedParamReasoningModel ────────────────────────────────────────────────

func TestIsFixedParamReasoningModel_TruePositives(t *testing.T) {
	cases := []struct {
		name  string
		model string
	}{
		{"gpt-5.6-sol", "gpt-5.6-sol"},
		{"jiekou prefixed gpt-5.6-sol", "jiekou/gpt-5.6-sol"},
		{"gpt-5o", "gpt-5o"},
		{"gpt-5-mini", "gpt-5-mini"},
		{"o1", "o1"},
		{"o1-mini", "o1-mini"},
		{"o1-preview", "o1-preview"},
		{"o1-pro", "o1-pro"},
		{"o3", "o3"},
		{"o3-mini", "o3-mini"},
		{"o4-mini", "o4-mini"},
		{"gpt-4.5-sol", "gpt-4.5-sol"},
		{"some-reasoning-model", "some-reasoning-model"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if !IsFixedParamReasoningModel(tc.model) {
				t.Errorf("IsFixedParamReasoningModel(%q) = false, want true", tc.model)
			}
		})
	}
}

func TestIsFixedParamReasoningModel_FalseNegatives(t *testing.T) {
	cases := []struct {
		name  string
		model string
	}{
		{"gpt-4o", "gpt-4o"},
		{"gpt-4o-mini", "gpt-4o-mini"},
		{"gpt-4-turbo", "gpt-4-turbo"},
		{"moonshotai/kimi-k3", "moonshotai/kimi-k3"},
		{"deepseek-chat", "deepseek-chat"},
		{"claude-3-5-sonnet", "claude-3-5-sonnet"},
		{"llama-3.3-70b", "llama-3.3-70b"},
	}
	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			if IsFixedParamReasoningModel(tc.model) {
				t.Errorf("IsFixedParamReasoningModel(%q) = true, want false", tc.model)
			}
		})
	}
}

// ── sanitizeJiekouRequest — Layer 1: Universal ────────────────────────────────

func TestSanitizeJiekouRequest_StripPrefix(t *testing.T) {
	input := `{"model":"jiekou/moonshotai/kimi-k3","messages":[{"role":"user","content":"hello"}],"temperature":1.0}`
	result := sanitizeJiekouRequest([]byte(input))

	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw: %s", err, result)
	}
	if model, _ := out["model"].(string); model != "moonshotai/kimi-k3" {
		t.Errorf("expected model=moonshotai/kimi-k3, got %q", model)
	}
}

func TestSanitizeJiekouRequest_UnsupportedFieldsRemoved(t *testing.T) {
	body := `{
		"model": "moonshotai/kimi-k3",
		"messages": [{"role":"user","content":"hi"}],
		"temperature": 0.7,
		"logit_bias": {"12345": 5},
		"user": "user-abc",
		"logprobs": true,
		"top_logprobs": 3,
		"stream_options": {"include_usage": true},
		"service_tier": "default"
	}`
	result := sanitizeJiekouRequest([]byte(body))
	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}

	removedFields := []string{"logit_bias", "user", "logprobs", "top_logprobs", "stream_options", "service_tier"}
	for _, f := range removedFields {
		if _, exists := out[f]; exists {
			t.Errorf("field %q should have been removed but is still present", f)
		}
	}
	if _, ok := out["model"]; !ok {
		t.Error("model field was incorrectly removed")
	}
	if _, ok := out["messages"]; !ok {
		t.Error("messages field was incorrectly removed")
	}
}

// ── sanitizeJiekouRequest — Layer 2: Kimi/Moonshot ───────────────────────────

func TestSanitizeJiekouRequest_KimiTemperatureClamp(t *testing.T) {
	tests := []struct {
		name    string
		tempIn  float64
		tempOut float64
	}{
		{"above max (1.2)", 1.2, 0.7},
		{"above max (2.0)", 2.0, 0.7},
		{"below min (-0.1)", -0.1, 0.1},
		{"at max boundary (1.0)", 1.0, 1.0},
		{"within range (0.7)", 0.7, 0.7},
	}
	for _, tc := range tests {
		t.Run(tc.name, func(t *testing.T) {
			body, _ := json.Marshal(map[string]interface{}{
				"model":       "moonshotai/kimi-k3",
				"temperature": tc.tempIn,
				"messages":    []interface{}{},
			})
			result := sanitizeJiekouRequest(body)
			var out map[string]interface{}
			if err := json.Unmarshal(result, &out); err != nil {
				t.Fatalf("invalid JSON: %v", err)
			}
			gotTemp, ok := out["temperature"].(float64)
			if !ok {
				t.Fatalf("temperature missing or not float64")
			}
			if gotTemp != tc.tempOut {
				t.Errorf("temperature: got %v, want %v", gotTemp, tc.tempOut)
			}
		})
	}
}

// ── sanitizeJiekouRequest — Layer 3: Reasoning/Beta GPT fixed-param ──────────

func TestSanitizeJiekouRequest_ReasoningModel_FixedParams(t *testing.T) {
	// The exact real-world failure: client sends temperature=0.2, presence_penalty=0.1
	// Upstream says: "temperature, top_p and n are fixed at 1, presence_penalty and
	// frequency_penalty are fixed at 0"
	body := `{
		"model": "gpt-5.6-sol",
		"messages": [{"role":"user","content":"hello"}],
		"temperature": 0.2,
		"top_p": 0.95,
		"presence_penalty": 0.1,
		"frequency_penalty": 0.05,
		"n": 3,
		"max_tokens": 512
	}`
	result := sanitizeJiekouRequest([]byte(body))
	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, result)
	}

	// All these must be overridden to the fixed values
	assertFloat(t, out, "temperature", 1.0)
	assertFloat(t, out, "top_p", 1.0)
	assertFloat(t, out, "n", 1.0)
	assertFloat(t, out, "presence_penalty", 0.0)
	assertFloat(t, out, "frequency_penalty", 0.0)

	// max_tokens must be migrated to max_completion_tokens
	if _, exists := out["max_tokens"]; exists {
		t.Error("max_tokens should have been removed (migrated to max_completion_tokens)")
	}
	if v, exists := out["max_completion_tokens"]; !exists {
		t.Error("max_completion_tokens should be present after migration")
	} else if v.(float64) != 512 {
		t.Errorf("max_completion_tokens: got %v, want 512", v)
	}
}

func TestSanitizeJiekouRequest_ReasoningModel_PrefixedFullFlow(t *testing.T) {
	// The original failing request: jiekou/gpt-5.6-sol with client defaults
	body := `{"model":"jiekou/gpt-5.6-sol","messages":[{"role":"user","content":"hi"}],"temperature":0.2,"top_p":0.95,"presence_penalty":0.1,"frequency_penalty":0.05}`
	result := sanitizeJiekouRequest([]byte(body))
	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("invalid JSON: %v\nraw: %s", err, result)
	}

	// Prefix must be stripped
	if model, _ := out["model"].(string); model != "gpt-5.6-sol" {
		t.Errorf("model prefix not stripped: got %q, want gpt-5.6-sol", model)
	}
	// Fixed params must be enforced
	assertFloat(t, out, "temperature", 1.0)
	assertFloat(t, out, "presence_penalty", 0.0)
	assertFloat(t, out, "frequency_penalty", 0.0)
}

func TestSanitizeJiekouRequest_ReasoningModel_VariantNames(t *testing.T) {
	variants := []string{"o1", "o1-mini", "o1-preview", "o3", "o3-mini", "o4-mini", "gpt-5o"}
	for _, model := range variants {
		t.Run(model, func(t *testing.T) {
			body, _ := json.Marshal(map[string]interface{}{
				"model":            model,
				"messages":         []interface{}{},
				"temperature":      0.5,
				"presence_penalty": 0.2,
			})
			result := sanitizeJiekouRequest(body)
			var out map[string]interface{}
			if err := json.Unmarshal(result, &out); err != nil {
				t.Fatalf("invalid JSON: %v", err)
			}
			assertFloat(t, out, "temperature", 1.0)
			assertFloat(t, out, "presence_penalty", 0.0)
		})
	}
}

func TestSanitizeJiekouRequest_ReasoningModel_MaxCompletionTokensNotOverwritten(t *testing.T) {
	// If max_completion_tokens is already set, max_tokens migration must not overwrite it.
	body := `{"model":"o1-mini","messages":[],"max_tokens":100,"max_completion_tokens":50}`
	result := sanitizeJiekouRequest([]byte(body))
	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if v, _ := out["max_completion_tokens"].(float64); v != 50 {
		t.Errorf("max_completion_tokens overwritten: got %v, want 50", v)
	}
	if _, exists := out["max_tokens"]; exists {
		t.Error("max_tokens should have been removed")
	}
}

// ── Kimi + prefix combo ───────────────────────────────────────────────────────

func TestSanitizeJiekouRequest_PrefixAndTemperatureTogether(t *testing.T) {
	body := `{"model":"jiekou/moonshotai/kimi-k3","messages":[{"role":"user","content":"hi"}],"temperature":1.5,"user":"test-user"}`
	result := sanitizeJiekouRequest([]byte(body))
	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("invalid JSON: %v", err)
	}
	if model, _ := out["model"].(string); model != "moonshotai/kimi-k3" {
		t.Errorf("model prefix not stripped: got %q", model)
	}
	if temp, ok := out["temperature"].(float64); !ok || temp != 0.7 {
		t.Errorf("temperature not clamped: got %v", out["temperature"])
	}
	if _, exists := out["user"]; exists {
		t.Error("user field should have been removed")
	}
}

// ── Semantic preservation ─────────────────────────────────────────────────────

func TestSanitizeJiekouRequest_CleanBody_SemanticPreserved(t *testing.T) {
	// Valid standard model body — temperature triggers full parse, but semantics preserved.
	input := []byte(`{"model":"moonshotai/kimi-k2","messages":[{"role":"user","content":"hello"}],"temperature":0.8}`)
	result := sanitizeJiekouRequest(input)

	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("result is not valid JSON: %v\nraw: %s", err, result)
	}
	if model, _ := out["model"].(string); model != "moonshotai/kimi-k2" {
		t.Errorf("model changed: got %q, want moonshotai/kimi-k2", model)
	}
	if temp, ok := out["temperature"].(float64); !ok || temp != 0.8 {
		t.Errorf("temperature changed: got %v, want 0.8", out["temperature"])
	}
	if _, ok := out["messages"]; !ok {
		t.Error("messages field was removed")
	}
	for _, f := range jiekouUnsupportedFields {
		if _, exists := out[f]; exists {
			t.Errorf("field %q should not be present in output", f)
		}
	}
}

// ── helpers ───────────────────────────────────────────────────────────────────

func assertFloat(t *testing.T, out map[string]interface{}, key string, want float64) {
	t.Helper()
	v, ok := out[key].(float64)
	if !ok {
		t.Errorf("field %q missing or wrong type (got %T=%v)", key, out[key], out[key])
		return
	}
	if v != want {
		t.Errorf("field %q: got %v, want %v", key, v, want)
	}
}

// ── Layer 1b: Role Normalization ──────────────────────────────────────────────

// TestSanitizeJiekouRequest_DeveloperRoleNormalized validates that IDE clients
// (Cursor, Cline, Roo Code) sending role:"developer" get it transparently
// rewritten to role:"system" so Moonshot/Kimi upstream does not reject with 400.
func TestSanitizeJiekouRequest_DeveloperRoleNormalized(t *testing.T) {
	input := `{
		"model": "jiekou/moonshotai/kimi-k3",
		"messages": [
			{"role": "developer", "content": "You are a helpful assistant."},
			{"role": "user", "content": "Hello!"}
		]
	}`
	result := sanitizeJiekouRequest([]byte(input))

	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw: %s", err, result)
	}

	messages, ok := out["messages"].([]interface{})
	if !ok || len(messages) < 2 {
		t.Fatalf("expected 2 messages in output, got: %v", out["messages"])
	}
	firstMsg, ok := messages[0].(map[string]interface{})
	if !ok {
		t.Fatalf("first message is not a map: %T", messages[0])
	}
	if role, _ := firstMsg["role"].(string); role != "system" {
		t.Errorf("expected role \"system\" after normalization, got %q", role)
	}
	// Second message (user) must remain unchanged.
	secondMsg, _ := messages[1].(map[string]interface{})
	if role, _ := secondMsg["role"].(string); role != "user" {
		t.Errorf("user role should remain unchanged, got %q", role)
	}
}

// TestSanitizeJiekouRequest_NullContentReplaced validates that null content
// in a message is replaced with an empty string to avoid Moonshot 400 rejections.
func TestSanitizeJiekouRequest_NullContentReplaced(t *testing.T) {
	input := `{
		"model": "moonshotai/kimi-k3",
		"messages": [
			{"role": "system", "content": null},
			{"role": "user", "content": "Hi"}
		]
	}`
	result := sanitizeJiekouRequest([]byte(input))

	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw: %s", err, result)
	}

	messages, _ := out["messages"].([]interface{})
	if len(messages) < 1 {
		t.Fatal("expected at least 1 message")
	}
	firstMsg, _ := messages[0].(map[string]interface{})
	if content, _ := firstMsg["content"].(string); content != "" {
		t.Errorf("expected empty string for null content, got %q", content)
	}
}

// TestSanitizeJiekouRequest_EmptyToolsRemoved validates that an empty tools array
// is removed (Moonshot returns 400 when tools:[] is present).
func TestSanitizeJiekouRequest_EmptyToolsRemoved(t *testing.T) {
	input := `{
		"model": "moonshotai/kimi-k3",
		"messages": [{"role": "user", "content": "hi"}],
		"tools": [],
		"tool_choice": "none"
	}`
	result := sanitizeJiekouRequest([]byte(input))

	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw: %s", err, result)
	}
	if _, ok := out["tools"]; ok {
		t.Error("expected 'tools' to be removed when empty, but it was present")
	}
	if _, ok := out["tool_choice"]; ok {
		t.Error("expected 'tool_choice' to be removed alongside empty tools, but it was present")
	}
}

// TestSanitizeJiekouRequest_KimiFieldsRemoved validates that Kimi/Moonshot models
// have presence_penalty, frequency_penalty, and seed stripped.
func TestSanitizeJiekouRequest_KimiFieldsRemoved(t *testing.T) {
	input := `{
		"model": "moonshotai/kimi-k3",
		"messages": [{"role": "user", "content": "hi"}],
		"temperature": 0.8,
		"presence_penalty": 0.5,
		"frequency_penalty": 0.3,
		"seed": 42
	}`
	result := sanitizeJiekouRequest([]byte(input))

	var out map[string]interface{}
	if err := json.Unmarshal(result, &out); err != nil {
		t.Fatalf("output is not valid JSON: %v\nraw: %s", err, result)
	}

	for _, field := range []string{"presence_penalty", "frequency_penalty", "seed"} {
		if _, ok := out[field]; ok {
			t.Errorf("field %q should have been removed for kimi/moonshot model, but was present", field)
		}
	}
	// temperature within [0,1] should be preserved
	assertFloat(t, out, "temperature", 0.8)
}

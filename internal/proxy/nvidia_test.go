package proxy

import (
	"encoding/json"
	"testing"
)

// --- supportsNvidiaReasoning tests ---

func TestSupportsNvidiaReasoning_TruePositives(t *testing.T) {
	cases := []struct {
		model string
	}{
		{"nvidia/nemotron-3-super-120b-a12b"},
		{"nemotron-70b"},
		{"deepseek/deepseek-r1"},
		{"deepseek-r1-distill-llama-70b"},
		{"qwen/qwq-32b-reasoning"},
		{"meta/llama-3.3-70b-think"},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if !supportsNvidiaReasoning(tc.model) {
				t.Errorf("supportsNvidiaReasoning(%q) = false; want true", tc.model)
			}
		})
	}
}

func TestSupportsNvidiaReasoning_FalsePositiveRegression(t *testing.T) {
	// These models must NOT trigger reasoning injection — they are standard
	// (non-thinking) models whose *organisation name* happens to contain a
	// reasoning keyword.
	cases := []struct {
		model   string
		comment string
	}{
		{
			"thinkingmachines/inkling",
			"org name 'thinkingmachines' contains 'think' but model 'inkling' does not",
		},
		{
			"reasoningcorp/llama-3-8b",
			"org name 'reasoningcorp' contains 'reasoning' but model 'llama-3-8b' does not",
		},
		{
			"r1-labs/mistral-7b",
			"org name 'r1-labs' — final segment is 'mistral-7b', no keyword match",
		},
	}
	for _, tc := range cases {
		t.Run(tc.model, func(t *testing.T) {
			if supportsNvidiaReasoning(tc.model) {
				t.Errorf("supportsNvidiaReasoning(%q) = true; want false (%s)", tc.model, tc.comment)
			}
		})
	}
}

func TestSupportsNvidiaReasoning_FalseNegatives(t *testing.T) {
	// Standard models that should never match.
	cases := []string{
		"z-ai/glm-5.1",
		"meta/llama-3.3-70b-instruct",
		"mistralai/mistral-small",
		"google/gemma-3-27b-it",
	}
	for _, model := range cases {
		t.Run(model, func(t *testing.T) {
			if supportsNvidiaReasoning(model) {
				t.Errorf("supportsNvidiaReasoning(%q) = true; want false", model)
			}
		})
	}
}

// --- sanitizeNvidiaRequest tests ---

func TestSanitizeNvidiaRequest_StripsUnsupportedFields(t *testing.T) {
	input := map[string]interface{}{
		"model":       "thinkingmachines/inkling",
		"messages":    []interface{}{map[string]interface{}{"role": "user", "content": "hi"}},
		"temperature": 0.7,
		"max_tokens":  512,
		"stream":      false,
		// OpenAI-only fields that NVIDIA rejects:
		"stream_options":      map[string]interface{}{"include_usage": true},
		"logprobs":            true,
		"top_logprobs":        5,
		"service_tier":        "auto",
		"store":               false,
		"metadata":            map[string]interface{}{"user_id": "abc"},
		"parallel_tool_calls": true,
		"logit_bias":          map[string]interface{}{},
		"suffix":              "done",
	}
	body, _ := json.Marshal(input)

	out := sanitizeNvidiaRequest(body)

	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("sanitizeNvidiaRequest produced invalid JSON: %v", err)
	}

	// Verify legal fields are preserved
	for _, keep := range []string{"model", "messages", "temperature", "max_tokens", "stream"} {
		if _, ok := result[keep]; !ok {
			t.Errorf("sanitizeNvidiaRequest removed valid field %q", keep)
		}
	}

	// Verify illegal fields are stripped
	for _, strip := range nvidiaUnsupportedFields {
		if _, ok := result[strip]; ok {
			t.Errorf("sanitizeNvidiaRequest kept unsupported field %q", strip)
		}
	}
}

func TestSanitizeNvidiaRequest_CleanPayloadNoAlloc(t *testing.T) {
	// A clean payload (no unsupported fields) must be returned unchanged.
	input := `{"model":"meta/llama-3.3-70b-instruct","messages":[{"role":"user","content":"hello"}],"max_tokens":256}`
	body := []byte(input)

	out := sanitizeNvidiaRequest(body)

	// When no unsupported fields are present the original slice is returned as-is
	// (the fast-probe exits early). Byte equality is sufficient for this check.
	if string(out) != string(body) {
		t.Errorf("sanitizeNvidiaRequest mutated a clean payload:\ngot:  %s\nwant: %s", out, body)
	}
}

func TestSanitizeNvidiaRequest_ReasoningBudgetNotStripped(t *testing.T) {
	// reasoning_budget is NOT in nvidiaUnsupportedFields — it should be preserved
	// when the upstream model supports it (injected earlier by injectNvidiaParams).
	input := `{"model":"nvidia/nemotron-3","messages":[],"reasoning_budget":4096,"chat_template_kwargs":{"enable_thinking":true}}`
	body := []byte(input)

	out := sanitizeNvidiaRequest(body)

	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("invalid JSON after sanitize: %v", err)
	}
	if _, ok := result["reasoning_budget"]; !ok {
		t.Error("sanitizeNvidiaRequest incorrectly stripped 'reasoning_budget' (should be preserved for reasoning models)")
	}
	if _, ok := result["chat_template_kwargs"]; !ok {
		t.Error("sanitizeNvidiaRequest incorrectly stripped 'chat_template_kwargs' (should be preserved for reasoning models)")
	}
}

func TestSanitizeNvidiaRequest_ClampsZeroTemperature(t *testing.T) {
	input := `{"model":"nvidia/google/gemma-2-2b-it","messages":[{"role":"user","content":"hi"}],"temperature":0}`
	body := []byte(input)

	out := sanitizeNvidiaRequest(body)

	var result map[string]interface{}
	if err := json.Unmarshal(out, &result); err != nil {
		t.Fatalf("invalid JSON after sanitize: %v", err)
	}

	temp, ok := result["temperature"].(float64)
	if !ok || temp <= 0 {
		t.Errorf("sanitizeNvidiaRequest did not clamp temperature 0 to > 0 value, got %v", result["temperature"])
	}
}

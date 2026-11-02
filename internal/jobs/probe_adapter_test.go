package jobs

import (
	"encoding/json"
	"strings"
	"testing"
)

func TestCleanUpstreamModel(t *testing.T) {
	cases := []struct {
		pattern  string
		provider string
		want     string
	}{
		{"nvidia/google/gemma-2-2b-it", "nvidia", "google/gemma-2-2b-it"},
		{"nvidia/nvidia/nemoretriever-parse", "nvidia", "nvidia/nemoretriever-parse"},
		{"cloudflare/@cf/meta/llama-3-8b-instruct", "cloudflare", "@cf/meta/llama-3-8b-instruct"},
		{"ollama/llama3", "ollama", "llama3"},
		{"1min/claude-3-5-sonnet", "1min", "claude-3-5-sonnet"},
		{"huggingface/meta-llama/Llama-3.3-70B-Instruct", "huggingface", "meta-llama/Llama-3.3-70B-Instruct"},
	}

	for _, tc := range cases {
		t.Run(tc.pattern, func(t *testing.T) {
			got := CleanUpstreamModel(tc.pattern, tc.provider)
			if got != tc.want {
				t.Errorf("CleanUpstreamModel(%q, %q) = %q; want %q", tc.pattern, tc.provider, got, tc.want)
			}
		})
	}
}

func TestBuildAdaptiveProbeRequest_Embedding(t *testing.T) {
	req, err := BuildAdaptiveProbeRequest("https://api.nvidia.com/v1", "key123", "nvidia", "nvidia/google/bge-m3", map[string]bool{"embedding": true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.HasSuffix(req.URL, "/v1/embeddings") && !strings.HasSuffix(req.URL, "/embeddings") {
		t.Errorf("expected embedding endpoint URL, got %s", req.URL)
	}

	var body map[string]interface{}
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatalf("invalid json body: %v", err)
	}

	if body["input"] != "health check" {
		t.Errorf("expected input 'health check', got %v", body["input"])
	}
	if body["model"] != "google/bge-m3" {
		t.Errorf("expected model 'google/bge-m3', got %v", body["model"])
	}
}

func TestBuildAdaptiveProbeRequest_NvidiaTemperatureOverride(t *testing.T) {
	req, err := BuildAdaptiveProbeRequest("https://integrate.api.nvidia.com/v1", "key123", "nvidia", "nvidia/google/gemma-2-2b-it", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var body map[string]interface{}
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatalf("invalid json body: %v", err)
	}

	temp, ok := body["temperature"].(float64)
	if !ok || temp <= 0 {
		t.Errorf("expected temperature > 0 for NVIDIA, got %v", body["temperature"])
	}
	if body["model"] != "google/gemma-2-2b-it" {
		t.Errorf("expected model 'google/gemma-2-2b-it', got %v", body["model"])
	}
}

func TestBuildAdaptiveProbeRequest_ParserMultimodal(t *testing.T) {
	req, err := BuildAdaptiveProbeRequest("https://integrate.api.nvidia.com/v1", "key123", "nvidia", "nvidia/nvidia/nemoretriever-parse", nil)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var body map[string]interface{}
	if err := json.Unmarshal(req.Body, &body); err != nil {
		t.Fatalf("invalid json body: %v", err)
	}

	if body["model"] != "nvidia/nemoretriever-parse" {
		t.Errorf("expected model 'nvidia/nemoretriever-parse', got %v", body["model"])
	}

	messages, ok := body["messages"].([]interface{})
	if !ok || len(messages) == 0 {
		t.Fatalf("expected non-empty messages array")
	}

	firstMsg := messages[0].(map[string]interface{})
	contentArray, ok := firstMsg["content"].([]interface{})
	if !ok || len(contentArray) == 0 {
		t.Errorf("expected non-empty content array for parse model, got %v", firstMsg["content"])
	}
}

func TestBuildAdaptiveProbeRequest_ImageGen(t *testing.T) {
	req, err := BuildAdaptiveProbeRequest("https://api.openai.com/v1", "key123", "openai", "dall-e-3", map[string]bool{"image_generation": true})
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	if !strings.Contains(req.URL, "/images/generations") {
		t.Errorf("expected image generation URL, got %s", req.URL)
	}
}

func TestBuildAdaptiveProbeRequest_AnthropicAndAgentRouter(t *testing.T) {
	// 1. Anthropic native endpoint
	antReq, err := BuildAdaptiveProbeRequest("https://api.anthropic.com/v1", "sk-ant-test", "anthropic", "anthropic/claude-3-5-sonnet", nil)
	if err != nil {
		t.Fatalf("unexpected error for anthropic: %v", err)
	}

	if !strings.HasSuffix(antReq.URL, "/messages") {
		t.Errorf("expected anthropic URL ending in /messages, got %s", antReq.URL)
	}
	if antReq.Headers["x-api-key"] != "sk-ant-test" {
		t.Errorf("expected x-api-key header, got %v", antReq.Headers["x-api-key"])
	}
	if antReq.Headers["anthropic-version"] != "2023-06-01" {
		t.Errorf("expected anthropic-version header, got %v", antReq.Headers["anthropic-version"])
	}

	var antBody map[string]interface{}
	if err := json.Unmarshal(antReq.Body, &antBody); err != nil {
		t.Fatalf("invalid json body: %v", err)
	}
	if antBody["model"] != "claude-3-5-sonnet" {
		t.Errorf("expected stripped model 'claude-3-5-sonnet', got %v", antBody["model"])
	}
	if antBody["max_tokens"] != float64(1) {
		t.Errorf("expected max_tokens: 1, got %v", antBody["max_tokens"])
	}

	// 2. AgentRouter native endpoint
	arReq, err := BuildAdaptiveProbeRequest("https://agentrouter.org", "ar-test-key", "agentrouter", "agentrouter/claude-3-5-sonnet", nil)
	if err != nil {
		t.Fatalf("unexpected error for agentrouter: %v", err)
	}
	if !strings.HasSuffix(arReq.URL, "/v1/messages") {
		t.Errorf("expected agentrouter URL ending in /v1/messages, got %s", arReq.URL)
	}
	if arReq.Headers["x-api-key"] != "ar-test-key" {
		t.Errorf("expected x-api-key header for agentrouter, got %v", arReq.Headers["x-api-key"])
	}
}


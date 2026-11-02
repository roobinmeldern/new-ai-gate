package proxy

import (
	"encoding/json"
	"testing"

	"github.com/buger/jsonparser"
)

func TestTranslateOpenAIToAnthropicResponse_TextAndThinking(t *testing.T) {
	// Case 1: Native reasoning_content
	openAIJSON1 := []byte(`{
		"id": "chatcmpl-test-1",
		"object": "chat.completion",
		"created": 1700000000,
		"model": "deepseek-reasoner",
		"choices": [
			{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "42 is the answer.",
					"reasoning_content": "Calculating the answer to life, universe, and everything."
				},
				"finish_reason": "stop"
			}
		],
		"usage": {
			"prompt_tokens": 25,
			"completion_tokens": 15,
			"total_tokens": 40
		}
	}`)

	resp1, promptTokens1, completionTokens1, respText1, err := TranslateOpenAIToAnthropicResponse(openAIJSON1, "claude-3-7-sonnet", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !json.Valid(resp1) {
		t.Fatalf("generated Anthropic response is not valid JSON: %s", string(resp1))
	}
	if promptTokens1 != 25 || completionTokens1 != 15 {
		t.Errorf("expected tokens (25, 15), got (%d, %d)", promptTokens1, completionTokens1)
	}
	if respText1 == "" {
		t.Error("expected non-empty responseText for telemetry")
	}

	model1, _ := jsonparser.GetString(resp1, "model")
	if model1 != "claude-3-7-sonnet" {
		t.Errorf("expected model 'claude-3-7-sonnet', got %q", model1)
	}

	stopReason1, _ := jsonparser.GetString(resp1, "stop_reason")
	if stopReason1 != "end_turn" {
		t.Errorf("expected stop_reason 'end_turn', got %q", stopReason1)
	}

	// Verify content blocks: thinking first, then text
	block0Type, _ := jsonparser.GetString(resp1, "content", "[0]", "type")
	if block0Type != "thinking" {
		t.Fatalf("expected block 0 to be 'thinking', got %q", block0Type)
	}
	thinkingText, _ := jsonparser.GetString(resp1, "content", "[0]", "thinking")
	if thinkingText != "Calculating the answer to life, universe, and everything." {
		t.Errorf("unexpected thinking text: %q", thinkingText)
	}
	sig, _ := jsonparser.GetString(resp1, "content", "[0]", "signature")
	if sig != "sig_clever_gate" {
		t.Errorf("expected signature 'sig_clever_gate', got %q", sig)
	}

	block1Type, _ := jsonparser.GetString(resp1, "content", "[1]", "type")
	if block1Type != "text" {
		t.Fatalf("expected block 1 to be 'text', got %q", block1Type)
	}
	textVal, _ := jsonparser.GetString(resp1, "content", "[1]", "text")
	if textVal != "42 is the answer." {
		t.Errorf("unexpected text content: %q", textVal)
	}

	// Case 2: Inline <think> tag
	openAIJSON2 := []byte(`{
		"id": "chatcmpl-test-2",
		"model": "qwen-2.5-max",
		"choices": [
			{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "<think>Let me evaluate 2+2.</think>The result is 4."
				},
				"finish_reason": "stop"
			}
		],
		"usage": {
			"prompt_tokens": 10,
			"completion_tokens": 8
		}
	}`)

	resp2, promptTokens2, completionTokens2, _, err := TranslateOpenAIToAnthropicResponse(openAIJSON2, "", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !json.Valid(resp2) {
		t.Fatalf("generated Anthropic response is not valid JSON: %s", string(resp2))
	}
	if promptTokens2 != 10 || completionTokens2 != 8 {
		t.Errorf("expected tokens (10, 8), got (%d, %d)", promptTokens2, completionTokens2)
	}

	// Should have thinking block then text block
	b0Type, _ := jsonparser.GetString(resp2, "content", "[0]", "type")
	if b0Type != "thinking" {
		t.Fatalf("expected block 0 to be 'thinking', got %q", b0Type)
	}
	b0Thinking, _ := jsonparser.GetString(resp2, "content", "[0]", "thinking")
	if b0Thinking != "Let me evaluate 2+2." {
		t.Errorf("unexpected thinking: %q", b0Thinking)
	}

	b1Type, _ := jsonparser.GetString(resp2, "content", "[1]", "type")
	if b1Type != "text" {
		t.Fatalf("expected block 1 to be 'text', got %q", b1Type)
	}
	b1Text, _ := jsonparser.GetString(resp2, "content", "[1]", "text")
	if b1Text != "The result is 4." {
		t.Errorf("unexpected text: %q", b1Text)
	}
}

func TestTranslateOpenAIToAnthropicResponse_ToolCalls(t *testing.T) {
	openAIJSON := []byte(`{
		"id": "chatcmpl-tool-test",
		"model": "gpt-4o",
		"choices": [
			{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "I will check the weather for you.",
					"tool_calls": [
						{
							"id": "call_123456",
							"type": "function",
							"function": {
								"name": "get_weather",
								"arguments": "{\"city\":\"San Francisco\",\"unit\":\"celsius\"}"
							}
						}
					]
				},
				"finish_reason": "tool_calls"
			}
		],
		"usage": {
			"prompt_tokens": 50,
			"completion_tokens": 20
		}
	}`)

	resp, pTokens, cTokens, _, err := TranslateOpenAIToAnthropicResponse(openAIJSON, "claude-3-5-sonnet", 0)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if !json.Valid(resp) {
		t.Fatalf("generated Anthropic response is not valid JSON: %s", string(resp))
	}
	if pTokens != 50 || cTokens != 20 {
		t.Errorf("expected (50, 20), got (%d, %d)", pTokens, cTokens)
	}

	stopReason, _ := jsonparser.GetString(resp, "stop_reason")
	if stopReason != "tool_use" {
		t.Errorf("expected stop_reason 'tool_use', got %q", stopReason)
	}

	// Content should contain text block followed by tool_use block
	b0Type, _ := jsonparser.GetString(resp, "content", "[0]", "type")
	if b0Type != "text" {
		t.Errorf("expected block 0 to be 'text', got %q", b0Type)
	}

	b1Type, _ := jsonparser.GetString(resp, "content", "[1]", "type")
	if b1Type != "tool_use" {
		t.Fatalf("expected block 1 to be 'tool_use', got %q", b1Type)
	}

	toolID, _ := jsonparser.GetString(resp, "content", "[1]", "id")
	if toolID != "call_123456" {
		t.Errorf("expected tool id 'call_123456', got %q", toolID)
	}

	toolName, _ := jsonparser.GetString(resp, "content", "[1]", "name")
	if toolName != "get_weather" {
		t.Errorf("expected tool name 'get_weather', got %q", toolName)
	}

	city, _ := jsonparser.GetString(resp, "content", "[1]", "input", "city")
	if city != "San Francisco" {
		t.Errorf("expected input.city 'San Francisco', got %q", city)
	}

	unit, _ := jsonparser.GetString(resp, "content", "[1]", "input", "unit")
	if unit != "celsius" {
		t.Errorf("expected input.unit 'celsius', got %q", unit)
	}
}

func BenchmarkTranslateOpenAIToAnthropicResponse(b *testing.B) {
	input := []byte(`{
		"id": "chatcmpl-bench",
		"object": "chat.completion",
		"created": 1729000000,
		"model": "gpt-4o",
		"choices": [
			{
				"index": 0,
				"message": {
					"role": "assistant",
					"content": "<think>Quick thought</think>Here is the response",
					"tool_calls": [
						{
							"id": "call_bench_1",
							"type": "function",
							"function": {
								"name": "lookup",
								"arguments": "{\"query\":\"test\"}"
							}
						}
					]
				},
				"finish_reason": "tool_calls"
			}
		],
		"usage": {
			"prompt_tokens": 50,
			"completion_tokens": 20,
			"total_tokens": 70,
			"prompt_tokens_details": {
				"cached_tokens": 25
			}
		}
	}`)

	b.ResetTimer()
	b.ReportAllocs()

	for i := 0; i < b.N; i++ {
		_, _, _, _, err := TranslateOpenAIToAnthropicResponse(input, "claude-3-5-sonnet", 50)
		if err != nil {
			b.Fatal(err)
		}
	}
}

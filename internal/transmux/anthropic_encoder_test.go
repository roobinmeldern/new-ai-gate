package transmux

import (
	"bufio"
	"bytes"
	"encoding/json"
	"strings"
	"testing"
)

// parseSSEEvents extracts all (event, data) pairs from a raw SSE byte stream.
func parseSSEEvents(t *testing.T, raw []byte) []struct{ event, data string } {
	t.Helper()
	var events []struct{ event, data string }
	scanner := bufio.NewScanner(bytes.NewReader(raw))
	currentEvent := ""
	currentData := ""
	for scanner.Scan() {
		line := scanner.Text()
		if line == "" {
			if currentEvent != "" || currentData != "" {
				events = append(events, struct{ event, data string }{currentEvent, currentData})
				currentEvent = ""
				currentData = ""
			}
			continue
		}
		if strings.HasPrefix(line, "event: ") {
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			currentData = strings.TrimPrefix(line, "data: ")
		}
	}
	if currentEvent != "" || currentData != "" {
		events = append(events, struct{ event, data string }{currentEvent, currentData})
	}
	return events
}

func TestAnthropicStreamEncoder_TextOnly(t *testing.T) {
	enc := NewAnthropicStreamEncoder("msg_123", "claude-3-5-sonnet", 10)

	chunks := [][]byte{
		[]byte(`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Hello"}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{"content":" world!"}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`),
		[]byte(`{"usage":{"prompt_tokens":12,"completion_tokens":5}}`),
	}

	var output []byte
	for _, chunk := range chunks {
		output = append(output, enc.EncodeOpenAIChunk(chunk)...)
	}
	output = append(output, enc.Finalize()...)

	events := parseSSEEvents(t, output)
	if len(events) == 0 {
		t.Fatal("expected SSE events, got none")
	}

	// Validate every data payload is valid JSON
	for i, ev := range events {
		if !json.Valid([]byte(ev.data)) {
			t.Errorf("event %d (%s) data is not valid JSON: %s", i, ev.event, ev.data)
		}
	}

	expectedSequence := []string{
		"message_start",
		"content_block_start",
		"content_block_delta",
		"content_block_delta",
		"content_block_stop",
		"message_delta",
		"message_stop",
	}

	if len(events) != len(expectedSequence) {
		t.Fatalf("expected %d events, got %d: %+v", len(expectedSequence), len(events), events)
	}

	for i, exp := range expectedSequence {
		if events[i].event != exp {
			t.Errorf("event %d: expected %s, got %s", i, exp, events[i].event)
		}
	}

	if enc.InputTokens() != 12 {
		t.Errorf("expected 12 input tokens, got %d", enc.InputTokens())
	}
	if enc.OutputTokens() != 5 {
		t.Errorf("expected 5 output tokens, got %d", enc.OutputTokens())
	}
	if enc.StopReason() != "end_turn" {
		t.Errorf("expected stop_reason end_turn, got %s", enc.StopReason())
	}
}

func TestAnthropicStreamEncoder_ThinkingThenText(t *testing.T) {
	enc := NewAnthropicStreamEncoder("msg_think", "claude-3-7-sonnet", 25)

	chunks := [][]byte{
		[]byte(`{"choices":[{"index":0,"delta":{"role":"assistant","reasoning_content":"Let me think"}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{"reasoning_content":" carefully."}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{"content":"The answer is 42."}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"stop"}],"usage":{"prompt_tokens":25,"completion_tokens":18}}`),
	}

	var output []byte
	for _, chunk := range chunks {
		output = append(output, enc.EncodeOpenAIChunk(chunk)...)
	}
	output = append(output, enc.Finalize()...)

	events := parseSSEEvents(t, output)
	for i, ev := range events {
		if !json.Valid([]byte(ev.data)) {
			t.Errorf("event %d (%s) data is not valid JSON: %s", i, ev.event, ev.data)
		}
	}

	// Verify thinking block index 0 followed by signature_delta, content_block_stop
	// then text block at index 1
	var foundSignature bool
	for _, ev := range events {
		if ev.event == "content_block_delta" && strings.Contains(ev.data, `"signature_delta"`) {
			foundSignature = true
			if !strings.Contains(ev.data, `"signature":"sig_clever_gate"`) {
				t.Errorf("expected sig_clever_gate in signature_delta, got %s", ev.data)
			}
			if !strings.Contains(ev.data, `"index":0`) {
				t.Errorf("expected signature_delta on index 0, got %s", ev.data)
			}
		}
		if ev.event == "content_block_start" && strings.Contains(ev.data, `"text"`) {
			if !strings.Contains(ev.data, `"index":1`) {
				t.Errorf("expected text block on index 1, got %s", ev.data)
			}
		}
	}
	if !foundSignature {
		t.Error("expected signature_delta before closing thinking block, but none was found")
	}
}

func TestAnthropicStreamEncoder_MultiToolCalls(t *testing.T) {
	enc := NewAnthropicStreamEncoder("msg_tools", "claude-3-5-sonnet", 50)

	chunks := [][]byte{
		[]byte(`{"choices":[{"index":0,"delta":{"role":"assistant","content":"Calling tools now."}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"id":"call_1","type":"function","function":{"name":"read_file","arguments":"{\"path\":"}}]}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"\"main.go\"}"}}]}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{"tool_calls":[{"index":1,"id":"call_2","type":"function","function":{"name":"list_dir","arguments":"{\"dir\":\"./\"}"}}]}}]}`),
		[]byte(`{"choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}],"usage":{"prompt_tokens":50,"completion_tokens":35}}`),
	}

	var output []byte
	for _, chunk := range chunks {
		output = append(output, enc.EncodeOpenAIChunk(chunk)...)
	}
	output = append(output, enc.Finalize()...)

	events := parseSSEEvents(t, output)
	for i, ev := range events {
		if !json.Valid([]byte(ev.data)) {
			t.Errorf("event %d (%s) data is not valid JSON: %s", i, ev.event, ev.data)
		}
	}

	var toolCallsStarted int
	for _, ev := range events {
		if ev.event == "content_block_start" && strings.Contains(ev.data, `"tool_use"`) {
			toolCallsStarted++
			if toolCallsStarted == 1 && !strings.Contains(ev.data, `"name":"read_file"`) {
				t.Errorf("expected first tool to be read_file, got %s", ev.data)
			}
			if toolCallsStarted == 2 && !strings.Contains(ev.data, `"name":"list_dir"`) {
				t.Errorf("expected second tool to be list_dir, got %s", ev.data)
			}
		}
	}

	if toolCallsStarted != 2 {
		t.Fatalf("expected 2 tool_use content blocks, got %d", toolCallsStarted)
	}
	if enc.StopReason() != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %s", enc.StopReason())
	}
}

func TestAnthropicStreamEncoder_GeminiMultiPartStream(t *testing.T) {
	enc := NewAnthropicStreamEncoder("msg_gemini", "claude-3-5-sonnet", 40)

	multiChunk := []byte("data: {\"choices\":[{\"index\":0,\"delta\":{\"reasoning_content\":\"thinking step\"}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{\"tool_calls\":[{\"index\":0,\"id\":\"call_g1\",\"type\":\"function\",\"function\":{\"name\":\"search\",\"arguments\":\"{}\"}}]}}]}\n\ndata: {\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"tool_calls\"}]}")

	out := enc.EncodeOpenAIChunk(multiChunk)
	out = append(out, enc.Finalize()...)

	events := parseSSEEvents(t, out)
	for i, ev := range events {
		if !json.Valid([]byte(ev.data)) {
			t.Errorf("event %d (%s) data is not valid JSON: %s", i, ev.event, ev.data)
		}
	}

	var hasThinking, hasToolUse bool
	for _, ev := range events {
		if ev.event == "content_block_start" {
			if strings.Contains(ev.data, `"thinking"`) {
				hasThinking = true
				if !strings.Contains(ev.data, `"index":0`) {
					t.Errorf("expected thinking on index 0, got %s", ev.data)
				}
			}
			if strings.Contains(ev.data, `"tool_use"`) {
				hasToolUse = true
				if !strings.Contains(ev.data, `"index":1`) {
					t.Errorf("expected tool_use on index 1, got %s", ev.data)
				}
			}
		}
	}

	if !hasThinking || !hasToolUse {
		t.Errorf("expected both thinking and tool_use blocks, got hasThinking=%v, hasToolUse=%v", hasThinking, hasToolUse)
	}
	if enc.StopReason() != "tool_use" {
		t.Errorf("expected stop_reason tool_use, got %s", enc.StopReason())
	}
}

func BenchmarkAnthropicStreamEncoder(b *testing.B) {
	chunk := []byte(`{"choices":[{"index":0,"delta":{"content":"Hello world, streaming fast!"}}]}`)
	b.ReportAllocs()
	b.ResetTimer()

	for i := 0; i < b.N; i++ {
		enc := NewAnthropicStreamEncoder("msg_bench", "claude-3-5-sonnet", 100)
		_ = enc.EncodeOpenAIChunk(chunk)
		_ = enc.Finalize()
	}
}

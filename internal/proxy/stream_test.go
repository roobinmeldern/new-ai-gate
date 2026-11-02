package proxy

import (
	"bytes"
	"encoding/json"
	"strings"
	"testing"

	"github.com/gin-gonic/gin"
)

func TestSanitizeStreamingChunk_CloudflareToolCallsNulls(t *testing.T) {
	// Exact chunk received from Cloudflare Workers AI that crashed Cline:
	// "delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"id":null,"index":0,"type":"function","function":{"name":null,"arguments":"{\"expression\": \"2+"}}]}
	rawChunk := []byte(`{"id":"id-1792661346675","created":1792661346,"model":"@cf/zai-org/glm-5.3-flash","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"role":null,"content":null,"reasoning_content":null,"tool_calls":[{"id":null,"index":0,"type":"function","function":{"name":null,"arguments":"{\"expression\": \"2+"}}]},"logprobs":null,"finish_reason":null,"matched_stop":null}],"usage":{"prompt_tokens":0,"completion_tokens":0,"total_tokens":0}}`)

	cleaned := sanitizeStreamingChunk(rawChunk)

	var parsed map[string]interface{}
	if err := json.Unmarshal(cleaned, &parsed); err != nil {
		t.Fatalf("failed to unmarshal cleaned chunk: %v", err)
	}

	choices := parsed["choices"].([]interface{})
	ch0 := choices[0].(map[string]interface{})
	delta := ch0["delta"].(map[string]interface{})

	// Verify null fields are removed from delta
	if _, exists := delta["role"]; exists {
		t.Errorf("expected 'role' to be removed from delta, but it exists: %v", delta["role"])
	}
	if _, exists := delta["content"]; exists {
		t.Errorf("expected 'content' to be removed from delta, but it exists: %v", delta["content"])
	}
	if _, exists := delta["reasoning_content"]; exists {
		t.Errorf("expected 'reasoning_content' to be removed from delta, but it exists: %v", delta["reasoning_content"])
	}

	tcs := delta["tool_calls"].([]interface{})
	tc0 := tcs[0].(map[string]interface{})

	// Verify 'id': null was removed
	if idVal, exists := tc0["id"]; exists {
		t.Errorf("expected 'id' to be removed from tool_calls[0], but got: %v", idVal)
	}

	fn := tc0["function"].(map[string]interface{})

	// Verify 'name': null was replaced with string so clients like Cline do not throw "Expected 'function.name' to be a string"
	if nameVal, exists := fn["name"]; !exists || nameVal != "" {
		t.Errorf("expected 'name' to be string in function, but got: %v", nameVal)
	}

	// Verify arguments are intact
	if args, ok := fn["arguments"].(string); !ok || args != `{"expression": "2+` {
		t.Errorf("unexpected arguments: %v", fn["arguments"])
	}
}

func TestSanitizeStreamingChunk_ChunkZeroPreserved(t *testing.T) {
	// Chunk 0 where tool call starts: has valid id, type, name, and empty arguments
	chunk0 := []byte(`{"id":"id-123","created":123,"model":"@cf/model","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_123","index":0,"type":"function","function":{"name":"calculate","arguments":""}}]},"finish_reason":null}]}`)

	cleaned := sanitizeStreamingChunk(chunk0)

	var parsed map[string]interface{}
	if err := json.Unmarshal(cleaned, &parsed); err != nil {
		t.Fatalf("failed to unmarshal cleaned chunk: %v", err)
	}

	choices := parsed["choices"].([]interface{})
	delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
	tcs := delta["tool_calls"].([]interface{})
	tc0 := tcs[0].(map[string]interface{})

	if tc0["id"] != "call_123" {
		t.Errorf("expected id 'call_123', got %v", tc0["id"])
	}
	fn := tc0["function"].(map[string]interface{})
	if fn["name"] != "calculate" {
		t.Errorf("expected function name 'calculate', got %v", fn["name"])
	}
	if fn["arguments"] != "" {
		t.Errorf("expected function arguments '', got %v", fn["arguments"])
	}
}

func TestSanitizeStreamingChunk_FastPathNoToolCalls(t *testing.T) {
	textChunk := []byte(`{"id":"id-123","choices":[{"delta":{"content":"Hello"}}]}`)
	cleaned := sanitizeStreamingChunk(textChunk)
	if string(cleaned) != string(textChunk) {
		t.Errorf("expected identical chunk on fast path, got %s", string(cleaned))
	}
}

type mockStreamWriter struct {
	gin.ResponseWriter
	buf bytes.Buffer
}

func (m *mockStreamWriter) Write(p []byte) (int, error) {
	return m.buf.Write(p)
}

func (m *mockStreamWriter) Flush() {}

func TestStreamToolSanitizer_EndToEnd(t *testing.T) {
	ts := newStreamToolSanitizer()

	chunk1 := []byte(`{"id":"id-1","created":100,"model":"@cf/glm-5.3","choices":[{"index":0,"delta":{"tool_calls":[{"id":"call_1","index":0,"type":"function","function":{"name":"read_files","arguments":""}}]},"finish_reason":null}]}`)
	chunk2 := []byte(`{"id":"id-1","created":100,"model":"@cf/glm-5.3","choices":[{"index":0,"delta":{"tool_calls":[{"index":0,"function":{"arguments":"` + "```json\\n{\\\"files\\\": [\\\"foo.go\\\"]}\\n```" + `"}}]},"finish_reason":null}]}`)

	absorbed1, rem1 := ts.Accumulate(chunk1)
	if !absorbed1 || len(rem1) > 0 {
		t.Fatalf("expected chunk1 to be fully absorbed")
	}

	absorbed2, rem2 := ts.Accumulate(chunk2)
	if !absorbed2 || len(rem2) > 0 {
		t.Fatalf("expected chunk2 to be fully absorbed")
	}

	if !ts.HasPending() {
		t.Fatalf("expected pending tools in sanitizer")
	}

	mockWriter := &mockStreamWriter{}
	sw := &sseWriter{w: mockWriter}

	if err := ts.Flush(sw, "@cf/glm-5.3"); err != nil {
		t.Fatalf("flush failed: %v", err)
	}

	out := mockWriter.buf.String()
	if !strings.HasPrefix(out, "data: ") || !strings.HasSuffix(out, "\n\n") {
		t.Fatalf("invalid SSE event output: %s", out)
	}

	payload := strings.TrimPrefix(out, "data: ")
	payload = strings.TrimSuffix(payload, "\n\n")

	var chunk map[string]interface{}
	if err := json.Unmarshal([]byte(payload), &chunk); err != nil {
		t.Fatalf("failed to unmarshal flushed chunk: %v", err)
	}

	choices := chunk["choices"].([]interface{})
	delta := choices[0].(map[string]interface{})["delta"].(map[string]interface{})
	tcs := delta["tool_calls"].([]interface{})
	tc0 := tcs[0].(map[string]interface{})

	if tc0["id"] != "call_1" {
		t.Errorf("expected id call_1, got %v", tc0["id"])
	}
	fn := tc0["function"].(map[string]interface{})
	if fn["name"] != "read_files" {
		t.Errorf("expected name read_files, got %v", fn["name"])
	}
	args := fn["arguments"].(string)
	if !strings.Contains(args, `{"path":"foo.go"}`) {
		t.Errorf("expected repaired read_files args, got %s", args)
	}
}


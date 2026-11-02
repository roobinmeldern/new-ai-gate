package proxy

import (
	"encoding/json"
	"strings"
	"testing"
)

// --- Test helpers ---

// parseJSON converts a JSON string to a map for assertions.
func parseJSON(t *testing.T, body string) map[string]any {
	t.Helper()
	var m map[string]any
	if err := json.Unmarshal([]byte(body), &m); err != nil {
		t.Fatalf("invalid JSON: %v\nbody: %s", err, body)
	}
	return m
}

// mustTranslate runs TranslateAnthropicToOpenAI and fails on error.
func mustTranslate(t *testing.T, raw []byte, model string, fallback int64) []byte {
	t.Helper()
	out, err := TranslateAnthropicToOpenAI(raw, model, fallback)
	if err != nil {
		t.Fatalf("TranslateAnthropicToOpenAI failed: %v", err)
	}
	return out
}

// --- Test 1: system string and blocks ---

func TestTranslateAnthropicToOpenAI_SystemStringAndBlocks(t *testing.T) {
	raw := []byte(`{
		"model": "claude-3-5-sonnet",
		"system": "You are a helpful assistant.",
		"max_tokens": 1024,
		"messages": [{"role": "user", "content": "Hello"}]
	}`)
	out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
	got := parseJSON(t, string(out))

	msgs, ok := got["messages"].([]any)
	if !ok || len(msgs) == 0 {
		t.Fatalf("expected non-empty messages, got: %s", out)
	}
	first, _ := msgs[0].(map[string]any)
	if first["role"] != "system" {
		t.Fatalf("expected leading system message, got: %v", first)
	}
	if first["content"] != "You are a helpful assistant." {
		t.Fatalf("unexpected system content: %v", first["content"])
	}
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages (system + user), got %d", len(msgs))
	}

	rawBlocks := []byte(`{
		"model": "claude-3-5-sonnet",
		"system": [
			{"type": "text", "text": "First block."},
			{"type": "text", "text": "Second block."}
		],
		"max_tokens": 512,
		"messages": [{"role": "user", "content": "Hi"}]
	}`)
	outBlocks := mustTranslate(t, rawBlocks, "claude-3-5-sonnet", 4096)
	gotBlocks := parseJSON(t, string(outBlocks))
	msgsBlocks, _ := gotBlocks["messages"].([]any)
	firstBlocks, _ := msgsBlocks[0].(map[string]any)
	if firstBlocks["role"] != "system" {
		t.Fatalf("expected leading system message for block form, got: %v", firstBlocks)
	}
	if firstBlocks["content"] != "First block.\n\nSecond block." {
		t.Fatalf("expected joined block system content, got: %v", firstBlocks["content"])
	}
}

// --- Test 2: multimodal images ---

func TestTranslateAnthropicToOpenAI_MultimodalImages(t *testing.T) {
	raw := []byte(`{
		"model": "claude-3-5-sonnet",
		"max_tokens": 100,
		"messages": [{
			"role": "user",
			"content": [
				{"type": "text", "text": "What is in this image?"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "iVBORw0KGgo="}},
				{"type": "image", "source": {"type": "url", "url": "https://example.com/cat.png"}}
			]
		}]
	}`)
	out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
	got := parseJSON(t, string(out))

	msgs, _ := got["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected 1 message, got %d", len(msgs))
	}
	user, _ := msgs[0].(map[string]any)
	if user["role"] != "user" {
		t.Fatalf("expected user role, got %v", user["role"])
	}
	content, ok := user["content"].([]any)
	if !ok {
		t.Fatalf("expected multimodal content array, got: %v", user["content"])
	}
	if len(content) != 3 {
		t.Fatalf("expected 3 content parts (text + 2 images), got %d", len(content))
	}

	textPart, _ := content[0].(map[string]any)
	if textPart["type"] != "text" || textPart["text"] != "What is in this image?" {
		t.Fatalf("unexpected text part: %v", textPart)
	}
	b64Part, _ := content[1].(map[string]any)
	if b64Part["type"] != "image_url" {
		t.Fatalf("expected image_url type for base64 source, got %v", b64Part)
	}
	b64URL, _ := b64Part["image_url"].(map[string]any)
	if b64URL == nil || b64URL["url"] != "data:image/png;base64,iVBORw0KGgo=" {
		t.Fatalf("unexpected base64 image url: %v", b64Part["image_url"])
	}
	urlPart, _ := content[2].(map[string]any)
	if urlPart["type"] != "image_url" {
		t.Fatalf("expected image_url type for url source, got %v", urlPart)
	}
	urlObj, _ := urlPart["image_url"].(map[string]any)
	if urlObj == nil || urlObj["url"] != "https://example.com/cat.png" {
		t.Fatalf("unexpected url image: %v", urlPart["image_url"])
	}
}

// --- Test 3: tool use and parallel tool results ---

func TestTranslateAnthropicToOpenAI_ToolUseAndParallelToolResults(t *testing.T) {
	raw := []byte(`{
		"model": "claude-3-5-sonnet",
		"max_tokens": 2048,
		"messages": [
			{"role": "user", "content": "What is the weather in Paris and London?"},
			{
				"role": "assistant",
				"content": [
					{"type": "text", "text": "Let me check both cities."},
					{"type": "thinking", "thinking": "I need two tool calls in parallel."},
					{"type": "tool_use", "id": "toolu_01", "name": "get_weather", "input": {"city": "Paris"}},
					{"type": "tool_use", "name": "get_weather", "input": {"city": "London"}}
				]
			},
			{
				"role": "user",
				"content": [
					{"type": "tool_result", "tool_use_id": "toolu_01", "content": "22C, sunny"},
					{"type": "tool_result", "tool_use_id": "toolu_02", "content": [{"type": "text", "text": "14C, raining"}], "is_error": true},
					{"type": "text", "text": "Now compare them."}
				]
			}
		]
	}`)
	out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
	got := parseJSON(t, string(out))

	msgs, _ := got["messages"].([]any)
	if len(msgs) != 5 {
		t.Fatalf("expected 5 messages (user, assistant, tool, tool, user), got %d: %s", len(msgs), out)
	}

	// Assistant message: text + reasoning + 2 tool calls.
	assistant, _ := msgs[1].(map[string]any)
	if assistant["role"] != "assistant" {
		t.Fatalf("expected assistant message at index 1, got %v", assistant)
	}
	if assistant["content"] != "Let me check both cities." {
		t.Fatalf("unexpected assistant content: %v", assistant["content"])
	}
	if assistant["reasoning_content"] != "I need two tool calls in parallel." {
		t.Fatalf("unexpected reasoning_content: %v", assistant["reasoning_content"])
	}
	toolCalls, ok := assistant["tool_calls"].([]any)
	if !ok || len(toolCalls) != 2 {
		t.Fatalf("expected 2 tool_calls, got: %v", assistant["tool_calls"])
	}
	call0, _ := toolCalls[0].(map[string]any)
	if call0["id"] != "toolu_01" {
		t.Fatalf("expected toolu_01 id, got %v", call0["id"])
	}
	if call0["type"] != "function" {
		t.Fatalf("expected function type, got %v", call0["type"])
	}
	fn0, _ := call0["function"].(map[string]any)
	if fn0["name"] != "get_weather" {
		t.Fatalf("unexpected function name: %v", fn0)
	}
	argsStr, _ := fn0["arguments"].(string)
	args := parseJSON(t, argsStr)
	if args["city"] != "Paris" {
		t.Fatalf("unexpected decoded arguments: %v", args)
	}

	// Second tool_use had no id — must get the call_gate fallback.
	call1, _ := toolCalls[1].(map[string]any)
	if !strings.HasPrefix(call1["id"].(string), "call_gate_") {
		t.Fatalf("expected call_gate fallback id, got %v", call1["id"])
	}

	// Tool messages come before the trailing user text.
	tool0, _ := msgs[2].(map[string]any)
	if tool0["role"] != "tool" || tool0["tool_call_id"] != "toolu_01" {
		t.Fatalf("unexpected first tool message: %v", tool0)
	}
	if tool0["content"] != "22C, sunny" {
		t.Fatalf("unexpected tool content: %v", tool0["content"])
	}
	tool1, _ := msgs[3].(map[string]any)
	if tool1["role"] != "tool" || tool1["tool_call_id"] != "toolu_02" {
		t.Fatalf("unexpected second tool message: %v", tool1)
	}
	// is_error: true must add the "Error: " prefix.
	if tool1["content"] != "Error: 14C, raining" {
		t.Fatalf("expected Error: prefix for is_error tool_result, got %v", tool1["content"])
	}
	last, _ := msgs[4].(map[string]any)
	if last["role"] != "user" || last["content"] != "Now compare them." {
		t.Fatalf("unexpected trailing user message: %v", last)
	}
}

// --- Test 4: tools and tool_choice ---

func TestTranslateAnthropicToOpenAI_ToolsAndToolChoice(t *testing.T) {
	base := `{
		"model": "claude-3-5-sonnet",
		"max_tokens": 100,
		"tools": [
			{
				"name": "get_weather",
				"description": "Get current weather",
				"input_schema": {"type": "object", "properties": {"city": {"type": "string"}}}
			},
			{
				"type": "function",
				"function": {"name": "legacy_tool", "description": "hybrid", "parameters": {"type": "object"}}
			}
		],
		"messages": [{"role": "user", "content": "hi"}],
		"tool_choice": %s
	}`

	cases := []struct {
		name       string
		toolChoice string
		want       any
	}{
		{"auto", `{"type": "auto"}`, "auto"},
		{"any becomes required", `{"type": "any"}`, "required"},
		{"none", `{"type": "none"}`, "none"},
		{"named tool", `{"type": "tool", "name": "get_weather"}`, nil},
	}

	for _, tc := range cases {
		t.Run(tc.name, func(t *testing.T) {
			raw := []byte(strings.Replace(base, "%s", tc.toolChoice, 1))
			out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
			got := parseJSON(t, string(out))

			tools, ok := got["tools"].([]any)
			if !ok || len(tools) != 2 {
				t.Fatalf("expected 2 tools, got: %v", got["tools"])
			}
			tool0, _ := tools[0].(map[string]any)
			if tool0["type"] != "function" {
				t.Fatalf("expected function type tool, got %v", tool0)
			}
			fn0, _ := tool0["function"].(map[string]any)
			if fn0 == nil || fn0["name"] != "get_weather" {
				t.Fatalf("unexpected converted tool: %v", tool0)
			}
			if _, hasSchema := fn0["parameters"]; !hasSchema {
				t.Fatalf("expected parameters copied from input_schema: %v", fn0)
			}
			tool1, _ := tools[1].(map[string]any)
			fn1, _ := tool1["function"].(map[string]any)
			if fn1 == nil || fn1["name"] != "legacy_tool" {
				t.Fatalf("expected hybrid tool preserved, got %v", tool1)
			}

			switch want := tc.want.(type) {
			case string:
				if got["tool_choice"] != want {
					t.Fatalf("expected tool_choice %q, got %v", want, got["tool_choice"])
				}
			case nil:
				obj, ok := got["tool_choice"].(map[string]any)
				if !ok {
					t.Fatalf("expected object tool_choice, got %v", got["tool_choice"])
				}
				fn, _ := obj["function"].(map[string]any)
				if fn["name"] != "get_weather" {
					t.Fatalf("expected named tool_choice, got %v", obj)
				}
			}
		})
	}

	// disable_parallel_tool_use maps to parallel_tool_calls: false.
	raw := []byte(strings.Replace(base, "%s", `{"type": "auto", "disable_parallel_tool_use": true}`, 1))
	out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
	got := parseJSON(t, string(out))
	if got["parallel_tool_calls"] != false {
		t.Fatalf("expected parallel_tool_calls false, got %v", got["parallel_tool_calls"])
	}
}

// --- Test 5: extended thinking and reasoning models ---

func TestTranslateAnthropicToOpenAI_ExtendedThinkingAndReasoningModels(t *testing.T) {
	// Budget mapping: <4096 low, >=16384 high, otherwise medium.
	cases := []struct {
		budget int
		want   string
	}{
		{1024, "low"},
		{4095, "low"},
		{4096, "medium"},
		{8192, "medium"},
		{16383, "medium"},
		{16384, "high"},
		{32768, "high"},
	}
	for _, tc := range cases {
		t.Run("budget_"+tc.want, func(t *testing.T) {
			raw := []byte(`{
				"model": "claude-3-5-sonnet",
				"max_tokens": 2048,
				"thinking": {"type": "enabled", "budget_tokens": ` + fmtInt(tc.budget) + `},
				"messages": [{"role": "user", "content": "hi"}]
			}`)
			out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
			got := parseJSON(t, string(out))
			if got["reasoning_effort"] != tc.want {
				t.Fatalf("budget %d: expected reasoning_effort %q, got %v", tc.budget, tc.want, got["reasoning_effort"])
			}
			if got["max_tokens"] != float64(2048) {
				t.Fatalf("expected max_tokens preserved, got %v", got["max_tokens"])
			}
		})
	}

	// Adaptive thinking also maps.
	raw := []byte(`{
		"model": "claude-3-5-sonnet",
		"max_tokens": 4096,
		"thinking": {"type": "adaptive", "budget_tokens": 5000},
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	out := mustTranslate(t, raw, "claude-3-5-sonnet", 4096)
	if got := parseJSON(t, string(out)); got["reasoning_effort"] != "medium" {
		t.Fatalf("expected adaptive medium, got %v", got["reasoning_effort"])
	}

	// Fixed-parameter reasoning models: gpt-5.6-sol.
	rawSol := []byte(`{
		"model": "claude-3-5-sonnet",
		"max_tokens": 3000,
		"temperature": 0.7,
		"top_p": 0.9,
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	outSol := mustTranslate(t, rawSol, "gpt-5.6-sol", 4096)
	gotSol := parseJSON(t, string(outSol))
	if _, hasMT := gotSol["max_tokens"]; hasMT {
		t.Fatalf("reasoning model must not carry max_tokens: %s", outSol)
	}
	if gotSol["max_completion_tokens"] != float64(3000) {
		t.Fatalf("expected max_completion_tokens 3000, got %v", gotSol["max_completion_tokens"])
	}
	if gotSol["temperature"] != float64(1) || gotSol["top_p"] != float64(1) || gotSol["n"] != float64(1) {
		t.Fatalf("expected fixed params 1/1/1, got %v/%v/%v", gotSol["temperature"], gotSol["top_p"], gotSol["n"])
	}
	if gotSol["presence_penalty"] != float64(0) || gotSol["frequency_penalty"] != float64(0) {
		t.Fatalf("expected zero penalties, got %v/%v", gotSol["presence_penalty"], gotSol["frequency_penalty"])
	}

	// o3-mini is also a fixed-parameter model.
	outO3 := mustTranslate(t, rawSol, "o3-mini", 4096)
	gotO3 := parseJSON(t, string(outO3))
	if gotO3["max_completion_tokens"] != float64(3000) {
		t.Fatalf("expected o3-mini to use max_completion_tokens, got %v", gotO3["max_completion_tokens"])
	}

	// Non-reasoning models keep max_tokens, copied sampling params, stop
	// sequences, user mapping and stream options.
	rawNorm := []byte(`{
		"model": "claude-3-5-sonnet",
		"max_tokens": 777,
		"temperature": 0.5,
		"top_p": 0.8,
		"stop_sequences": ["END", "STOP"],
		"metadata": {"user_id": "user_123"},
		"stream": true,
		"messages": [{"role": "user", "content": "hi"}]
	}`)
	outNorm := mustTranslate(t, rawNorm, "claude-3-5-sonnet", 4096)
	gotNorm := parseJSON(t, string(outNorm))
	if gotNorm["max_tokens"] != float64(777) {
		t.Fatalf("expected max_tokens 777, got %v", gotNorm["max_tokens"])
	}
	if _, has := gotNorm["max_completion_tokens"]; has {
		t.Fatalf("non-reasoning model must not have max_completion_tokens")
	}
	if gotNorm["temperature"] != float64(0.5) || gotNorm["top_p"] != float64(0.8) {
		t.Fatalf("expected copied sampling params, got %v/%v", gotNorm["temperature"], gotNorm["top_p"])
	}
	stop, _ := gotNorm["stop"].([]any)
	if len(stop) != 2 || stop[0] != "END" || stop[1] != "STOP" {
		t.Fatalf("expected stop sequences mapped, got %v", gotNorm["stop"])
	}
	if gotNorm["user"] != "user_123" {
		t.Fatalf("expected user from metadata.user_id, got %v", gotNorm["user"])
	}
	if gotNorm["stream"] != true {
		t.Fatalf("expected stream true, got %v", gotNorm["stream"])
	}
	so, _ := gotNorm["stream_options"].(map[string]any)
	if so == nil || so["include_usage"] != true {
		t.Fatalf("expected stream_options.include_usage, got %v", gotNorm["stream_options"])
	}
}

// fmtInt formats an integer without strconv import churn in test literals.
func fmtInt(n int) string {
	if n == 0 {
		return "0"
	}
	neg := n < 0
	if neg {
		n = -n
	}
	var buf [20]byte
	i := len(buf)
	for n > 0 {
		i--
		buf[i] = byte('0' + n%10)
		n /= 10
	}
	if neg {
		i--
		buf[i] = '-'
	}
	return string(buf[i:])
}

// --- Test 6: PrepareNativeAnthropicPayload ---

func TestPrepareNativeAnthropicPayload(t *testing.T) {
	raw := []byte(`{
		"model": "anthropic/claude-sonnet-4-5",
		"system": "Be terse.",
		"max_tokens": 1024,
		"temperature": 0.4,
		"top_p": 0.9,
		"top_k": 40,
		"stream": true,
		"stop_sequences": ["END"],
		"thinking": {"type": "enabled", "budget_tokens": 8192},
		"metadata": {"user_id": "u1"},
		"messages": [
			{"role": "user", "content": [
				{"type": "text", "text": "hello"},
				{"type": "image", "source": {"type": "base64", "media_type": "image/jpeg", "data": "AAAA"}}
			]}
		],
		"stream_options": {"include_usage": true},
		"logprobs": 5,
		"n": 2
	}`)

	out, err := PrepareNativeAnthropicPayload(raw, "claude-sonnet-4-5")
	if err != nil {
		t.Fatalf("PrepareNativeAnthropicPayload failed: %v", err)
	}
	got := parseJSON(t, string(out))

	if got["model"] != "claude-sonnet-4-5" {
		t.Fatalf("expected stripped model, got %v", got["model"])
	}
	if got["system"] != "Be terse." {
		t.Fatalf("expected system intact, got %v", got["system"])
	}
	if got["max_tokens"] != float64(1024) {
		t.Fatalf("expected max_tokens intact, got %v", got["max_tokens"])
	}
	if got["temperature"] != float64(0.4) || got["top_p"] != float64(0.9) || got["top_k"] != float64(40) {
		t.Fatalf("expected sampling params intact")
	}
	if got["stream"] != true {
		t.Fatalf("expected stream intact")
	}
	if got["stop_sequences"] == nil {
		t.Fatalf("expected stop_sequences intact")
	}
	th, _ := got["thinking"].(map[string]any)
	if th == nil || th["budget_tokens"] != float64(8192) {
		t.Fatalf("expected thinking intact, got %v", got["thinking"])
	}
	meta, _ := got["metadata"].(map[string]any)
	if meta == nil || meta["user_id"] != "u1" {
		t.Fatalf("expected metadata intact, got %v", got["metadata"])
	}
	msgs, _ := got["messages"].([]any)
	if len(msgs) != 1 {
		t.Fatalf("expected messages intact")
	}
	user, _ := msgs[0].(map[string]any)
	content, _ := user["content"].([]any)
	if len(content) != 2 {
		t.Fatalf("expected content blocks intact")
	}
	img, _ := content[1].(map[string]any)
	if img["type"] != "image" {
		t.Fatalf("expected image block intact, got %v", img)
	}

	// OpenAI-only fields removed.
	for _, field := range []string{"stream_options", "logprobs", "n"} {
		if _, present := got[field]; present {
			t.Fatalf("expected %s stripped, got %v", field, got[field])
		}
	}

	// No-prefix model passes through unchanged (except stripped fields).
	raw2 := []byte(`{"model": "claude-3-5-haiku", "max_tokens": 1, "messages": [{"role": "user", "content": "x"}], "seed": 42}`)
	out2, err := PrepareNativeAnthropicPayload(raw2, "claude-3-5-haiku")
	if err != nil {
		t.Fatalf("PrepareNativeAnthropicPayload failed: %v", err)
	}
	got2 := parseJSON(t, string(out2))
	if got2["model"] != "claude-3-5-haiku" {
		t.Fatalf("expected model unchanged, got %v", got2["model"])
	}
	if _, present := got2["seed"]; present {
		t.Fatalf("expected seed stripped")
	}
}

// --- Test 7: EstimateAnthropicInputTokens ---

func TestEstimateAnthropicInputTokens(t *testing.T) {
	// One user message with 8 chars of text: 4 overhead + ceil(8/4)=2 => 6.
	raw := []byte(`{"model": "m", "max_tokens": 1, "messages": [{"role": "user", "content": "12345678"}]}`)
	n, err := EstimateAnthropicInputTokens(raw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n != 6 {
		t.Fatalf("expected 6 tokens, got %d", n)
	}

	// System + image + tools.
	raw2 := []byte(`{
		"model": "m",
		"system": "abcd",
		"messages": [{"role": "user", "content": [
			{"type": "text", "text": "abcdefgh"},
			{"type": "image", "source": {"type": "base64", "media_type": "image/png", "data": "xx"}}
		]}],
		"tools": [{"name": "get_weather", "description": "abcd", "input_schema": {"type": "object"}}]
	}`)
	n2, err := EstimateAnthropicInputTokens(raw2)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if n2 != 1228 {
		t.Fatalf("expected 1228 tokens, got %d", n2)
	}

	// tool_use and tool_result payloads counted.
	raw3 := []byte(`{
		"model": "m",
		"messages": [
			{"role": "assistant", "content": [{"type": "tool_use", "name": "fn", "input": {"key": "value"}}]},
			{"role": "user", "content": [{"type": "tool_result", "tool_use_id": "x", "content": "abcdefgh"}]}
		]
	}`)
	n3, err := EstimateAnthropicInputTokens(raw3)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	// msg1: 4 + name(1) + input(4) ; msg2: 4 + content(2) = 15.
	if n3 != 15 {
		t.Fatalf("expected 15 tokens, got %d", n3)
	}

	// Malformed JSON errors.
	if _, err := EstimateAnthropicInputTokens([]byte(`{not json`)); err == nil {
		t.Fatal("expected error for malformed JSON")
	}
	if _, err := EstimateAnthropicInputTokens([]byte(`{"model": "m"}`)); err == nil {
		t.Fatal("expected error for missing messages")
	}
	if _, err := EstimateAnthropicInputTokens(nil); err == nil {
		t.Fatal("expected error for empty body")
	}
}

// --- Test 8: benchmark ---

func BenchmarkTranslateAnthropicToOpenAI(b *testing.B) {
	raw := []byte(`{
		"model": "claude-3-5-sonnet",
		"system": "You are a coding assistant.",
		"max_tokens": 4096,
		"temperature": 0.7,
		"stream": true,
		"messages": [
			{"role": "user", "content": "Write me a function that computes fibonacci numbers."},
			{"role": "assistant", "content": [
				{"type": "text", "text": "Here is a recursive implementation."},
				{"type": "tool_use", "id": "toolu_01", "name": "run_code", "input": {"code": "def fib(n):"}}
			]},
			{"role": "user", "content": [
				{"type": "tool_result", "tool_use_id": "toolu_01", "content": "55"},
				{"type": "text", "text": "Great, now optimize it."}
			]}
		],
		"tools": [{"name": "run_code", "description": "Run code", "input_schema": {"type": "object", "properties": {"code": {"type": "string"}}}}],
		"tool_choice": {"type": "auto"}
	}`)

	b.ReportAllocs()
	b.ResetTimer()
	for i := 0; i < b.N; i++ {
		if _, err := TranslateAnthropicToOpenAI(raw, "claude-3-5-sonnet", 4096); err != nil {
			b.Fatal(err)
		}
	}
}

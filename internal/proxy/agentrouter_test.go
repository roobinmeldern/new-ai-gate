package proxy

import (
	"encoding/json"
	"testing"
)

func TestTranslateOpenAIToAgentRouter(t *testing.T) {
	openAIRaw := []byte(`{
		"model": "agentrouter/claude-3-5-sonnet-20241022",
		"messages": [
			{"role": "system", "content": "You are a helpful coding assistant."},
			{"role": "user", "content": "Hello, write a hello world in Go"}
		],
		"temperature": 0.7,
		"stream": true
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error during translation: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse translated json: %v", err)
	}

	if parsed["model"] != "claude-3-5-sonnet-20241022" {
		t.Errorf("expected clean model name 'claude-3-5-sonnet-20241022', got '%v'", parsed["model"])
	}

	if parsed["system"] != "You are a helpful coding assistant." {
		t.Errorf("expected system prompt 'You are a helpful coding assistant.', got '%v'", parsed["system"])
	}

	if parsed["stream"] != true {
		t.Errorf("expected stream to be true, got %v", parsed["stream"])
	}

	msgs, ok := parsed["messages"].([]interface{})
	if !ok || len(msgs) != 1 {
		t.Fatalf("expected 1 user message, got %d", len(msgs))
	}
}

func TestTranslateCoalescesConsecutiveUserMessages(t *testing.T) {
	// Kilo/Cline sends multiple consecutive user messages (context, files, question).
	// Anthropic API requires strict user/assistant alternation.
	openAIRaw := []byte(`{
		"model": "agentrouter/gpt-5.6-sol",
		"messages": [
			{"role": "system", "content": "You are a coding assistant."},
			{"role": "user", "content": "Here is file1.go:\npackage main"},
			{"role": "user", "content": "Here is file2.go:\npackage utils"},
			{"role": "user", "content": "What does this project do?"}
		],
		"stream": true
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	// System should be extracted
	if parsed["system"] != "You are a coding assistant." {
		t.Errorf("unexpected system: %v", parsed["system"])
	}

	// Three consecutive user messages should be coalesced into one
	msgs := parsed["messages"].([]interface{})
	if len(msgs) != 1 {
		t.Fatalf("expected 1 coalesced message, got %d", len(msgs))
	}

	msg := msgs[0].(map[string]interface{})
	if msg["role"] != "user" {
		t.Errorf("expected role 'user', got '%v'", msg["role"])
	}

	// Content should be a content array with 3 text blocks
	content, ok := msg["content"].([]interface{})
	if !ok {
		t.Fatalf("expected content to be array after coalescing, got %T", msg["content"])
	}
	if len(content) != 3 {
		t.Errorf("expected 3 content blocks, got %d", len(content))
	}
}

func TestTranslateHandlesArraySystemContent(t *testing.T) {
	// Some tools send system messages with array content format
	openAIRaw := []byte(`{
		"model": "agentrouter/claude-3-5-sonnet-20241022",
		"messages": [
			{"role": "system", "content": [{"type": "text", "text": "You are a coding assistant with project context."}]},
			{"role": "user", "content": "What does this do?"}
		],
		"stream": true
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	// System should be extracted from array content
	if parsed["system"] != "You are a coding assistant with project context." {
		t.Errorf("expected extracted system text, got '%v'", parsed["system"])
	}
}

func TestTranslateHandlesNullContent(t *testing.T) {
	// Assistant messages with tool_calls often have null content
	openAIRaw := []byte(`{
		"model": "agentrouter/claude-3-5-sonnet-20241022",
		"messages": [
			{"role": "user", "content": "Run the tests"},
			{"role": "assistant", "content": null},
			{"role": "user", "content": "OK the tests passed"}
		],
		"stream": true
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	msgs := parsed["messages"].([]interface{})
	// Should have user, assistant, user (3 messages, properly alternating)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages with null assistant preserved, got %d", len(msgs))
	}

	msg1 := msgs[1].(map[string]interface{})
	if msg1["role"] != "assistant" {
		t.Errorf("expected assistant role preserved, got '%v'", msg1["role"])
	}
}

func TestTranslateAgentRouterResponseToOpenAI(t *testing.T) {
	anthropicRaw := []byte(`{
		"id": "msg_013Z55f4k5g6h7j8",
		"type": "message",
		"role": "assistant",
		"model": "claude-3-5-sonnet-20241022",
		"content": [
			{"type": "thinking", "thinking": "Let us solve this problem step by step..."},
			{"type": "text", "text": "Here is the code in Go:\nfunc main() {}"}
		],
		"stop_reason": "end_turn",
		"usage": {
			"input_tokens": 15,
			"output_tokens": 42
		}
	}`)

	openAIPayload, ok := translateAgentRouterResponseToOpenAI(anthropicRaw, "claude-3-5-sonnet-20241022")
	if !ok {
		t.Fatalf("expected successful translation of Anthropic non-streaming response")
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(openAIPayload, &parsed); err != nil {
		t.Fatalf("failed to parse output json: %v", err)
	}

	if parsed["object"] != "chat.completion" {
		t.Errorf("expected object 'chat.completion', got '%v'", parsed["object"])
	}

	choices := parsed["choices"].([]interface{})
	choice0 := choices[0].(map[string]interface{})
	msg := choice0["message"].(map[string]interface{})

	if msg["content"] != "Here is the code in Go:\nfunc main() {}" {
		t.Errorf("unexpected content: %v", msg["content"])
	}

	if msg["reasoning_content"] != "Let us solve this problem step by step..." {
		t.Errorf("unexpected reasoning_content: %v", msg["reasoning_content"])
	}

	if choice0["finish_reason"] != "stop" {
		t.Errorf("expected finish_reason 'stop', got '%v'", choice0["finish_reason"])
	}

	usage := parsed["usage"].(map[string]interface{})
	if usage["prompt_tokens"].(float64) != 15 || usage["completion_tokens"].(float64) != 42 {
		t.Errorf("unexpected usage numbers: %v", usage)
	}
}

func TestTranslateOpenAIToAgentRouterWithTools(t *testing.T) {
	openAIRaw := []byte(`{
		"model": "agentrouter/claude-3-5-sonnet-20241022",
		"messages": [
			{"role": "user", "content": "List files in directory"},
			{
				"role": "assistant",
				"content": null,
				"tool_calls": [
					{
						"id": "call_999",
						"type": "function",
						"function": {
							"name": "list_dir",
							"arguments": "{\"path\":\"/workspace\"}"
						}
					}
				]
			},
			{
				"role": "tool",
				"tool_call_id": "call_999",
				"content": "file1.go\nfile2.go"
			}
		],
		"tools": [
			{
				"type": "function",
				"function": {
					"name": "list_dir",
					"description": "Lists contents of a directory",
					"parameters": {
						"type": "object",
						"properties": {
							"path": {"type": "string"}
						}
					}
				}
			}
		],
		"tool_choice": "auto"
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error during translation: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse translated json: %v", err)
	}

	tools, ok := parsed["tools"].([]interface{})
	if !ok || len(tools) != 1 {
		t.Fatalf("expected 1 tool in anthropic payload, got %v", parsed["tools"])
	}

	tool0 := tools[0].(map[string]interface{})
	if tool0["name"] != "list_dir" {
		t.Errorf("expected tool name 'list_dir', got '%v'", tool0["name"])
	}
	if tool0["input_schema"] == nil {
		t.Errorf("expected input_schema to be populated")
	}

	toolChoice, ok := parsed["tool_choice"].(map[string]interface{})
	if !ok || toolChoice["type"] != "auto" {
		t.Errorf("expected tool_choice auto, got %v", parsed["tool_choice"])
	}

	msgs, ok := parsed["messages"].([]interface{})
	if !ok {
		t.Fatalf("expected messages array")
	}

	if len(msgs) != 3 {
		t.Fatalf("expected 3 turns (user, assistant, user with tool_result), got %d", len(msgs))
	}

	// Turn 1: Assistant with tool_use
	astMsg := msgs[1].(map[string]interface{})
	if astMsg["role"] != "assistant" {
		t.Errorf("expected role assistant, got %v", astMsg["role"])
	}
	astBlocks := astMsg["content"].([]interface{})
	toolUseFound := false
	for _, b := range astBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_use" {
			toolUseFound = true
			if block["id"] != "call_999" || block["name"] != "list_dir" {
				t.Errorf("unexpected tool_use block: %v", block)
			}
		}
	}
	if !toolUseFound {
		t.Errorf("expected tool_use block in assistant content")
	}

	// Turn 2: User with tool_result
	usrMsg := msgs[2].(map[string]interface{})
	if usrMsg["role"] != "user" {
		t.Errorf("expected role user for tool result, got %v", usrMsg["role"])
	}
	usrBlocks := usrMsg["content"].([]interface{})
	toolResultFound := false
	for _, b := range usrBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_result" {
			toolResultFound = true
			if block["tool_use_id"] != "call_999" || block["content"] != "file1.go\nfile2.go" {
				t.Errorf("unexpected tool_result block: %v", block)
			}
		}
	}
	if !toolResultFound {
		t.Errorf("expected tool_result block in user content")
	}
}

func TestTranslateOpenAIToAgentRouterEmptyToolIDAndObjectArgs(t *testing.T) {
	openAIRaw := []byte(`{
		"model": "agentrouter/gpt-5.6-sol",
		"messages": [
			{"role": "user", "content": "Read file"},
			{
				"role": "assistant",
				"content": "",
				"tool_calls": [
					{
						"id": "",
						"type": "function",
						"function": {
							"name": "read",
							"arguments": {"filePath": "/path/to/file.txt"}
						}
					}
				]
			}
		]
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse json: %v", err)
	}

	msgs := parsed["messages"].([]interface{})
	if len(msgs) != 2 {
		t.Fatalf("expected 2 messages, got %d", len(msgs))
	}

	astMsg := msgs[1].(map[string]interface{})
	blocks := astMsg["content"].([]interface{})
	if len(blocks) != 1 {
		t.Fatalf("expected 1 tool_use block, got %d", len(blocks))
	}

	toolUse := blocks[0].(map[string]interface{})
	if toolUse["type"] != "tool_use" {
		t.Errorf("expected type tool_use, got %v", toolUse["type"])
	}

	idStr, _ := toolUse["id"].(string)
	if idStr == "" {
		t.Errorf("expected auto-generated non-empty id for tool_use, got empty string")
	}

	inputObj, ok := toolUse["input"].(map[string]interface{})
	if !ok || inputObj["filePath"] != "/path/to/file.txt" {
		t.Errorf("expected filePath in input schema object, got %v", toolUse["input"])
	}
}

func TestTranslateEmptyToolIDsMatchResults(t *testing.T) {
	// Simulates GPT models through agentrouter where the Anthropic SSE omits
	// content_block.id — the client accumulates tool_calls with empty IDs and
	// tool results with empty tool_call_id. The translator must generate
	// deterministic IDs that match each tool_use with its tool_result.
	openAIRaw := []byte(`{
		"model": "agentrouter/gpt-5.6-sol",
		"messages": [
			{"role": "user", "content": "Read two files"},
			{
				"role": "assistant",
				"content": "",
				"tool_calls": [
					{"id": "", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.txt\"}"}},
					{"id": "", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"b.txt\"}"}}
				]
			},
			{"role": "tool", "tool_call_id": "", "content": "content of a"},
			{"role": "tool", "tool_call_id": "", "content": "content of b"}
		]
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	msgs := parsed["messages"].([]interface{})
	// user, assistant(tool_use x2), user(tool_result x2 coalesced)
	if len(msgs) != 3 {
		t.Fatalf("expected 3 messages, got %d", len(msgs))
	}

	// Assistant message: should have 2 tool_use blocks with non-empty, distinct IDs
	astMsg := msgs[1].(map[string]interface{})
	astBlocks := astMsg["content"].([]interface{})
	var toolUseIDs []string
	for _, b := range astBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_use" {
			id, _ := block["id"].(string)
			if id == "" {
				t.Errorf("expected non-empty tool_use id, got empty")
			}
			toolUseIDs = append(toolUseIDs, id)
		}
	}
	if len(toolUseIDs) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got %d", len(toolUseIDs))
	}
	if toolUseIDs[0] == toolUseIDs[1] {
		t.Errorf("expected distinct tool_use IDs, got duplicate: %s", toolUseIDs[0])
	}

	// User message with tool_results: each tool_result's tool_use_id must match a tool_use ID
	usrMsg := msgs[2].(map[string]interface{})
	usrBlocks := usrMsg["content"].([]interface{})
	var toolResultIDs []string
	for _, b := range usrBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_result" {
			id, _ := block["tool_use_id"].(string)
			toolResultIDs = append(toolResultIDs, id)
		}
	}
	if len(toolResultIDs) != 2 {
		t.Fatalf("expected 2 tool_result blocks, got %d", len(toolResultIDs))
	}

	// Each tool_result ID must match a tool_use ID in order
	for i, resultID := range toolResultIDs {
		if resultID != toolUseIDs[i] {
			t.Errorf("tool_result[%d] id %q does not match tool_use[%d] id %q",
				i, resultID, i, toolUseIDs[i])
		}
	}
}

func TestTranslateMixedToolIDsMatchResults(t *testing.T) {
	// Mixed scenario: one tool call has a proper ID, one has empty.
	// The tool results follow the same pattern.
	openAIRaw := []byte(`{
		"model": "agentrouter/gpt-5.6-sol",
		"messages": [
			{"role": "user", "content": "Read two files"},
			{
				"role": "assistant",
				"content": "",
				"tool_calls": [
					{"id": "call_proper_1", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"a.txt\"}"}},
					{"id": "", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"b.txt\"}"}}
				]
			},
			{"role": "tool", "tool_call_id": "call_proper_1", "content": "content of a"},
			{"role": "tool", "tool_call_id": "", "content": "content of b"}
		]
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	msgs := parsed["messages"].([]interface{})

	// Assistant message: first tool_use has "call_proper_1", second has generated ID
	astMsg := msgs[1].(map[string]interface{})
	astBlocks := astMsg["content"].([]interface{})
	var toolUseIDs []string
	for _, b := range astBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_use" {
			id, _ := block["id"].(string)
			toolUseIDs = append(toolUseIDs, id)
		}
	}
	if len(toolUseIDs) != 2 {
		t.Fatalf("expected 2 tool_use blocks, got %d", len(toolUseIDs))
	}
	if toolUseIDs[0] != "call_proper_1" {
		t.Errorf("expected first tool_use id 'call_proper_1', got %q", toolUseIDs[0])
	}
	if toolUseIDs[1] == "" {
		t.Errorf("expected non-empty generated id for second tool_use")
	}

	// User message with tool_results: first matches "call_proper_1", second matches generated
	usrMsg := msgs[2].(map[string]interface{})
	usrBlocks := usrMsg["content"].([]interface{})
	var toolResultIDs []string
	for _, b := range usrBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_result" {
			id, _ := block["tool_use_id"].(string)
			toolResultIDs = append(toolResultIDs, id)
		}
	}
	if toolResultIDs[0] != "call_proper_1" {
		t.Errorf("expected first tool_result id 'call_proper_1', got %q", toolResultIDs[0])
	}
	if toolResultIDs[1] != toolUseIDs[1] {
		t.Errorf("expected second tool_result id %q to match generated tool_use id, got %q",
			toolUseIDs[1], toolResultIDs[1])
	}
}

func TestTranslateContentWithToolUseSkipsToolCalls(t *testing.T) {
	// When the assistant content array already contains tool_use blocks (Anthropic
	// format embedded in OpenAI content), the translator must skip tool_calls
	// processing to avoid duplicate tool_use blocks.
	openAIRaw := []byte(`{
		"model": "agentrouter/claude-opus-5",
		"messages": [
			{"role": "user", "content": "Read file"},
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "toolu_abc", "name": "read", "input": {"path": "/file.txt"}}
				],
				"tool_calls": [
					{"id": "toolu_abc", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"/file.txt\"}"}}
				]
			},
			{"role": "tool", "tool_call_id": "toolu_abc", "content": "file content"}
		]
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	msgs := parsed["messages"].([]interface{})

	// Assistant message: should have exactly 1 tool_use (not duplicated)
	astMsg := msgs[1].(map[string]interface{})
	astBlocks := astMsg["content"].([]interface{})
	toolUseCount := 0
	for _, b := range astBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_use" {
			toolUseCount++
			if block["id"] != "toolu_abc" {
				t.Errorf("expected tool_use id 'toolu_abc', got %v", block["id"])
			}
		}
	}
	if toolUseCount != 1 {
		t.Errorf("expected 1 tool_use block (no duplicate), got %d", toolUseCount)
	}

	// Tool result should match
	usrMsg := msgs[2].(map[string]interface{})
	usrBlocks := usrMsg["content"].([]interface{})
	for _, b := range usrBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_result" {
			if block["tool_use_id"] != "toolu_abc" {
				t.Errorf("expected tool_result tool_use_id 'toolu_abc', got %v", block["tool_use_id"])
			}
		}
	}
}

func TestTranslateContentWithEmptyToolUseID(t *testing.T) {
	// Content has tool_use blocks with empty IDs (from GPT model streaming).
	// The translator should generate IDs and match them with tool results.
	openAIRaw := []byte(`{
		"model": "agentrouter/gpt-5.6-sol",
		"messages": [
			{"role": "user", "content": "Read file"},
			{
				"role": "assistant",
				"content": [
					{"type": "tool_use", "id": "", "name": "read", "input": {"path": "/file.txt"}}
				],
				"tool_calls": [
					{"id": "", "type": "function", "function": {"name": "read", "arguments": "{\"path\":\"/file.txt\"}"}}
				]
			},
			{"role": "tool", "tool_call_id": "", "content": "file content"}
		]
	}`)

	agentBody, err := TranslateOpenAIToAgentRouter(openAIRaw)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(agentBody, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	msgs := parsed["messages"].([]interface{})

	// Assistant: exactly 1 tool_use with generated non-empty ID
	astMsg := msgs[1].(map[string]interface{})
	astBlocks := astMsg["content"].([]interface{})
	var toolUseID string
	toolUseCount := 0
	for _, b := range astBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_use" {
			toolUseCount++
			toolUseID, _ = block["id"].(string)
		}
	}
	if toolUseCount != 1 {
		t.Errorf("expected 1 tool_use block, got %d", toolUseCount)
	}
	if toolUseID == "" {
		t.Errorf("expected non-empty generated tool_use id")
	}

	// Tool result should match the generated ID
	usrMsg := msgs[2].(map[string]interface{})
	usrBlocks := usrMsg["content"].([]interface{})
	for _, b := range usrBlocks {
		block := b.(map[string]interface{})
		if block["type"] == "tool_result" {
			if block["tool_use_id"] != toolUseID {
				t.Errorf("expected tool_result tool_use_id %q, got %v", toolUseID, block["tool_use_id"])
			}
		}
	}
}

func TestTranslateAgentRouterResponseEmptyToolID(t *testing.T) {
	// Non-streaming Anthropic response with a tool_use block that has an empty ID
	// (GPT models proxied through agentrouter). The translator should generate
	// a stable, non-empty ID.
	anthropicRaw := []byte(`{
		"id": "msg_001",
		"type": "message",
		"role": "assistant",
		"model": "gpt-5.6-sol",
		"content": [
			{"type": "text", "text": "Let me read that file."},
			{"type": "tool_use", "id": "", "name": "read_file", "input": {"path": "/test.go"}}
		],
		"stop_reason": "tool_use",
		"usage": {"input_tokens": 10, "output_tokens": 20}
	}`)

	openAIPayload, ok := translateAgentRouterResponseToOpenAI(anthropicRaw, "gpt-5.6-sol")
	if !ok {
		t.Fatalf("expected successful translation")
	}

	var parsed map[string]interface{}
	if err := json.Unmarshal(openAIPayload, &parsed); err != nil {
		t.Fatalf("failed to parse: %v", err)
	}

	choices := parsed["choices"].([]interface{})
	choice0 := choices[0].(map[string]interface{})
	msg := choice0["message"].(map[string]interface{})
	toolCalls := msg["tool_calls"].([]interface{})

	if len(toolCalls) != 1 {
		t.Fatalf("expected 1 tool call, got %d", len(toolCalls))
	}

	tc := toolCalls[0].(map[string]interface{})
	id, _ := tc["id"].(string)
	if id == "" {
		t.Errorf("expected non-empty generated tool call id")
	}

	fn := tc["function"].(map[string]interface{})
	if fn["name"] != "read_file" {
		t.Errorf("expected tool name 'read_file', got %v", fn["name"])
	}

	if choice0["finish_reason"] != "tool_calls" {
		t.Errorf("expected finish_reason 'tool_calls', got %v", choice0["finish_reason"])
	}
}

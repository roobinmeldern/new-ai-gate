package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"strconv"
	"strings"
	"time"
	"unicode/utf8"

	"github.com/buger/jsonparser"
)

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic ⇄ provider outbound response translation (Phase 3).
//
// TranslateOpenAIToAnthropicResponse is the outbound half of the
// Anthropic-compatible front API: it converts a canonical OpenAI Chat
// Completions response body (as produced by every non-native provider
// pipeline) into the Anthropic Messages response envelope that /v1/messages
// clients expect:
//
//	{"id":"...","type":"message","role":"assistant","model":"...","content":[…],
//	 "stop_reason":"...","stop_sequence":null,"usage":{…}}
//
// Mapping table:
//	message.reasoning_content (or <think> tag) → {"type":"thinking","thinking":…,"signature":"sig_clever_gate"}
//	message.content (string/array)             → {"type":"text","text":…}
//	message.tool_calls[]                       → {"type":"tool_use","id":…,"name":…,"input":{…}}
//	finish_reason stop                         → end_turn
//	finish_reason length                       → max_tokens
//	finish_reason tool_calls                   → tool_use
//	finish_reason content_filter               → refusal
//	usage.prompt_tokens                        → usage.input_tokens
//	usage.completion_tokens                    → usage.output_tokens
//	usage.prompt_tokens_details.cached_tokens  → usage.cache_read_input_tokens
// ─────────────────────────────────────────────────────────────────────────────

type anthropicToolCall struct {
	id   string
	name string
	args string
}

// TranslateOpenAIToAnthropicResponse converts an OpenAI chat completion JSON
// body into an Anthropic Messages response JSON body.
//
// Parameters:
//   - openAIResp: Raw canonical OpenAI chat completion JSON bytes.
//   - requestedModel: The client-facing model identifier (e.g. "claude-sonnet-4").
//   - fallbackInputTokens: Used if usage.prompt_tokens is not provided.
//
// Returns:
//   - anthropicBytes: Formatted Anthropic message response JSON.
//   - promptTokens: Extracted or estimated input tokens.
//   - completionTokens: Extracted or estimated output tokens.
//   - responseText: Combined text content for telemetry recording.
//   - err: Any unmarshaling or upstream error.
func TranslateOpenAIToAnthropicResponse(openAIResp []byte, requestedModel string, fallbackInputTokens int) (anthropicBytes []byte, promptTokens int, completionTokens int, responseText string, err error) {
	if len(bytes.TrimSpace(openAIResp)) == 0 {
		return nil, 0, 0, "", errors.New("empty response body")
	}

	// 1. Check for upstream error payload embedded in JSON
	if errVal, _, _, parseErr := jsonparser.Get(openAIResp, "error"); parseErr == nil && len(errVal) > 0 {
		errMsg, _ := jsonparser.GetString(openAIResp, "error", "message")
		if errMsg == "" {
			errMsg = string(errVal)
		}
		return nil, 0, 0, "", fmt.Errorf("upstream error: %s", errMsg)
	}

	// 2. Extract message id and model
	id, _ := jsonparser.GetString(openAIResp, "id")
	if id == "" {
		id = "msg_gate_" + strconv.FormatInt(time.Now().UnixNano(), 36)
	}
	model := requestedModel
	if model == "" {
		model, _ = jsonparser.GetString(openAIResp, "model")
	}
	if model == "" {
		model = "unknown"
	}

	// 3. Extract reasoning and content
	reasoning, _ := jsonparser.GetString(openAIResp, "choices", "[0]", "message", "reasoning_content")
	content, _ := jsonparser.GetString(openAIResp, "choices", "[0]", "message", "content")

	// If content is an array of text parts (emitted by some OpenAI-compatible providers)
	if content == "" {
		cVal, cType, _, cErr := jsonparser.Get(openAIResp, "choices", "[0]", "message", "content")
		if cErr == nil && cType == jsonparser.Array {
			var parts []string
			jsonparser.ArrayEach(cVal, func(part []byte, dataType jsonparser.ValueType, _ int, _ error) {
				if dataType != jsonparser.Object {
					return
				}
				if t, _ := jsonparser.GetString(part, "type"); t == "text" {
					if txt, _ := jsonparser.GetString(part, "text"); txt != "" {
						parts = append(parts, txt)
					}
				}
			})
			content = strings.Join(parts, "\n")
		}
	}

	// If reasoning is empty, check for inline <think>...</think> tags in content
	if reasoning == "" && strings.Contains(content, "<think>") {
		var reasoningParts []string
		var cleanBuilder strings.Builder
		remaining := content
		for {
			startIdx := strings.Index(remaining, "<think>")
			if startIdx == -1 {
				cleanBuilder.WriteString(remaining)
				break
			}
			cleanBuilder.WriteString(remaining[:startIdx])
			afterTag := remaining[startIdx+len("<think>"):]
			endIdx := strings.Index(afterTag, "</think>")
			if endIdx == -1 {
				if trimmed := strings.TrimSpace(afterTag); trimmed != "" {
					reasoningParts = append(reasoningParts, trimmed)
				}
				break
			}
			if trimmed := strings.TrimSpace(afterTag[:endIdx]); trimmed != "" {
				reasoningParts = append(reasoningParts, trimmed)
			}
			remaining = afterTag[endIdx+len("</think>"):]
		}
		if len(reasoningParts) > 0 {
			reasoning = strings.Join(reasoningParts, "\n")
			content = strings.TrimSpace(cleanBuilder.String())
		}
	}

	// 4. Extract tool calls
	var toolCalls []anthropicToolCall
	tcVal, tcType, _, tcErr := jsonparser.Get(openAIResp, "choices", "[0]", "message", "tool_calls")
	if tcErr == nil && tcType == jsonparser.Array {
		idx := 0
		jsonparser.ArrayEach(tcVal, func(call []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			name, _ := jsonparser.GetString(call, "function", "name")
			if name == "" {
				name = "unknown_tool"
			}
			callID, _ := jsonparser.GetString(call, "id")
			if callID == "" {
				callID = fmt.Sprintf("toolu_gate_%d", idx)
			}
			argsRaw, _ := jsonparser.GetString(call, "function", "arguments")
			inputJSON := "{}"
			if argsRaw != "" && json.Valid([]byte(argsRaw)) {
				inputJSON = argsRaw
			} else if argsRaw != "" {
				var b bytes.Buffer
				b.WriteString(`{"_raw":`)
				appendJSONString(&b, argsRaw)
				b.WriteByte('}')
				inputJSON = b.String()
			}
			toolCalls = append(toolCalls, anthropicToolCall{
				id:   callID,
				name: name,
				args: inputJSON,
			})
			idx++
		})
	}

	// 5. Map finish_reason to Anthropic stop_reason
	finishReason, _ := jsonparser.GetString(openAIResp, "choices", "[0]", "finish_reason")
	var stopReason string
	if len(toolCalls) > 0 || finishReason == "tool_calls" || finishReason == "function_call" {
		stopReason = "tool_use"
	} else {
		switch finishReason {
		case "stop", "":
			stopReason = "end_turn"
		case "length":
			stopReason = "max_tokens"
		case "content_filter":
			stopReason = "refusal"
		default:
			stopReason = "end_turn"
		}
	}

	// 6. Build Anthropic JSON response
	var buf bytes.Buffer
	buf.WriteString(`{"id":`)
	appendJSONString(&buf, id)
	buf.WriteString(`,"type":"message","role":"assistant","model":`)
	appendJSONString(&buf, model)
	buf.WriteString(`,"content":[`)

	first := true

	// Thinking block first
	if reasoning != "" {
		buf.WriteString(`{"type":"thinking","thinking":`)
		appendJSONString(&buf, reasoning)
		buf.WriteString(`,"signature":"sig_clever_gate"}`)
		first = false
	}

	// Text block
	if content != "" || (reasoning == "" && len(toolCalls) == 0) {
		if !first {
			buf.WriteByte(',')
		}
		buf.WriteString(`{"type":"text","text":`)
		appendJSONString(&buf, content)
		buf.WriteByte('}')
		first = false
	}

	// Tool use blocks
	for _, tc := range toolCalls {
		if !first {
			buf.WriteByte(',')
		}
		buf.WriteString(`{"type":"tool_use","id":`)
		appendJSONString(&buf, tc.id)
		buf.WriteString(`,"name":`)
		appendJSONString(&buf, tc.name)
		buf.WriteString(`,"input":`)
		buf.WriteString(tc.args)
		buf.WriteByte('}')
		first = false
	}

	buf.WriteString(`],"stop_reason":`)
	appendJSONString(&buf, stopReason)
	buf.WriteString(`,"stop_sequence":null,"usage":{`)

	// 7. Token accounting
	pTokens, _ := jsonparser.GetInt(openAIResp, "usage", "prompt_tokens")
	cTokens, _ := jsonparser.GetInt(openAIResp, "usage", "completion_tokens")
	cachedTokens, _ := jsonparser.GetInt(openAIResp, "usage", "prompt_tokens_details", "cached_tokens")
	if cachedTokens == 0 {
		cachedTokens, _ = jsonparser.GetInt(openAIResp, "usage", "cache_read_input_tokens")
	}
	cacheCreationTokens, _ := jsonparser.GetInt(openAIResp, "usage", "cache_creation_input_tokens")

	promptTokens = int(pTokens)
	if promptTokens <= 0 {
		promptTokens = fallbackInputTokens
	}
	if promptTokens <= 0 {
		promptTokens = 1
	}

	completionTokens = int(cTokens)
	if completionTokens <= 0 {
		runes := utf8.RuneCountInString(content) + utf8.RuneCountInString(reasoning)
		for _, tc := range toolCalls {
			runes += utf8.RuneCountInString(tc.name) + utf8.RuneCountInString(tc.args)
		}
		completionTokens = (runes + 3) / 4
		if completionTokens <= 0 && (content != "" || reasoning != "" || len(toolCalls) > 0) {
			completionTokens = 1
		}
	}

	buf.WriteString(`"input_tokens":`)
	buf.WriteString(strconv.Itoa(promptTokens))
	buf.WriteString(`,"output_tokens":`)
	buf.WriteString(strconv.Itoa(completionTokens))
	buf.WriteString(`,"cache_creation_input_tokens":`)
	buf.WriteString(strconv.Itoa(int(cacheCreationTokens)))
	buf.WriteString(`,"cache_read_input_tokens":`)
	buf.WriteString(strconv.Itoa(int(cachedTokens)))
	buf.WriteString(`}}`)

	// 8. Prepare telemetry response text
	if content != "" && reasoning != "" {
		responseText = reasoning + "\n" + content
	} else if content != "" {
		responseText = content
	} else {
		responseText = reasoning
	}

	return buf.Bytes(), promptTokens, completionTokens, responseText, nil
}

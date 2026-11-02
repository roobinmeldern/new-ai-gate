package transmux

import (
	"bytes"
	"fmt"
	"sync/atomic"
	"time"
	"unicode/utf8"

	"github.com/buger/jsonparser"
)

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicStreamEncoder — Universal Outbound Reverse SSE Stream Transmuxer
// (OpenAI chat.completion.chunk ➔ Anthropic Messages SSE protocol).
//
// Complies with official Anthropic SDKs, Claude Code CLI, Cursor, and Cline:
//   message_start ➔ (content_block_start ➔ content_block_delta* ➔ content_block_stop)*
//   ➔ message_delta ➔ message_stop
//
// Invariants:
//   - Zero reflection, buger/jsonparser field extraction on canonical OpenAI chunks.
//   - Reuses chunkBufPool (getBuffer/putBuffer), escapeJSONString, and writeInt.
//   - Automatically emits signature_delta when closing thinking blocks.
//   - Correctly splits multi-chunk frames (e.g. Gemini "\n\ndata: " combinations).
// ─────────────────────────────────────────────────────────────────────────────

type AnthropicBlockType int

const (
	BlockNone AnthropicBlockType = iota
	BlockThinking
	BlockText
	BlockToolUse
)

var anthropicMessageIDCounter uint64

// AnthropicStreamEncoder converts canonical OpenAI chat.completion.chunk JSON
// into Anthropic Messages SSE frames.
type AnthropicStreamEncoder struct {
	messageID        string
	model            string
	messageStarted   bool
	messageStopped   bool
	activeBlockType  AnthropicBlockType
	nextBlockIndex   int
	activeBlockIndex int
	currentToolIndex int // tracks OpenAI tool_calls[].index
	toolUseSeen      bool
	stopReason       string
	inputTokens              int
	outputTokens             int
	cacheCreationInputTokens int
	cacheReadInputTokens     int
}

// NewAnthropicStreamEncoder creates a stateful stream encoder for an Anthropic message.
func NewAnthropicStreamEncoder(messageID, model string, estimatedInputTokens int) *AnthropicStreamEncoder {
	if messageID == "" {
		messageID = fmt.Sprintf("msg_gate_%x_%d", time.Now().UnixNano(), atomic.AddUint64(&anthropicMessageIDCounter, 1))
	}
	if model == "" {
		model = "unknown"
	}
	return &AnthropicStreamEncoder{
		messageID:        messageID,
		model:            model,
		activeBlockIndex: -1,
		activeBlockType:  BlockNone,
		inputTokens:      estimatedInputTokens,
	}
}

// InputTokens returns the input tokens (reported by upstream or estimated).
func (e *AnthropicStreamEncoder) InputTokens() int {
	return e.inputTokens
}

// OutputTokens returns the accumulated completion tokens.
func (e *AnthropicStreamEncoder) OutputTokens() int {
	return e.outputTokens
}

// CacheCreationInputTokens returns the number of tokens written to cache.
func (e *AnthropicStreamEncoder) CacheCreationInputTokens() int {
	return e.cacheCreationInputTokens
}

// CacheReadInputTokens returns the number of tokens read from cache.
func (e *AnthropicStreamEncoder) CacheReadInputTokens() int {
	return e.cacheReadInputTokens
}

// MessageStopped reports whether stream generation stopped or an error was emitted.
func (e *AnthropicStreamEncoder) MessageStopped() bool {
	return e.messageStopped
}

// MessageID returns the client-visible Anthropic message ID.
func (e *AnthropicStreamEncoder) MessageID() string {
	return e.messageID
}

// Model returns the target model identifier.
func (e *AnthropicStreamEncoder) Model() string {
	return e.model
}

// StopReason returns the finalized Anthropic stop reason.
func (e *AnthropicStreamEncoder) StopReason() string {
	return e.stopReason
}

// EncodeOpenAIChunk converts one or more OpenAI chat completion chunk(s)
// into formatted Anthropic SSE events ("event: <type>\ndata: <json>\n\n").
// Handles concatenated multi-chunk transmuxer outputs (such as Gemini
// thinking + tool_call frames separated by "\n\ndata: ").
func (e *AnthropicStreamEncoder) EncodeOpenAIChunk(rawChunk []byte) []byte {
	if e.messageStopped || len(rawChunk) == 0 {
		return nil
	}

	buf := getBuffer()
	defer putBuffer(buf)

	// Split concatenated SSE chunks if present
	subchunks := bytes.Split(rawChunk, []byte("\n\ndata: "))
	for _, sub := range subchunks {
		sub = bytes.TrimSpace(sub)
		sub = bytes.TrimPrefix(sub, []byte("data: "))
		sub = bytes.TrimSpace(sub)
		if len(sub) == 0 {
			continue
		}
		if bytes.Equal(sub, []byte("[DONE]")) {
			continue
		}
		e.processSingleChunk(buf, sub)
	}

	if buf.Len() == 0 {
		return nil
	}
	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())
	return result
}

func (e *AnthropicStreamEncoder) processSingleChunk(buf *bytes.Buffer, chunk []byte) {
	// 1. Check for mid-stream provider error payload
	if errVal, _, _, err := jsonparser.Get(chunk, "error"); err == nil && len(errVal) > 0 {
		errType, _ := jsonparser.GetString(chunk, "error", "type")
		errMsg, _ := jsonparser.GetString(chunk, "error", "message")
		if errMsg == "" {
			errMsg = string(errVal)
		}
		if errType == "" {
			errType = "api_error"
		}
		e.emitStreamErrorInto(buf, errType, errMsg)
		return
	}

	// 2. Extract token usage if provided
	if promptTokens, pErr := jsonparser.GetInt(chunk, "usage", "prompt_tokens"); pErr == nil && promptTokens > 0 {
		e.inputTokens = int(promptTokens)
	}
	if completionTokens, cErr := jsonparser.GetInt(chunk, "usage", "completion_tokens"); cErr == nil && completionTokens > 0 {
		e.outputTokens = int(completionTokens)
	}
	if cachedTokens, err := jsonparser.GetInt(chunk, "usage", "prompt_tokens_details", "cached_tokens"); err == nil && cachedTokens > 0 {
		e.cacheReadInputTokens = int(cachedTokens)
	} else if cacheRead, err := jsonparser.GetInt(chunk, "usage", "cache_read_input_tokens"); err == nil && cacheRead > 0 {
		e.cacheReadInputTokens = int(cacheRead)
	}
	if cacheCreation, err := jsonparser.GetInt(chunk, "usage", "cache_creation_input_tokens"); err == nil && cacheCreation > 0 {
		e.cacheCreationInputTokens = int(cacheCreation)
	}

	// 3. Extract finish_reason if provided
	if fr, err := jsonparser.GetString(chunk, "choices", "[0]", "finish_reason"); err == nil && fr != "" && fr != "null" {
		e.stopReason = MapOpenAIFinishReasonToAnthropic(fr, e.toolUseSeen)
	}

	// 4. Reasoning / Thinking delta
	if reasoning, err := jsonparser.GetString(chunk, "choices", "[0]", "delta", "reasoning_content"); err == nil && reasoning != "" {
		if e.activeBlockType != BlockThinking {
			e.emitBlockStop(buf)
			e.ensureMessageStart(buf)
			e.activeBlockIndex = e.nextBlockIndex
			e.nextBlockIndex++
			e.activeBlockType = BlockThinking

			buf.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":")
			writeInt(buf, e.activeBlockIndex)
			buf.WriteString(",\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n")
		}

		buf.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":")
		writeInt(buf, e.activeBlockIndex)
		buf.WriteString(",\"delta\":{\"type\":\"thinking_delta\",\"thinking\":")
		buf.WriteByte('"')
		escapeJSONString(buf, []byte(reasoning))
		buf.WriteString("\"}}\n\n")

		e.outputTokens += (utf8.RuneCountInString(reasoning) + 3) / 4
	}

	// 5. Text content delta
	if content, err := jsonparser.GetString(chunk, "choices", "[0]", "delta", "content"); err == nil && content != "" {
		if e.activeBlockType != BlockText {
			e.emitBlockStop(buf)
			e.ensureMessageStart(buf)
			e.activeBlockIndex = e.nextBlockIndex
			e.nextBlockIndex++
			e.activeBlockType = BlockText

			buf.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":")
			writeInt(buf, e.activeBlockIndex)
			buf.WriteString(",\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		}

		buf.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":")
		writeInt(buf, e.activeBlockIndex)
		buf.WriteString(",\"delta\":{\"type\":\"text_delta\",\"text\":")
		buf.WriteByte('"')
		escapeJSONString(buf, []byte(content))
		buf.WriteString("\"}}\n\n")

		e.outputTokens += (utf8.RuneCountInString(content) + 3) / 4
	}

	// 6. Tool call deltas
	if tcVal, tcType, _, tcErr := jsonparser.Get(chunk, "choices", "[0]", "delta", "tool_calls"); tcErr == nil && tcType == jsonparser.Array {
		jsonparser.ArrayEach(tcVal, func(call []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			e.toolUseSeen = true

			tcIndex := 0
			if idx, err := jsonparser.GetInt(call, "index"); err == nil {
				tcIndex = int(idx)
			}
			tcID, _ := jsonparser.GetString(call, "id")
			fnName, _ := jsonparser.GetString(call, "function", "name")
			fnArgs, _ := jsonparser.GetString(call, "function", "arguments")

			if e.activeBlockType != BlockToolUse || tcIndex != e.currentToolIndex || (tcID != "" && fnName != "") {
				e.emitBlockStop(buf)
				e.ensureMessageStart(buf)
				e.currentToolIndex = tcIndex
				e.activeBlockIndex = e.nextBlockIndex
				e.nextBlockIndex++
				e.activeBlockType = BlockToolUse

				if tcID == "" {
					tcID = fmt.Sprintf("toolu_gate_%d", e.activeBlockIndex)
				}
				if fnName == "" {
					fnName = "unknown_tool"
				}

				buf.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":")
				writeInt(buf, e.activeBlockIndex)
				buf.WriteString(",\"content_block\":{\"type\":\"tool_use\",\"id\":\"")
				escapeJSONString(buf, []byte(tcID))
				buf.WriteString("\",\"name\":\"")
				escapeJSONString(buf, []byte(fnName))
				buf.WriteString("\",\"input\":{}}}\n\n")
			}

			if fnArgs != "" {
				buf.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":")
				writeInt(buf, e.activeBlockIndex)
				buf.WriteString(",\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":")
				buf.WriteByte('"')
				escapeJSONString(buf, []byte(fnArgs))
				buf.WriteString("\"}}\n\n")
			}
		})
	}
}

// ensureMessageStart emits the message_start event if not already sent.
func (e *AnthropicStreamEncoder) ensureMessageStart(buf *bytes.Buffer) {
	if e.messageStarted {
		return
	}
	e.messageStarted = true
	buf.WriteString("event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"")
	escapeJSONString(buf, []byte(e.messageID))
	buf.WriteString("\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"")
	escapeJSONString(buf, []byte(e.model))
	buf.WriteString("\",\"content\":[],\"stop_reason\":null,\"stop_sequence\":null,\"usage\":{\"input_tokens\":")
	writeInt(buf, e.inputTokens)
	buf.WriteString(",\"output_tokens\":0,\"cache_creation_input_tokens\":")
	writeInt(buf, e.cacheCreationInputTokens)
	buf.WriteString(",\"cache_read_input_tokens\":")
	writeInt(buf, e.cacheReadInputTokens)
	buf.WriteString("}}}\n\n")
}

// emitBlockStop closes the current active content block.
// For thinking blocks, emits a signature_delta event before content_block_stop
// to guarantee multi-turn validation compliance for Anthropic SDKs.
func (e *AnthropicStreamEncoder) emitBlockStop(buf *bytes.Buffer) {
	if e.activeBlockType == BlockNone || e.activeBlockIndex < 0 {
		return
	}

	if e.activeBlockType == BlockThinking {
		buf.WriteString("event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":")
		writeInt(buf, e.activeBlockIndex)
		buf.WriteString(",\"delta\":{\"type\":\"signature_delta\",\"signature\":\"sig_clever_gate\"}}\n\n")
	}

	buf.WriteString("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":")
	writeInt(buf, e.activeBlockIndex)
	buf.WriteString("}\n\n")

	e.activeBlockType = BlockNone
	e.activeBlockIndex = -1
}

// Finalize completes the Anthropic SSE stream: closes any open blocks,
// guarantees at least one content block was opened, and emits message_delta
// followed by message_stop.
func (e *AnthropicStreamEncoder) Finalize() []byte {
	if e.messageStopped {
		return nil
	}
	e.messageStopped = true

	buf := getBuffer()
	defer putBuffer(buf)

	e.ensureMessageStart(buf)

	// Degenerate empty stream: Anthropic SDKs expect at least one content block
	if e.activeBlockType == BlockNone && e.nextBlockIndex == 0 {
		buf.WriteString("event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n")
		buf.WriteString("event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n")
	} else if e.activeBlockType != BlockNone {
		e.emitBlockStop(buf)
	}

	if e.stopReason == "" {
		if e.toolUseSeen {
			e.stopReason = "tool_use"
		} else {
			e.stopReason = "end_turn"
		}
	}

	buf.WriteString("event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"")
	escapeJSONString(buf, []byte(e.stopReason))
	buf.WriteString("\",\"stop_sequence\":null},\"usage\":{\"output_tokens\":")
	writeInt(buf, e.outputTokens)
	buf.WriteString(",\"cache_creation_input_tokens\":")
	writeInt(buf, e.cacheCreationInputTokens)
	buf.WriteString(",\"cache_read_input_tokens\":")
	writeInt(buf, e.cacheReadInputTokens)
	buf.WriteString("}}\n\n")

	buf.WriteString("event: message_stop\ndata: {\"type\":\"message_stop\"}\n\n")

	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())
	return result
}

// EncodeStreamError emits an Anthropic error event frame and terminates the message.
func (e *AnthropicStreamEncoder) EncodeStreamError(errType, errMsg string) []byte {
	if e.messageStopped {
		return nil
	}
	buf := getBuffer()
	defer putBuffer(buf)
	e.emitStreamErrorInto(buf, errType, errMsg)

	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())
	return result
}

func (e *AnthropicStreamEncoder) emitStreamErrorInto(buf *bytes.Buffer, errType, errMsg string) {
	e.messageStopped = true
	if errType == "" {
		errType = "api_error"
	}
	buf.WriteString("event: error\ndata: {\"type\":\"error\",\"error\":{\"type\":\"")
	escapeJSONString(buf, []byte(errType))
	buf.WriteString("\",\"message\":\"")
	escapeJSONString(buf, []byte(errMsg))
	buf.WriteString("\"}}\n\n")
}

// MapOpenAIFinishReasonToAnthropic maps an OpenAI finish_reason string to
// its Anthropic stop_reason equivalent.
func MapOpenAIFinishReasonToAnthropic(finishReason string, toolUseSeen bool) string {
	if toolUseSeen || finishReason == "tool_calls" || finishReason == "function_call" {
		return "tool_use"
	}
	switch finishReason {
	case "stop":
		return "end_turn"
	case "length":
		return "max_tokens"
	case "content_filter":
		return "refusal"
	default:
		return "end_turn"
	}
}

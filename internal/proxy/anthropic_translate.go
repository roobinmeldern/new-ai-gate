package proxy

import (
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"net/http"
	"strconv"
	"strings"
	"sync"
	"unicode/utf8"

	"github.com/buger/jsonparser"
	"github.com/tidwall/sjson"
)

// ─────────────────────────────────────────────────────────────────────────────
// Anthropic ⇄ provider inbound request translation engine (Phase 2)
//
// This file implements the inbound half of the Anthropic-compatible front API:
//
//  1. Native zero-copy passthrough — when the acquired upstream credential is
//     an Anthropic-protocol provider ("anthropic", "agentrouter"), the request
//     body is forwarded in its original Anthropic Messages shape with only the
//     routing-prefixed "model" field rewritten (PrepareNativeAnthropicPayload).
//
//  2. Cross-provider translation — for every other provider (OpenAI, Gemini,
//     NVIDIA NIM, Ollama, 1min.ai, Cloudflare, Jiekou, DeepSeek, Groq,
//     OpenRouter, ZenMux, …), TranslateAnthropicToOpenAI converts the
//     Anthropic Messages payload into the canonical OpenAI Chat Completions
//     JSON that the existing provider pipeline already transforms into each
//     provider's native upstream request.
//
// Design invariants:
//   - no reflection-heavy struct graphs on the hot path: fields are located
//     with github.com/buger/jsonparser and assembled with a single-pass
//     hand-rolled JSON writer (same style as transmux.buildDelta);
//   - github.com/tidwall/sjson is used only for O(1)-sized point edits in the
//     native passthrough path;
//   - scratch buffers come from a sync.Pool and the output is a single copy.
// ─────────────────────────────────────────────────────────────────────────────

// isNativeAnthropicProvider reports whether the given provider speaks the
// Anthropic Messages wire protocol natively, so Anthropic front-door requests
// can be passed through without translation.
func isNativeAnthropicProvider(provider string) bool {
	return provider == "anthropic" || provider == "agentrouter"
}

// anthropicNativeStripFields lists OpenAI-only top-level fields that make a
// strict Anthropic Messages endpoint reject the payload. They are removed only
// when present (e.g. gateway-generated OpenAI bodies rerouted to a native
// provider); a genuine Anthropic request never carries them.
var anthropicNativeStripFields = []string{
	"stream_options",        // OpenAI streaming usage control
	"logit_bias",            // OpenAI token bias
	"logprobs",              // OpenAI log probabilities
	"top_logprobs",          // OpenAI top-N log probabilities
	"max_completion_tokens", // reasoning-model alias of max_tokens
	"n",                     // OpenAI parallel completions
	"seed",                  // OpenAI deterministic sampling
	"response_format",       // OpenAI structured outputs
	"service_tier",          // OpenAI priority tier
	"store",                 // OpenAI conversation storage
	"parallel_tool_calls",   // OpenAI parallel function calling
	"reasoning_effort",      // OpenAI reasoning effort
	"user",                  // OpenAI end-user identifier (Anthropic uses metadata.user_id)
}

// PrepareNativeAnthropicPayload prepares a raw Anthropic Messages body for a
// native Anthropic-protocol upstream:
//
//   - "model" is replaced with the stripped upstream model so routing prefixes
//     ("anthropic/", "agentrouter/" or the credential prefix) never reach the
//     upstream endpoint;
//   - everything Anthropic-native (system, messages, tools, tool_choice,
//     thinking, max_tokens, stream, temperature, top_p, top_k,
//     stop_sequences, metadata, ...) is preserved untouched;
//   - OpenAI-only fields (see anthropicNativeStripFields) are removed when
//     present, because strict Anthropic endpoints reject unknown keys.
func PrepareNativeAnthropicPayload(rawBody []byte, upstreamModel string) ([]byte, error) {
	out, err := sjson.SetBytes(rawBody, "model", upstreamModel)
	if err != nil {
		return nil, fmt.Errorf("prepare native anthropic payload: %w", err)
	}
	for _, field := range anthropicNativeStripFields {
		if _, _, _, getErr := jsonparser.Get(out, field); getErr == nil {
			out, err = sjson.DeleteBytes(out, field)
			if err != nil {
				return nil, fmt.Errorf("prepare native anthropic payload: strip %q: %w", field, err)
			}
		}
	}
	return out, nil
}

// anthropicPromptCachingBeta is the beta feature flag that enables prompt
// caching on Anthropic Messages endpoints. Bodies that carry `cache_control`
// breakpoints are rejected with a 400 ("feature not enabled") unless the
// upstream request advertises it, so the gateway injects the flag
// automatically when the client forgot (the documented Claude Code / SDK
// default is to send it, but hand-rolled clients frequently omit it).
const anthropicPromptCachingBeta = "prompt-caching-2024-07-31"

// AnthropicPromptCachingBeta returns the prompt-caching beta flag when the
// native Anthropic body contains at least one `cache_control` breakpoint and
// the caller has not already set an anthropic-beta header. Returning "" means
// "leave the header untouched" — existing client headers always win, and
// bodies without cache_control must not opt into the beta at all.
//
// The presence probe is a byte scan (allocations-free); a false positive from
// the literal appearing inside user text is harmless — enabling the beta
// flag changes no semantics for bodies that don't use it.
func AnthropicPromptCachingBeta(body []byte, existingBeta string) string {
	if existingBeta != "" {
		return ""
	}
	if !bytes.Contains(body, []byte(`"cache_control"`)) {
		return ""
	}
	return anthropicPromptCachingBeta
}

// translateBufPool recycles scratch buffers for the Anthropic→OpenAI
// translation. Buffers start at 16KB — large enough for typical chat payloads
// so growth reallocations are rare.
var translateBufPool = sync.Pool{
	New: func() interface{} {
		return bytes.NewBuffer(make([]byte, 0, 16*1024))
	},
}

// appendJSONString writes s as a JSON string literal (escaped) into buf.
// Only the mandatory escapes are emitted; every other byte passes through —
// valid UTF-8 multi-byte sequences are JSON-safe as-is.
func appendJSONString(buf *bytes.Buffer, s string) {
	buf.WriteByte('"')
	for i := 0; i < len(s); i++ {
		c := s[i]
		switch c {
		case '"':
			buf.WriteString(`\"`)
		case '\\':
			buf.WriteString(`\\`)
		case '\n':
			buf.WriteString(`\n`)
		case '\r':
			buf.WriteString(`\r`)
		case '\t':
			buf.WriteString(`\t`)
		default:
			if c < 0x20 {
				const hex = "0123456789abcdef"
				buf.WriteString(`\u00`)
				buf.WriteByte(hex[c>>4])
				buf.WriteByte(hex[c&0xF])
			} else {
				buf.WriteByte(c)
			}
		}
	}
	buf.WriteByte('"')
}

// emitComma writes the JSON array element separator unless this is the first
// element, then clears the flag.
func emitComma(buf *bytes.Buffer, first *bool) {
	if !*first {
		buf.WriteByte(',')
	}
	*first = false
}

// TranslateAnthropicToOpenAI converts an Anthropic Messages API request body
// into the canonical OpenAI Chat Completions JSON consumed by every
// non-Anthropic provider pipeline in this gateway.
//
// targetModel is written into the OpenAI "model" field verbatim (callers strip
// routing prefixes first, see stripAnthropicRoutingModel). fallbackMaxTokens is
// used when the Anthropic body carries no usable "max_tokens".
//
// Prompt caching (Phase 4): `cache_control` breakpoints have no OpenAI
// counterpart — they are stripped implicitly because the output body is
// assembled from scratch (only translatable fields are emitted). Prompt cache
// economics therefore do not survive cross-provider translation by design.
func TranslateAnthropicToOpenAI(rawBody []byte, targetModel string, fallbackMaxTokens int64) ([]byte, error) {
	if len(bytes.TrimSpace(rawBody)) == 0 {
		return nil, errors.New("empty anthropic request body")
	}
	if !json.Valid(rawBody) {
		return nil, errors.New("anthropic request body is not valid JSON")
	}

	msgVal, msgType, _, msgErr := jsonparser.Get(rawBody, "messages")
	if msgErr != nil || msgType != jsonparser.Array {
		return nil, errors.New("missing or invalid 'messages' field: expected an array of message objects")
	}

	fixedParams := IsFixedParamReasoningModel(targetModel)

	buf := translateBufPool.Get().(*bytes.Buffer)
	buf.Reset()
	defer translateBufPool.Put(buf)

	buf.WriteString(`{"model":`)
	appendJSONString(buf, targetModel)

	// ── messages: system prompt first, then the conversation ─────────────
	buf.WriteString(`,"messages":[`)
	first := true
	if systemText := collectAnthropicSystemText(rawBody); systemText != "" {
		buf.WriteString(`{"role":"system","content":`)
		appendJSONString(buf, systemText)
		buf.WriteByte('}')
		first = false
	}

	toolCallCounter := 0
	var convErr error
	jsonparser.ArrayEach(msgVal, func(msg []byte, dataType jsonparser.ValueType, _ int, elemErr error) {
		if convErr != nil {
			return
		}
		if elemErr != nil {
			convErr = errors.New("malformed entry in 'messages' array")
			return
		}
		if dataType != jsonparser.Object || len(msg) == 0 {
			return
		}
		role, _ := jsonparser.GetString(msg, "role")
		switch role {
		case "user":
			appendAnthropicUserMessage(buf, msg, &first, &toolCallCounter)
		case "assistant":
			appendAnthropicAssistantMessage(buf, msg, &first, &toolCallCounter)
		default:
			if role == "" {
				return
			}
			appendAnthropicPlainMessage(buf, msg, role, &first)
		}
	})
	if convErr != nil {
		return nil, convErr
	}
	buf.WriteByte(']')

	translateAnthropicTools(buf, rawBody)
	translateAnthropicToolChoice(buf, rawBody)
	translateAnthropicStopSequences(buf, rawBody)

	if effort := anthropicThinkingToReasoningEffort(rawBody); effort != "" {
		buf.WriteString(`,"reasoning_effort":`)
		appendJSONString(buf, effort)
	}

	translateAnthropicGenerationParams(buf, rawBody, fallbackMaxTokens, fixedParams)

	buf.WriteByte('}')

	out := make([]byte, buf.Len())
	copy(out, buf.Bytes())
	return out, nil
}

// collectAnthropicSystemText extracts the Anthropic "system" prompt as a
// single string: either the raw string form, or all "text" content blocks of
// the array form joined with blank lines (matching Anthropic's own
// block-concatenation behavior).
func collectAnthropicSystemText(rawBody []byte) string {
	sysVal, sysType, _, sysErr := jsonparser.Get(rawBody, "system")
	if sysErr != nil || len(sysVal) == 0 {
		return ""
	}
	switch sysType {
	case jsonparser.String:
		s, _ := jsonparser.GetString(rawBody, "system") // unescaped
		return s
	case jsonparser.Array:
		var parts []string
		jsonparser.ArrayEach(sysVal, func(block []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			if t, _ := jsonparser.GetString(block, "type"); t == "text" {
				if txt, _ := jsonparser.GetString(block, "text"); txt != "" {
					parts = append(parts, txt)
				}
			}
		})
		return strings.Join(parts, "\n\n")
	}
	return ""
}

// appendAnthropicPlainMessage passes through a message with a non-standard
// role (e.g. a stray "system" entry inside messages[]) when its content is a
// plain string. Everything else is skipped defensively.
func appendAnthropicPlainMessage(buf *bytes.Buffer, msg []byte, role string, first *bool) {
	_, contentType, _, contentErr := jsonparser.Get(msg, "content")
	if contentErr != nil || contentType != jsonparser.String {
		return
	}
	text, _ := jsonparser.GetString(msg, "content")
	emitComma(buf, first)
	buf.WriteString(`{"role":`)
	appendJSONString(buf, role)
	buf.WriteString(`,"content":`)
	appendJSONString(buf, text)
	buf.WriteByte('}')
}

// appendAnthropicUserMessage converts one Anthropic user message into the
// OpenAI message sequence. Parallel tool_result blocks become role:"tool"
// messages first (matching the order OpenAI tool calls expect), then the
// remaining text/image/document blocks become a single user message. Handles
// the JSON array comma via *first.
func appendAnthropicUserMessage(buf *bytes.Buffer, msg []byte, first *bool, _ *int) {
	contentVal, contentType, _, contentErr := jsonparser.Get(msg, "content")
	if contentErr != nil {
		return
	}

	// String content: one plain user message.
	if contentType == jsonparser.String {
		text, _ := jsonparser.GetString(msg, "content")
		emitComma(buf, first)
		buf.WriteString(`{"role":"user","content":`)
		appendJSONString(buf, text)
		buf.WriteByte('}')
		return
	}
	if contentType != jsonparser.Array {
		return
	}

	// Pass 1 — tool_result blocks become OpenAI tool messages first.
	jsonparser.ArrayEach(contentVal, func(block []byte, dataType jsonparser.ValueType, _ int, _ error) {
		if dataType != jsonparser.Object {
			return
		}
		if bt, _ := jsonparser.GetString(block, "type"); bt != "tool_result" {
			return
		}
		toolUseID, _ := jsonparser.GetString(block, "tool_use_id")
		if toolUseID == "" {
			return
		}
		text := extractAnthropicToolResultText(block)
		if isErr, _ := jsonparser.GetBoolean(block, "is_error"); isErr && !strings.HasPrefix(text, "Error: ") {
			text = "Error: " + text
		}
		emitComma(buf, first)
		buf.WriteString(`{"role":"tool","tool_call_id":`)
		appendJSONString(buf, toolUseID)
		buf.WriteString(`,"content":`)
		appendJSONString(buf, text)
		buf.WriteByte('}')
	})

	// Pass 2 — remaining text/image/document blocks become one user message.
	var textValues []string // text-only turn: joined into a single string
	var parts []string      // multimodal turn: ordered OpenAI content parts
	hasRichParts := false   // image/file parts present → array content required
	jsonparser.ArrayEach(contentVal, func(block []byte, dataType jsonparser.ValueType, _ int, _ error) {
		if dataType != jsonparser.Object {
			return
		}
		switch bt, _ := jsonparser.GetString(block, "type"); bt {
		case "text":
			txt, _ := jsonparser.GetString(block, "text")
			if txt == "" {
				return
			}
			textValues = append(textValues, txt)
			var p bytes.Buffer
			p.WriteString(`{"type":"text","text":`)
			appendJSONString(&p, txt)
			p.WriteByte('}')
			parts = append(parts, p.String())
		case "image":
			hasRichParts = true
			if part := buildAnthropicImagePart(block); part != "" {
				parts = append(parts, part)
			}
		case "document":
			// Phase 4: documents are no longer flattened to a lossy note —
			// text/plain sources forward their text, nested-content sources
			// forward the joined text blocks, and base64 PDFs become OpenAI
			// `file` content parts carrying a data URI. Only binary sources
			// without an OpenAI representation degrade to a text note.
			part, docText, isRich := buildAnthropicDocumentPart(block)
			if part == "" {
				return
			}
			if docText != "" {
				textValues = append(textValues, docText)
			}
			if isRich {
				hasRichParts = true
			}
			parts = append(parts, part)
		}
	})
	if len(parts) == 0 {
		return // tool_result-only message — nothing else to emit
	}

	emitComma(buf, first)
	if hasRichParts {
		buf.WriteString(`{"role":"user","content":[`)
		for i, part := range parts {
			if i > 0 {
				buf.WriteByte(',')
			}
			buf.WriteString(part)
		}
		buf.WriteString("]}")
	} else {
		// Text-only turn: a single string maximizes compatibility with
		// providers that reject array content on plain-text turns.
		buf.WriteString(`{"role":"user","content":`)
		appendJSONString(buf, strings.Join(textValues, "\n"))
		buf.WriteByte('}')
	}
}

// buildAnthropicImagePart converts an Anthropic image block into an OpenAI
// image_url content part. Base64 sources become data URLs; URL sources are
// passed through. Returns "" for unsupported or malformed sources.
func buildAnthropicImagePart(block []byte) string {
	srcVal, srcType, _, srcErr := jsonparser.Get(block, "source")
	if srcErr != nil || srcType != jsonparser.Object {
		return ""
	}
	srcKind, _ := jsonparser.GetString(srcVal, "type")
	mediaType, _ := jsonparser.GetString(srcVal, "media_type")
	var url string
	switch srcKind {
	case "base64":
		data, dataErr := jsonparser.GetString(srcVal, "data")
		if dataErr != nil || data == "" {
			return ""
		}
		url = "data:" + mediaType + ";base64," + data
	case "url":
		u, uErr := jsonparser.GetString(srcVal, "url")
		if uErr != nil || u == "" {
			return ""
		}
		url = u
	default:
		return ""
	}
	var p bytes.Buffer
	p.WriteString(`{"type":"image_url","image_url":{"url":`)
	appendJSONString(&p, url)
	p.WriteString("}}")
	return p.String()
}

// buildAnthropicDocumentNote degrades an Anthropic document (PDF) block into a
// short text note, since OpenAI-compatible chat APIs have no equivalent.
func buildAnthropicDocumentNote(block []byte) string {
	title, _ := jsonparser.GetString(block, "title")
	if title == "" {
		if srcVal, srcType, _, srcErr := jsonparser.Get(block, "source"); srcErr == nil && srcType == jsonparser.Object {
			if mt, _ := jsonparser.GetString(srcVal, "media_type"); mt != "" {
				title = mt
			}
		}
	}
	if title == "" {
		title = "attachment"
	}
	return "[document: " + title + " - binary document content is not forwarded to this provider]"
}

// buildAnthropicDocumentPart converts an Anthropic document (PDF/text) block
// into the best available OpenAI representation:
//
//   - source.type == "text"   → a text content part carrying the document's
//     text payload (the plain-string fast path also receives it via docText);
//   - source.type == "content" → the joined text sub-blocks of the nested
//     content array (Anthropic's structured document form);
//   - source.type == "base64" with media_type "application/pdf" → an OpenAI
//     `file` content part with a data URI ({"type":"file","file":{…}}), the
//     shape gpt-4o-class models accept for PDF input — isRich=true;
//   - anything else (binary, unsupported media) → a short text note so the
//     intent is not lost — isRich=false, docText="" (note only).
//
// part is "" only for malformed blocks with nothing meaningful to forward.
func buildAnthropicDocumentPart(block []byte) (part string, docText string, isRich bool) {
	srcVal, srcType, _, srcErr := jsonparser.Get(block, "source")
	if srcErr != nil || srcType != jsonparser.Object {
		// Malformed source — fall back to the title-only note.
		note := buildAnthropicDocumentNote(block)
		return notePart(note), "", false
	}
	srcKind, _ := jsonparser.GetString(srcVal, "type")
	mediaType, _ := jsonparser.GetString(srcVal, "media_type")
	title, _ := jsonparser.GetString(block, "title")

	switch {
	case srcKind == "text" || mediaType == "text/plain":
		data, _ := jsonparser.GetString(srcVal, "data")
		if data == "" {
			data, _ = jsonparser.GetString(srcVal, "text")
		}
		if data == "" {
			note := buildAnthropicDocumentNote(block)
			return notePart(note), "", false
		}
		if title != "" {
			data = fmt.Sprintf("[Document: %s]\n%s", title, data)
		}
		var p bytes.Buffer
		p.WriteString(`{"type":"text","text":`)
		appendJSONString(&p, data)
		p.WriteByte('}')
		return p.String(), data, false

	case srcKind == "content":
		// Nested structured document: forward the text sub-blocks joined
		// with newlines, mirroring extractAnthropicToolResultText.
		var sb strings.Builder
		cVal, cType, _, cErr := jsonparser.Get(srcVal, "content")
		if cErr == nil && cType == jsonparser.Array {
			jsonparser.ArrayEach(cVal, func(sub []byte, dataType jsonparser.ValueType, _ int, _ error) {
				if dataType != jsonparser.Object {
					return
				}
				if t, _ := jsonparser.GetString(sub, "type"); t != "text" {
					return
				}
				if txt, _ := jsonparser.GetString(sub, "text"); txt != "" {
					if sb.Len() > 0 {
						sb.WriteString("\n")
					}
					sb.WriteString(txt)
				}
			})
		}
		if sb.Len() == 0 {
			note := buildAnthropicDocumentNote(block)
			return notePart(note), "", false
		}
		text := sb.String()
		if title != "" {
			text = fmt.Sprintf("[Document: %s]\n%s", title, text)
		}
		var p bytes.Buffer
		p.WriteString(`{"type":"text","text":`)
		appendJSONString(&p, text)
		p.WriteByte('}')
		return p.String(), text, false

	case srcKind == "base64" && mediaType == "application/pdf":
		data, _ := jsonparser.GetString(srcVal, "data")
		if data != "" {
			// OpenAI PDF input: {"type":"file","file":{"filename":…,
			// "file_data":"data:application/pdf;base64,…"}}.
			if title == "" {
				title = "document.pdf"
			}
			var p bytes.Buffer
			p.WriteString(`{"type":"file","file":{"filename":`)
			appendJSONString(&p, title)
			p.WriteString(`,"file_data":"data:application/pdf;base64,`)
			p.WriteString(data) // base64 alphabet is JSON-safe — copy verbatim
			p.WriteString(`"}}`)
			return p.String(), "", true
		}
	}

	// Binary or unsupported source — degrade to the lossy text note.
	note := buildAnthropicDocumentNote(block)
	return notePart(note), "", false
}

// notePart wraps a document note string as a text content part.
func notePart(note string) string {
	var p bytes.Buffer
	p.WriteString(`{"type":"text","text":`)
	appendJSONString(&p, note)
	p.WriteByte('}')
	return p.String()
}

// extractAnthropicToolResultText extracts the textual content of an Anthropic
// tool_result block: string content directly, array content as the joined text
// sub-blocks, and an empty string when omitted.
func extractAnthropicToolResultText(block []byte) string {
	cVal, cType, _, cErr := jsonparser.Get(block, "content")
	if cErr != nil {
		return ""
	}
	switch cType {
	case jsonparser.String:
		s, _ := jsonparser.GetString(block, "content")
		return s
	case jsonparser.Array:
		var sb strings.Builder
		jsonparser.ArrayEach(cVal, func(sub []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			if t, _ := jsonparser.GetString(sub, "type"); t != "text" {
				return
			}
			if txt, _ := jsonparser.GetString(sub, "text"); txt != "" {
				if sb.Len() > 0 {
					sb.WriteString("\n")
				}
				sb.WriteString(txt)
			}
		})
		return sb.String()
	}
	return ""
}

// appendAnthropicAssistantMessage converts one Anthropic assistant message
// into a single OpenAI assistant message: text blocks become "content",
// thinking blocks become "reasoning_content", and tool_use blocks become the
// OpenAI "tool_calls" array. Handles the JSON array comma via *first.
func appendAnthropicAssistantMessage(buf *bytes.Buffer, msg []byte, first *bool, toolCallCounter *int) {
	contentVal, contentType, _, contentErr := jsonparser.Get(msg, "content")

	// Top-level string content (rare but valid on assistant messages).
	if contentErr == nil && contentType == jsonparser.String {
		text, _ := jsonparser.GetString(msg, "content")
		emitComma(buf, first)
		buf.WriteString(`{"role":"assistant","content":`)
		appendJSONString(buf, text)
		buf.WriteByte('}')
		return
	}

	var textValues []string
	var thinkingValues []string
	var toolCalls []string
	if contentErr == nil && contentType == jsonparser.Array {
		jsonparser.ArrayEach(contentVal, func(block []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			switch bt, _ := jsonparser.GetString(block, "type"); bt {
			case "text":
				if txt, _ := jsonparser.GetString(block, "text"); txt != "" {
					textValues = append(textValues, txt)
				}
			case "thinking":
				if th, _ := jsonparser.GetString(block, "thinking"); th != "" {
					thinkingValues = append(thinkingValues, th)
				}
			case "redacted_thinking":
				// explicitly ignored: carries no translatable reasoning for non-Anthropic providers
			case "tool_use":
				if call := buildAnthropicToolCall(block, toolCallCounter); call != "" {
					toolCalls = append(toolCalls, call)
				}
			}
		})
	}

	content := strings.Join(textValues, "\n")
	hasToolCalls := len(toolCalls) > 0
	if content == "" && !hasToolCalls && len(thinkingValues) == 0 {
		return // nothing translatable in this message
	}

	emitComma(buf, first)
	buf.WriteString(`{"role":"assistant","content":`)
	if content == "" && hasToolCalls {
		if len(thinkingValues) > 0 {
			buf.WriteString(`""`)
		} else {
			// OpenAI convention: null content when the turn is a pure tool call without reasoning.
			buf.WriteString(`null`)
		}
	} else {
		appendJSONString(buf, content)
	}
	if len(thinkingValues) > 0 {
		buf.WriteString(`,"reasoning_content":`)
		appendJSONString(buf, strings.Join(thinkingValues, "\n"))
	}
	if hasToolCalls {
		buf.WriteString(`,"tool_calls":[`)
		for i, call := range toolCalls {
			if i > 0 {
				buf.WriteByte(',')
			}
			buf.WriteString(call)
		}
		buf.WriteByte(']')
	}
	buf.WriteByte('}')
}

// buildAnthropicToolCall converts an Anthropic tool_use block into an OpenAI
// tool_calls entry. Note the key asymmetry: Anthropic carries the arguments as
// a JSON *object* in "input", while OpenAI requires a JSON-encoded *string* in
// "function.arguments". Empty ids get a deterministic call_gate_<n> fallback.
func buildAnthropicToolCall(block []byte, toolCallCounter *int) string {
	name, _ := jsonparser.GetString(block, "name")
	if name == "" {
		return ""
	}
	id, _ := jsonparser.GetString(block, "id")
	if id == "" {
		id = fmt.Sprintf("call_gate_%d", *toolCallCounter)
		*toolCallCounter++
	}
	args := "{}"
	if inputRaw, inputType, _, inputErr := jsonparser.Get(block, "input"); inputErr == nil &&
		(inputType == jsonparser.Object || inputType == jsonparser.Array) && len(inputRaw) > 0 {
		args = string(inputRaw)
	}
	var b bytes.Buffer
	b.WriteString(`{"id":`)
	appendJSONString(&b, id)
	b.WriteString(`,"type":"function","function":{"name":`)
	appendJSONString(&b, name)
	b.WriteString(`,"arguments":`)
	appendJSONString(&b, args)
	b.WriteString(`}}`)
	return b.String()
}

// translateAnthropicTools appends the OpenAI "tools" array when the Anthropic
// body defines tools. Anthropic {name, description, input_schema} definitions
// become OpenAI {type:"function", function:{...}} entries; entries that
// already carry a "function" object (hybrid clients) pass through unchanged.
func translateAnthropicTools(buf *bytes.Buffer, rawBody []byte) {
	toolsVal, toolsType, _, toolsErr := jsonparser.Get(rawBody, "tools")
	if toolsErr != nil || toolsType != jsonparser.Array || len(toolsVal) == 0 {
		return
	}
	var out bytes.Buffer
	out.WriteByte('[')
	n := 0
	jsonparser.ArrayEach(toolsVal, func(tool []byte, dataType jsonparser.ValueType, _ int, _ error) {
		if dataType != jsonparser.Object || len(tool) == 0 {
			return
		}
		var entry string
		if _, fnType, _, fnErr := jsonparser.Get(tool, "function"); fnErr == nil && fnType == jsonparser.Object {
			// Hybrid client: already OpenAI-shaped — preserve as-is.
			entry = string(tool)
		} else {
			name, _ := jsonparser.GetString(tool, "name")
			if name == "" {
				return
			}
			schemaVal, schemaType, _, schemaErr := jsonparser.Get(tool, "input_schema")
			if schemaErr != nil || schemaType != jsonparser.Object {
				return // Anthropic requires input_schema; nothing to map
			}
			var e bytes.Buffer
			e.WriteString(`{"type":"function","function":{"name":`)
			appendJSONString(&e, name)
			if desc, dErr := jsonparser.GetString(tool, "description"); dErr == nil && desc != "" {
				e.WriteString(`,"description":`)
				appendJSONString(&e, desc)
			}
			e.WriteString(`,"parameters":`)
			e.Write(schemaVal) // raw schema copy — zero re-encoding
			e.WriteString(`}}`)
			entry = e.String()
		}
		if entry == "" {
			return
		}
		if n > 0 {
			out.WriteByte(',')
		}
		out.WriteString(entry)
		n++
	})
	if n == 0 {
		return
	}
	out.WriteByte(']')
	buf.WriteString(`,"tools":`)
	buf.Write(out.Bytes())
}

// translateAnthropicToolChoice appends the OpenAI "tool_choice" (and
// "parallel_tool_calls") fields when the Anthropic body constrains tool usage.
//
//	Anthropic {"type":"auto"}  -> "auto"
//	Anthropic {"type":"any"}   -> "required"
//	Anthropic {"type":"none"}  -> "none"
//	Anthropic {"type":"tool","name":X} -> {"type":"function","function":{"name":X}}
func translateAnthropicToolChoice(buf *bytes.Buffer, rawBody []byte) {
	tcVal, tcType, _, tcErr := jsonparser.Get(rawBody, "tool_choice")
	if tcErr != nil || tcType != jsonparser.Object {
		return
	}
	kind, _ := jsonparser.GetString(tcVal, "type")
	switch kind {
	case "auto":
		buf.WriteString(`,"tool_choice":"auto"`)
	case "any":
		buf.WriteString(`,"tool_choice":"required"`)
	case "none":
		buf.WriteString(`,"tool_choice":"none"`)
	case "tool":
		name, _ := jsonparser.GetString(tcVal, "name")
		if name == "" {
			return
		}
		buf.WriteString(`,"tool_choice":{"type":"function","function":{"name":`)
		appendJSONString(buf, name)
		buf.WriteString(`}}`)
	default:
		return
	}
	if disabled, _ := jsonparser.GetBoolean(tcVal, "disable_parallel_tool_use"); disabled {
		buf.WriteString(`,"parallel_tool_calls":false`)
	}
}

// anthropicThinkingToReasoningEffort maps the Anthropic extended-thinking
// budget to the OpenAI reasoning_effort scale used by this gateway:
//
//	budget < 4096   -> "low"
//	budget >= 16384 -> "high"
//	otherwise       -> "medium"
//
// Returns "" when thinking is absent, disabled, or carries no usable budget.
func anthropicThinkingToReasoningEffort(rawBody []byte) string {
	thVal, thType, _, thErr := jsonparser.Get(rawBody, "thinking")
	if thErr != nil || thType != jsonparser.Object {
		return ""
	}
	kind, _ := jsonparser.GetString(thVal, "type")
	if kind != "enabled" && kind != "adaptive" {
		return ""
	}
	budget, bErr := jsonparser.GetInt(thVal, "budget_tokens")
	if bErr != nil || budget <= 0 {
		return ""
	}
	switch {
	case budget < 4096:
		return "low"
	case budget >= 16384:
		return "high"
	default:
		return "medium"
	}
}

// translateAnthropicStopSequences appends the OpenAI "stop" array when the
// Anthropic body constrains generation stops.
func translateAnthropicStopSequences(buf *bytes.Buffer, rawBody []byte) {
	ssVal, ssType, _, ssErr := jsonparser.Get(rawBody, "stop_sequences")
	if ssErr != nil || ssType != jsonparser.Array || len(ssVal) == 0 {
		return
	}
	var stops []string
	if err := json.Unmarshal(ssVal, &stops); err != nil || len(stops) == 0 {
		return
	}
	buf.WriteString(`,"stop":[`)
	for i, s := range stops {
		if i > 0 {
			buf.WriteByte(',')
		}
		appendJSONString(buf, s)
	}
	buf.WriteByte(']')
}

// translateAnthropicGenerationParams appends the sampling parameters: the max
// completion budget (max_tokens, or max_completion_tokens plus the fixed
// parameter set for reasoning models), temperature/top_p, the streaming flags,
// and the OpenAI "user" field derived from metadata.user_id.
func translateAnthropicGenerationParams(buf *bytes.Buffer, rawBody []byte, fallbackMaxTokens int64, fixedParams bool) {
	maxTokens, mtErr := jsonparser.GetInt(rawBody, "max_tokens")
	if mtErr != nil || maxTokens <= 0 {
		maxTokens = fallbackMaxTokens
	}
	if maxTokens <= 0 {
		maxTokens = 4096 // last-resort guard (Anthropic's documented minimum budget)
	}

	if fixedParams {
		// Fixed-parameter reasoning models (gpt-5*, *-sol, o1/o3/o4*, *reasoning*)
		// enforce these values and use max_completion_tokens instead of max_tokens.
		buf.WriteString(`,"max_completion_tokens":`)
		buf.WriteString(strconv.FormatInt(maxTokens, 10))
		buf.WriteString(`,"temperature":1,"top_p":1,"n":1,"presence_penalty":0,"frequency_penalty":0`)
	} else {
		buf.WriteString(`,"max_tokens":`)
		buf.WriteString(strconv.FormatInt(maxTokens, 10))
		if tVal, tType, _, tErr := jsonparser.Get(rawBody, "temperature"); tErr == nil && tType == jsonparser.Number && len(tVal) > 0 {
			buf.WriteString(`,"temperature":`)
			buf.Write(tVal)
		}
		if pVal, pType, _, pErr := jsonparser.Get(rawBody, "top_p"); pErr == nil && pType == jsonparser.Number && len(pVal) > 0 {
			buf.WriteString(`,"top_p":`)
			buf.Write(pVal)
		}
	}

	// Streaming: mirror the flag and request usage accounting at the end of
	// the stream (OpenAI-compatible upstreams).
	if isStream, sErr := jsonparser.GetBoolean(rawBody, "stream"); sErr == nil && isStream {
		buf.WriteString(`,"stream":true,"stream_options":{"include_usage":true}`)
	}

	// Anthropic metadata.user_id -> OpenAI user (end-user identification).
	if uid, uErr := jsonparser.GetString(rawBody, "metadata", "user_id"); uErr == nil && uid != "" {
		buf.WriteString(`,"user":`)
		appendJSONString(buf, uid)
	}
}

// stripAnthropicRoutingModel removes the gateway routing prefix from a
// requested model so the upstream receives the clean model id.
//
// Priority:
//  1. exact pool pattern match -> the pattern's clean form
//  2. explicit Anthropic-front namespaces ("anthropic/", "agentrouter/")
//  3. any other known provider routing namespace ("nvidia/", "open router/", ...)
//
// ExtractAnthropicPromptText extracts a compact text summary of an Anthropic
// request for telemetry and semantic vector logging: the system prompt (if
// any) plus the last user message's text content. Output is capped to keep
// log entries bounded.
func ExtractAnthropicPromptText(body []byte) string {
	if len(body) == 0 {
		return ""
	}
	const maxChars = 4096
	var sb strings.Builder

	if sys := collectAnthropicSystemText(body); sys != "" {
		sb.WriteString(truncateRunes(sys, maxChars/2))
	}

	var lastUser string
	if msgVal, msgType, _, msgErr := jsonparser.Get(body, "messages"); msgErr == nil && msgType == jsonparser.Array {
		jsonparser.ArrayEach(msgVal, func(msg []byte, dataType jsonparser.ValueType, _ int, _ error) {
			if dataType != jsonparser.Object {
				return
			}
			if role, _ := jsonparser.GetString(msg, "role"); role != "user" {
				return
			}
			contentVal, contentType, _, contentErr := jsonparser.Get(msg, "content")
			if contentErr != nil {
				return
			}
			switch contentType {
			case jsonparser.String:
				if s, _ := jsonparser.GetString(msg, "content"); s != "" {
					lastUser = s
				}
			case jsonparser.Array:
				var texts []string
				jsonparser.ArrayEach(contentVal, func(block []byte, blockType jsonparser.ValueType, _ int, _ error) {
					if blockType != jsonparser.Object {
						return
					}
					if t, _ := jsonparser.GetString(block, "type"); t == "text" {
						if txt, _ := jsonparser.GetString(block, "text"); txt != "" {
							texts = append(texts, txt)
						}
					}
				})
				if len(texts) > 0 {
					lastUser = strings.Join(texts, "\n")
				}
			}
		})
	}
	if lastUser != "" {
		if sb.Len() > 0 {
			sb.WriteString("\n")
		}
		sb.WriteString(truncateRunes(lastUser, maxChars/2))
	}
	return sb.String()
}

// truncateRunes caps a string at max Unicode code points.
func truncateRunes(s string, max int) string {
	if max <= 0 || utf8.RuneCountInString(s) <= max {
		return s
	}
	runes := []rune(s)
	return string(runes[:max])
}

// anthropicErrorTypeForStatus maps an HTTP status to the Anthropic error type.
func anthropicErrorTypeForStatus(status int) string {
	switch status {
	case http.StatusBadRequest, http.StatusRequestEntityTooLarge, http.StatusUnsupportedMediaType:
		return "invalid_request_error"
	case http.StatusUnauthorized:
		return "authentication_error"
	case http.StatusForbidden:
		return "permission_error"
	case http.StatusNotFound:
		return "not_found_error"
	case http.StatusTooManyRequests:
		return "rate_limit_error"
	case 529:
		return "overloaded_error"
	default:
		return "api_error"
	}
}

// formatAnthropicGatewayError builds the canonical Anthropic error envelope for
// gateway-level failures (pool exhaustion, upstream rejection) on the
// Anthropic front door — the counterpart of formatOpenError for the OpenAI door.
func formatAnthropicGatewayError(status int, rawBody []byte, summary string) []byte {
	// Prefer an upstream error message when one is extractable.
	upstreamMsg := ""
	if len(rawBody) > 0 {
		if msg, err := jsonparser.GetString(rawBody, "error", "message"); err == nil && msg != "" {
			upstreamMsg = msg
		}
		if upstreamMsg == "" {
			if msg, err := jsonparser.GetString(rawBody, "message"); err == nil && msg != "" {
				upstreamMsg = msg
			}
		}
		if upstreamMsg == "" {
			// FastAPI / Python upstream pattern.
			if msg, err := jsonparser.GetString(rawBody, "detail"); err == nil && msg != "" {
				upstreamMsg = msg
			}
		}
	}

	message := summary
	if upstreamMsg != "" {
		message = summary + ". Last upstream message: " + upstreamMsg
	} else if len(rawBody) > 0 {
		raw := string(rawBody)
		if strings.Contains(raw, "<html") || strings.Contains(raw, "<!DOCTYPE") {
			raw = "[HTML response from upstream load balancer or CDN]"
		} else if len(raw) > 512 {
			raw = raw[:512] + "…"
		}
		if isPrintable(raw) {
			message = summary + ". Last upstream body: " + raw
		}
	}

	var buf bytes.Buffer
	buf.WriteString(`{"type":"error","error":{"type":`)
	appendJSONString(&buf, anthropicErrorTypeForStatus(status))
	buf.WriteString(`,"message":`)
	appendJSONString(&buf, message)
	buf.WriteString(`}}`)
	return buf.Bytes()
}

// 4. unchanged (no routing prefix present)
func stripAnthropicRoutingModel(model, poolPattern string) string {
	if poolPattern != "" && model == poolPattern {
		return FormatUpstreamModelForPattern(poolPattern)
	}
	if strings.HasPrefix(model, "anthropic/") {
		return strings.TrimPrefix(model, "anthropic/")
	}
	if strings.HasPrefix(model, "agentrouter/") {
		return strings.TrimPrefix(model, "agentrouter/")
	}
	if clean, provider := StripProviderPrefix(model); provider != "" {
		return clean
	}
	return model
}

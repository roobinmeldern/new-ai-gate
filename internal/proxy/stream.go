package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"fmt"
	"io"
	"net/http"
	"runtime/debug"
	"sort"
	"strings"
	"sync"
	"sync/atomic"
	"time"

	"github.com/buger/jsonparser"
	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/transmux"
	"go.uber.org/zap"
)

// StreamProxy handles SSE (Server-Sent Events) streaming from upstream providers.
// It reads upstream chunks line-by-line, passes them through a provider-specific
// transmuxer for format normalization, and flushes each chunk to the client immediately.
type StreamProxy struct {
	client      *http.Client
	logger      *zap.Logger
	scannerPool sync.Pool
}

// NewStreamProxy creates a new stream proxy handler.
func NewStreamProxy(client *http.Client, logger *zap.Logger) *StreamProxy {
	return &StreamProxy{
		client: client,
		logger: logger,
		scannerPool: sync.Pool{
			New: func() interface{} {
				return make([]byte, 0, 64*1024) // 64KB scanner buffer
			},
		},
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Stream liveness knobs
//
// Production incident (2026-09-23, 52xueai/claude-opus-5): the model streamed
// its opening answer, then went silent for ~109s while composing a large code
// block. The gateway had nothing to relay, so every intermediary between the
// client and the gateway (nginx / Cloudflare / hosting load balancer, idle
// timeouts of ~60-110s) tore the client connection down — surfacing to the
// end user as "unexpected EOF (incomplete chunked read)".
//
// Two mechanisms below fix that class of failure:
//  1. an SSE heartbeat (`: keepalive` comment) emitted while the upstream is
//     silent, which resets every intermediary's idle timer;
//  2. a stall watchdog that gives up on a totally-silent upstream and
//     terminates the client stream cleanly with a visible error event.
//
// Package-level vars (not consts) so tests can shrink them without waiting
// wall-clock minutes.
var (
	// sseKeepaliveInterval is how often the heartbeat fires while the
	// upstream is silent.
	sseKeepaliveInterval = 15 * time.Second

	// streamStallTimeout is the maximum upstream silence tolerated mid-stream
	// before the credential is abandoned. Generous by design: even long
	// Claude-style thinking gaps stream keepalive deltas and never approach
	// this window.
	streamStallTimeout = 5 * time.Minute
)

// sseWriter serializes writes to the client response between the main relay
// loop and the keepalive heartbeat goroutine. SSE comments (`: …`) are legal
// between events per the SSE spec, so interleaving heartbeats with data
// events is protocol-safe. The mutex is released via defer inside every
// method, so a panic mid-write can never leave it locked for the recovery
// path.
type sseWriter struct {
	w  gin.ResponseWriter
	mu sync.Mutex
}

// writeDataEvent emits one `data: <payload>\n\n` SSE event and flushes it.
func (s *sseWriter) writeDataEvent(payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.w.Write([]byte("data: ")); err != nil {
		return err
	}
	if _, err := s.w.Write(payload); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("\n\n")); err != nil {
		return err
	}
	s.w.Flush()
	return nil
}

// writeEvent emits one named SSE event (`event: <name>\ndata: <payload>\n\n`)
// and flushes it. Used for Anthropic-protocol error termination.
func (s *sseWriter) writeEvent(name string, payload []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.w.Write([]byte("event: " + name + "\ndata: ")); err != nil {
		return err
	}
	if _, err := s.w.Write(payload); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("\n\n")); err != nil {
		return err
	}
	s.w.Flush()
	return nil
}

// writeRawLine relays one pre-formatted SSE line byte-for-byte, restoring the
// line ending, and flushes. Used by ProxyStreamRaw for zero-copy relaying.
func (s *sseWriter) writeRawLine(line []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.w.Write(line); err != nil {
		return err
	}
	if _, err := s.w.Write([]byte("\n")); err != nil {
		return err
	}
	s.w.Flush()
	return nil
}

// writeComment emits one SSE comment line (`: <comment>\n\n`) and flushes it.
// Comments are ignored by every SSE parser — the standard heartbeat channel.
func (s *sseWriter) writeComment(comment string) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if _, err := s.w.Write([]byte(": " + comment + "\n\n")); err != nil {
		return err
	}
	s.w.Flush()
	return nil
}

// writeRawBytes writes pre-formatted SSE frames directly to the response writer and flushes.
func (s *sseWriter) writeRawBytes(data []byte) error {
	s.mu.Lock()
	defer s.mu.Unlock()
	if len(data) == 0 {
		return nil
	}
	if _, err := s.w.Write(data); err != nil {
		return err
	}
	s.w.Flush()
	return nil
}

// signalProgressChan notifies the heartbeat goroutine that upstream data was
// just relayed, resetting the stall watchdog. Non-blocking: a dropped signal
// only delays the watchdog by one heartbeat tick.
func signalProgressChan(progress chan struct{}) {
	select {
	case progress <- struct{}{}:
	default:
	}
}

// EmitAnthropicEvent satisfies transmux.AnthropicSSEWriter: it writes one
// named Anthropic SSE event frame and flushes it. An empty event name falls
// back to an anonymous data event.
func (s *sseWriter) EmitAnthropicEvent(event string, data []byte) error {
	if event == "" {
		return s.writeDataEvent(data)
	}
	return s.writeEvent(event, data)
}

// emitStreamInterrupted terminates an SSE stream with a visible, OpenAI-style
// error event followed by [DONE] so clients render a clear "stream
// interrupted" error instead of a silently truncated (or hanging) response.
// The client already holds partial content at this point — a retry is
// impossible, so the honest, graceful end is an explicit error signal.
func emitStreamInterrupted(sw *sseWriter, cause string) {
	if sw == nil {
		return
	}
	payload := fmt.Sprintf(
		`{"error":{"message":"Upstream connection interrupted before the response completed (%s). Partial content may be missing — retry the request.","type":"server_error","code":"stream_interrupted"}}`,
		sanitizeSSEJSONString(cause, 200),
	)
	_ = sw.writeDataEvent([]byte(payload))
	_ = sw.writeDataEvent([]byte("[DONE]"))
}

// sanitizeSSEJSONString makes an arbitrary error string safe to embed inside a
// JSON string literal (strips quotes, backslashes and control characters).
func sanitizeSSEJSONString(s string, maxLen int) string {
	var b strings.Builder
	for _, r := range s {
		if r == '"' || r == '\\' || r < 0x20 {
			continue
		}
		b.WriteRune(r)
		if b.Len() >= maxLen {
			break
		}
	}
	return b.String()
}

// ─────────────────────────────────────────────────────────────────────────────
// ProxyStream options
// ─────────────────────────────────────────────────────────────────────────────

// ProxyStreamOption customizes optional ProxyStream relay behavior.
type ProxyStreamOption func(*proxyStreamOptions)

type proxyStreamOptions struct {
	// anthropicModel is non-empty when the client speaks the Anthropic
	// Messages protocol: the relay loop converts every upstream OpenAI-format
	// chunk into Anthropic SSE events addressed to this model id instead of
	// writing chat.completion.chunk data events.
	anthropicModel       string
	anthropicMsgID       string
	anthropicInputTokens int
}

// WithAnthropicModel enables Anthropic front-door encoding mode: upstream
// OpenAI chunks are re-encoded into the Anthropic Messages SSE protocol
// (message_start / content_block_* / message_delta / message_stop) before
// being flushed to the client.
func WithAnthropicModel(model string) ProxyStreamOption {
	return func(o *proxyStreamOptions) {
		o.anthropicModel = model
	}
}

// WithAnthropicStream enables Anthropic front-door encoding mode with full
// message ID and input token metadata.
func WithAnthropicStream(msgID, model string, inputTokens int) ProxyStreamOption {
	return func(o *proxyStreamOptions) {
		o.anthropicMsgID = msgID
		o.anthropicModel = model
		o.anthropicInputTokens = inputTokens
	}
}

// ProxyStream pipes SSE chunks from upstream to client with format translation,
// and returns the fully accumulated response text along with estimated completion tokens.
//
// Optional ProxyStreamOption arguments customize the relay. The Anthropic
// front door passes WithAnthropicModel to convert the OpenAI-format chunks
// into Anthropic Messages SSE events before they reach the client.
func (sp *StreamProxy) ProxyStream(c *gin.Context, upstream *http.Response, provider string, requestedModel string, opts ...ProxyStreamOption) (responseText string, completionTokens int) {
	// Resolve optional behaviors (zero-cost when no options are given).
	var o proxyStreamOptions
	for _, opt := range opts {
		if opt != nil {
			opt(&o)
		}
	}
	var anthropicEnc *transmux.AnthropicStreamEncoder
	if o.anthropicModel != "" {
		anthropicEnc = transmux.NewAnthropicStreamEncoder(o.anthropicMsgID, o.anthropicModel, o.anthropicInputTokens)
	}

	// Acquire pooled scanner buffer
	scanBuf := sp.scannerPool.Get().([]byte)

	var responseBuilder strings.Builder
	var tokenEstimate int
	streamStart := time.Now()

	var sw *sseWriter // created once the flusher is confirmed; nil-safe everywhere

	defer func() {
		// Always return the scanner buffer to the pool
		sp.scannerPool.Put(scanBuf[:0])

		// Always close the upstream response body
		upstream.Body.Close()

		// Catch any panic from the transmuxer or write path
		if r := recover(); r != nil {
			sp.logger.Error("recovered from stream processing panic",
				zap.Any("panic", r),
				zap.String("provider", provider),
				zap.ByteString("stack", debug.Stack()),
			)
			// Attempt to signal stream termination to client if connection is still alive
			if sw != nil {
				_ = sw.writeDataEvent([]byte("[DONE]"))
			}
		}
	}()

	// Step 1: Set SSE headers for streaming
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("Transfer-Encoding", "chunked")
	c.Writer.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering
	c.Writer.WriteHeader(http.StatusOK)

	// Step 2: Create the appropriate transmuxer for this provider
	tmx := transmux.NewTransmuxer(provider)
	defer tmx.Close()

	// Step 3: Create scanner with pooled buffer
	scanner := bufio.NewScanner(upstream.Body)
	scanner.Buffer(scanBuf, 1024*1024) // Max 1MB line (for base64 images)

	if _, ok := c.Writer.(http.Flusher); !ok {
		sp.logger.Error("response writer does not support flushing")
		return "", 0
	}
	sw = &sseWriter{w: c.Writer}

	// Step 3.5: keepalive heartbeat + stall watchdog.
	//
	// Long generations regularly go silent for minutes while the model
	// composes a large code block. Intermediaries between the client and the
	// gateway (nginx, Cloudflare, hosting load balancers) kill "idle"
	// connections after ~60-110s. The heartbeat goroutine below:
	//
	//   1. emits `: keepalive` SSE comments while the upstream is silent,
	//      keeping every intermediary's idle timer reset;
	//   2. aborts the upstream read once the client connection is gone
	//      (heartbeat write fails) so the relay loop exits promptly;
	//   3. aborts the upstream read after streamStallTimeout of total
	//      silence — a truly dead upstream — so the client receives a clean
	//      error event + [DONE] instead of hanging forever.
	kaDone := make(chan struct{})
	progress := make(chan struct{}, 1)
	var clientGone int32   // atomic: heartbeat write failed → client disconnected
	var stallAborted int32 // atomic: watchdog closed the upstream body

	go func() {
		ticker := time.NewTicker(sseKeepaliveInterval)
		defer ticker.Stop()
		lastData := time.Now()
		for {
			select {
			case <-kaDone:
				return
			case <-progress:
				lastData = time.Now()
			case <-ticker.C:
				if err := sw.writeComment("keepalive"); err != nil {
					// Client connection is gone. Unblock the relay loop by
					// closing the upstream body — its next Read errors out.
					atomic.StoreInt32(&clientGone, 1)
					upstream.Body.Close()
					return
				}
				if time.Since(lastData) >= streamStallTimeout {
					atomic.StoreInt32(&stallAborted, 1)
					upstream.Body.Close()
					return
				}
			}
		}
	}()
	defer close(kaDone)

	// Step 4: Read and transmux each SSE line
	var sseEventType string
	var sawToolCalls bool    // tracks if any translated chunk contained tool_calls
	var sawFinishReason bool // tracks if a non-null finish_reason was already sent
	toolSanitizer := newStreamToolSanitizer()
	for scanner.Scan() {
		signalProgressChan(progress)
		line := scanner.Bytes()

		// Skip empty lines (SSE delimiter)
		if len(line) == 0 {
			continue
		}

		// Handle SSE data lines
		if bytes.HasPrefix(line, []byte("data: ")) {
			data := line[6:] // Strip "data: " prefix

			// Check for stream termination
			if bytes.Equal(data, []byte("[DONE]")) {
				if anthropicEnc != nil {
					// Anthropic protocol termination: the encoder emits the
					// single message_delta (stop_reason + usage) followed by
					// message_stop — there is no [DONE] on this protocol.
					finalBytes := anthropicEnc.Finalize()
					if len(finalBytes) > 0 {
						_ = sw.writeRawBytes(finalBytes)
					}
					responseText = responseBuilder.String()
					completionTokens = anthropicEnc.OutputTokens()
					return
				}
				if toolSanitizer.HasPending() {
					_ = toolSanitizer.Flush(sw, requestedModel)
				}
				if sawToolCalls && !sawFinishReason {
					finishChunk := fmt.Sprintf(`{"id":"chatcmpl-gate","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
					_ = sw.writeDataEvent([]byte(finishChunk))
					sawFinishReason = true
				}
				_ = sw.writeDataEvent([]byte("[DONE]"))
				responseText = responseBuilder.String()
				completionTokens = tokenEstimate
				return
			}

			// Transmux the chunk to OpenAI format
			translated, err := tmx.TranslateChunk(data)
			if err != nil {
				sp.logger.Debug("transmux error, forwarding raw",
					zap.String("provider", provider),
					zap.Error(err),
				)
				translated = data
			}

			// Debug: log raw agentrouter SSE events to diagnose GPT tool call issues
			if provider == "agentrouter" && len(data) > 0 && sseEventType != "ping" {
				blockType, _, _, _ := jsonparser.Get(data, "content_block", "type")
				hasTC := bytes.Contains(translated, []byte(`"tool_calls"`))
				logFields := []zap.Field{
					zap.String("event", sseEventType),
					zap.ByteString("block_type", blockType),
					zap.Bool("has_tool_calls", hasTC),
					zap.Int("raw_len", len(data)),
					zap.Int("translated_len", len(translated)),
				}
				// For content_block_start and input_json_delta, log raw data
				if sseEventType == "content_block_start" || (hasTC && sseEventType == "content_block_delta") {
					truncData := data
					if len(truncData) > 500 {
						truncData = truncData[:500]
					}
					logFields = append(logFields, zap.ByteString("raw_data", truncData))
				}
				sp.logger.Info("agentrouter SSE event", logFields...)
			}

			// Track tool calls and finish_reason for synthetic finish injection
			hasToolCalls := bytes.Contains(translated, []byte(`"tool_calls"`))
			if hasToolCalls {
				sawToolCalls = true
			}
			if bytes.Contains(translated, []byte(`"finish_reason":"`)) {
				sawFinishReason = true
			}

			if len(translated) > 0 {
				if anthropicEnc != nil {
					// ── Anthropic front door (Phase 3+4) ──────────────────
					// Re-encode the OpenAI-format chunk into Anthropic SSE
					// events. The encoder owns block lifecycle (text,
					// thinking, tool_use), usage absorption and mid-stream
					// error detection — no OpenAI-client sanitization runs
					// on this path. A top-level "error" chunk surfaces as an
					// `event: error` frame and stops the message.
					if content, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "content"); err == nil {
						responseBuilder.WriteString(content)
					} else if reasoning, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "reasoning_content"); err == nil {
						responseBuilder.WriteString(reasoning)
					}
					anthBytes := anthropicEnc.EncodeOpenAIChunk(translated)
					if len(anthBytes) > 0 {
						if writeErr := sw.writeRawBytes(anthBytes); writeErr != nil {
							sp.logger.Debug("client disconnected during anthropic stream",
								zap.String("provider", provider),
								zap.Error(writeErr),
							)
							responseText = responseBuilder.String()
							completionTokens = anthropicEnc.OutputTokens()
							return
						}
					}
					if provider == "1minai" && sseEventType == "done" {
						finalBytes := anthropicEnc.Finalize()
						if len(finalBytes) > 0 {
							_ = sw.writeRawBytes(finalBytes)
						}
						responseText = responseBuilder.String()
						completionTokens = anthropicEnc.OutputTokens()
						return
					}
					continue
				}
				if requestedModel != "" {
					if _, err := jsonparser.GetString(translated, "model"); err == nil {
						if updated, err := jsonparser.Set(translated, []byte(`"`+requestedModel+`"`), "model"); err == nil {
							translated = updated
						}
					}
				}

				if hasToolCalls {
					absorbed, remainder := toolSanitizer.Accumulate(translated)
					if absorbed {
						continue
					}
					translated = remainder
				} else if toolSanitizer.HasPending() {
					// Non-tool chunk arrived (e.g. text or finish_reason) after tool calls:
					// flush buffered tool calls first!
					_ = toolSanitizer.Flush(sw, requestedModel)
				}

				// Sanitize chunk for OpenAI client compliance (removes invalid null fields in tool_calls, etc.)
				translated = sanitizeStreamingChunk(translated)

				// Capture both content and reasoning_content deltas so the
				// accumulated response text and token estimate include the
				// model's thinking process, not just the final answer. A
				// single OpenAI delta carries one or the other, never both.
				if content, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "content"); err == nil {
					responseBuilder.WriteString(content)
					tokenEstimate++
				} else if reasoning, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "reasoning_content"); err == nil {
					responseBuilder.WriteString(reasoning)
					tokenEstimate++
				}

				if writeErr := sw.writeDataEvent(translated); writeErr != nil {
					sp.logger.Debug("client disconnected during stream",
						zap.String("provider", provider),
						zap.Error(writeErr),
					)
					responseText = responseBuilder.String()
					completionTokens = tokenEstimate
					return
				}

				if provider == "1minai" && sseEventType == "done" {
					_ = sw.writeDataEvent([]byte("[DONE]"))
					responseText = responseBuilder.String()
					completionTokens = tokenEstimate
					return
				}
			}
			continue
		}

		// Handle non-data SSE events (some providers send event types)
		if bytes.HasPrefix(line, []byte("event: ")) {
			eventType := string(line[7:])
			sseEventType = eventType
			if provider == "anthropic" || provider == "1minai" || provider == "agentrouter" {
				tmx.SetEventType(eventType)
			}
			continue
		}

		// Handle provider-specific non-SSE streaming (e.g., Gemini JSON array)
		if provider == "gemini" && len(line) > 0 && (line[0] == '[' || line[0] == ',' || line[0] == '{') {
			text, tok := sp.handleGeminiStream(sw, tmx, line, scanner, progress, anthropicEnc)
			responseBuilder.WriteString(text)
			tokenEstimate += tok
			responseText = responseBuilder.String()
			if anthropicEnc != nil {
				completionTokens = anthropicEnc.OutputTokens()
			} else {
				completionTokens = tokenEstimate
			}
			return
		}

		// Handle Ollama native NDJSON streaming (/api/chat and /api/generate).
		if provider == "ollama" && transmux.IsOllamaNativeChunk(line) {
			content := sp.processOllamaChunk(sw, tmx, line, progress, anthropicEnc)
			if content != "" {
				responseBuilder.WriteString(content)
				tokenEstimate++
			}
			if isDone, err := jsonparser.GetBoolean(line, "done"); err == nil && isDone {
				if anthropicEnc != nil {
					finalBytes := anthropicEnc.Finalize()
					if len(finalBytes) > 0 {
						_ = sw.writeRawBytes(finalBytes)
					}
					responseText = responseBuilder.String()
					completionTokens = anthropicEnc.OutputTokens()
					return
				}
				_ = sw.writeDataEvent([]byte("[DONE]"))
				responseText = responseBuilder.String()
				completionTokens = tokenEstimate
				return
			}
			continue
		}

		// Forward any other lines as-is (comments, retry directives, etc.)
		if bytes.HasPrefix(line, []byte(":")) {
			continue
		}
	}

	// ── Upstream stream failure: terminate cleanly and visibly ──────────────
	// scanner.Err() != nil means the upstream connection broke mid-stream
	// (unexpected EOF, connection reset, watchdog abort, …). The client
	// already holds partial content, so a retry is impossible — but the old
	// behavior of masking the truncation as a successful [DONE] left clients
	// with silently truncated answers and left nothing in production logs.
	if scanErr := scanner.Err(); scanErr != nil {
		elapsed := time.Since(streamStart)
		if anthropicEnc != nil {
			// Anthropic protocol termination: an `event: error` frame with the
			// failure cause (message_stop is deliberately omitted — Anthropic
			// streams terminate on the error event).
			switch {
			case atomic.LoadInt32(&clientGone) == 1:
				// Nothing we write can be delivered.
			case atomic.LoadInt32(&stallAborted) == 1:
				sp.logger.Warn("upstream stream stalled — terminating anthropic stream with error event",
					zap.String("provider", provider),
					zap.String("model", requestedModel),
					zap.Duration("silent_for", streamStallTimeout),
					zap.Duration("elapsed", elapsed),
				)
				errBytes := anthropicEnc.EncodeStreamError("api_error",
					"upstream produced no data for "+streamStallTimeout.String()+"; stream terminated")
				if len(errBytes) > 0 {
					_ = sw.writeRawBytes(errBytes)
				}
			default:
				sp.logger.Warn("upstream stream interrupted mid-response — terminating anthropic stream with error event",
					zap.String("provider", provider),
					zap.String("model", requestedModel),
					zap.Duration("elapsed", elapsed),
					zap.Error(scanErr),
				)
				errBytes := anthropicEnc.EncodeStreamError("api_error",
					"upstream connection interrupted before the response completed ("+sanitizeSSEJSONString(scanErr.Error(), 200)+")")
				if len(errBytes) > 0 {
					_ = sw.writeRawBytes(errBytes)
				}
			}
			responseText = responseBuilder.String()
			completionTokens = anthropicEnc.OutputTokens()
			return
		}
		switch {
		case atomic.LoadInt32(&clientGone) == 1:
			// Client (or an intermediary in front of it) already tore the
			// connection down — nothing we write can be delivered.
			sp.logger.Warn("client disconnected during upstream stream — aborting relay",
				zap.String("provider", provider),
				zap.String("model", requestedModel),
				zap.Duration("elapsed", elapsed),
				zap.Int("estimated_tokens", tokenEstimate),
				zap.Error(scanErr),
			)
		case atomic.LoadInt32(&stallAborted) == 1:
			sp.logger.Warn("upstream stream stalled — terminating client stream with error event",
				zap.String("provider", provider),
				zap.String("model", requestedModel),
				zap.Duration("silent_for", streamStallTimeout),
				zap.Duration("elapsed", elapsed),
				zap.Int("estimated_tokens", tokenEstimate),
			)
			emitStreamInterrupted(sw, "upstream produced no data for "+streamStallTimeout.String())
		default:
			sp.logger.Warn("upstream stream interrupted mid-response — terminating client stream with error event",
				zap.String("provider", provider),
				zap.String("model", requestedModel),
				zap.Duration("elapsed", elapsed),
				zap.Int("estimated_tokens", tokenEstimate),
				zap.Error(scanErr),
			)
			emitStreamInterrupted(sw, scanErr.Error())
		}
		responseText = responseBuilder.String()
		completionTokens = tokenEstimate
		return
	}

	// Gap 5 Fix: For 1min.ai, emit a synthetic stop chunk if the upstream
	// connection dropped before the "done" event was transmitted. Without this,
	// downstream clients hang waiting for the terminal finish_reason marker.
	// If the "done" event was already processed, a duplicate stop chunk is
	// harmless — OpenAI clients handle multiple finish_reason chunks gracefully.
	if provider == "1minai" && anthropicEnc == nil {
		tmx.SetEventType("done")
		stopChunk, _ := tmx.TranslateChunk([]byte(`{}`))
		if len(stopChunk) > 0 {
			_ = sw.writeDataEvent(stopChunk)
		}
	}

	// ── Stream termination ────────────────────────────────────────────────
	if anthropicEnc != nil {
		// Anthropic front door: the encoder emits the terminal sequence
		// (content_block_stop → message_delta with stop_reason + usage →
		// message_stop). Upstreams that closed without a finish_reason get
		// the default end_turn.
		finalBytes := anthropicEnc.Finalize()
		if len(finalBytes) > 0 {
			_ = sw.writeRawBytes(finalBytes)
		}
		responseText = responseBuilder.String()
		completionTokens = anthropicEnc.OutputTokens()
		return
	}

	if toolSanitizer.HasPending() {
		_ = toolSanitizer.Flush(sw, requestedModel)
	}

	// If tool calls were present but upstream never sent a finish_reason, inject "tool_calls".
	// Otherwise for agentrouter/anthropic, inject "stop" if finish_reason was omitted.
	if sawToolCalls && !sawFinishReason {
		finishChunk := fmt.Sprintf(`{"id":"chatcmpl-gate","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"tool_calls"}]}`)
		sp.logger.Info("injecting synthetic finish_reason",
			zap.String("provider", provider),
			zap.String("reason", "tool_calls"),
			zap.Bool("saw_tool_calls", sawToolCalls),
		)
		_ = sw.writeDataEvent([]byte(finishChunk))
		sawFinishReason = true
	} else if (provider == "agentrouter" || provider == "anthropic") && !sawFinishReason {
		finishChunk := fmt.Sprintf(`{"id":"chatcmpl-gate","object":"chat.completion.chunk","choices":[{"index":0,"delta":{},"finish_reason":"stop"}]}`)
		sp.logger.Info("injecting synthetic finish_reason",
			zap.String("provider", provider),
			zap.String("reason", "stop"),
			zap.Bool("saw_tool_calls", sawToolCalls),
		)
		_ = sw.writeDataEvent([]byte(finishChunk))
	}

	// Ensure [DONE] is sent even if upstream didn't send it
	_ = sw.writeDataEvent([]byte("[DONE]"))

	responseText = responseBuilder.String()
	completionTokens = tokenEstimate
	return
}

// handleGeminiStream processes Gemini's non-SSE JSON streaming format.
func (sp *StreamProxy) handleGeminiStream(sw *sseWriter, tmx transmux.Transmuxer, firstLine []byte, scanner *bufio.Scanner, progress chan struct{}, anthropicEnc *transmux.AnthropicStreamEncoder) (string, int) {
	var sb strings.Builder
	var tokens int

	// Process the first line
	if val := sp.processGeminiChunk(sw, tmx, firstLine, progress, anthropicEnc); val != "" {
		sb.WriteString(val)
		tokens++
	}

	// Continue reading
	for scanner.Scan() {
		signalProgressChan(progress)
		line := scanner.Bytes()
		if len(line) == 0 || bytes.Equal(line, []byte("]")) {
			continue
		}
		if val := sp.processGeminiChunk(sw, tmx, line, progress, anthropicEnc); val != "" {
			sb.WriteString(val)
			tokens++
		}
	}

	if anthropicEnc != nil {
		finalBytes := anthropicEnc.Finalize()
		if len(finalBytes) > 0 {
			_ = sw.writeRawBytes(finalBytes)
		}
		return sb.String(), anthropicEnc.OutputTokens()
	}

	_ = sw.writeDataEvent([]byte("[DONE]"))

	return sb.String(), tokens
}

// processOllamaChunk translates a single Ollama native NDJSON line into an
// OpenAI-compatible SSE chunk and flushes it to the client.
// ProxyStreamRaw relays an Anthropic-protocol SSE stream to the client
// byte-for-byte with zero translation — the zero-copy path used when an
// Anthropic front-door request is served by a native Anthropic upstream.
//
// It keeps the same liveness machinery as ProxyStream (SSE keepalive comments
// while the upstream is silent, stall watchdog, client-gone detection) so the
// incident protections documented on ProxyStream apply identically. The final
// error, if any, is emitted as an Anthropic-protocol error event.
//
// Returns the number of completion tokens advertised by the upstream's usage
// events (0 when the upstream never reported usage).
func (sp *StreamProxy) ProxyStreamRaw(c *gin.Context, upstream *http.Response) (responseText string, completionTokens int) {
	scanBuf := sp.scannerPool.Get().([]byte)
	var sw *sseWriter

	defer func() {
		// Always return the scanner buffer to the pool
		sp.scannerPool.Put(scanBuf[:0])

		// Always close the upstream response body
		upstream.Body.Close()

		// Catch any panic from the relay path
		if r := recover(); r != nil {
			sp.logger.Error("recovered from raw stream relay panic",
				zap.Any("panic", r),
				zap.ByteString("stack", debug.Stack()),
			)
			// Anthropic-protocol termination: emit an error event if possible.
			if sw != nil {
				_ = sw.writeEvent("error", []byte(`{"type":"error","error":{"type":"api_error","message":"stream relay failed internally"}}`))
			}
		}
	}()

	// Step 1: Set SSE headers for streaming
	c.Writer.Header().Set("Content-Type", "text/event-stream")
	c.Writer.Header().Set("Cache-Control", "no-cache")
	c.Writer.Header().Set("Connection", "keep-alive")
	c.Writer.Header().Set("Transfer-Encoding", "chunked")
	c.Writer.Header().Set("X-Accel-Buffering", "no") // Disable nginx buffering
	c.Writer.WriteHeader(http.StatusOK)

	if _, ok := c.Writer.(http.Flusher); !ok {
		sp.logger.Error("response writer does not support flushing")
		return "", 0
	}
	sw = &sseWriter{w: c.Writer}

	// Step 2: keepalive heartbeat + stall watchdog (same rationale as
	// ProxyStream: long thinking gaps must not kill idle connections).
	kaDone := make(chan struct{})
	progress := make(chan struct{}, 1)
	var clientGone int32
	var stallAborted int32

	go func() {
		ticker := time.NewTicker(sseKeepaliveInterval)
		defer ticker.Stop()
		lastData := time.Now()
		for {
			select {
			case <-kaDone:
				return
			case <-progress:
				lastData = time.Now()
			case <-ticker.C:
				if err := sw.writeComment("keepalive"); err != nil {
					// Client connection is gone. Unblock the relay loop by
					// closing the upstream body — its next Read errors out.
					atomic.StoreInt32(&clientGone, 1)
					upstream.Body.Close()
					return
				}
				if time.Since(lastData) >= streamStallTimeout {
					atomic.StoreInt32(&stallAborted, 1)
					upstream.Body.Close()
					return
				}
			}
		}
	}()
	defer close(kaDone)

	// Step 3: line-oriented verbatim relay. Reading line-wise keeps the
	// keepalive progress signaling per SSE event (as in ProxyStream) while
	// every byte between newlines is forwarded unmodified.
	var responseBuilder strings.Builder
	scanner := bufio.NewScanner(upstream.Body)
	scanner.Buffer(scanBuf, 1024*1024)
	for scanner.Scan() {
		signalProgressChan(progress)
		line := scanner.Bytes() // valid only until the next Scan

		// Track Anthropic usage events for token accounting and text for telemetry
		if len(line) > 6 && bytes.HasPrefix(line, []byte("data: ")) {
			payload := line[6:]
			if t, err := jsonparser.GetString(payload, "delta", "text"); err == nil && t != "" {
				responseBuilder.WriteString(t)
			} else if th, err := jsonparser.GetString(payload, "delta", "thinking"); err == nil && th != "" {
				responseBuilder.WriteString(th)
			}
			if tok, err := jsonparser.GetInt(payload, "usage", "output_tokens"); err == nil && tok > 0 {
				completionTokens = int(tok)
			}
		}

		if err := sw.writeRawLine(line); err != nil {
			// Client went away or the connection broke — stop relaying.
			return responseBuilder.String(), completionTokens
		}
	}

	// Step 4: error termination. A stalled upstream (or scanner error) gets a
	// clean Anthropic-protocol error event so the client sees an explicit
	// failure instead of a silent truncation.
	if atomic.LoadInt32(&stallAborted) == 1 {
		payload := []byte(`{"type":"error","error":{"type":"api_error","message":"upstream stream stalled; relay terminated"}}`)
		_ = sw.writeEvent("error", payload)
	} else if err := scanner.Err(); err != nil && atomic.LoadInt32(&clientGone) == 0 {
		payload := []byte(`{"type":"error","error":{"type":"api_error","message":"upstream stream interrupted before completion"}}`)
		_ = sw.writeEvent("error", payload)
	}

	return responseBuilder.String(), completionTokens
}

// processOllamaChunk parses one Ollama NDJSON streaming chunk and emits the
// equivalent OpenAI chat.completion.chunk SSE event or Anthropic SSE event.
func (sp *StreamProxy) processOllamaChunk(sw *sseWriter, tmx transmux.Transmuxer, chunk []byte, progress chan struct{}, anthropicEnc *transmux.AnthropicStreamEncoder) string {
	translated, err := tmx.TranslateChunk(chunk)
	if err != nil {
		sp.logger.Debug("ollama chunk transmux error", zap.Error(err))
		return ""
	}

	if len(translated) > 0 {
		signalProgressChan(progress)
		if anthropicEnc != nil {
			anthBytes := anthropicEnc.EncodeOpenAIChunk(translated)
			if len(anthBytes) > 0 {
				_ = sw.writeRawBytes(anthBytes)
			}
		} else {
			translated = sanitizeStreamingChunk(translated)
			if writeErr := sw.writeDataEvent(translated); writeErr != nil {
				return ""
			}
		}

		if content, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "content"); err == nil {
			return content
		} else if reasoning, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "reasoning_content"); err == nil {
			return reasoning
		}
	}
	return ""
}

// processGeminiChunk translates a single Gemini JSON chunk into OpenAI SSE format or Anthropic SSE event.
func (sp *StreamProxy) processGeminiChunk(sw *sseWriter, tmx transmux.Transmuxer, chunk []byte, progress chan struct{}, anthropicEnc *transmux.AnthropicStreamEncoder) string {
	chunk = bytes.TrimLeft(chunk, "[,")
	chunk = bytes.TrimRight(chunk, "]")
	chunk = bytes.TrimSpace(chunk)

	if len(chunk) == 0 {
		return ""
	}

	translated, err := tmx.TranslateChunk(chunk)
	if err != nil {
		return ""
	}

	if len(translated) > 0 {
		signalProgressChan(progress)
		if anthropicEnc != nil {
			anthBytes := anthropicEnc.EncodeOpenAIChunk(translated)
			if len(anthBytes) > 0 {
				_ = sw.writeRawBytes(anthBytes)
			}
		} else {
			translated = sanitizeStreamingChunk(translated)
			if writeErr := sw.writeDataEvent(translated); writeErr != nil {
				return ""
			}
		}

		if content, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "content"); err == nil {
			return content
		} else if reasoning, err := jsonparser.GetString(translated, "choices", "[0]", "delta", "reasoning_content"); err == nil {
			return reasoning
		}
	}
	return ""
}

// ExtractStreamFlag checks the body for the stream flag without full unmarshalling.
// Uses strings.Contains for minimal overhead.
func ExtractStreamFlag(body []byte) bool {
	return strings.Contains(string(body), `"stream":true`) ||
		strings.Contains(string(body), `"stream": true`)
}

// StreamBodyReader wraps an io.ReadCloser to tee into a buffer for retry scenarios.
type StreamBodyReader struct {
	io.ReadCloser
	buf *bytes.Buffer
}

func NewStreamBodyReader(body io.ReadCloser) *StreamBodyReader {
	return &StreamBodyReader{
		ReadCloser: body,
		buf:        &bytes.Buffer{},
	}
}

func (r *StreamBodyReader) Read(p []byte) (int, error) {
	n, err := r.ReadCloser.Read(p)
	if n > 0 {
		r.buf.Write(p[:n])
	}
	return n, err
}

// sanitizeStreamingChunk normalizes and cleans up tool_calls deltas in SSE chunks.
// Upstream providers (such as Cloudflare Workers AI) serialize omitted/optional delta
// fields as JSON `null` instead of omitting them (e.g. `"id": null`, `"function": {"name": null, "arguments": "..."}`).
// Strict OpenAI client validators (like Zod in Cline, Cursor, Vercel AI SDK) throw:
// "Expected 'function.name' to be a string." when `name` is null.
// This function strips invalid null fields from tool_calls deltas so the chunk strictly conforms
// to the OpenAI streaming specification.
func sanitizeStreamingChunk(chunk []byte) []byte {
	if len(chunk) == 0 {
		return chunk
	}
	// Fast path: if chunk does not contain tool_calls, skip JSON parsing entirely.
	if !bytes.Contains(chunk, []byte(`"tool_calls"`)) {
		return chunk
	}
	// Fast path: if chunk does not contain any null value, skip JSON parsing.
	if !bytes.Contains(chunk, []byte(`:null`)) && !bytes.Contains(chunk, []byte(`: null`)) {
		return chunk
	}

	var payload map[string]interface{}
	if err := json.Unmarshal(chunk, &payload); err != nil {
		return chunk
	}

	choices, ok := payload["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return chunk
	}

	modified := false
	for _, choice := range choices {
		chMap, ok := choice.(map[string]interface{})
		if !ok {
			continue
		}

		delta, ok := chMap["delta"].(map[string]interface{})
		if !ok {
			continue
		}

		// Clean null fields from delta root that upstreams emit alongside tool_calls
		if val, exists := delta["role"]; exists && val == nil {
			delete(delta, "role")
			modified = true
		}
		if val, exists := delta["content"]; exists && val == nil {
			delete(delta, "content")
			modified = true
		}
		if val, exists := delta["reasoning_content"]; exists && val == nil {
			delete(delta, "reasoning_content")
			modified = true
		}

		tcs, ok := delta["tool_calls"].([]interface{})
		if !ok {
			continue
		}

		for _, tc := range tcs {
			tcMap, ok := tc.(map[string]interface{})
			if !ok {
				continue
			}

			// Clean tool_call id if null
			if val, exists := tcMap["id"]; exists && val == nil {
				delete(tcMap, "id")
				modified = true
			}

			// Clean tool_call type if null
			if val, exists := tcMap["type"]; exists && val == nil {
				delete(tcMap, "type")
				modified = true
			}

			// Clean function object
			if fn, ok := tcMap["function"].(map[string]interface{}); ok {
				// If function name is null, set to empty string so clients see string
				if val, exists := fn["name"]; exists && val == nil {
					fn["name"] = ""
					modified = true
				}
				// If arguments is null, set to empty string
				if val, exists := fn["arguments"]; exists && val == nil {
					fn["arguments"] = ""
					modified = true
				}
			}
		}
	}

	if !modified {
		return chunk
	}

	cleaned, err := json.Marshal(payload)
	if err != nil {
		return chunk
	}
	return cleaned
}

// streamToolCallInfo holds buffered streaming tool call parts for a single tool index.
type streamToolCallInfo struct {
	Index int
	ID    string
	Name  string
	Args  strings.Builder
}

// streamToolSanitizer buffers, repairs, and emits compliant tool call chunks
// across streaming deltas. This solves both:
// 1. Zod "Expected 'function.name' to be a string" (ensures name is non-null on every chunk)
// 2. Client JSON parsing errors (removes markdown fences, unescapes newlines, normalizes read_files arguments).
type streamToolSanitizer struct {
	tools   map[int]*streamToolCallInfo
	id      string
	model   string
	created int64
}

func newStreamToolSanitizer() *streamToolSanitizer {
	return &streamToolSanitizer{
		tools: make(map[int]*streamToolCallInfo),
	}
}

func (ts *streamToolSanitizer) HasPending() bool {
	return len(ts.tools) > 0
}

func (ts *streamToolSanitizer) Accumulate(chunk []byte) (bool, []byte) {
	var payload map[string]interface{}
	if err := json.Unmarshal(chunk, &payload); err != nil {
		return false, chunk
	}

	if idStr, ok := payload["id"].(string); ok && idStr != "" {
		ts.id = idStr
	}
	if modelStr, ok := payload["model"].(string); ok && modelStr != "" {
		ts.model = modelStr
	}
	if createdNum, ok := payload["created"].(float64); ok && createdNum != 0 {
		ts.created = int64(createdNum)
	}

	choices, ok := payload["choices"].([]interface{})
	if !ok || len(choices) == 0 {
		return false, chunk
	}

	ch0, ok := choices[0].(map[string]interface{})
	if !ok {
		return false, chunk
	}

	delta, ok := ch0["delta"].(map[string]interface{})
	if !ok {
		return false, chunk
	}

	tcs, ok := delta["tool_calls"].([]interface{})
	if !ok || len(tcs) == 0 {
		return false, chunk
	}

	for _, tc := range tcs {
		tcMap, ok := tc.(map[string]interface{})
		if !ok {
			continue
		}
		idx := 0
		if idxVal, ok := tcMap["index"].(float64); ok {
			idx = int(idxVal)
		}
		tool, exists := ts.tools[idx]
		if !exists {
			tool = &streamToolCallInfo{Index: idx}
			ts.tools[idx] = tool
		}

		if idStr, ok := tcMap["id"].(string); ok && idStr != "" {
			tool.ID = idStr
		}
		if fn, ok := tcMap["function"].(map[string]interface{}); ok {
			if nameStr, ok := fn["name"].(string); ok && nameStr != "" {
				tool.Name = nameStr
			}
			if argsStr, ok := fn["arguments"].(string); ok && argsStr != "" {
				tool.Args.WriteString(argsStr)
			}
		}
	}

	// Check if this chunk also carried non-empty text content
	hasContent := false
	if content, ok := delta["content"].(string); ok && content != "" {
		hasContent = true
	}
	if reasoning, ok := delta["reasoning_content"].(string); ok && reasoning != "" {
		hasContent = true
	}

	if !hasContent {
		return true, nil
	}

	// Chunk had both text and tool_calls: strip tool_calls and forward text
	delete(delta, "tool_calls")
	remainder, err := json.Marshal(payload)
	if err != nil {
		return false, chunk
	}
	return false, remainder
}

func (ts *streamToolSanitizer) Flush(sw *sseWriter, requestedModel string) error {
	if len(ts.tools) == 0 {
		return nil
	}
	tools := ts.tools
	ts.tools = make(map[int]*streamToolCallInfo)

	var indices []int
	for idx := range tools {
		indices = append(indices, idx)
	}
	sort.Ints(indices)

	var tcs []map[string]interface{}
	for _, idx := range indices {
		tool := tools[idx]
		if tool.ID == "" {
			tool.ID = fmt.Sprintf("call_%x_%d", time.Now().UnixNano(), tool.Index)
		}
		name := tool.Name
		repairedArgs := RepairToolArguments(name, tool.Args.String())
		tcs = append(tcs, map[string]interface{}{
			"index": tool.Index,
			"id":    tool.ID,
			"type":  "function",
			"function": map[string]interface{}{
				"name":      name,
				"arguments": repairedArgs,
			},
		})
	}

	if len(tcs) == 0 {
		return nil
	}

	model := ts.model
	if requestedModel != "" {
		model = requestedModel
	}
	if model == "" {
		model = "gateway"
	}
	id := ts.id
	if id == "" {
		id = fmt.Sprintf("chatcmpl-gate-%x", time.Now().UnixNano())
	}
	created := ts.created
	if created == 0 {
		created = time.Now().Unix()
	}

	chunkMap := map[string]interface{}{
		"id":      id,
		"object":  "chat.completion.chunk",
		"created": created,
		"model":   model,
		"choices": []map[string]interface{}{
			{
				"index": 0,
				"delta": map[string]interface{}{
					"role":       "assistant",
					"tool_calls": tcs,
				},
				"finish_reason": nil,
			},
		},
	}

	chunkBytes, err := json.Marshal(chunkMap)
	if err != nil {
		return err
	}
	return sw.writeDataEvent(chunkBytes)
}

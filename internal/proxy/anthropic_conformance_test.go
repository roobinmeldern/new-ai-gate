package proxy

import (
	"bufio"
	"bytes"
	"encoding/json"
	"errors"
	"fmt"
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"testing"
	"time"

	"github.com/buger/jsonparser"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
)

// ─────────────────────────────────────────────────────────────────────────────
// AnthropicSDKStreamAccumulator: Deterministic Wire-Level SSE Accumulator
//
// Strictly mirrors official github.com/anthropics/anthropic-sdk-go
// `Message.Accumulate(event)` validation logic so that conformance tests
// run offline with zero external network dependencies.
// ─────────────────────────────────────────────────────────────────────────────

type ConformanceBlock struct {
	Type        string         `json:"type"`
	Text        string         `json:"text,omitempty"`
	Thinking    string         `json:"thinking,omitempty"`
	Signature   string         `json:"signature,omitempty"`
	ID          string         `json:"id,omitempty"`
	Name        string         `json:"name,omitempty"`
	Input       map[string]any `json:"input,omitempty"`
	PartialJSON string         `json:"-"`
}

type ConformanceUsage struct {
	InputTokens              int `json:"input_tokens"`
	OutputTokens             int `json:"output_tokens"`
	CacheCreationInputTokens int `json:"cache_creation_input_tokens,omitempty"`
	CacheReadInputTokens     int `json:"cache_read_input_tokens,omitempty"`
}

type ConformanceMessage struct {
	ID           string             `json:"id"`
	Type         string             `json:"type"`
	Role         string             `json:"role"`
	Model        string             `json:"model"`
	Content      []ConformanceBlock `json:"content"`
	StopReason   string             `json:"stop_reason"`
	StopSequence *string            `json:"stop_sequence"`
	Usage        ConformanceUsage   `json:"usage"`
}

type AnthropicStreamError struct {
	Type    string `json:"type"`
	Message string `json:"message"`
}

func (e *AnthropicStreamError) Error() string {
	return fmt.Sprintf("anthropic stream error (%s): %s", e.Type, e.Message)
}

type AnthropicSDKStreamAccumulator struct {
	Message        *ConformanceMessage
	activeBlockIdx int
	started        bool
	stopped        bool
	StreamError    *AnthropicStreamError
	EventsReceived []string
}

// NewAnthropicSDKStreamAccumulator creates a new accumulator ready to parse an Anthropic SSE stream.
func NewAnthropicSDKStreamAccumulator() *AnthropicSDKStreamAccumulator {
	return &AnthropicSDKStreamAccumulator{
		activeBlockIdx: -1,
	}
}

// Accumulate processes a single SSE event and validates Anthropic protocol invariants.
func (a *AnthropicSDKStreamAccumulator) Accumulate(eventType string, data []byte) error {
	a.EventsReceived = append(a.EventsReceived, eventType)

	if a.stopped {
		return fmt.Errorf("wire-level violation: received event %q after stream had already stopped", eventType)
	}

	// Mid-stream error frame
	if eventType == "error" {
		errType, _ := jsonparser.GetString(data, "error", "type")
		errMsg, _ := jsonparser.GetString(data, "error", "message")
		if errMsg == "" {
			errMsg = string(data)
		}
		if errType == "" {
			errType = "api_error"
		}
		a.StreamError = &AnthropicStreamError{Type: errType, Message: errMsg}
		a.stopped = true
		return a.StreamError
	}

	// Invariant 1: First event MUST be message_start
	if !a.started {
		if eventType != "message_start" {
			return fmt.Errorf("wire-level violation: first event must be 'message_start', got %q", eventType)
		}
	}

	switch eventType {
	case "message_start":
		if a.started {
			return errors.New("wire-level violation: duplicate message_start event")
		}
		a.started = true
		a.activeBlockIdx = -1

		id, _ := jsonparser.GetString(data, "message", "id")
		mType, _ := jsonparser.GetString(data, "message", "type")
		role, _ := jsonparser.GetString(data, "message", "role")
		model, _ := jsonparser.GetString(data, "message", "model")
		inTok, _ := jsonparser.GetInt(data, "message", "usage", "input_tokens")
		cCreate, _ := jsonparser.GetInt(data, "message", "usage", "cache_creation_input_tokens")
		cRead, _ := jsonparser.GetInt(data, "message", "usage", "cache_read_input_tokens")

		a.Message = &ConformanceMessage{
			ID:      id,
			Type:    mType,
			Role:    role,
			Model:   model,
			Content: make([]ConformanceBlock, 0),
			Usage: ConformanceUsage{
				InputTokens:              int(inTok),
				CacheCreationInputTokens: int(cCreate),
				CacheReadInputTokens:     int(cRead),
			},
		}

	case "content_block_start":
		idx64, err := jsonparser.GetInt(data, "index")
		if err != nil {
			return fmt.Errorf("content_block_start missing index: %w", err)
		}
		idx := int(idx64)

		// Invariant 2: Sequential index continuity without gaps
		if idx != len(a.Message.Content) {
			return fmt.Errorf("wire-level violation: content_block_start skipped index: expected %d, got %d", len(a.Message.Content), idx)
		}
		if a.activeBlockIdx != -1 {
			return fmt.Errorf("wire-level violation: content_block_start for index %d received while block %d is still open", idx, a.activeBlockIdx)
		}

		cbType, _ := jsonparser.GetString(data, "content_block", "type")
		cbID, _ := jsonparser.GetString(data, "content_block", "id")
		cbName, _ := jsonparser.GetString(data, "content_block", "name")
		cbText, _ := jsonparser.GetString(data, "content_block", "text")
		cbThinking, _ := jsonparser.GetString(data, "content_block", "thinking")

		block := ConformanceBlock{
			Type:     cbType,
			ID:       cbID,
			Name:     cbName,
			Text:     cbText,
			Thinking: cbThinking,
		}
		a.Message.Content = append(a.Message.Content, block)
		a.activeBlockIdx = idx

	case "content_block_delta":
		idx64, err := jsonparser.GetInt(data, "index")
		if err != nil {
			return fmt.Errorf("content_block_delta missing index: %w", err)
		}
		idx := int(idx64)

		// Invariant 3: Delta must reference currently open block index
		if idx != a.activeBlockIdx {
			return fmt.Errorf("wire-level violation: content_block_delta for index %d, but active block is %d", idx, a.activeBlockIdx)
		}
		if idx < 0 || idx >= len(a.Message.Content) {
			return fmt.Errorf("wire-level violation: content_block_delta index %d out of bounds", idx)
		}

		deltaType, _ := jsonparser.GetString(data, "delta", "type")
		block := &a.Message.Content[idx]
		switch deltaType {
		case "text_delta":
			txt, _ := jsonparser.GetString(data, "delta", "text")
			block.Text += txt
		case "thinking_delta":
			th, _ := jsonparser.GetString(data, "delta", "thinking")
			block.Thinking += th
		case "signature_delta":
			sig, _ := jsonparser.GetString(data, "delta", "signature")
			block.Signature = sig
		case "input_json_delta":
			pjson, _ := jsonparser.GetString(data, "delta", "partial_json")
			block.PartialJSON += pjson
		default:
			return fmt.Errorf("wire-level violation: unknown delta type %q", deltaType)
		}

	case "content_block_stop":
		idx64, err := jsonparser.GetInt(data, "index")
		if err != nil {
			return fmt.Errorf("content_block_stop missing index: %w", err)
		}
		idx := int(idx64)

		if idx != a.activeBlockIdx {
			return fmt.Errorf("wire-level violation: content_block_stop for index %d, but active block is %d", idx, a.activeBlockIdx)
		}

		block := &a.Message.Content[idx]
		// Invariant 4: Tool call JSON arguments must accumulate into valid JSON
		if block.Type == "tool_use" {
			rawArgs := strings.TrimSpace(block.PartialJSON)
			if rawArgs == "" || rawArgs == "{}" {
				block.Input = make(map[string]any)
			} else {
				var argsMap map[string]any
				if err := json.Unmarshal([]byte(rawArgs), &argsMap); err != nil {
					return fmt.Errorf("wire-level violation: tool_use block %d partial_json is not valid JSON: %w (raw: %s)", idx, err, rawArgs)
				}
				block.Input = argsMap
			}
		} else if block.Type == "thinking" {
			// Invariant 5: Thinking block must populate signature before stop
			if block.Signature == "" {
				return fmt.Errorf("wire-level violation: thinking block %d closed without signature_delta", idx)
			}
		}

		a.activeBlockIdx = -1

	case "message_delta":
		if a.activeBlockIdx != -1 {
			return fmt.Errorf("wire-level violation: message_delta received while block %d is still open", a.activeBlockIdx)
		}
		stopReason, _ := jsonparser.GetString(data, "delta", "stop_reason")
		validReasons := map[string]bool{
			"end_turn":      true,
			"max_tokens":    true,
			"tool_use":      true,
			"stop_sequence": true,
			"refusal":       true,
		}
		// Invariant 6: Valid stop_reason required
		if !validReasons[stopReason] {
			return fmt.Errorf("wire-level violation: invalid stop_reason %q in message_delta", stopReason)
		}
		a.Message.StopReason = stopReason
		if outTok, err := jsonparser.GetInt(data, "usage", "output_tokens"); err == nil {
			a.Message.Usage.OutputTokens = int(outTok)
		}
		if cCreate, err := jsonparser.GetInt(data, "usage", "cache_creation_input_tokens"); err == nil {
			a.Message.Usage.CacheCreationInputTokens = int(cCreate)
		}
		if cRead, err := jsonparser.GetInt(data, "usage", "cache_read_input_tokens"); err == nil {
			a.Message.Usage.CacheReadInputTokens = int(cRead)
		}

	case "message_stop":
		if a.activeBlockIdx != -1 {
			return fmt.Errorf("wire-level violation: message_stop received while block %d is still open", a.activeBlockIdx)
		}
		a.stopped = true

	case "ping":
		// Ping events are keepalive frames allowed by Anthropic SSE specification
	default:
		return fmt.Errorf("wire-level violation: unexpected Anthropic SSE event type %q", eventType)
	}

	return nil
}

// AccumulateStream consumes an SSE stream from an io.Reader and builds a complete ConformanceMessage.
func (a *AnthropicSDKStreamAccumulator) AccumulateStream(r io.Reader) (*ConformanceMessage, error) {
	scanner := bufio.NewScanner(r)
	var currentEvent string
	var currentData bytes.Buffer

	flushEvent := func() error {
		if currentEvent != "" {
			err := a.Accumulate(currentEvent, currentData.Bytes())
			currentEvent = ""
			currentData.Reset()
			if err != nil {
				return err
			}
		}
		return nil
	}

	for scanner.Scan() {
		line := scanner.Text()
		if strings.HasPrefix(line, "event: ") {
			if err := flushEvent(); err != nil {
				return nil, err
			}
			currentEvent = strings.TrimPrefix(line, "event: ")
		} else if strings.HasPrefix(line, "data: ") {
			dataPayload := strings.TrimPrefix(line, "data: ")
			if currentData.Len() > 0 {
				currentData.WriteByte('\n')
			}
			currentData.WriteString(dataPayload)
		} else if strings.TrimSpace(line) == "" {
			if err := flushEvent(); err != nil {
				return nil, err
			}
		}
	}
	if err := scanner.Err(); err != nil {
		return nil, err
	}
	if err := flushEvent(); err != nil {
		return nil, err
	}

	if a.StreamError != nil {
		return nil, a.StreamError
	}
	if !a.stopped {
		return nil, errors.New("wire-level violation: stream ended prematurely without message_stop")
	}

	return a.Message, nil
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 1: Conformance Unit Tests for AnthropicSDKStreamAccumulator
// ─────────────────────────────────────────────────────────────────────────────

func TestConformance_Accumulator_StrictValidationRules(t *testing.T) {
	t.Run("RejectsMissingMessageStart", func(t *testing.T) {
		acc := NewAnthropicSDKStreamAccumulator()
		raw := "event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
		_, err := acc.AccumulateStream(strings.NewReader(raw))
		if err == nil || !strings.Contains(err.Error(), "first event must be 'message_start'") {
			t.Fatalf("expected first event violation error, got: %v", err)
		}
	})

	t.Run("RejectsIndexSkip", func(t *testing.T) {
		acc := NewAnthropicSDKStreamAccumulator()
		raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":1,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n"
		_, err := acc.AccumulateStream(strings.NewReader(raw))
		if err == nil || !strings.Contains(err.Error(), "skipped index") {
			t.Fatalf("expected skipped index error, got: %v", err)
		}
	})

	t.Run("RejectsInvalidToolJSONArguments", func(t *testing.T) {
		acc := NewAnthropicSDKStreamAccumulator()
		raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"tool_use\",\"id\":\"tool_1\",\"name\":\"calc\",\"input\":{}}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"input_json_delta\",\"partial_json\":\"{broken_json\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
		_, err := acc.AccumulateStream(strings.NewReader(raw))
		if err == nil || !strings.Contains(err.Error(), "not valid JSON") {
			t.Fatalf("expected invalid JSON error, got: %v", err)
		}
	})

	t.Run("RejectsThinkingBlockWithoutSignature", func(t *testing.T) {
		acc := NewAnthropicSDKStreamAccumulator()
		raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"thinking\",\"thinking\":\"\"}}\n\n" +
			"event: content_block_delta\ndata: {\"type\":\"content_block_delta\",\"index\":0,\"delta\":{\"type\":\"thinking_delta\",\"thinking\":\"pondering\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n"
		_, err := acc.AccumulateStream(strings.NewReader(raw))
		if err == nil || !strings.Contains(err.Error(), "closed without signature_delta") {
			t.Fatalf("expected missing signature error, got: %v", err)
		}
	})

	t.Run("RejectsInvalidStopReason", func(t *testing.T) {
		acc := NewAnthropicSDKStreamAccumulator()
		raw := "event: message_start\ndata: {\"type\":\"message_start\",\"message\":{\"id\":\"msg_1\",\"type\":\"message\",\"role\":\"assistant\",\"model\":\"claude\",\"content\":[],\"usage\":{\"input_tokens\":10,\"output_tokens\":0}}}\n\n" +
			"event: content_block_start\ndata: {\"type\":\"content_block_start\",\"index\":0,\"content_block\":{\"type\":\"text\",\"text\":\"\"}}\n\n" +
			"event: content_block_stop\ndata: {\"type\":\"content_block_stop\",\"index\":0}\n\n" +
			"event: message_delta\ndata: {\"type\":\"message_delta\",\"delta\":{\"stop_reason\":\"unknown_reason\"},\"usage\":{\"output_tokens\":5}}\n\n"
		_, err := acc.AccumulateStream(strings.NewReader(raw))
		if err == nil || !strings.Contains(err.Error(), "invalid stop_reason") {
			t.Fatalf("expected invalid stop_reason error, got: %v", err)
		}
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 2: Conformance Multi-Turn Agent Loop Across Providers (Gemini -> Nvidia)
// ─────────────────────────────────────────────────────────────────────────────

func TestConformance_MultiTurnAgentLoop_AcrossProviders(t *testing.T) {
	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	// ── Upstream 1: Mock Gemini Server ───────────────────────────────────────
	// Returns a multi-part streaming response: thought: true + functionCall
	geminiServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)

		// Gemini SSE payload with thought + function call
		geminiChunk := `data: {"candidates":[{"content":{"parts":[{"thought":true,"text":"Let me query Seattle weather."},{"functionCall":{"name":"lookup_weather","args":{"city":"Seattle"}}}]},"finishReason":"STOP"}],"usageMetadata":{"promptTokenCount":25,"candidatesTokenCount":18,"thoughtsTokenCount":30}}` + "\n\n"
		_, _ = w.Write([]byte(geminiChunk))
		_, _ = w.Write([]byte("data: [DONE]\n\n"))
		flusher.Flush()
	}))
	defer geminiServer.Close()

	// ── Upstream 2: Mock NVIDIA NIM Server ───────────────────────────────────
	// Expects the accumulated assistant message with reasoning + tool_calls,
	// followed by the tool_result and final user question.
	var receivedNvidiaBody []byte
	nvidiaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		receivedNvidiaBody, _ = io.ReadAll(r.Body)

		w.Header().Set("Content-Type", "text/event-stream")
		w.Header().Set("Cache-Control", "no-cache")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)

		events := []string{
			"data: {\"id\":\"chatcmpl-turn2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{\"role\":\"assistant\",\"content\":\"Seattle is 62 degrees and partly cloudy. A light jacket is recommended.\"},\"finish_reason\":null}]}\n\n",
			"data: {\"id\":\"chatcmpl-turn2\",\"object\":\"chat.completion.chunk\",\"choices\":[{\"index\":0,\"delta\":{},\"finish_reason\":\"stop\"}],\"usage\":{\"prompt_tokens\":120,\"completion_tokens\":16}}\n\n",
			"data: [DONE]\n\n",
		}
		for _, ev := range events {
			_, _ = w.Write([]byte(ev))
			flusher.Flush()
		}
	}))
	defer nvidiaServer.Close()

	// Register routing pools:
	// 1. "gemini-pool" -> geminiServer
	geminiCred := &credentials.RuntimeCredential{
		ID:       101,
		Provider: "gemini",
		APIKey:   "gemini-key",
		BaseURL:  geminiServer.URL,
		Weight:   1,
	}
	geminiPool := credentials.NewBalancedPool("gemini-pool", "round-robin", []*credentials.RuntimeCredential{geminiCred}, nil)
	cacheStore.Set(cache.PoolKey("gemini-pool"), geminiPool, 1)

	// 2. "nvidia-pool" -> nvidiaServer
	nvidiaCred := &credentials.RuntimeCredential{
		ID:       102,
		Provider: "nvidia",
		APIKey:   "nv-key",
		BaseURL:  nvidiaServer.URL,
		Weight:   1,
	}
	nvidiaPool := credentials.NewBalancedPool("nvidia-pool", "round-robin", []*credentials.RuntimeCredential{nvidiaCred}, nil)
	cacheStore.Set(cache.PoolKey("nvidia-pool"), nvidiaPool, 1)

	cacheStore.Wait()
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"gemini-pool": geminiPool,
		"nvidia-pool": nvidiaPool,
	})

	client := gwServer.Client()

	// ── Turn 1: Client -> Gateway (POST /v1/messages, Gemini upstream) ───────
	turn1Payload := `{
		"model": "gemini-pool",
		"max_tokens": 4096,
		"system": [
			{"type": "text", "text": "You are a helpful meteorologist.", "cache_control": {"type": "ephemeral"}}
		],
		"tools": [
			{
				"name": "lookup_weather",
				"description": "Get current weather in city",
				"input_schema": {
					"type": "object",
					"properties": {"city": {"type": "string"}},
					"required": ["city"]
				}
			}
		],
		"thinking": {"type": "enabled", "budget_tokens": 10000},
		"messages": [
			{"role": "user", "content": "What is the weather in Seattle?"}
		],
		"stream": true
	}`

	req1, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(turn1Payload))
	req1.Header.Set("Content-Type", "application/json")
	req1.Header.Set("x-api-key", "admin-test-key")
	req1.Header.Set("anthropic-version", "2023-06-01")

	resp1, err := client.Do(req1)
	if err != nil {
		t.Fatalf("Turn 1 request failed: %v", err)
	}
	defer resp1.Body.Close()

	if resp1.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp1.Body)
		t.Fatalf("Turn 1 expected 200 OK, got %d: %s", resp1.StatusCode, string(b))
	}

	acc1 := NewAnthropicSDKStreamAccumulator()
	msg1, err := acc1.AccumulateStream(resp1.Body)
	if err != nil {
		t.Fatalf("Turn 1 stream accumulation failed wire validation: %v", err)
	}

	if msg1.Role != "assistant" {
		t.Errorf("Turn 1 expected role 'assistant', got %q", msg1.Role)
	}
	if len(msg1.Content) != 2 {
		t.Fatalf("Turn 1 expected 2 content blocks (thinking + tool_use), got %d", len(msg1.Content))
	}
	if msg1.Content[0].Type != "thinking" || !strings.Contains(msg1.Content[0].Thinking, "Seattle weather") {
		t.Errorf("Turn 1 expected thinking block, got %+v", msg1.Content[0])
	}
	if msg1.Content[0].Signature == "" {
		t.Errorf("Turn 1 expected signature in thinking block, got empty")
	}
	if msg1.Content[1].Type != "tool_use" || msg1.Content[1].Name != "lookup_weather" {
		t.Errorf("Turn 1 expected tool_use block, got %+v", msg1.Content[1])
	}
	if city, _ := msg1.Content[1].Input["city"].(string); city != "Seattle" {
		t.Errorf("Turn 1 expected input.city='Seattle', got %v", msg1.Content[1].Input)
	}
	if msg1.StopReason != "tool_use" {
		t.Errorf("Turn 1 expected stop_reason='tool_use', got %q", msg1.StopReason)
	}
	if msg1.Usage.OutputTokens <= 0 {
		t.Errorf("Turn 1 expected output_tokens > 0, got %d", msg1.Usage.OutputTokens)
	}

	// ── Turn 2: Feed accumulated Message into Turn 2 (Nvidia upstream) ───────
	turn2PayloadMap := map[string]any{
		"model":      "nvidia-pool",
		"max_tokens": 4096,
		"messages": []any{
			map[string]any{
				"role":    "user",
				"content": "What is the weather in Seattle?",
			},
			map[string]any{
				"role": "assistant",
				"content": []any{
					map[string]any{
						"type":      "thinking",
						"thinking":  msg1.Content[0].Thinking,
						"signature": msg1.Content[0].Signature,
					},
					map[string]any{
						"type":  "tool_use",
						"id":    msg1.Content[1].ID,
						"name":  msg1.Content[1].Name,
						"input": msg1.Content[1].Input,
					},
				},
			},
			map[string]any{
				"role": "user",
				"content": []any{
					map[string]any{
						"type":         "tool_result",
						"tool_use_id":  msg1.Content[1].ID,
						"content":      `{"temperature": 62, "condition": "Partly Cloudy"}`,
					},
					map[string]any{
						"type": "text",
						"text": "What clothing should I wear?",
					},
				},
			},
		},
		"stream": true,
	}

	turn2Bytes, _ := json.Marshal(turn2PayloadMap)
	req2, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", bytes.NewReader(turn2Bytes))
	req2.Header.Set("Content-Type", "application/json")
	req2.Header.Set("x-api-key", "admin-test-key")
	req2.Header.Set("anthropic-version", "2023-06-01")

	resp2, err := client.Do(req2)
	if err != nil {
		t.Fatalf("Turn 2 request failed: %v", err)
	}
	defer resp2.Body.Close()

	if resp2.StatusCode != http.StatusOK {
		b, _ := io.ReadAll(resp2.Body)
		t.Fatalf("Turn 2 expected 200 OK, got %d: %s", resp2.StatusCode, string(b))
	}

	acc2 := NewAnthropicSDKStreamAccumulator()
	msg2, err := acc2.AccumulateStream(resp2.Body)
	if err != nil {
		t.Fatalf("Turn 2 stream accumulation failed wire validation: %v", err)
	}

	if msg2.Role != "assistant" {
		t.Errorf("Turn 2 expected role 'assistant', got %q", msg2.Role)
	}
	if len(msg2.Content) != 1 || msg2.Content[0].Type != "text" {
		t.Fatalf("Turn 2 expected 1 text content block, got %+v", msg2.Content)
	}
	if !strings.Contains(msg2.Content[0].Text, "Seattle is 62 degrees") {
		t.Errorf("Turn 2 unexpected text: %s", msg2.Content[0].Text)
	}
	if msg2.StopReason != "end_turn" {
		t.Errorf("Turn 2 expected stop_reason='end_turn', got %q", msg2.StopReason)
	}
	if msg2.Usage.OutputTokens != 16 {
		t.Errorf("Turn 2 expected output_tokens=16, got %d", msg2.Usage.OutputTokens)
	}

	// Verify inbound translation in Turn 2 to NVIDIA upstream
	if !bytes.Contains(receivedNvidiaBody, []byte(`"role":"tool"`)) {
		t.Errorf("Nvidia upstream should have received role:'tool' message, got: %s", string(receivedNvidiaBody))
	}
	if !bytes.Contains(receivedNvidiaBody, []byte(msg1.Content[1].ID)) {
		t.Errorf("Nvidia upstream should have received tool_call_id %s, got: %s", msg1.Content[1].ID, string(receivedNvidiaBody))
	}
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 3: Ollama (NDJSON) and 1min.ai (Named SSE) to Anthropic Stream Conformance
// ─────────────────────────────────────────────────────────────────────────────

func TestConformance_OllamaAndOneMinAI_ToAnthropic(t *testing.T) {
	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	client := gwServer.Client()

	// ── Subtest A: Ollama NDJSON Stream ──────────────────────────────────────
	ollamaServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "application/x-ndjson")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)

		chunks := []string{
			"{\"model\":\"llama3\",\"message\":{\"role\":\"assistant\",\"content\":\"Hello \"},\"done\":false}\n",
			"{\"model\":\"llama3\",\"message\":{\"role\":\"assistant\",\"content\":\"world from Ollama!\"},\"done\":false}\n",
			"{\"model\":\"llama3\",\"done\":true,\"done_reason\":\"stop\",\"prompt_eval_count\":15,\"eval_count\":6}\n",
		}
		for _, c := range chunks {
			_, _ = w.Write([]byte(c))
			flusher.Flush()
		}
	}))
	defer ollamaServer.Close()

	ollamaCred := &credentials.RuntimeCredential{
		ID:       201,
		Provider: "ollama",
		APIKey:   "ollama-local",
		BaseURL:  ollamaServer.URL,
		Weight:   1,
	}
	ollamaPool := credentials.NewBalancedPool("ollama-pool", "round-robin", []*credentials.RuntimeCredential{ollamaCred}, nil)
	cacheStore.Set(cache.PoolKey("ollama-pool"), ollamaPool, 1)

	// ── Subtest B: 1min.ai Named SSE Stream ──────────────────────────────────
	oneMinServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
		w.Header().Set("Content-Type", "text/event-stream")
		w.WriteHeader(http.StatusOK)

		flusher := w.(http.Flusher)

		events := []string{
			"event: content\ndata: {\"content\": \"Hello from \"}\n\n",
			"event: content\ndata: {\"content\": \"1min.ai transmuxing!\"}\n\n",
			"event: done\ndata: {\"message\": \"Stream completed\"}\n\n",
			"data: [DONE]\n\n",
		}
		for _, ev := range events {
			_, _ = w.Write([]byte(ev))
			flusher.Flush()
		}
	}))
	defer oneMinServer.Close()

	oneMinCred := &credentials.RuntimeCredential{
		ID:       202,
		Provider: "1minai",
		APIKey:   "1min-secret",
		BaseURL:  oneMinServer.URL,
		Weight:   1,
	}
	oneMinPool := credentials.NewBalancedPool("1min/gpt-4o", "round-robin", []*credentials.RuntimeCredential{oneMinCred}, nil)
	cacheStore.Set(cache.PoolKey("1min/gpt-4o"), oneMinPool, 1)

	cacheStore.Wait()
	sm.SetPools(map[string]*credentials.BalancedChannelPool{
		"ollama-pool": ollamaPool,
		"1min/gpt-4o": oneMinPool,
	})

	t.Run("Ollama_NDJSON_To_Anthropic", func(t *testing.T) {
		reqPayload := `{
			"model": "ollama-pool",
			"max_tokens": 1024,
			"messages": [{"role": "user", "content": "Hi Ollama"}],
			"stream": true
		}`
		req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(reqPayload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "admin-test-key")
		req.Header.Set("anthropic-version", "2023-06-01")

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("Ollama request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected 200 OK, got %d: %s", resp.StatusCode, string(b))
		}

		acc := NewAnthropicSDKStreamAccumulator()
		msg, err := acc.AccumulateStream(resp.Body)
		if err != nil {
			t.Fatalf("Ollama accumulator error: %v", err)
		}

		if msg.Role != "assistant" {
			t.Errorf("expected role 'assistant', got %q", msg.Role)
		}
		if len(msg.Content) != 1 || msg.Content[0].Type != "text" {
			t.Fatalf("expected 1 text content block, got %+v", msg.Content)
		}
		if msg.Content[0].Text != "Hello world from Ollama!" {
			t.Errorf("expected 'Hello world from Ollama!', got %q", msg.Content[0].Text)
		}
		if msg.StopReason != "end_turn" {
			t.Errorf("expected stop_reason='end_turn', got %q", msg.StopReason)
		}
	})

	t.Run("OneMinAI_NamedSSE_To_Anthropic", func(t *testing.T) {
		reqPayload := `{
			"model": "1min/gpt-4o",
			"max_tokens": 1024,
			"messages": [{"role": "user", "content": "Hi 1min"}],
			"stream": true
		}`
		req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(reqPayload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "admin-test-key")
		req.Header.Set("anthropic-version", "2023-06-01")

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("1min.ai request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusOK {
			b, _ := io.ReadAll(resp.Body)
			t.Fatalf("expected 200 OK, got %d: %s", resp.StatusCode, string(b))
		}

		acc := NewAnthropicSDKStreamAccumulator()
		msg, err := acc.AccumulateStream(resp.Body)
		if err != nil {
			t.Fatalf("1min.ai accumulator error: %v", err)
		}

		if msg.Role != "assistant" {
			t.Errorf("expected role 'assistant', got %q", msg.Role)
		}
		if len(msg.Content) != 1 || msg.Content[0].Type != "text" {
			t.Fatalf("expected 1 text content block, got %+v", msg.Content)
		}
		if msg.Content[0].Text != "Hello from 1min.ai transmuxing!" {
			t.Errorf("expected 'Hello from 1min.ai transmuxing!', got %q", msg.Content[0].Text)
		}
		if msg.StopReason != "end_turn" {
			t.Errorf("expected stop_reason='end_turn', got %q", msg.StopReason)
		}
	})
}

// ─────────────────────────────────────────────────────────────────────────────
// Test 4: Error Envelopes and Mid-Stream Error Conformance
// ─────────────────────────────────────────────────────────────────────────────

func TestConformance_ErrorEnvelopesAndMidStreamError(t *testing.T) {
	_, cacheStore, sm, engine := setupE2ETestEnv(t)
	gwServer := httptest.NewServer(engine)
	defer gwServer.Close()

	client := gwServer.Client()

	t.Run("HTTP400_InvalidRequestError", func(t *testing.T) {
		req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(`{"model":"foo"}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "admin-test-key")

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusBadRequest {
			t.Errorf("expected 400, got %d", resp.StatusCode)
		}

		var errEnvelope map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&errEnvelope)
		if errEnvelope["type"] != "error" {
			t.Errorf("expected type='error', got %v", errEnvelope["type"])
		}
		errObj, _ := errEnvelope["error"].(map[string]any)
		if errObj["type"] != "invalid_request_error" {
			t.Errorf("expected error.type='invalid_request_error', got %v", errObj["type"])
		}
	})

	t.Run("HTTP401_AuthenticationError", func(t *testing.T) {
		req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(`{"model":"claude","messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "invalid-key")

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusUnauthorized {
			t.Errorf("expected 401, got %d", resp.StatusCode)
		}

		var errEnvelope map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&errEnvelope)
		if errEnvelope["type"] != "error" {
			t.Errorf("expected type='error', got %v", errEnvelope["type"])
		}
		errObj, _ := errEnvelope["error"].(map[string]any)
		if errObj["type"] != "authentication_error" {
			t.Errorf("expected error.type='authentication_error', got %v", errObj["type"])
		}
	})

	t.Run("HTTP404_NotFoundError", func(t *testing.T) {
		req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(`{"model":"non-existent-pool","max_tokens":100,"messages":[{"role":"user","content":"hi"}]}`))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "admin-test-key")

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		if resp.StatusCode != http.StatusNotFound {
			t.Errorf("expected 404, got %d", resp.StatusCode)
		}

		var errEnvelope map[string]any
		_ = json.NewDecoder(resp.Body).Decode(&errEnvelope)
		errObj, _ := errEnvelope["error"].(map[string]any)
		if errObj["type"] != "not_found_error" {
			t.Errorf("expected error.type='not_found_error', got %v", errObj["type"])
		}
	})

	t.Run("FormatAnthropicGatewayError_TypeMappings", func(t *testing.T) {
		statusToType := []struct {
			status       int
			expectedType string
		}{
			{http.StatusBadRequest, "invalid_request_error"},
			{http.StatusUnauthorized, "authentication_error"},
			{http.StatusForbidden, "permission_error"},
			{http.StatusNotFound, "not_found_error"},
			{http.StatusTooManyRequests, "rate_limit_error"},
			{529, "overloaded_error"},
			{http.StatusBadGateway, "api_error"},
			{http.StatusInternalServerError, "api_error"},
		}

		for _, tc := range statusToType {
			body := formatAnthropicGatewayError(tc.status, nil, "gateway error")
			var env map[string]any
			if err := json.Unmarshal(body, &env); err != nil {
				t.Fatalf("status %d produced invalid JSON: %v", tc.status, err)
			}
			errObj, _ := env["error"].(map[string]any)
			if errObj["type"] != tc.expectedType {
				t.Errorf("status %d expected error.type=%q, got %q", tc.status, tc.expectedType, errObj["type"])
			}
		}
	})

	t.Run("MidStreamError_EventFrameEmitted", func(t *testing.T) {
		// Mock upstream that sends a chunk then emits a mid-stream error object
		errServer := httptest.NewServer(http.HandlerFunc(func(w http.ResponseWriter, r *http.Request) {
			w.Header().Set("Content-Type", "text/event-stream")
			w.WriteHeader(http.StatusOK)

			flusher := w.(http.Flusher)

			_, _ = w.Write([]byte("data: {\"choices\":[{\"delta\":{\"content\":\"Initial tokens before failure...\"}}]}\n\n"))
			flusher.Flush()

			time.Sleep(10 * time.Millisecond)

			// OpenAI format mid-stream error payload
			_, _ = w.Write([]byte("data: {\"error\":{\"type\":\"api_error\",\"message\":\"CUDA out of memory mid-stream\"}}\n\n"))
			flusher.Flush()
		}))
		defer errServer.Close()

		errCred := &credentials.RuntimeCredential{
			ID:       301,
			Provider: "openai",
			APIKey:   "test-key",
			BaseURL:  errServer.URL,
			Weight:   1,
		}
		errPool := credentials.NewBalancedPool("err-pool", "round-robin", []*credentials.RuntimeCredential{errCred}, nil)
		cacheStore.Set(cache.PoolKey("err-pool"), errPool, 1)
		cacheStore.Wait()
		sm.SetPools(map[string]*credentials.BalancedChannelPool{
			"err-pool": errPool,
		})

		reqPayload := `{
			"model": "err-pool",
			"max_tokens": 1024,
			"messages": [{"role": "user", "content": "crash test"}],
			"stream": true
		}`
		req, _ := http.NewRequest("POST", gwServer.URL+"/v1/messages", strings.NewReader(reqPayload))
		req.Header.Set("Content-Type", "application/json")
		req.Header.Set("x-api-key", "admin-test-key")

		resp, err := client.Do(req)
		if err != nil {
			t.Fatalf("request failed: %v", err)
		}
		defer resp.Body.Close()

		acc := NewAnthropicSDKStreamAccumulator()
		_, accErr := acc.AccumulateStream(resp.Body)
		if accErr == nil {
			t.Fatal("expected mid-stream error, got nil")
		}

		streamErr, ok := accErr.(*AnthropicStreamError)
		if !ok {
			t.Fatalf("expected *AnthropicStreamError, got: %T (%v)", accErr, accErr)
		}
		if streamErr.Type != "api_error" || !strings.Contains(streamErr.Message, "CUDA out of memory") {
			t.Errorf("unexpected stream error: %+v", streamErr)
		}
	})
}

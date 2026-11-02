package proxy

import (
	"io"
	"net/http"
	"net/http/httptest"
	"strings"
	"sync"
	"testing"
	"time"

	"github.com/gin-gonic/gin"
	"github.com/skadraneshghn/clever-ai-gate/internal/cache"
	"github.com/skadraneshghn/clever-ai-gate/internal/config"
	"github.com/skadraneshghn/clever-ai-gate/internal/credentials"
	"go.uber.org/zap"
)

// pollinationsLowBalanceMsg is the exact error message reported by users,
// returned by Pollinations with HTTP 200 in place of the model's answer.
const pollinationsLowBalanceMsg = "The account behind this API key doesn't have enough credits. Please [top up](https://enter.pollinations.ai/top-up?ref=agent_low_balance_topup) or [complete a quest](https://enter.pollinations.ai/quests?ref=agent_low_balance_quests), then try again.\n\nIf this isn’t your Pollinations account, contact whoever runs the app or service you're using."

func TestDetectSuccessBodyError_PollinationsPlainText(t *testing.T) {
	v := detectSuccessBodyError([]byte(pollinationsLowBalanceMsg), "text/plain; charset=utf-8", true)
	if v == nil {
		t.Fatal("expected pollinations low-balance message to be detected as error")
	}
	if v.Reason != "pollinations_low_balance" {
		t.Errorf("expected reason pollinations_low_balance, got %q", v.Reason)
	}
}

func TestDetectSuccessBodyError_PollinationsCurlyQuotes(t *testing.T) {
	// Typographic apostrophe variant ("doesn’t" with U+2019) must still match.
	msg := "The account behind this API key doesn’t have enough credits. Please top up or complete a quest, then try again."
	v := detectSuccessBodyError([]byte(msg), "text/plain", true)
	if v == nil {
		t.Fatal("expected curly-quote variant to be detected")
	}
}

func TestDetectSuccessBodyError_PollinationsJSONChatCarrier(t *testing.T) {
	body := `{"id":"chatcmpl-1","object":"chat.completion","choices":[{"index":0,"message":{"role":"assistant","content":"` +
		strings.ReplaceAll(strings.ReplaceAll(pollinationsLowBalanceMsg, `"`, `\"`), "\n", `\n`) +
		`"},"finish_reason":"stop"}],"usage":{"prompt_tokens":10,"completion_tokens":80}}`
	v := detectSuccessBodyError([]byte(body), "application/json", true)
	if v == nil {
		t.Fatal("expected JSON chat-completion carrier to be detected")
	}
	if v.Reason != "pollinations_low_balance" {
		t.Errorf("expected reason pollinations_low_balance, got %q", v.Reason)
	}
}

func TestDetectSuccessBodyError_PollinationsNoLinksVariant(t *testing.T) {
	// Signature message without the URLs — must still hit tier-1 via the
	// verbatim opener phrase.
	msg := "The account behind this API key doesn't have enough credits. Please top up or complete a quest, then try again."
	v := detectSuccessBodyError([]byte(msg), "text/markdown", true)
	if v == nil {
		t.Fatal("expected no-links pollinations variant to be detected")
	}
}

func TestDetectSuccessBodyError_TopLevelErrorObject(t *testing.T) {
	v := detectSuccessBodyError([]byte(`{"error":{"message":"Insufficient credits","type":"soft_error","code":"credits"}}`), "application/json", true)
	if v == nil {
		t.Fatal("expected top-level error object in 200 to be detected")
	}
	if v.Reason != "json_error_object" {
		t.Errorf("expected reason json_error_object, got %q", v.Reason)
	}
}

func TestDetectSuccessBodyError_TopLevelErrorString(t *testing.T) {
	v := detectSuccessBodyError([]byte(`{"error":"credit balance too low"}`), "application/json", true)
	if v == nil {
		t.Fatal("expected top-level error string in 200 to be detected")
	}
}

func TestDetectSuccessBodyError_ErrorPlaceholdersIgnored(t *testing.T) {
	cases := []string{
		`{"data":[],"error":null}`,
		`{"data":[],"error":""}`,
		`{"data":[],"error":false}`,
		`{"data":[],"error":0}`,
		`{"data":[],"error":{}}`,
	}
	for _, body := range cases {
		if v := detectSuccessBodyError([]byte(body), "application/json", true); v != nil {
			t.Errorf("placeholder error must not be flagged: %q -> %+v", body, v)
		}
	}
}

func TestDetectSuccessBodyError_SuccessFalseNoContent(t *testing.T) {
	v := detectSuccessBodyError([]byte(`{"success":false,"errors":[{"code":7003,"message":"rate limit exceeded for account"}]}`), "application/json", true)
	if v == nil {
		t.Fatal("expected success:false without content to be detected")
	}
	if v.Reason != "success_false" {
		t.Errorf("expected reason success_false, got %q", v.Reason)
	}
}

func TestDetectSuccessBodyError_SuccessFalseWithContentIgnored(t *testing.T) {
	v := detectSuccessBodyError([]byte(`{"success":false,"choices":[{"message":{"content":"here is your answer anyway"}}]}`), "application/json", true)
	if v != nil {
		t.Fatalf("success:false WITH completion content must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_SuccessTrueClean(t *testing.T) {
	v := detectSuccessBodyError([]byte(`{"success":true,"result":{"output":"all good"}}`), "application/json", true)
	if v != nil {
		t.Fatalf("success:true response must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_GenericBalanceCarrier(t *testing.T) {
	msg := "Your credit balance is too low to process this request. Please top up your account."
	v := detectSuccessBodyError([]byte(msg), "text/plain", true)
	if v == nil {
		t.Fatal("expected generic balance error carrier to be detected")
	}
	if v.Reason != "credit_balance_error_text" {
		t.Errorf("expected reason credit_balance_error_text, got %q", v.Reason)
	}
}

func TestDetectSuccessBodyError_HTMLCarrier(t *testing.T) {
	body := "<html><body><h1>Account Notice</h1><p>Your account has insufficient credits. Please <a href=\"https://billing.example.com\">add funds</a> to continue using your API key.</p></body></html>"
	v := detectSuccessBodyError([]byte(body), "text/html; charset=utf-8", true)
	if v == nil {
		t.Fatal("expected HTML billing interstitial to be detected")
	}
}

// ── False-positive guards ─────────────────────────────────────────────────────

func TestDetectSuccessBodyError_NormalCompletionClean(t *testing.T) {
	body := `{"choices":[{"message":{"role":"assistant","content":"The capital of France is Paris."}}]}`
	if v := detectSuccessBodyError([]byte(body), "application/json", true); v != nil {
		t.Fatalf("normal completion must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_LongBillingDiscussionClean(t *testing.T) {
	// An educational answer about HTTP 402: mentions payment, accounts and
	// "payment required", but is prose for a human, not operator-facing.
	msg := "HTTP 402 Payment Required means the server requires payment before processing the request. APIs use it when an account has no funds available."
	if v := detectSuccessBodyError([]byte(msg), "text/plain", true); v != nil {
		t.Fatalf("educational answer must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_TopUpEmailDraftClean(t *testing.T) {
	// A copywriting request ("write a top-up reminder email") produces a
	// short billing-flavored completion — must NOT be flagged.
	msg := "Subject: Your balance is low — please top up your account to keep your service active. Sent from our billing system."
	if v := detectSuccessBodyError([]byte(msg), "text/plain", true); v != nil {
		t.Fatalf("email draft must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_QuotaExplanationClean(t *testing.T) {
	msg := "If you exceed your quota, the request fails with HTTP 429. You may need to upgrade your plan with the provider."
	if v := detectSuccessBodyError([]byte(msg), "text/plain", true); v != nil {
		t.Fatalf("quota explanation must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_TruncatedLongBodySkipsTier2(t *testing.T) {
	// Long legit completion, truncated at the sniff limit — tier-2 must be
	// skipped (complete=false), only tier-1/structural apply.
	long := `{"choices":[{"message":{"content":"` + strings.Repeat("Once upon a time a lighthouse keeper counted his stones. ", 500) + `"}}]}`
	if v := detectSuccessBodyError([]byte(long), "application/json", false); v != nil {
		t.Fatalf("truncated long body must not be flagged: %+v", v)
	}
}

func TestDetectSuccessBodyError_Guards(t *testing.T) {
	if v := detectSuccessBodyError(nil, "application/json", true); v != nil {
		t.Fatalf("empty body must not be flagged: %+v", v)
	}
	if isSniffableContentType("image/png") {
		t.Fatal("image/png must not be sniffable")
	}
	if isSniffableContentType("audio/mpeg") {
		t.Fatal("audio/mpeg must not be sniffable")
	}
	if !isSniffableContentType("application/json; charset=utf-8") {
		t.Fatal("application/json must be sniffable")
	}
	if !isSniffableContentType("text/event-stream") {
		t.Fatal("text/event-stream must not be flagged")
	}
}

func TestSyntheticBodyShape(t *testing.T) {
	v := &successBodyError{Reason: "pollinations_low_balance", Message: pollinationsLowBalanceMsg}
	body := string(v.SyntheticBody())
	if !strings.Contains(body, `"insufficient_quota"`) {
		t.Fatalf("synthetic body missing type: %s", body)
	}
	if !strings.Contains(body, "The account behind this API key") {
		t.Fatalf("synthetic body must embed original message: %s", body)
	}
	// formatOpenAIError must pass this envelope through unchanged.
	passed := formatOpenAIError(http.StatusPaymentRequired, v.SyntheticBody(), "summary")
	if !strings.Contains(string(passed), "The account behind this API key") {
		t.Fatalf("formatOpenAIError should preserve the envelope: %s", passed)
	}
}

// ── Stream sniffer tests ──────────────────────────────────────────────────────

// chunkReader serves its input one chunk per Read, simulating an SSE stream
// arriving over the network in pieces. It tracks a byte offset within the
// current chunk so small destination buffers never drop data (io.ReadAll
// starts with a 512-byte buffer, for example).
type chunkReader struct {
	chunks []string
	idx    int
	off    int
}

func (r *chunkReader) Read(p []byte) (int, error) {
	if r.idx >= len(r.chunks) {
		return 0, io.EOF
	}
	chunk := r.chunks[r.idx]
	n := copy(p, chunk[r.off:])
	r.off += n
	if r.off >= len(chunk) {
		r.idx++
		r.off = 0
	}
	return n, nil
}

func TestSniffStream_PollinationsWordByWord(t *testing.T) {
	// The error message streamed as tiny per-word SSE deltas — the hardest
	// case: no single event contains a complete signature.
	var sb strings.Builder
	for _, w := range strings.Fields("The account behind this API key doesn't have enough credits. Please top up or complete a quest, then try again. Visit https://enter.pollinations.ai/top-up?ref=agent_low_balance_topup") {
		sb.WriteString("data: {\"choices\":[{\"delta\":{\"content\":\"" + w + " \"}}]}\n\n")
	}
	sb.WriteString("data: [DONE]\n\n")
	res := sniffStreamForError(strings.NewReader(sb.String()))
	if res.verdict == nil {
		t.Fatal("expected word-by-word pollinations stream to be detected")
	}
	if res.verdict.Reason != "pollinations_low_balance" {
		t.Errorf("expected reason pollinations_low_balance, got %q", res.verdict.Reason)
	}
}

func TestSniffStream_SingleEventError(t *testing.T) {
	esc := strings.ReplaceAll(pollinationsLowBalanceMsg, `"`, `\"`)
	esc = strings.ReplaceAll(esc, "\n\n", `\n\n`)
	stream := "data: {\"choices\":[{\"delta\":{\"content\":\"" + esc + "\"}}]}\n\ndata: [DONE]\n\n"
	res := sniffStreamForError(strings.NewReader(stream))
	if res.verdict == nil {
		t.Fatal("expected single-event pollinations stream to be detected")
	}
}

func TestSniffStream_FastCommitOnFirstContent(t *testing.T) {
	chunks := []string{
		"data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\"The answer is 42.\"}}]}\n\n",
		"data: {\"choices\":[{\"delta\":{\"content\":\" More text follows here.\"}}]}\n\n",
		"data: [DONE]\n\n",
	}
	r := &chunkReader{chunks: chunks}
	res := sniffStreamForError(r)
	if res.verdict != nil {
		t.Fatalf("normal stream must not be flagged: %+v", res.verdict)
	}
	// The first content event carries no watch-keywords — the sniffer must
	// commit immediately instead of buffering the whole stream (TTFB).
	if r.idx > 2 {
		t.Errorf("expected fast commit after the first content event, consumed %d chunks", r.idx)
	}
	combined, err := io.ReadAll(res.body)
	if err != nil {
		t.Fatalf("read combined: %v", err)
	}
	if string(combined) != strings.Join(chunks, "") {
		t.Fatal("fast-commit replay is not byte-exact")
	}
}

func TestSniffStream_NDJSONErrorEvent(t *testing.T) {
	// An NDJSON upstream (Ollama-style) that errors immediately: the first
	// record is a top-level error object smuggled inside a 200 OK stream.
	r := &chunkReader{chunks: []string{
		"{\"error\":\"insufficient balance: please top up your account\",\"done\":true}\n",
	}}
	res := sniffStreamForError(r)
	if res.verdict == nil || res.verdict.Reason != "json_error_object" {
		t.Fatalf("expected NDJSON error event to be detected, got %+v", res.verdict)
	}
}

func TestSniffStream_BareJSONErrorBody(t *testing.T) {
	// stream:true but the upstream answers with a full JSON body instead of
	// an SSE stream.
	body := `{"choices":[{"message":{"role":"assistant","content":"The account behind this API key doesn't have enough credits. Please top up, then try again."}}]}`
	res := sniffStreamForError(strings.NewReader(body))
	if res.verdict == nil {
		t.Fatal("expected bare JSON pollinations carrier to be detected")
	}
}

func TestSniffStream_EmptyDoneClean(t *testing.T) {
	stream := "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\",\"content\":\"\"}}]}\n\ndata: [DONE]\n\n"
	res := sniffStreamForError(strings.NewReader(stream))
	if res.verdict != nil {
		t.Fatalf("empty stream must not be flagged: %+v", res.verdict)
	}
}

func TestSniffStream_WatchKeywordThenLongContentClean(t *testing.T) {
	// A legit completion that OPENS with billing vocabulary and then keeps
	// going past the commit threshold — must commit clean, never error.
	first := "data: {\"choices\":[{\"delta\":{\"content\":\"Your credit balance is calculated monthly. \"}}]}\n\n"
	second := "data: {\"choices\":[{\"delta\":{\"content\":\"" +
		strings.Repeat("Here is how the statement works, line by line. ", 20) +
		"\"}}]}\n\ndata: [DONE]\n\n"
	r := &chunkReader{chunks: []string{first, second}}
	res := sniffStreamForError(r)
	if res.verdict != nil {
		t.Fatalf("watch-keyword legit stream must not be flagged: %+v", res.verdict)
	}
	combined, err := io.ReadAll(res.body)
	if err != nil {
		t.Fatalf("read combined: %v", err)
	}
	if string(combined) != first+second {
		t.Fatal("watch-hold replay is not byte-exact")
	}
}

func TestSniffStream_RawCapCommitClean(t *testing.T) {
	// A stream of role-only deltas (no content, no signatures) must commit as
	// clean once the raw buffer hits its cap — with byte-exact replay.
	var chunks []string
	for i := 0; i < 800; i++ {
		chunks = append(chunks, "data: {\"choices\":[{\"delta\":{\"role\":\"assistant\"}}]}\n\n")
	}
	r := &chunkReader{chunks: chunks}
	res := sniffStreamForError(r)
	if res.verdict != nil {
		t.Fatalf("role-only stream must not be flagged: %+v", res.verdict)
	}
	combined, err := io.ReadAll(res.body)
	if err != nil {
		t.Fatalf("read combined: %v", err)
	}
	if string(combined) != strings.Join(chunks, "") {
		t.Fatal("raw-cap replay is not byte-exact")
	}
}

func TestSniffStream_LargeLegitContentCommitsFast(t *testing.T) {
	// A large legit completion commits as clean on its first content event
	// (content threshold reached), never buffering the whole stream.
	var chunks []string
	for i := 0; i < 10; i++ {
		chunks = append(chunks, "data: {\"choices\":[{\"delta\":{\"content\":\""+
			strings.Repeat("plain filler text without markers. ", 200)+"\"}}]}\n\n")
	}
	r := &chunkReader{chunks: chunks}
	res := sniffStreamForError(r)
	if res.verdict != nil {
		t.Fatalf("large filler stream must not be flagged: %+v", res.verdict)
	}
	if r.idx > 1 {
		t.Errorf("expected commit after the first content chunk, consumed %d chunks", r.idx)
	}
	combined, err := io.ReadAll(res.body)
	if err != nil {
		t.Fatalf("read combined: %v", err)
	}
	if string(combined) != strings.Join(chunks, "") {
		t.Fatal("fast-commit replay is not byte-exact")
	}
}

// ── Handler-level tests ───────────────────────────────────────────────────────

// mockRoundTripper is defined in handler_test.go and reused here.

func newSniffTestHandler(t *testing.T, transport http.RoundTripper) *Handler {
	t.Helper()
	logger := zap.NewNop()
	cfg := &config.Config{CacheMaxSizeMB: 10, CacheNumCounters: 100}
	cacheStore, err := cache.New(cfg, logger)
	if err != nil {
		t.Fatalf("failed to create cache: %v", err)
	}
	t.Cleanup(func() { cacheStore.Close() })
	client := &http.Client{Transport: transport}
	return NewHandler(client, cacheStore, nil, logger, nil, nil, nil)
}

func newProxyTestContext(t *testing.T, body string) (*gin.Context, *httptest.ResponseRecorder) {
	t.Helper()
	req := httptest.NewRequest("POST", "/v1/chat/completions", strings.NewReader(body))
	req.Header.Set("Content-Type", "application/json")
	w := httptest.NewRecorder()
	c, _ := gin.CreateTestContext(w)
	c.Request = req
	return c, w
}

// escapeJSONString makes a Go string embeddable inside a JSON string value.
func escapeJSONString(s string) string {
	s = strings.ReplaceAll(s, `\`, `\\`)
	s = strings.ReplaceAll(s, `"`, `\"`)
	return strings.ReplaceAll(s, "\n", `\n`)
}

func TestForwardRequest_PollinationsErrorIn200_NonStream(t *testing.T) {
	errBody := `{"id":"x","choices":[{"message":{"role":"assistant","content":"` +
		escapeJSONString(pollinationsLowBalanceMsg) + `"},"finish_reason":"stop"}]}`
	transport := &mockRoundTripper{roundTripFunc: func(req *http.Request) (*http.Response, error) {
		resp := &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(errBody))}
		resp.Header.Set("Content-Type", "application/json")
		return resp, nil
	}}
	h := newSniffTestHandler(t, transport)

	cred := &credentials.RuntimeCredential{ID: 1, Provider: "custom", APIKey: "sk-x", BaseURL: "https://up.example/v1", Weight: 1}
	pool := credentials.NewBalancedPool("test/claude", "round-robin", []*credentials.RuntimeCredential{cred}, nil)
	c, w := newProxyTestContext(t, `{"model":"test/claude","messages":[]}`)
	pctx := &proxyContext{
		model:          "test/claude",
		requestedModel: "test/claude",
		body:           []byte(`{"model":"test/claude","messages":[]}`),
		pool:           pool,
		credential:     &credentials.AcquireResult{Credential: cred, Index: 0, FromPool: pool},
	}

	statusCode, _, errBodyRet, err := h.forwardRequest(c, pctx)
	if err != nil {
		t.Fatalf("forwardRequest returned error: %v", err)
	}
	if statusCode != http.StatusPaymentRequired {
		t.Fatalf("expected synthetic 402 for error-in-success body, got %d", statusCode)
	}
	if !strings.Contains(string(errBodyRet), "insufficient_quota") {
		t.Fatalf("expected OpenAI error envelope, got: %s", errBodyRet)
	}
	if w.Body.Len() != 0 {
		t.Fatalf("nothing must be written to the client on detection, got: %s", w.Body.String())
	}
}

func TestForwardRequest_PollinationsErrorIn200_Stream(t *testing.T) {
	var sb strings.Builder
	for _, w := range strings.Fields("The account behind this API key doesn't have enough credits. Please top up or complete a quest, then try again. Visit https://enter.pollinations.ai/top-up?ref=agent_low_balance_topup") {
		sb.WriteString("data: {\"choices\":[{\"delta\":{\"content\":\"" + w + " \"}}]}\n\n")
	}
	sb.WriteString("data: [DONE]\n\n")
	transport := &mockRoundTripper{roundTripFunc: func(req *http.Request) (*http.Response, error) {
		resp := &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(sb.String()))}
		resp.Header.Set("Content-Type", "text/event-stream")
		return resp, nil
	}}
	h := newSniffTestHandler(t, transport)

	cred := &credentials.RuntimeCredential{ID: 1, Provider: "custom", APIKey: "sk-x", BaseURL: "https://up.example/v1", Weight: 1}
	pool := credentials.NewBalancedPool("test/claude", "round-robin", []*credentials.RuntimeCredential{cred}, nil)
	c, w := newProxyTestContext(t, `{"model":"test/claude","stream":true,"messages":[]}`)
	pctx := &proxyContext{
		model:          "test/claude",
		requestedModel: "test/claude",
		body:           []byte(`{"model":"test/claude","stream":true,"messages":[]}`),
		isStream:       true,
		pool:           pool,
		credential:     &credentials.AcquireResult{Credential: cred, Index: 0, FromPool: pool},
	}

	statusCode, _, errBodyRet, err := h.forwardRequest(c, pctx)
	if err != nil {
		t.Fatalf("forwardRequest returned error: %v", err)
	}
	if statusCode != http.StatusPaymentRequired {
		t.Fatalf("expected synthetic 402 for streamed error-in-success body, got %d", statusCode)
	}
	if !strings.Contains(string(errBodyRet), "insufficient_quota") {
		t.Fatalf("expected OpenAI error envelope, got: %s", errBodyRet)
	}
	if w.Body.Len() != 0 {
		t.Fatalf("nothing must be written to the client on detection, got: %s", w.Body.String())
	}
}

// TestExecuteWithRetry_RotatesOnSuccessBodyError proves the full pipeline:
// key #1 answers 200 OK with the Pollinations low-balance marketing text in
// place of a completion — the gateway classifies it as a retryable failure,
// penalizes the credential, rotates to key #2, and the client receives the
// healthy completion with HTTP 200 (never seeing the marketing text).
func TestExecuteWithRetry_RotatesOnSuccessBodyError(t *testing.T) {
	badBody := `{"id":"x","choices":[{"message":{"role":"assistant","content":"` +
		escapeJSONString(pollinationsLowBalanceMsg) + `"},"finish_reason":"stop"}]}`
	goodBody := `{"id":"y","choices":[{"message":{"role":"assistant","content":"healthy answer from the second key"},"finish_reason":"stop"}]}`

	var mu sync.Mutex
	callsByKey := map[string]int{}
	transport := &mockRoundTripper{roundTripFunc: func(req *http.Request) (*http.Response, error) {
		auth := req.Header.Get("Authorization")
		mu.Lock()
		callsByKey[auth]++
		mu.Unlock()
		var body string
		if auth == "Bearer sk-bad" {
			body = badBody
		} else {
			body = goodBody
		}
		resp := &http.Response{StatusCode: http.StatusOK, Header: make(http.Header), Body: io.NopCloser(strings.NewReader(body))}
		resp.Header.Set("Content-Type", "application/json")
		return resp, nil
	}}
	h := newSniffTestHandler(t, transport)

	// Pool order matters: AcquireActiveToken's cursor picks index 1 first,
	// so the BAD credential must sit at index 1 to be tried before the good one.
	goodCred := &credentials.RuntimeCredential{ID: 1, Provider: "custom", APIKey: "sk-good", BaseURL: "https://up.example/v1", Weight: 1}
	badCred := &credentials.RuntimeCredential{ID: 2, Provider: "custom", APIKey: "sk-bad", BaseURL: "https://up.example/v1", Weight: 1}
	pool := credentials.NewBalancedPool("test/claude", "round-robin", []*credentials.RuntimeCredential{goodCred, badCred}, nil)

	c, w := newProxyTestContext(t, `{"model":"test/claude","messages":[]}`)
	pctx := &proxyContext{
		model:          "test/claude",
		requestedModel: "test/claude",
		body:           []byte(`{"model":"test/claude","messages":[]}`),
		pool:           pool,
	}

	h.executeWithRetry(c, pctx, time.Now(), 2)

	if w.Code != http.StatusOK {
		t.Fatalf("expected rotation to the healthy key to yield HTTP 200, got %d: %s", w.Code, w.Body.String())
	}
	if !strings.Contains(w.Body.String(), "healthy answer from the second key") {
		t.Fatalf("client should receive the healthy completion, got: %s", w.Body.String())
	}
	if strings.Contains(w.Body.String(), "pollinations") || strings.Contains(w.Body.String(), "top-up") {
		t.Fatalf("marketing text must never reach the client, got: %s", w.Body.String())
	}
	mu.Lock()
	badCalls := callsByKey["Bearer sk-bad"]
	goodCalls := callsByKey["Bearer sk-good"]
	mu.Unlock()
	if badCalls != 1 {
		t.Errorf("expected the bad key to be called exactly once, got %d", badCalls)
	}
	if goodCalls < 1 {
		t.Error("expected the good key to be called at least once")
	}
}

func TestSniffStream_CleanReplayByteExact(t *testing.T) {
	stream := "data: {\"choices\":[{\"delta\":{\"content\":\"Hello\"}}]}\n\n" +
		"data: {\"choices\":[{\"delta\":{\"content\":\" world from a healthy model!\"},\"finish_reason\":\"stop\"}]}\n\n" +
		"data: [DONE]\n\n"
	body := strings.NewReader(stream)
	res := sniffStreamForError(body)
	if res.verdict != nil {
		t.Fatalf("clean stream must not be flagged: %+v", res.verdict)
	}
	// Replay + remaining unread bytes must equal the original stream exactly.
	combined, err := io.ReadAll(res.body)
	if err != nil {
		t.Fatalf("read combined: %v", err)
	}
	if string(combined) != stream {
		t.Fatal("clean stream replay is not byte-exact")
	}
}

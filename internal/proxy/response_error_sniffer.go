package proxy

import (
	"bytes"
	"encoding/json"
	"io"
	"strings"
	"sync"
	"time"

	"github.com/buger/jsonparser"
)

// ─────────────────────────────────────────────────────────────────────────────
// Error-in-success-body detection ("success sniffer")
//
// Some upstreams — most notably Pollinations — answer with HTTP 200 OK even
// when the request actually FAILED, embedding the error message in place of
// the model's answer:
//
//	"The account behind this API key doesn't have enough credits. Please
//	 [top up](https://enter.pollinations.ai/top-up?ref=agent_low_balance_topup)
//	 or [complete a quest](https://enter.pollinations.ai/quests?ref=agent_low_balance_quests),
//	 then try again. ..."
//
// To the gateway this is indistinguishable from a successful completion: the
// client receives the marketing text as the "model answer", success telemetry
// is recorded, and the pool never rotates to a credential that still has
// credits. This module classifies 2xx bodies BEFORE anything is written to
// the client and converts detected error carriers into a synthetic
// 402 Payment Required, which flows through the existing retry/failover
// machinery (penalize → rotate → exact-model cross-provider fallback →
// canonical OpenAI error envelope at total exhaustion).
//
// Detection is three-tiered and tuned for near-zero false positives:
//
//  1. Structural — a top-level JSON "error" object/string, or success:false
//     with no completion content. OpenAI-compatible providers never emit
//     these on a genuine completion.
//  2. Tier-1     — exact high-confidence signatures (Pollinations referral
//     links, top-up/quests URLs, the verbatim low-balance opener). These can
//     never legitimately appear inside a real model answer, so they match
//     anywhere in the body, at any length, including mid-stream.
//  3. Tier-2     — a generic credit/balance/quota classifier for unknown
//     providers pulling the same trick. It only fires on the WHOLE response
//     text (≤600 chars) and requires at least two distinct billing-family
//     keywords (credits, balance, quota, top up, …), one problem-state
//     marker (insufficient, not enough, too low, exceeded, …) AND one
//     operator-signal (api key, https://, try again, your account, …). A
//     legitimate assistant answer would have to be composed entirely of
//     billing-error phrasing in under 600 characters to be misclassified.
// ─────────────────────────────────────────────────────────────────────────────

// Sniffing limits.
const (
	// nonStreamSniffLimit bounds how many leading bytes of a non-stream 2xx
	// body are inspected. Error carriers are tiny (a few hundred bytes); any
	// body larger than this is treated as a genuine completion.
	nonStreamSniffLimit = 64 * 1024

	// streamSniffRawCap bounds the raw bytes buffered while sniffing a
	// streaming (SSE/NDJSON/bare-JSON/plain-text) response before committing
	// to clean. Everything buffered is replayed byte-exactly afterwards.
	streamSniffRawCap = 32 * 1024

	// streamSniffDeadline bounds how long the sniffer may delay the first
	// client-visible byte of a stream when the opening content contains
	// billing watch-keywords (suspicious-but-not-conclusive text).
	streamSniffDeadline = 3 * time.Second

	// streamSniffCommitChars: once this many characters of extracted content
	// have accumulated without any error signature, the stream is
	// definitively clean — real completions keep flowing, error carriers are
	// never this long.
	streamSniffCommitChars = 600

	// wholeBodyErrorMaxChars is the maximum length for tier-2 classification:
	// only short whole-response texts can be generic error carriers.
	wholeBodyErrorMaxChars = 600
)

// streamFirstByteIdleTimeout bounds how long a single blocking Read inside
// sniffStreamForError may wait for upstream bytes. Real providers answer with
// the first SSE event within seconds; a stream that produces nothing within
// this window is declared stalled and the request rotates to another
// credential (observed in production: 52xueai accepted the request, then sat
// silent for 109s until the client's intermediary killed the connection).
// Package-level var (not const) so tests can shrink it.
var streamFirstByteIdleTimeout = 90 * time.Second

// successBodyError is the verdict returned when a 2xx body is classified as
// an error carrier.
type successBodyError struct {
	// Reason is a short machine-readable classification (for logs/alerts).
	Reason string
	// Message is the most informative human-readable text extracted from the
	// upstream body (embedded into the synthetic error envelope).
	Message string
}

// SyntheticBody renders a canonical OpenAI error envelope for this verdict.
// formatOpenAIError passes well-formed envelopes through unchanged, so at
// total-exhaustion the client always sees a clean, meaningful error instead
// of the upstream's marketing text. The envelope keywords also feed the
// isDepletedAccount heuristics in executeWithRetry (15s transient cooldown
// + immediate rotation).
func (e *successBodyError) SyntheticBody() []byte {
	code := "insufficient_credits"
	typ := "insufficient_quota"
	if e.Reason == "json_error_object" || e.Reason == "success_false" {
		code = "upstream_error_in_success_body"
		typ = "upstream_error"
	}

	msg := "Upstream provider returned an error message inside a 200 OK response — " +
		"converted to a retryable failure so the gateway can rotate to a healthy credential."
	if snippet := truncateSnippet(e.Message, 400); snippet != "" {
		msg += " Original response: " + snippet
	}

	body, _ := json.Marshal(map[string]any{
		"error": map[string]any{
			"message": msg,
			"type":    typ,
			"param":   nil,
			"code":    code,
		},
	})
	return body
}

// truncateSnippet trims a snippet for embedding in logs/synthetic bodies.
func truncateSnippet(s string, max int) string {
	s = strings.TrimSpace(s)
	if len(s) > max {
		s = s[:max] + "…"
	}
	return s
}

// normalizeForMatch prepares text for signature matching: curly/typographic
// quotes folded to ASCII, whitespace collapsed, lowercased. Upstream error
// messages frequently use typographic apostrophes ("doesn’t") that would
// otherwise evade byte-exact phrase matching.
func normalizeForMatch(s string) string {
	var b strings.Builder
	b.Grow(len(s))
	for _, r := range s {
		switch r {
		case '\u2019', '\u2018', '\u02bc': // ’ ‘ ʼ
			b.WriteByte('\'')
		case '\u201c', '\u201d': // “ ”
			b.WriteByte('"')
		case '\u2013', '\u2014': // – —
			b.WriteByte('-')
		case '\u00a0': // non-breaking space
			b.WriteByte(' ')
		case '\u200b', '\ufeff': // zero-width characters
			// dropped
		default:
			b.WriteRune(r)
		}
	}
	// strings.Fields collapses every run of whitespace (including newlines)
	// into single spaces and trims the result.
	return strings.ToLower(strings.Join(strings.Fields(b.String()), " "))
}

// tier1Signatures are exact, high-confidence error signatures. They may be
// matched anywhere in the body, at any length, at any time (including
// mid-stream before the response completes).
var tier1Signatures = []string{
	// Pollinations low-balance agent referral links (top-up and quest variants).
	"ref=agent_low_balance",
	"enter.pollinations.ai/top-up",
	"enter.pollinations.ai/quests",
	// Pollinations verbatim low-balance opener (quote style normalized).
	"the account behind this api key doesn't have enough credits",
	"the account behind this api key does not have enough credits",
}

// tier1SignatureHit scans normalized text for a known signature, plus the
// combined pollinations + credits check.
func tier1SignatureHit(norm string) bool {
	for _, sig := range tier1Signatures {
		if strings.Contains(norm, sig) {
			return true
		}
	}
	if strings.Contains(norm, "pollinations") &&
		(strings.Contains(norm, "not enough credits") ||
			strings.Contains(norm, "doesn't have enough credits") ||
			strings.Contains(norm, "does not have enough credits") ||
			strings.Contains(norm, "complete a quest")) {
		return true
	}
	return false
}

// Tier-2 classifier word lists. All matching happens on normalized text.
var (
	// familyKeywords: billing/balance vocabulary. At least TWO distinct hits
	// are required so single-word coincidences ("payment" in an educational
	// answer) never classify on their own.
	familyKeywords = []string{
		"credit", "balance", "quota", "billing", "top up", "top-up", "topup",
		"subscription", "usage limit", "api key", "api_key", "payment",
		"account", "plan", "add funds",
	}
	// problemStateMarkers: the text must state an actual problem, not merely
	// mention billing vocabulary ("your balance is too low" vs "balance is
	// a concept in accounting").
	problemStateMarkers = []string{
		"insufficient", "not enough", "out of credit", "out of funds",
		"exceed", "exhausted", "too low", "depleted", "suspended",
		"deactivated", "hard limit", "limit reached", "reached your limit",
		"upgrade your", "payment required", "complete a quest", "expired",
	}
	// operatorSignals: the text must be addressed to the app operator (the
	// holder of the API key), not read like prose for an end user. Signals
	// that appear naturally in ordinary answers about billing ("your quota",
	// "your usage") are deliberately excluded — real carriers always carry
	// at least one of the unambiguous signals below.
	operatorSignals = []string{
		"api key", "api_key", "http://", "https://", "try again", "contact",
		"account behind", "this account", "your account", "your credit",
		"your balance", "your api", "your subscription",
	}
	// watchKeywords trigger extended stream buffering (suspicious content —
	// not yet conclusive). Superset of the family and state lists plus the
	// Pollinations opener prefix.
	watchKeywords = func() []string {
		kw := append([]string{}, familyKeywords...)
		kw = append(kw, problemStateMarkers...)
		return append(kw, "the account behind")
	}()
)

// looksLikeBalanceErrorText classifies a complete, whole-response text as a
// generic credit/balance/quota error carrier addressed to the app operator.
func looksLikeBalanceErrorText(text string) bool {
	norm := strings.TrimSpace(normalizeForMatch(text))
	if norm == "" || len(norm) > wholeBodyErrorMaxChars {
		return false
	}
	family := 0
	for _, kw := range familyKeywords {
		if strings.Contains(norm, kw) {
			family++
		}
	}
	if family < 2 {
		return false
	}
	state := 0
	for _, m := range problemStateMarkers {
		if strings.Contains(norm, m) {
			state++
		}
	}
	if state < 1 {
		return false
	}
	for _, sig := range operatorSignals {
		if strings.Contains(norm, sig) {
			return true
		}
	}
	return false
}

// structuralSuccessError detects unambiguous error envelopes in JSON bodies:
// a top-level "error" object/non-empty string, or success:false with no
// completion content anywhere. Null/empty/false error fields are ignored —
// some APIs include them as no-op placeholders on successful responses.
func structuralSuccessError(raw []byte) (msg string, hit bool) {
	if len(raw) == 0 {
		return "", false
	}
	if v, dt, _, err := jsonparser.Get(raw, "error"); err == nil {
		switch dt {
		case jsonparser.Object:
			if m, mErr := jsonparser.GetString(v, "message"); mErr == nil && m != "" {
				return m, true
			}
			if c, cErr := jsonparser.GetString(v, "code"); cErr == nil && c != "" {
				return c, true
			}
		case jsonparser.String:
			if s, _ := jsonparser.GetString(raw, "error"); s != "" {
				return s, true
			}
		}
	}
	// success:false with no completion payload anywhere (Cloudflare-style
	// {success, errors} envelopes and friends).
	if success, err := jsonparser.GetBoolean(raw, "success"); err == nil && !success {
		hasContent := false
		for _, key := range []string{"choices", "response", "content", "result", "data", "output"} {
			if _, _, _, kErr := jsonparser.Get(raw, key); kErr == nil {
				hasContent = true
				break
			}
		}
		if !hasContent {
			if m, mErr := jsonparser.GetString(raw, "errors", "[0]", "message"); mErr == nil && m != "" {
				return m, true
			}
			if m, mErr := jsonparser.GetString(raw, "message"); mErr == nil && m != "" {
				return m, true
			}
			return "upstream reported success=false inside a 200 OK response", true
		}
	}
	return "", false
}

// candidateExtractionPaths enumerates every response shape the gateway
// proxies (OpenAI chat/completions/legacy, Anthropic, Ollama, Gemini,
// FastAPI detail errors, …). Only STRING values are returned; arrays and
// objects (multimodal content blocks, embeddings) are skipped.
var candidateExtractionPaths = [][]string{
	{"choices", "[0]", "message", "content"},
	{"choices", "[0]", "delta", "content"},
	{"choices", "[0]", "text"},
	{"message", "content"},
	{"message", "content", "[0]", "text"},
	{"content"},
	{"content", "[0]", "text"},
	{"response"},
	{"detail"},
	{"message"},
	{"candidates", "[0]", "content", "parts", "[0]", "text"},
	{"output_text"},
}

// extractCandidateTexts pulls human-readable completion/error text out of any
// supported response envelope.
func extractCandidateTexts(raw []byte) []string {
	if len(raw) == 0 {
		return nil
	}
	var out []string
	seen := map[string]bool{}
	for _, p := range candidateExtractionPaths {
		if s, err := jsonparser.GetString(raw, p...); err == nil && s != "" {
			if !seen[s] {
				seen[s] = true
				out = append(out, s)
			}
		}
	}
	return out
}

// stripHTMLTags removes tags from an HTML body so error text inside billing
// interstitials becomes classifiable. Tag boundaries become spaces so
// phrases spanning tags ("not</p><p>enough") still normalize correctly.
func stripHTMLTags(raw []byte) string {
	var b strings.Builder
	b.Grow(len(raw))
	inTag := false
	for _, r := range string(raw) {
		switch {
		case r == '<':
			inTag = true
			b.WriteByte(' ')
		case r == '>':
			inTag = false
			b.WriteByte(' ')
		case !inTag:
			b.WriteRune(r)
		}
	}
	return b.String()
}

// isJSONishContent reports whether a normalized (lowercase, parameter-free)
// content type may carry JSON text.
func isJSONishContent(ct string) bool {
	return ct == "application/json" ||
		ct == "application/json-seq" ||
		ct == "application/jsonl" ||
		ct == "application/x-ndjson" ||
		ct == "application/problem+json" ||
		strings.HasPrefix(ct, "text/json") ||
		strings.HasSuffix(ct, "+json")
}

// isSniffableContentType reports whether bodies of this type may carry
// textual error messages and are worth inspecting. Binary media (images,
// audio, video, archives) are never sniffed. An empty/unknown content type
// is sniffed too, gated by isLikelyTextual on the actual bytes.
func isSniffableContentType(contentType string) bool {
	ct := strings.ToLower(strings.TrimSpace(contentType))
	if i := strings.IndexByte(ct, ';'); i >= 0 {
		ct = strings.TrimSpace(ct[:i])
	}
	if ct == "" {
		return true
	}
	return strings.HasPrefix(ct, "text/") || isJSONishContent(ct)
}

// isLikelyTextual reports whether the leading bytes look like printable text
// (used when the upstream omitted Content-Type entirely).
func isLikelyTextual(prefix []byte) bool {
	if len(prefix) == 0 {
		return false
	}
	sample := prefix
	if len(sample) > 512 {
		sample = sample[:512]
	}
	printable := 0
	for _, b := range sample {
		if b == '\n' || b == '\r' || b == '\t' {
			continue
		}
		if (b >= 0x20 && b < 0x7f) || b >= 0x80 { // ASCII printable or UTF-8 multibyte
			printable++
		}
	}
	return printable*10/len(sample) >= 7
}

// readPrefix consumes at most `limit+one-chunk` leading bytes from r and
// reports whether the body continues beyond the returned prefix. Every byte
// consumed MUST be replayed to the client byte-exactly (see prefixReplayReader).
func readPrefix(r io.Reader, limit int) (prefix []byte, truncated bool) {
	buf := make([]byte, 0, 64*1024)
	tmp := make([]byte, 32*1024)
	for {
		n, err := r.Read(tmp)
		if n > 0 {
			buf = append(buf, tmp[:n]...)
			if len(buf) > limit {
				return buf, true
			}
		}
		if err != nil {
			return buf, false // EOF or transport error — nothing more to read
		}
	}
}

// prefixReplayReader replays the sniffed prefix and then continues with the
// original body. A transport error observed while sniffing is re-surfaced
// after the prefix so downstream readers see the exact same failure.
type prefixReplayReader struct {
	prefix []byte
	rest   io.Reader
	err    error
}

func (r *prefixReplayReader) Read(p []byte) (int, error) {
	if len(r.prefix) > 0 {
		n := copy(p, r.prefix)
		r.prefix = r.prefix[n:]
		return n, nil
	}
	if r.err != nil {
		return 0, r.err
	}
	return r.rest.Read(p)
}

// detectSuccessBodyError classifies a (possibly truncated) 2xx body prefix.
//
//	raw         — leading body bytes (up to nonStreamSniffLimit for
//	               non-streaming responses, or the full buffered stream prefix).
//	contentType — upstream Content-Type ("" when unknown).
//	complete    — true when raw contains the ENTIRE body (EOF within the
//	               sniff limit). Tier-2 heuristics only run on complete
//	               bodies so a truncated legit completion can never be
//	               misclassified.
func detectSuccessBodyError(raw []byte, contentType string, complete bool) *successBodyError {
	if len(raw) == 0 {
		return nil
	}

	ct := strings.ToLower(strings.TrimSpace(contentType))
	baseCT := ct
	if i := strings.IndexByte(baseCT, ';'); i >= 0 {
		baseCT = strings.TrimSpace(baseCT[:i])
	}

	// 1. Structural: top-level error envelope / success:false.
	if msg, hit := structuralSuccessError(raw); hit {
		reason := "json_error_object"
		if !bytes.Contains(raw, []byte(`"error"`)) {
			reason = "success_false"
		}
		return &successBodyError{Reason: reason, Message: msg}
	}

	// 2. Tier-1: exact signatures anywhere in the raw bytes. Markers contain
	// no JSON-escapable characters, so they appear verbatim inside JSON
	// string values as well.
	rawNorm := normalizeForMatch(string(raw))
	if tier1SignatureHit(rawNorm) {
		return &successBodyError{Reason: "pollinations_low_balance", Message: truncateSnippet(string(raw), 800)}
	}

	// 3. Tier-2 generic classifier — complete bodies only.
	if complete {
		candidates := extractCandidateTexts(raw)
		isHTML := strings.HasPrefix(baseCT, "text/html") ||
			(!isJSONishContent(baseCT) && !strings.HasPrefix(baseCT, "text/") && strings.Contains(rawNorm, "<html"))
		if isHTML {
			candidates = append(candidates, stripHTMLTags(raw))
		}
		for _, cand := range candidates {
			if tier1SignatureHit(normalizeForMatch(cand)) {
				return &successBodyError{Reason: "pollinations_low_balance", Message: cand}
			}
			if looksLikeBalanceErrorText(cand) {
				return &successBodyError{Reason: "credit_balance_error_text", Message: cand}
			}
		}
		if len(candidates) == 0 {
			// No structured content extracted — classify the raw text itself
			// (plain-text / markdown / HTML carriers).
			if looksLikeBalanceErrorText(string(raw)) {
				return &successBodyError{Reason: "credit_balance_error_text", Message: truncateSnippet(string(raw), 800)}
			}
		}
	}
	return nil
}

// extractEventContent pulls delta/message content and top-level error values
// out of a single SSE/NDJSON event payload.
func extractEventContent(evt []byte) (content, errMsg string) {
	if len(evt) == 0 {
		return "", ""
	}
	if v, dt, _, err := jsonparser.Get(evt, "error"); err == nil {
		switch dt {
		case jsonparser.Object:
			if m, mErr := jsonparser.GetString(v, "message"); mErr == nil && m != "" {
				errMsg = m
			} else if len(v) > 2 { // non-empty object without a message field
				errMsg = "upstream error object inside stream event"
			}
		case jsonparser.String:
			if sv, _ := jsonparser.GetString(evt, "error"); sv != "" {
				errMsg = sv
			}
		}
	}
	for _, p := range [][]string{
		{"choices", "[0]", "delta", "content"},
		{"choices", "[0]", "message", "content"},
		{"choices", "[0]", "text"},
		{"response"},
		{"candidates", "[0]", "content", "parts", "[0]", "text"},
		{"message", "content", "[0]", "text"},
		{"content"},
	} {
		if sv, err := jsonparser.GetString(evt, p...); err == nil && sv != "" {
			return sv, errMsg
		}
	}
	return "", errMsg
}

// streamSniffResult carries the outcome of sniffing a streaming response.
// When verdict is nil the stream is clean; replay holds every byte consumed
// while sniffing and MUST be replayed to the client byte-exactly.
// body provides the unified, continuous stream (replay prefix followed by pump reads)
// so that no concurrent reads occur and no bytes are lost.
type streamSniffResult struct {
	verdict *successBodyError
	replay  []byte
	// stalled is set when the upstream went silent mid-sniff (no byte within
	// streamFirstByteIdleTimeout). The caller surfaces this as a retryable
	// transport failure so the rotation loop moves to the next credential
	// instead of hanging the client until its own timeout fires.
	stalled bool
	body    io.ReadCloser
}

// streamSniffer holds the incremental state while sniffing a stream.
type streamSniffer struct {
	raw      []byte
	content  strings.Builder
	eventErr string
	sawDone  bool
	sawSSE   bool
	complete int // offset within raw of bytes already processed as whole lines
}

// ingest appends a chunk and processes every newly completed line.
func (s *streamSniffer) ingest(chunk []byte) {
	s.raw = append(s.raw, chunk...)
	for {
		idx := bytes.IndexByte(s.raw[s.complete:], '\n')
		if idx < 0 {
			break
		}
		line := s.raw[s.complete : s.complete+idx]
		s.complete += idx + 1
		s.processLine(line)
	}
}

// processTail processes the final unterminated line (EOF without newline).
func (s *streamSniffer) processTail() {
	if s.complete < len(s.raw) {
		line := s.raw[s.complete:]
		s.complete = len(s.raw)
		s.processLine(line)
	}
}

// processLine classifies one complete stream line (SSE data event, NDJSON
// record, or plain text).
func (s *streamSniffer) processLine(line []byte) {
	trimmed := bytes.TrimSpace(line)
	if len(trimmed) == 0 {
		return // SSE event delimiter
	}
	if bytes.HasPrefix(trimmed, []byte("data:")) {
		s.sawSSE = true
		payload := bytes.TrimSpace(trimmed[5:])
		if bytes.Equal(payload, []byte("[DONE]")) {
			s.sawDone = true
			return
		}
		content, errMsg := extractEventContent(payload)
		if errMsg != "" && s.eventErr == "" {
			s.eventErr = errMsg
		}
		if content != "" {
			s.content.WriteString(content)
		}
		return
	}
	if trimmed[0] == ':' {
		return // SSE keep-alive comment
	}
	if trimmed[0] == '{' || trimmed[0] == '[' {
		// NDJSON record (Ollama, Gemini) or a bare JSON document chunk.
		content, errMsg := extractEventContent(trimmed)
		if errMsg != "" && s.eventErr == "" {
			s.eventErr = errMsg
		}
		if content != "" {
			s.content.WriteString(content)
		}
		return
	}
	// Plain text line.
	s.content.Write(trimmed)
	s.content.WriteByte(' ')
}

// bareJSONContent attempts content extraction from a partially buffered bare
// (non-SSE) JSON document — some providers answer stream:true with a full
// JSON body instead of an SSE stream.
func (s *streamSniffer) bareJSONContent() string {
	if s.sawSSE || len(s.raw) == 0 || s.raw[0] != '{' {
		return ""
	}
	for _, p := range [][]string{
		{"choices", "[0]", "message", "content"},
		{"response"},
		{"content"},
		{"detail"},
		{"message"},
	} {
		if v, err := jsonparser.GetString(s.raw, p...); err == nil && v != "" {
			return v
		}
	}
	return ""
}

// effectiveContent returns the best available accumulated stream text.
func (s *streamSniffer) effectiveContent() string {
	if s.content.Len() > 0 {
		return s.content.String()
	}
	return s.bareJSONContent()
}

// hasWatchKeyword reports whether the stream content so far contains billing
// vocabulary worth holding the buffer for.
func hasWatchKeyword(content string) bool {
	norm := normalizeForMatch(content)
	for _, kw := range watchKeywords {
		if strings.Contains(norm, kw) {
			return true
		}
	}
	return false
}

// sniffStreamForError buffers the opening of a streaming 2xx response (SSE,
// NDJSON, bare JSON or plain text) and decides — before a single byte is
// written to the client — whether the stream is actually an error carrier.
//
// Decision rules, evaluated after every read:
//   - A JSON error object/string inside an SSE/NDJSON event  → immediate error.
//   - A tier-1 signature in the raw bytes or content        → immediate error.
//   - Stream finished ([DONE] or EOF)                       → final classification
//     (tier-2 generic heuristics run on the complete accumulated text).
//   - ≥600 chars of content with no signature              → definitively clean.
//   - Raw buffer at cap                                      → clean (byte-exact replay).
//   - Content with no billing watch-keywords                 → clean immediately
//     (normal streams commit on their first content event — TTFB preserved).
//   - Watch-keywords present but nothing conclusive          → keep buffering,
//     bounded by the raw cap and the deadline.
//
// streamPipeReader replays the sniffed prefix and then transparently
// consumes remaining chunks produced by the sniffer's pump goroutine.
// This prevents concurrent reads on the underlying body and ensures
// zero bytes are lost between sniffing and streaming relay.
type streamPipeReader struct {
	prefix    []byte
	reqRead   chan<- struct{}
	reads     <-chan readResult
	leftover  []byte
	closer    io.Closer
	done      chan struct{}
	closeOnce sync.Once
	err       error
}

func (r *streamPipeReader) Read(p []byte) (int, error) {
	if len(r.prefix) > 0 {
		n := copy(p, r.prefix)
		r.prefix = r.prefix[n:]
		return n, nil
	}
	if len(r.leftover) > 0 {
		n := copy(p, r.leftover)
		r.leftover = r.leftover[n:]
		return n, nil
	}
	if r.err != nil {
		return 0, r.err
	}
	select {
	case <-r.done:
		return 0, io.EOF
	case r.reqRead <- struct{}{}:
	}

	select {
	case <-r.done:
		return 0, io.EOF
	case res, ok := <-r.reads:
		if !ok {
			return 0, io.EOF
		}
		if len(res.buf) > 0 {
			n := copy(p, res.buf)
			if n < len(res.buf) {
				r.leftover = res.buf[n:]
			}
			if res.err != nil {
				r.err = res.err
			}
			return n, nil
		}
		if res.err != nil {
			return 0, res.err
		}
		return 0, io.EOF
	}
}

func (r *streamPipeReader) Close() error {
	r.closeOnce.Do(func() {
		close(r.done)
	})
	if r.closer != nil {
		return r.closer.Close()
	}
	return nil
}

type readResult struct {
	buf []byte
	err error
}

// sniffStreamForError buffers the opening of a streaming 2xx response (SSE,
// Ollama NDJSON, bare JSON) to detect HTTP 200 OK error bodies.
func sniffStreamForError(body io.Reader) *streamSniffResult {
	s := &streamSniffer{}
	start := time.Now()
	spins := 0

	var closer io.Closer
	if c, ok := body.(io.Closer); ok {
		closer = c
	}

	reqRead := make(chan struct{})
	reads := make(chan readResult)
	done := make(chan struct{})

	go func() {
		tmp := make([]byte, 16*1024)
		for {
			select {
			case <-reqRead:
			case <-done:
				return
			}
			n, err := body.Read(tmp)
			var buf []byte
			if n > 0 {
				buf = make([]byte, n)
				copy(buf, tmp[:n])
			}
			select {
			case reads <- readResult{buf: buf, err: err}:
			case <-done:
				return
			}
			if err != nil {
				return
			}
		}
	}()

	makePipe := func() *streamPipeReader {
		return &streamPipeReader{
			prefix:  s.raw,
			reqRead: reqRead,
			reads:   reads,
			closer:  closer,
			done:    done,
		}
	}

	var n int
	var readErr error
	for {
		select {
		case reqRead <- struct{}{}:
		case <-done:
			return &streamSniffResult{verdict: nil, replay: s.raw, stalled: true, body: makePipe()}
		}

		select {
		case r := <-reads:
			n, readErr = len(r.buf), r.err
			if n > 0 {
				s.ingest(r.buf)
			}
		case <-time.After(streamFirstByteIdleTimeout):
			// Upstream accepted the request but went silent before producing
			// the first byte — a retryable transport failure, not an error
			// body. The rotation loop moves to the next credential.
			close(done)
			return &streamSniffResult{verdict: nil, replay: s.raw, stalled: true, body: makePipe()}
		}

		// Unambiguous error events fire immediately.
		if s.eventErr != "" {
			return &streamSniffResult{
				verdict: &successBodyError{Reason: "json_error_object", Message: s.eventErr},
				replay:  s.raw,
				body:    makePipe(),
			}
		}

		contentText := s.effectiveContent()

		// Tier-1 signatures anywhere in the raw bytes or accumulated content.
		if tier1SignatureHit(normalizeForMatch(string(s.raw))) ||
			(contentText != "" && tier1SignatureHit(normalizeForMatch(contentText))) {
			msg := contentText
			if msg == "" {
				msg = truncateSnippet(string(s.raw), 800)
			}
			return &streamSniffResult{
				verdict: &successBodyError{Reason: "pollinations_low_balance", Message: msg},
				replay:  s.raw,
				body:    makePipe(),
			}
		}

		// Stream complete: final classification on the whole text.
		if s.sawDone || readErr != nil {
			s.processTail()
			contentText = s.effectiveContent()
			if contentText != "" && looksLikeBalanceErrorText(contentText) {
				return &streamSniffResult{
					verdict: &successBodyError{Reason: "credit_balance_error_text", Message: contentText},
					replay:  s.raw,
					body:    makePipe(),
				}
			}
			// Bare JSON documents get the full non-stream classifier.
			if !s.sawSSE && len(s.raw) > 0 && s.raw[0] == '{' {
				if v := detectSuccessBodyError(s.raw, "", true); v != nil {
					return &streamSniffResult{verdict: v, replay: s.raw, body: makePipe()}
				}
			}
			return &streamSniffResult{verdict: nil, replay: s.raw, body: makePipe()}
		}

		// Clean-commit conditions (buffered bytes are replayed byte-exactly).
		if len(contentText) >= streamSniffCommitChars {
			return &streamSniffResult{verdict: nil, replay: s.raw, body: makePipe()}
		}
		if len(s.raw) >= streamSniffRawCap {
			return &streamSniffResult{verdict: nil, replay: s.raw, body: makePipe()}
		}
		if contentText != "" && !hasWatchKeyword(contentText) {
			return &streamSniffResult{verdict: nil, replay: s.raw, body: makePipe()}
		}
		if time.Since(start) > streamSniffDeadline {
			return &streamSniffResult{verdict: nil, replay: s.raw, body: makePipe()}
		}

		if n == 0 {
			spins++
			if spins > 4096 { // pathological reader — bail out cleanly
				return &streamSniffResult{verdict: nil, replay: s.raw, body: makePipe()}
			}
		}
	}
}

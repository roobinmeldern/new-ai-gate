package proxy

import (
	"bytes"
	"strings"
	"testing"
)

// TestModelDisallowsSamplingParams pins the model gate for the Claude
// generation that hard-rejects sampling parameters. Opus 4.1 and older still
// honour temperature and must never match.
func TestModelDisallowsSamplingParams(t *testing.T) {
	cases := []struct {
		model string
		want  bool
	}{
		{"claude-opus-5", true},
		{"claude-opus-5-thinking", true},
		{"claude-opus-5-5", true},
		{"anthropic/claude-opus-5", true},
		{"cc/claude-opus-5", true},
		{"anthropic/claude-opus-4.5", true},
		{"claude-opus-4-5-20251101", true},
		{"Claude-Opus-5", true},             // case-insensitive
		{"claude-opus-4-1-20250805", false}, // Opus 4.1 still honours temperature
		{"claude-sonnet-5", false},          // no rejection observed — untouched
		{"gpt-5", false},
		{"mimo-v2.6-flash", false},
		{"", false},
	}
	for _, tc := range cases {
		if got := modelDisallowsSamplingParams(tc.model); got != tc.want {
			t.Errorf("modelDisallowsSamplingParams(%q) = %v, want %v", tc.model, got, tc.want)
		}
	}
}

// TestStripSamplingParams verifies temperature/top_p/top_k removal while
// every other field survives byte-exact, plus the allocation-free clean path.
func TestStripSamplingParams(t *testing.T) {
	in := `{"model":"claude-opus-5","messages":[{"role":"user","content":"hi"}],"temperature":0.7,"top_p":0.9,"top_k":40,"max_tokens":1024,"stream":true}`
	out := stripSamplingParams([]byte(in))

	if bytes.Contains(out, []byte(`"temperature"`)) ||
		bytes.Contains(out, []byte(`"top_p"`)) ||
		bytes.Contains(out, []byte(`"top_k"`)) {
		t.Fatalf("sampling params survived the strip: %s", out)
	}
	for _, keep := range []string{
		`"model":"claude-opus-5"`,
		`"role":"user"`,
		`"content":"hi"`,
		`"max_tokens":1024`,
		`"stream":true`,
	} {
		if !strings.Contains(string(out), keep) {
			t.Errorf("field %s must be preserved byte-exact, got: %s", keep, out)
		}
	}

	// Clean request: the original slice must come back unchanged.
	clean := []byte(`{"model":"claude-opus-5","messages":[]}`)
	if got := stripSamplingParams(clean); !bytes.Equal(got, clean) {
		t.Errorf("clean body must be returned unchanged, got: %s", got)
	}

	// Idempotent: a second pass must be a no-op.
	if got := stripSamplingParams(out); !bytes.Equal(got, out) {
		t.Errorf("strip must be idempotent, got: %s", got)
	}

	// Non-JSON body passes through untouched.
	notJSON := []byte("not-a-json-body")
	if got := stripSamplingParams(notJSON); !bytes.Equal(got, notJSON) {
		t.Errorf("non-JSON body must be returned unchanged, got: %s", got)
	}
}

// TestParseUnsupportedSamplingParam replays the exact production 400 body
// (gateway-2026-09-22.log, Cloudflare code 7003 for anthropic/claude-opus-5)
// and guards against over-eager parsing.
func TestParseUnsupportedSamplingParam(t *testing.T) {
	body := []byte(`{"errors":[{"message":"Model execution failed (User Input Error): ` +
		"Validation error at temperature: `temperature` is not supported on this model. " +
		`Remove it from your request. See https://docs.anthropic.com/en/docs/about-claude/models/migration-guide for details.","code":7003}],"success":false,"result":{},"messages":[]}`)

	param, ok := parseUnsupportedSamplingParam(body)
	if !ok || param != "temperature" {
		t.Fatalf("expected temperature, got %q ok=%v", param, ok)
	}

	// Structural params are never strippable, even if the upstream names them.
	if _, ok := parseUnsupportedSamplingParam([]byte("`messages` is not supported on this model")); ok {
		t.Error("structural params must never be strippable")
	}
	if _, ok := parseUnsupportedSamplingParam([]byte("`model` is not supported on this model")); ok {
		t.Error("structural params must never be strippable (model)")
	}

	// Unrelated 400 bodies must not parse.
	if _, ok := parseUnsupportedSamplingParam([]byte(`{"error":{"message":"Invalid model or alias"}}`)); ok {
		t.Error("unrelated body must not parse")
	}
	if _, ok := parseUnsupportedSamplingParam(nil); ok {
		t.Error("empty body must not parse")
	}
}

// TestIsTransientUpstream400 replays the exact production 400 body from the
// inference provider (claude-opus-5) that used to kill rotation mid-pool.
func TestIsTransientUpstream400(t *testing.T) {
	if !isTransientUpstream400([]byte(`{"error":{"message":"The upstream provider returned an error while processing this request.","type":"upstream_error"}}`)) {
		t.Error("inference upstream failure must classify as transient")
	}
	if isTransientUpstream400([]byte("Validation error at temperature: `temperature` is not supported on this model")) {
		t.Error("payload rejections must not classify as transient")
	}
	if isTransientUpstream400([]byte(`{"error":{"message":"Invalid model or alias"}}`)) {
		t.Error("unrelated 400 bodies must not classify as transient")
	}
	if isTransientUpstream400(nil) {
		t.Error("empty body must not classify as transient")
	}
}

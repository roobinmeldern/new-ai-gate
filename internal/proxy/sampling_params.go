package proxy

import (
	"bytes"
	"encoding/json"
	"regexp"
	"strings"
)

// ─────────────────────────────────────────────────────────────────────────────
// Claude "no sampling parameters" handling
//
// Production evidence (gateway-2026-09-22.log): Cloudflare Workers AI serving
// anthropic/claude-opus-5 rejects OpenAI sampling parameters with a hard 400:
//
//	{"errors":[{"message":"Model execution failed (User Input Error):
//	 Validation error at temperature: `temperature` is not supported on this
//	 model. Remove it from your request. See
//	 https://docs.anthropic.com/en/docs/about-claude/models/migration-guide
//	 for details.","code":7003}]}
//
// Clients that routinely send temperature/top_p (Cline, Cursor, the admin
// playground, the NewModelChecker probe) therefore burn two 400 attempts and
// hit the gateway's "payload schema error — aborting rotation" fast-fail, even
// though every pool credential is perfectly healthy. The same generation
// returns opaque transient 400s ("The upstream provider returned an error
// while processing this request.") that also must not abort rotation.
//
// Two defences, one per direction:
//
//   - Proactive (forwardRequest): for models of the Claude ≥ Opus 4.5
//     generation, strip temperature/top_p/top_k from the outbound body before
//     the request is ever sent. Harmless on aggregators (OpenRouter & friends
//     already drop them) and required on strict Anthropic-schema endpoints.
//
//   - Reactive (executeWithRetry): when any upstream answers 400 with an
//     explicit "«param» is not supported on this model" rejection, strip that
//     param from the request body and retry the same healthy credential
//     immediately. This self-heals future models/params the proactive list
//     does not know about.
// ─────────────────────────────────────────────────────────────────────────────

// claudeNoSamplingModelFragments matches the Claude generation documented in
// Anthropic's migration guide that removed support for sampling overrides.
// Opus 4.1 and older still honour temperature — do NOT add them here.
var claudeNoSamplingModelFragments = []string{
	"claude-opus-5",   // claude-opus-5, claude-opus-5-thinking, claude-opus-5-5, cc/claude-opus-5…
	"claude-opus-4.5", // dotted alias form
	"claude-opus-4-5", // dashed release form (claude-opus-4-5-20251101)
}

// strippableSamplingParams is the closed set of top-level request fields the
// reactive self-heal path may strip after an explicit upstream rejection.
// Deliberately excludes anything structural (model, messages, tools…) so a
// buggy or hostile upstream error message can never gut a request.
var strippableSamplingParams = map[string]bool{
	"temperature": true,
	"top_p":       true,
	"top_k":       true,
}

// modelDisallowsSamplingParams reports whether the model belongs to the Claude
// generation that hard-rejects OpenAI sampling parameters. Matching is
// substring-based so it covers provider routing prefixes (cc/…, anthropic/…)
// and suffix variants (-thinking, -5-5, dated releases).
func modelDisallowsSamplingParams(model string) bool {
	if model == "" {
		return false
	}
	m := strings.ToLower(model)
	for _, frag := range claudeNoSamplingModelFragments {
		if strings.Contains(m, frag) {
			return true
		}
	}
	return false
}

// stripSamplingParams removes temperature/top_p/top_k from an OpenAI-format
// chat completion body. It returns the original slice when nothing needs to
// change (the common clean-request path is allocation-free).
func stripSamplingParams(body []byte) []byte {
	stripped, changed := stripJSONTopLevelKeys(body, []string{"temperature", "top_p", "top_k"})
	if !changed {
		return body
	}
	return stripped
}

// stripJSONTopLevelKeys removes the given top-level keys from a JSON object
// body, preserving every other field byte-exact (json.RawMessage round-trip).
// It returns (newBody, true) only when at least one key was actually present
// and removed; otherwise the original body is returned unchanged.
func stripJSONTopLevelKeys(body []byte, keys []string) ([]byte, bool) {
	if len(body) == 0 {
		return body, false
	}
	// Fast presence probe before paying for the full parse.
	present := false
	for _, k := range keys {
		if bytes.Contains(body, []byte(`"`+k+`"`)) {
			present = true
			break
		}
	}
	if !present {
		return body, false
	}

	var payload map[string]json.RawMessage
	if err := json.Unmarshal(body, &payload); err != nil {
		return body, false // not a JSON object (e.g. multipart) — leave untouched
	}

	changed := false
	for _, k := range keys {
		if _, ok := payload[k]; ok {
			delete(payload, k)
			changed = true
		}
	}
	if !changed {
		return body, false
	}

	out, err := json.Marshal(payload)
	if err != nil {
		return body, false
	}
	return out, true
}

// unsupportedParamRe matches Anthropic-style rejection messages:
// "`temperature` is not supported on this model. Remove it from your request."
var unsupportedParamRe = regexp.MustCompile("`([a-zA-Z_][a-zA-Z0-9_]*)` is not supported on this model")

// parseUnsupportedSamplingParam extracts the sampling parameter name from an
// upstream 400 body that explicitly rejects it, e.g. Cloudflare's code-7003
// "Validation error at temperature: `temperature` is not supported on this
// model." Returns ok=false when the body does not name a strippable sampling
// parameter.
func parseUnsupportedSamplingParam(errBody []byte) (string, bool) {
	if len(errBody) == 0 {
		return "", false
	}
	m := unsupportedParamRe.FindSubmatch(errBody)
	if len(m) != 2 {
		return "", false
	}
	param := string(m[1])
	if !strippableSamplingParams[param] {
		return "", false
	}
	return param, true
}

// isTransientUpstream400 reports whether a 400 body actually describes an
// upstream-side processing failure ("The upstream provider returned an error
// while processing this request." — observed from the inference provider on
// claude-opus-5) rather than a client payload problem. Such failures are
// transient on the provider's side: rotating to another key can still succeed,
// so they must not count toward the payload-schema 400 fast-fail abort.
func isTransientUpstream400(errBody []byte) bool {
	if len(errBody) == 0 {
		return false
	}
	return strings.Contains(
		strings.ToLower(string(errBody)),
		"the upstream provider returned an error while processing this request",
	)
}

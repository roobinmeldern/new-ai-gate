package proxy

import (
	"bytes"

	"github.com/buger/jsonparser"
)

// nvidiaUnsupportedFields lists standard OpenAI Chat Completions request fields
// that NVIDIA NIM's API gateway does not accept. Sending them triggers an upstream
// 400 "Unsupported parameter(s)" error.
//
// The gateway's 400 fast-fail block (executeWithRetry) would otherwise break the
// retry loop after the very first attempt, producing a misleading 502
// gateway_exhaustion_error to the downstream client. Stripping these fields
// upstream prevents the validation failure entirely.
//
// This list covers the most common OpenAI-specific extensions that client SDKs
// and coding assistants (Cline, Continue, Kilo Code, …) silently include in
// every request regardless of the target provider.
var nvidiaUnsupportedFields = []string{
	"stream_options",      // OpenAI streaming usage metadata (usage_on_every_chunk etc.)
	"logprobs",            // OpenAI per-token log probabilities
	"top_logprobs",        // OpenAI top-N log probabilities
	"service_tier",        // OpenAI priority routing tier
	"store",               // OpenAI conversation storage flag
	"metadata",            // OpenAI request metadata object
	"parallel_tool_calls", // OpenAI parallel function-calling flag
	"logit_bias",          // OpenAI per-token logit bias map
	"suffix",              // OpenAI text-completion suffix
}

// sanitizeNvidiaRequest removes OpenAI-only request fields that NVIDIA NIM's strict
// JSON schema rejects with HTTP 400 "Unsupported parameter(s)".
//
// Implementation follows the same pattern as sanitizeSarvamRequest and
// sanitizePuterRequest:
//  1. Fast byte-scan probe — if none of the fields are present, return the
//     original slice unchanged (zero allocation on the hot-path).
//  2. jsonparser.Delete pass — removes only the keys that are present,
//     producing a trimmed JSON object without a full unmarshal/marshal cycle.
//
// The function is called in forwardRequest, gated on cred.Provider == "nvidia",
// so it covers both the "nvidia/…" prefixed routing form and any clean alias
// that resolves to an NVIDIA credential.
func sanitizeNvidiaRequest(body []byte) []byte {
	needed := false
	for _, f := range nvidiaUnsupportedFields {
		if bytes.Contains(body, []byte(`"`+f+`"`)) {
			needed = true
			break
		}
	}

	// Check if temperature is present and non-positive (NVIDIA NIM requires temperature > 0)
	if !needed && bytes.Contains(body, []byte(`"temperature"`)) {
		if temp, err := jsonparser.GetFloat(body, "temperature"); err == nil && temp <= 0 {
			needed = true
		}
	}

	if !needed {
		return body
	}

	out := body
	for _, f := range nvidiaUnsupportedFields {
		if bytes.Contains(out, []byte(`"`+f+`"`)) {
			out = jsonparser.Delete(out, f)
		}
	}

	if bytes.Contains(out, []byte(`"temperature"`)) {
		if temp, err := jsonparser.GetFloat(out, "temperature"); err == nil && temp <= 0 {
			if updated, err := jsonparser.Set(out, []byte("0.7"), "temperature"); err == nil {
				out = updated
			}
		}
	}

	return out
}

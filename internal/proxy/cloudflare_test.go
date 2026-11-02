package proxy

import (
	"encoding/json"
	"strings"
	"testing"

	"github.com/buger/jsonparser"
)

func TestTranslateCloudflareImageRequest(t *testing.T) {
	// OpenAI image-generation body sent by the chat app.
	openAIBody := []byte(`{"model":"cloudflare/@cf/black-forest-labs/flux-1-schnell","prompt":"a cyberpunk lizard","n":1,"size":"1024x1024","response_format":"b64_json"}`)

	out, ct, err := translateCloudflareImageRequest(openAIBody)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ct != "application/json" {
		t.Fatalf("expected application/json content type, got %s", ct)
	}

	// Cloudflare must receive ONLY the prompt — extra fields trigger code 8002.
	var got map[string]interface{}
	if err := json.Unmarshal(out, &got); err != nil {
		t.Fatalf("invalid JSON output: %v", err)
	}
	if len(got) != 1 {
		t.Fatalf("expected exactly 1 field, got %d: %v", len(got), got)
	}
	if got["prompt"] != "a cyberpunk lizard" {
		t.Fatalf("unexpected prompt: %v", got["prompt"])
	}
	for _, banned := range []string{"model", "n", "size", "response_format"} {
		if _, ok := got[banned]; ok {
			t.Fatalf("banned field %q present in translated body", banned)
		}
	}
}

func TestTranslateCloudflareImageRequest_MissingPrompt(t *testing.T) {
	if _, _, err := translateCloudflareImageRequest([]byte(`{"model":"@cf/x","n":1}`)); err == nil {
		t.Fatal("expected error for missing prompt")
	}
}

func TestTranslateCloudflareImageResponse(t *testing.T) {
	// Cloudflare /ai/run text-to-image response shape.
	cfBody := []byte(`{"success":true,"result":{"image":"aGVsbG8="},"errors":[]}`)

	out, ct, err := translateCloudflareImageResponse(cfBody)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ct != "application/json" {
		t.Fatalf("expected application/json content type, got %s", ct)
	}

	created, _ := jsonparser.GetInt(out, "created")
	if created <= 0 {
		t.Fatalf("expected created > 0, got %d", created)
	}
	b64, _ := jsonparser.GetString(out, "data", "[0]", "b64_json")
	if b64 != "aGVsbG8=" {
		t.Fatalf("unexpected b64_json: %s", b64)
	}
	// Must NOT leak the Cloudflare envelope.
	if strings.Contains(string(out), "success") || strings.Contains(string(out), "result") {
		t.Fatalf("translated response leaked cloudflare envelope: %s", out)
	}
}

func TestTranslateCloudflareImageResponse_BinaryImage(t *testing.T) {
	// Raw binary PNG bytes
	rawPNG := []byte{0x89, 'P', 'N', 'G', '\r', '\n', 0x1a, '\n', 0, 0, 0, 13, 'I', 'H', 'D', 'R'}

	out, ct, err := translateCloudflareImageResponse(rawPNG)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ct != "application/json" {
		t.Fatalf("expected application/json content type, got %s", ct)
	}

	b64, _ := jsonparser.GetString(out, "data", "[0]", "b64_json")
	if b64 == "" {
		t.Fatalf("expected non-empty b64_json for binary PNG input, got: %s", out)
	}
}

func TestTranslateCloudflareImageResponse_MissingImage(t *testing.T) {
	if _, _, err := translateCloudflareImageResponse([]byte(`{"success":true,"result":{}}`)); err == nil {
		t.Fatal("expected error for missing image")
	}
}

func TestIsCloudflareImageRequest(t *testing.T) {
	cases := map[string]bool{
		"/v1/images/generations": true,
		"/v1/images/edits":       true,
		"/v1/chat/completions":   false,
		"/v1/embeddings":         false,
		"/v1/audio/speech":       false,
	}
	for path, want := range cases {
		if got := isCloudflareImageRequest(path); got != want {
			t.Errorf("isCloudflareImageRequest(%q) = %v, want %v", path, got, want)
		}
	}
}

func TestIsCloudflareModelAgreementError(t *testing.T) {
	agreementErrors := [][]byte{
		[]byte(`{"errors":[{"message":"AiError: Model Agreement: Prior to using this model, you must submit the prompt 'agree'. By submitting ‘agree’, you hereby agree to the llama-3.2-11b-vision-instruct Community License"}]}`),
		[]byte(`{"errors":[{"code":1000,"message":"Prior to using this model, you must submit the prompt 'agree'"}]}`),
		[]byte(`{"error":"Model Agreement required, submit agree"}`),
	}

	for _, errBody := range agreementErrors {
		if !isCloudflareModelAgreementError(errBody) {
			t.Errorf("expected isCloudflareModelAgreementError to be true for: %s", string(errBody))
		}
	}

	otherErrors := [][]byte{
		[]byte(`{"errors":[{"message":"Rate limit exceeded"}]}`),
		[]byte(`{"error":"Invalid API key"}`),
		[]byte(`{"success":false,"errors":[{"message":"Not found"}]}`),
	}

	for _, errBody := range otherErrors {
		if isCloudflareModelAgreementError(errBody) {
			t.Errorf("expected isCloudflareModelAgreementError to be false for: %s", string(errBody))
		}
	}
}

func TestSanitizeCloudflareRequest_VisionWithoutText(t *testing.T) {
	// Request containing image without text
	rawJSON := []byte(`{
		"model": "cloudflare/@cf/meta/llama-3.2-11b-vision-instruct",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "image_url", "image_url": {"url": "data:image/png;base64,iVBORw0KGgo="}}
				]
			}
		],
		"stream_options": {"include_usage": true},
		"tools": []
	}`)

	sanitized := sanitizeCloudflareRequest(rawJSON)

	// Verify model prefix stripped
	model, err := jsonparser.GetString(sanitized, "model")
	if err != nil || model != "@cf/meta/llama-3.2-11b-vision-instruct" {
		t.Errorf("expected model @cf/meta/llama-3.2-11b-vision-instruct, got %s", model)
	}

	// Verify stream_options and tools removed
	if _, _, _, err := jsonparser.Get(sanitized, "stream_options"); err == nil {
		t.Errorf("expected stream_options to be removed")
	}
	if _, _, _, err := jsonparser.Get(sanitized, "tools"); err == nil {
		t.Errorf("expected empty tools to be removed")
	}

	// Verify text prompt injected for image
	var req struct {
		Messages []struct {
			Role    string `json:"role"`
			Content []struct {
				Type string `json:"type"`
				Text string `json:"text,omitempty"`
			} `json:"content"`
		} `json:"messages"`
	}

	if err := json.Unmarshal(sanitized, &req); err != nil {
		t.Fatalf("failed to unmarshal sanitized JSON: %v", err)
	}

	if len(req.Messages) == 0 {
		t.Fatalf("expected messages, got 0")
	}

	hasText := false
	for _, part := range req.Messages[0].Content {
		if part.Type == "text" && part.Text != "" {
			hasText = true
		}
	}

	if !hasText {
		t.Errorf("expected text part to be injected in vision message, got %+v", req.Messages[0].Content)
	}
}

func TestIsCloudflareNativeVisionModel(t *testing.T) {
	cases := map[string]bool{
		"@cf/meta/llama-3.2-11b-vision-instruct":            true,
		"@cf/meta/llama-3.2-90b-vision-instruct":            true,
		"@cf/llava-hf/llava-1.5-7b-hf":                      true,
		"@cf/unum/uform-gen2-qwen-500m":                     true,
		"cloudflare/@cf/meta/llama-3.2-11b-vision-instruct": true,
		"@cf/meta/llama-3.1-8b-instruct":                    false,
		"@cf/stabilityai/stable-diffusion-xl-base-1.0":      false,
		"openai/gpt-4o": false,
	}

	for model, want := range cases {
		if got := isCloudflareNativeVisionModel(model); got != want {
			t.Errorf("isCloudflareNativeVisionModel(%q) = %v, want %v", model, got, want)
		}
	}
}

func TestTranslateCloudflareVisionRequest(t *testing.T) {
	openAIBody := []byte(`{
		"model": "@cf/meta/llama-3.2-11b-vision-instruct",
		"messages": [
			{
				"role": "user",
				"content": [
					{"type": "text", "text": "Describe this image in detail and list all key visual elements."},
					{"type": "image_url", "image_url": {"url": "data:image/jpeg;base64,/9j/4AAQSkZJRgABAQEASABIAAD/2wBDAP//////////////////////////////////////////////////////////////////////////////////////wgALCAABAAEBAREA/8QAFBABAAAAAAAAAAAAAAAAAAAAAP/aAAgBAQABPxA="}}
				]
			}
		]
	}`)

	out, ct, err := translateCloudflareVisionRequest(openAIBody)
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ct != "application/json" {
		t.Fatalf("expected application/json content type, got %s", ct)
	}

	prompt, _ := jsonparser.GetString(out, "prompt")
	if !strings.Contains(prompt, "Describe this image in detail") {
		t.Errorf("unexpected prompt in translated vision request: %s", prompt)
	}

	// Verify image array is present
	_, dataType, _, _ := jsonparser.Get(out, "image")
	if dataType != jsonparser.Array {
		t.Errorf("expected image field to be array in translated vision request, got: %s", out)
	}
}

func TestTranslateCloudflareVisionResponse(t *testing.T) {
	cfVisionBody := []byte(`{
		"result": {
			"response": "This image shows an electronic circuit board with conductive traces and an IC chip."
		},
		"success": true,
		"errors": [],
		"messages": []
	}`)

	out, ct, err := translateCloudflareVisionResponse(cfVisionBody, "@cf/meta/llama-3.2-11b-vision-instruct")
	if err != nil {
		t.Fatalf("unexpected error: %v", err)
	}
	if ct != "application/json" {
		t.Fatalf("expected application/json content type, got %s", ct)
	}

	content, _ := jsonparser.GetString(out, "choices", "[0]", "message", "content")
	if !strings.Contains(content, "electronic circuit board") {
		t.Errorf("unexpected content in translated response: %s", content)
	}

	model, _ := jsonparser.GetString(out, "model")
	if model != "@cf/meta/llama-3.2-11b-vision-instruct" {
		t.Errorf("expected model @cf/meta/llama-3.2-11b-vision-instruct, got %s", model)
	}
}

package proxy

import (
	"bytes"
	"context"
	"encoding/json"
	"fmt"
	"net/http"
	"strings"
	"unicode/utf8"

	"github.com/buger/jsonparser"
)

// ─── Gemini Request Structures ────────────────────────────────────────────────

// geminiRequest is the body sent to Google AI Studio generateContent endpoint.
type geminiRequest struct {
	SystemInstruction *geminiContent  `json:"systemInstruction,omitempty"`
	Contents          []geminiContent `json:"contents"`
	Tools             []geminiTool    `json:"tools,omitempty"`
	ToolConfig        *geminiToolCfg  `json:"toolConfig,omitempty"`
	GenerationConfig  geminiGenCfg    `json:"generationConfig,omitempty"`
	SafetySettings    []geminiSafety  `json:"safetySettings,omitempty"`
}

type geminiContent struct {
	Role  string       `json:"role,omitempty"` // "user" or "model"
	Parts []geminiPart `json:"parts"`
}

type geminiPart struct {
	Text             string                  `json:"text,omitempty"`
	InlineData       *geminiInlineData       `json:"inlineData,omitempty"`
	FileData         *geminiFileData         `json:"fileData,omitempty"`
	FunctionCall     *geminiFunctionCall     `json:"functionCall,omitempty"`
	FunctionResponse *geminiFunctionResponse `json:"functionResponse,omitempty"`
	// ThoughtSignature carries Google's opaque reasoning-trail token.
	// Gemini 3 models attach a thought_signature to every functionCall part
	// they emit so the model can validate its internal reasoning state across
	// turns. OpenAI's tool_calls schema has no slot for it, so OpenAI-only
	// clients (Kilo Code, Cline, …) discard it on receipt and cannot replay it
	// on the next turn. Gemini 3 then rejects the history with HTTP 400
	// INVALID_ARGUMENT. For Gemini 3 models we inject the official bypass
	// sentinel (geminiThoughtSignatureBypass) so the validator is satisfied
	// without the gateway needing to cache authentic signatures.
	ThoughtSignature string `json:"thoughtSignature,omitempty"`
}

type geminiInlineData struct {
	MimeType string `json:"mimeType"`
	Data     string `json:"data"` // base64-encoded
}

type geminiFileData struct {
	MimeType string `json:"mimeType,omitempty"`
	FileURI  string `json:"fileUri"`
}

type geminiFunctionCall struct {
	Name string          `json:"name"`
	Args json.RawMessage `json:"args"`
}

type geminiFunctionResponse struct {
	Name     string          `json:"name"`
	Response json.RawMessage `json:"response"`
}

type geminiTool struct {
	FunctionDeclarations []geminiFuncDecl `json:"functionDeclarations,omitempty"`
}

type geminiFuncDecl struct {
	Name        string          `json:"name"`
	Description string          `json:"description,omitempty"`
	Parameters  json.RawMessage `json:"parameters,omitempty"`
}

type geminiToolCfg struct {
	FunctionCallingConfig geminiFuncCallingCfg `json:"functionCallingConfig"`
}

type geminiFuncCallingCfg struct {
	Mode             string   `json:"mode"` // "AUTO", "ANY", "NONE"
	AllowedFunctions []string `json:"allowedFunctionNames,omitempty"`
}

type geminiGenCfg struct {
	Temperature      *float64        `json:"temperature,omitempty"`
	TopP             *float64        `json:"topP,omitempty"`
	MaxOutputTokens  *int            `json:"maxOutputTokens,omitempty"`
	StopSequences    []string        `json:"stopSequences,omitempty"`
	ResponseMimeType string          `json:"responseMimeType,omitempty"`
	ResponseSchema   json.RawMessage `json:"responseSchema,omitempty"`
	ThinkingConfig   *geminiThinking `json:"thinkingConfig,omitempty"`
}

type geminiThinking struct {
	// ThinkingBudget: 0=disabled, -1=dynamic (let Gemini decide), N=token budget cap
	ThinkingBudget  int  `json:"thinkingBudget"`
	IncludeThoughts bool `json:"includeThoughts,omitempty"`
}

type geminiSafety struct {
	Category  string `json:"category"`
	Threshold string `json:"threshold"`
}

// defaultGeminiSafetySettings uses permissive thresholds so legitimate coding
// queries (security research, code analysis, vulnerability discussion) are not
// blocked by Gemini's content filters at the gateway layer.
var defaultGeminiSafetySettings = []geminiSafety{
	{Category: "HARM_CATEGORY_HATE_SPEECH", Threshold: "BLOCK_ONLY_HIGH"},
	{Category: "HARM_CATEGORY_DANGEROUS_CONTENT", Threshold: "BLOCK_ONLY_HIGH"},
	{Category: "HARM_CATEGORY_SEXUALLY_EXPLICIT", Threshold: "BLOCK_ONLY_HIGH"},
	{Category: "HARM_CATEGORY_HARASSMENT", Threshold: "BLOCK_ONLY_HIGH"},
}

// geminiThoughtSignatureBypass is Google's official sentinel value that tells
// a Gemini 3 model to skip thought_signature validation for a part. It is
// injected on historical functionCall parts (which OpenAI-only clients cannot
// round-trip) so the validator is satisfied without the gateway caching
// authentic per-turn signatures.
const geminiThoughtSignatureBypass = "skip_thought_signature_validator"

// isGemini3Model reports whether model is a Gemini 3.x family member. Gemini 3
// is the first generation that strictly validates thought_signature presence on
// replayed functionCall parts; earlier generations (1.5, 2.5) accept the field
// but do not enforce it, so the sentinel is only injected for 3.x to keep the
// payload minimal for older models.
func isGemini3Model(model string) bool {
	return strings.HasPrefix(strings.ToLower(model), "gemini-3")
}

// ─── OpenAI Request Structures (for parsing incoming body) ───────────────────

type openAIMessage struct {
	Role       string           `json:"role"`
	Content    json.RawMessage  `json:"content"` // string or []contentPart
	Name       string           `json:"name,omitempty"`
	ToolCallID string           `json:"tool_call_id,omitempty"`
	ToolCalls  []openAIToolCall `json:"tool_calls,omitempty"`
}

type openAIToolCall struct {
	ID       string `json:"id"`
	Type     string `json:"type"`
	Function struct {
		Name      string `json:"name"`
		Arguments string `json:"arguments"`
	} `json:"function"`
}

type openAIContentPart struct {
	Type     string          `json:"type"`
	Text     string          `json:"text,omitempty"`
	ImageURL *openAIImageURL `json:"image_url,omitempty"`
}

type openAIImageURL struct {
	URL    string `json:"url"`
	Detail string `json:"detail,omitempty"`
}

type openAITool struct {
	Type     string `json:"type"`
	Function struct {
		Name        string          `json:"name"`
		Description string          `json:"description,omitempty"`
		Parameters  json.RawMessage `json:"parameters,omitempty"`
	} `json:"function"`
}

type openAIRequest struct {
	Model               string          `json:"model"`
	Messages            []openAIMessage `json:"messages"`
	Tools               []openAITool    `json:"tools,omitempty"`
	ToolChoice          json.RawMessage `json:"tool_choice,omitempty"`
	Temperature         *float64        `json:"temperature,omitempty"`
	TopP                *float64        `json:"top_p,omitempty"`
	MaxTokens           *int            `json:"max_tokens,omitempty"`
	MaxCompletionTokens *int            `json:"max_completion_tokens,omitempty"`
	Stop                json.RawMessage `json:"stop,omitempty"`
	Stream              bool            `json:"stream"`
	ResponseFormat      *struct {
		Type       string          `json:"type"`
		JSONSchema json.RawMessage `json:"json_schema,omitempty"`
	} `json:"response_format,omitempty"`
	ReasoningEffort string `json:"reasoning_effort,omitempty"`
	// Non-standard thinking budget extension used by some coding agents
	ThinkingBudget int `json:"thinking_budget,omitempty"`
}

// ─── Request Transpilation ────────────────────────────────────────────────────

// transpileOpenAIToGemini converts an OpenAI chat completions request body into
// a Google AI Studio generateContent request body.
//
// Handles: message role mapping, multi-modal content parts (text + image_url),
// tool/function call translation, generation config mapping, reasoning budget
// activation, and JSON / structured output mode.
func transpileOpenAIToGemini(openAIBody []byte) ([]byte, error) {
	var req openAIRequest
	if err := json.Unmarshal(openAIBody, &req); err != nil {
		return nil, fmt.Errorf("failed to parse openai request body: %w", err)
	}

	gemReq := geminiRequest{
		SafetySettings: defaultGeminiSafetySettings,
	}

	// ── System / Developer messages → systemInstruction ──────────────────────
	var systemParts []string
	for _, msg := range req.Messages {
		if msg.Role == "system" || msg.Role == "developer" {
			text := extractTextContent(msg.Content)
			if text != "" {
				systemParts = append(systemParts, text)
			}
		}
	}
	if len(systemParts) > 0 {
		gemReq.SystemInstruction = &geminiContent{
			Parts: []geminiPart{{Text: strings.Join(systemParts, "\n\n")}},
		}
	}

	// ── Conversation messages → contents ─────────────────────────────────────
	// Gemini 3.x strictly validates thought_signature presence on replayed
	// functionCall parts. OpenAI-only clients cannot round-trip the signature,
	// so for Gemini 3 models we inject the bypass sentinel on every historical
	// assistant functionCall (see buildGeminiAssistantParts).
	injectThoughtSig := isGemini3Model(req.Model)
	var prevRole string
	// toolCallNames maps tool_call_id → function name, populated from
	// assistant tool_calls so that later "tool" result messages can resolve the
	// matching functionCall name even when the client omits the optional "name"
	// field (Gemini requires functionResponse.name to match functionCall.name).
	toolCallNames := make(map[string]string)
	for _, msg := range req.Messages {
		switch msg.Role {
		case "system", "developer":
			continue // already handled above

		case "user":
			parts, err := buildGeminiPartsFromContent(msg.Content)
			if err != nil {
				return nil, fmt.Errorf("failed to parse user message content: %w", err)
			}
			// Merge consecutive user turns (Gemini rejects same-role adjacency)
			if prevRole == "user" && len(gemReq.Contents) > 0 {
				last := &gemReq.Contents[len(gemReq.Contents)-1]
				last.Parts = append(last.Parts, parts...)
			} else {
				gemReq.Contents = append(gemReq.Contents, geminiContent{
					Role:  "user",
					Parts: parts,
				})
			}
			prevRole = "user"

		case "assistant":
			// Register tool_call_id → function name so later tool-result
			// messages can resolve the matching functionCall name even when
			// the client omits the optional "name" field.
			for _, tc := range msg.ToolCalls {
				if tc.ID != "" && tc.Function.Name != "" {
					toolCallNames[tc.ID] = tc.Function.Name
				}
			}
			parts, err := buildGeminiAssistantParts(msg, injectThoughtSig)
			if err != nil {
				return nil, fmt.Errorf("failed to parse assistant message: %w", err)
			}
			if len(parts) > 0 {
				gemReq.Contents = append(gemReq.Contents, geminiContent{
					Role:  "model",
					Parts: parts,
				})
				prevRole = "model"
			}

		case "tool":
			// Tool results → user turn with functionResponse parts.
			funcName := msg.Name
			if funcName == "" {
				if name, ok := toolCallNames[msg.ToolCallID]; ok {
					funcName = name
				} else {
					funcName = "function_result"
				}
			}
			resultRaw := msg.Content
			if len(resultRaw) == 0 {
				resultRaw = json.RawMessage(`""`)
			}
			// Gemini requires functionResponse.response to be a JSON object
			// (google.protobuf.Struct). Coding-agent tool results are
			// frequently plain strings (file contents), JSON arrays (directory
			// listings), or numbers, so coerce anything that is not already a
			// JSON object into {"output": <value>}. See coerceToJSONObject.
			resultJSON := coerceToJSONObject(resultRaw)

			fnResp := &geminiFunctionResponse{
				Name:     funcName,
				Response: resultJSON,
			}
			// Group consecutive tool results into a single user turn.
			if prevRole == "tool" && len(gemReq.Contents) > 0 {
				last := &gemReq.Contents[len(gemReq.Contents)-1]
				last.Parts = append(last.Parts, geminiPart{FunctionResponse: fnResp})
			} else {
				gemReq.Contents = append(gemReq.Contents, geminiContent{
					Role:  "user",
					Parts: []geminiPart{{FunctionResponse: fnResp}},
				})
			}
			prevRole = "tool"
		}
	}

	// ── Tools → functionDeclarations ─────────────────────────────────────────
	// Blank-tool safeguard: coding agents (Kilo Code, Cline, …) may send
	// "tools": [] (an empty array) on initial validation sweeps. We emit
	// functionDeclarations ONLY when real function tools are present; otherwise
	// gemReq.Tools stays nil and is omitted from the payload via the `omitempty`
	// tag. Google AI Studio's schema parser rejects an empty tools array and an
	// empty functionDeclarations list, so both length checks below are load-
	// bearing. Tool parameter schemas are also sanitized (see
	// sanitizeGeminiToolSchema) to strip JSON-Schema draft keywords Google rejects.
	if len(req.Tools) > 0 {
		var decls []geminiFuncDecl
		for _, t := range req.Tools {
			if t.Type != "function" {
				continue
			}
			decls = append(decls, geminiFuncDecl{
				Name:        t.Function.Name,
				Description: t.Function.Description,
				Parameters:  sanitizeGeminiToolSchema(t.Function.Parameters),
			})
		}
		if len(decls) > 0 {
			gemReq.Tools = []geminiTool{{FunctionDeclarations: decls}}
		} else {
			// Only non-function tools were present — emit nothing rather than an
			// empty functionDeclarations list, which Google rejects.
			gemReq.Tools = nil
		}

		// tool_choice → functionCallingConfig
		if len(req.ToolChoice) > 0 {
			choiceStr := strings.Trim(string(req.ToolChoice), `"`)
			var mode string
			var allowedFns []string
			switch choiceStr {
			case "none":
				mode = "NONE"
			case "required", "any":
				mode = "ANY"
			case "auto":
				mode = "AUTO"
			default:
				if name, err := jsonparser.GetString(req.ToolChoice, "function", "name"); err == nil && name != "" {
					mode = "ANY"
					allowedFns = []string{name}
				} else {
					mode = "AUTO"
				}
			}
			gemReq.ToolConfig = &geminiToolCfg{
				FunctionCallingConfig: geminiFuncCallingCfg{
					Mode:             mode,
					AllowedFunctions: allowedFns,
				},
			}
		}
	}

	// ── Generation config ─────────────────────────────────────────────────────
	cfg := geminiGenCfg{}
	if req.Temperature != nil {
		cfg.Temperature = req.Temperature
	}
	if req.TopP != nil {
		cfg.TopP = req.TopP
	}
	// max_completion_tokens takes precedence (newer OpenAI spec)
	if req.MaxCompletionTokens != nil {
		cfg.MaxOutputTokens = req.MaxCompletionTokens
	} else if req.MaxTokens != nil {
		cfg.MaxOutputTokens = req.MaxTokens
	}

	// Stop sequences (string or array)
	if len(req.Stop) > 0 && string(req.Stop) != "null" {
		var stopStr string
		var stopArr []string
		if err := json.Unmarshal(req.Stop, &stopStr); err == nil {
			cfg.StopSequences = []string{stopStr}
		} else if err := json.Unmarshal(req.Stop, &stopArr); err == nil {
			cfg.StopSequences = stopArr
		}
	}

	// JSON mode / structured outputs
	if req.ResponseFormat != nil {
		switch req.ResponseFormat.Type {
		case "json_object":
			cfg.ResponseMimeType = "application/json"
		case "json_schema":
			cfg.ResponseMimeType = "application/json"
			if schemaRaw, _, _, sErr := jsonparser.Get(req.ResponseFormat.JSONSchema, "schema"); sErr == nil && len(schemaRaw) > 0 {
				cfg.ResponseSchema = sanitizeGeminiToolSchema(schemaRaw)
			} else {
				cfg.ResponseSchema = sanitizeGeminiToolSchema(req.ResponseFormat.JSONSchema)
			}
		}
	}

	// Reasoning / thinking budget
	if req.ThinkingBudget > 0 {
		cfg.ThinkingConfig = &geminiThinking{
			ThinkingBudget:  req.ThinkingBudget,
			IncludeThoughts: true,
		}
	} else if req.ReasoningEffort != "" {
		budget := geminiReasoningBudget(req.ReasoningEffort)
		if budget != 0 {
			cfg.ThinkingConfig = &geminiThinking{
				ThinkingBudget:  budget,
				IncludeThoughts: true,
			}
		}
	}

	gemReq.GenerationConfig = cfg
	return json.Marshal(gemReq)
}

// geminiReasoningBudget maps OpenAI reasoning_effort strings to Gemini
// thinkingBudget token counts.
func geminiReasoningBudget(effort string) int {
	switch strings.ToLower(effort) {
	case "low":
		return 1024
	case "medium":
		return 8192
	case "high":
		return 24576
	case "max":
		return -1 // dynamic — Gemini decides
	case "none":
		return 0
	default:
		return 0
	}
}

// normalizeGeminiModel maps unknown/unsupported Gemini model names to actual
// existing Google AI Studio model IDs. This is ONLY used by findPoolByPrefix
// to locate a fallback pool when the exact requested model is not registered.
// It is intentionally NOT called in geminiPath (URL generation) — the pool
// system guarantees that resolved model IDs are real and should be passed
// through unchanged to Google's API.
func normalizeGeminiModel(model string) string {
	lower := strings.ToLower(model)

	// Known real Gemini model IDs — pass through as-is.
	// This list covers both historic and newly discovered models.
	knownModels := map[string]bool{
		// 2.5 generation
		"gemini-2.5-pro": true, "gemini-2.5-flash": true, "gemini-2.5-flash-lite": true,
		"gemini-2.5-computer-use-preview-10-2025": true,
		"gemini-2.5-flash-image":                  true, "gemini-2.5-flash-preview-tts": true,
		"gemini-2.5-pro-preview-tts": true,
		// 2.0 generation
		"gemini-2.0-flash": true, "gemini-2.0-flash-001": true,
		"gemini-2.0-flash-lite": true, "gemini-2.0-flash-lite-001": true,
		"gemini-2.0-flash-exp": true, "gemini-2.0-flash-thinking-exp-1219": true,
		"gemini-2.0-pro-exp-02-05": true,
		// 3.x generation (newly discovered real models)
		"gemini-3.5-flash":       true,
		"gemini-3.1-flash-image": true, "gemini-3.1-flash-image-preview": true,
		"gemini-3.1-flash-lite": true, "gemini-3.1-flash-lite-image": true,
		"gemini-3.1-flash-lite-preview": true, "gemini-3.1-flash-tts-preview": true,
		"gemini-3.1-pro-preview": true, "gemini-3.1-pro-preview-customtools": true,
		"gemini-3-flash-preview": true,
		"gemini-3-pro-image":     true, "gemini-3-pro-image-preview": true, "gemini-3-pro-preview": true,
		// 1.5 generation
		"gemini-1.5-pro": true, "gemini-1.5-flash": true,
		// Embeddings
		"gemini-embedding-001": true, "gemini-embedding-2": true, "gemini-embedding-2-preview": true,
		"text-embedding-004": true, "embedding-001": true,
		// Latest aliases
		"gemini-flash-latest": true, "gemini-flash-lite-latest": true, "gemini-pro-latest": true,
		"gemini-omni-flash-preview": true,
		// Gemma / other models under AI Studio
		"gemma-4-26b-a4b-it": true, "gemma-4-31b-it": true,
	}
	if knownModels[lower] {
		return model
	}

	// Dynamic fallback mapping for truly unknown/fictional model aliases.
	// Prefer the newest generation available.
	if strings.Contains(lower, "pro") {
		return "gemini-2.5-pro"
	}
	if strings.Contains(lower, "flash-lite") {
		return "gemini-2.5-flash-lite"
	}
	// Default to gemini-3.5-flash as it is the most capable generally available model
	return "gemini-3.5-flash"
}

// sanitizeGeminiToolSchema strips JSON-Schema draft keywords that Google AI
// Studio's strict OpenAPI 3.0 schema validator rejects.
//
// Google's functionDeclarations[].parameters and responseSchema fields are
// parsed as a fixed-field OpenAPI 3.0 Schema object. Any JSON Schema draft
// keyword that is not a member of that object is rejected wholesale with
// HTTP 400:
//
//	Invalid JSON payload received. Unknown name "<keyword>" at
//	'tools[0].function_declarations[i].parameters...': Cannot find field.
//
// Coding agents (Kilo Code, Cline, ...) emit strict OpenAI-compliant schemas
// that freely use these draft keywords — most notably "exclusiveMinimum" and
// "exclusiveMaximum", but also "$schema", "$ref", "const", "patternProperties",
// "if/then/else", "examples", etc. sanitizeGeminiToolSchema deletes them
// recursively before the payload is forwarded. Removing them is safe: Google
// cannot enforce them anyway, and the OpenAPI 3.0 keywords that DO carry
// validation semantics (type, format, properties, items, required, enum,
// minimum, maximum, minLength, maxLength, pattern, additionalProperties,
// anyOf/oneOf/allOf, ...) are intentionally preserved.
//
// Non-object input (booleans, primitives, invalid JSON) is returned unchanged.
func sanitizeGeminiToolSchema(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return raw
	}
	var node any
	if err := json.Unmarshal(raw, &node); err != nil {
		return raw
	}
	if !stripSchemaKeywords(&node) {
		return raw
	}
	b, err := json.Marshal(node)
	if err != nil {
		return raw
	}
	return b
}

// geminiUnsupportedSchemaKeywords is the set of JSON Schema draft keywords that
// Google AI Studio's functionDeclarations/responseSchema parser does not model.
// These are stripped recursively (see stripSchemaKeywords) to avoid the 400
// "Unknown name ... Cannot find field" rejection. OpenAPI 3.0 Schema keywords
// are deliberately excluded so real validation constraints survive.
var geminiUnsupportedSchemaKeywords = map[string]bool{
	// Draft version / reference machinery
	"$schema": true, "$id": true, "$ref": true, "$defs": true,
	"definitions": true, "$comment": true, "$anchor": true,
	"$dynamicRef": true, "$dynamicAnchor": true,
	// Numeric bounds expressed as separate draft keywords (OAS 3.0 uses
	// minimum/maximum only; Google rejects the exclusive* variants).
	"exclusiveMinimum": true, "exclusiveMaximum": true,
	// Draft-only constraint / conditional keywords
	"const": true, "contains": true, "minContains": true, "maxContains": true,
	"propertyNames": true, "patternProperties": true,
	"dependencies": true, "dependentRequired": true, "dependentSchemas": true,
	"if": true, "then": true, "else": true,
	"unevaluatedProperties": true, "unevaluatedItems": true,
	// Content annotation (OAS 3.0 has no content*; Google rejects them)
	"contentEncoding": true, "contentMediaType": true, "contentSchema": true,
	// JSON Schema uses "examples" (plural); OAS 3.0 uses "example" (singular).
	"examples": true,
}

// stripSchemaKeywords recursively deletes every key listed in
// geminiUnsupportedSchemaKeywords from any map reachable from n, descending
// through nested objects and arrays so unsupported keywords at any depth (e.g.
// inside properties.*.items or anyOf[] entries) are removed. It reports
// whether any modification was made.
func stripSchemaKeywords(n *any) bool {
	switch v := (*n).(type) {
	case map[string]any:
		changed := false
		for k := range v {
			if geminiUnsupportedSchemaKeywords[k] {
				delete(v, k)
				changed = true
			}
		}
		for k, val := range v {
			vv := val
			if stripSchemaKeywords(&vv) {
				v[k] = vv
				changed = true
			}
		}
		return changed
	case []any:
		changed := false
		for i := range v {
			if stripSchemaKeywords(&v[i]) {
				changed = true
			}
		}
		return changed
	}
	return false
}

// extractTextContent extracts plain text from an OpenAI message content field
// that may be a plain string or an array of content parts.
func extractTextContent(raw json.RawMessage) string {
	if len(raw) == 0 {
		return ""
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return s
	}
	var parts []openAIContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return string(raw)
	}
	var sb strings.Builder
	for _, p := range parts {
		if p.Type == "text" {
			sb.WriteString(p.Text)
		}
	}
	return sb.String()
}

// buildGeminiPartsFromContent converts an OpenAI message content (string or
// []contentPart) into Gemini parts, handling text and image_url parts.
func buildGeminiPartsFromContent(raw json.RawMessage) ([]geminiPart, error) {
	if len(raw) == 0 {
		return []geminiPart{{Text: ""}}, nil
	}
	var s string
	if err := json.Unmarshal(raw, &s); err == nil {
		return []geminiPart{{Text: s}}, nil
	}
	var parts []openAIContentPart
	if err := json.Unmarshal(raw, &parts); err != nil {
		return []geminiPart{{Text: string(raw)}}, nil
	}
	var gemParts []geminiPart
	for _, p := range parts {
		switch p.Type {
		case "text":
			gemParts = append(gemParts, geminiPart{Text: p.Text})
		case "image_url":
			if p.ImageURL == nil {
				continue
			}
			gp, err := convertImageURL(p.ImageURL.URL)
			if err != nil {
				return nil, err
			}
			gemParts = append(gemParts, gp)
		}
	}
	if len(gemParts) == 0 {
		gemParts = []geminiPart{{Text: ""}}
	}
	return gemParts, nil
}

// convertImageURL converts an OpenAI image_url value into a Gemini part.
// data: URIs → inlineData (base64). http(s):// URLs → fetched and converted to inlineData.
func convertImageURL(url string) (geminiPart, error) {
	if strings.HasPrefix(url, "data:") {
		mimeType, _, b64data, err := ExtractBase64FromDataURI(url)
		if err != nil {
			return geminiPart{}, err
		}
		return geminiPart{
			InlineData: &geminiInlineData{
				MimeType: mimeType,
				Data:     b64data,
			},
		}, nil
	}

	if strings.HasPrefix(url, "http://") || strings.HasPrefix(url, "https://") {
		mimeType, b64data, _, err := FetchAndEncodeImage(context.Background(), url)
		if err != nil {
			return geminiPart{}, fmt.Errorf("failed to fetch remote image for gemini: %w", err)
		}
		return geminiPart{
			InlineData: &geminiInlineData{
				MimeType: mimeType,
				Data:     b64data,
			},
		}, nil
	}

	return geminiPart{}, fmt.Errorf("unsupported image URL format: must be data URI or http(s) URL")
}

// buildGeminiAssistantParts converts an OpenAI assistant message (text content
// and/or tool_calls) into Gemini parts.
//
// When injectThoughtSig is true (Gemini 3 models), every functionCall part
// receives the thought_signature bypass sentinel. Gemini 3 attaches an opaque
// reasoning signature to each functionCall it emits and rejects history
// replays that omit it with HTTP 400. OpenAI's tool_calls schema has no field
// for it, so OpenAI-only clients (Kilo Code, Cline, …) drop it and cannot
// echo it back. The sentinel satisfies the validator statelessly.
func buildGeminiAssistantParts(msg openAIMessage, injectThoughtSig bool) ([]geminiPart, error) {
	var parts []geminiPart
	text := extractTextContent(msg.Content)
	if text != "" {
		parts = append(parts, geminiPart{Text: text})
	}
	for _, tc := range msg.ToolCalls {
		if tc.Type != "function" {
			continue
		}
		// Gemini requires functionCall.args to be a JSON object. Kilo/Cline
		// always emit object arguments, but coerce defensively so a stray
		// array/string argument cannot 400 the whole request.
		argsRaw := coerceToJSONObject(json.RawMessage(tc.Function.Arguments))
		part := geminiPart{
			FunctionCall: &geminiFunctionCall{
				Name: tc.Function.Name,
				Args: argsRaw,
			},
		}
		if injectThoughtSig {
			part.ThoughtSignature = geminiThoughtSignatureBypass
		}
		parts = append(parts, part)
	}
	return parts, nil
}

// coerceToJSONObject ensures raw is a JSON object, as required by the Gemini
// API for functionCall.args and functionResponse.response (both are parsed as
// google.protobuf.Struct). Gemini rejects arrays, strings, numbers, booleans,
// and null at those locations with HTTP 400:
//
//	Invalid value at 'contents[...].parts[...].functionResponse.response': ...
//
// Coding agents (Kilo Code, Cline, …) routinely return tool results that are
// plain strings (file contents), JSON arrays (directory listings), or numbers.
// coerceToJSONObject passes objects through unchanged and wraps every other
// shape as {"output": <value>}. Non-JSON strings are wrapped as a quoted
// string. An empty/null input becomes "{}".
func coerceToJSONObject(raw json.RawMessage) json.RawMessage {
	if len(raw) == 0 || string(raw) == "null" {
		return json.RawMessage(`{}`)
	}
	trimmed := bytes.TrimSpace(raw)
	if len(trimmed) > 0 && trimmed[0] == '{' && json.Valid(raw) {
		return raw
	}
	if json.Valid(raw) {
		wrapped, _ := json.Marshal(map[string]json.RawMessage{"output": raw})
		return wrapped
	}
	wrapped, _ := json.Marshal(map[string]string{"output": string(raw)})
	return wrapped
}

// ─── Non-Streaming Response Translation ──────────────────────────────────────

// translateGeminiResponse converts a Google AI Studio generateContent response
// body into an OpenAI-compatible chat completion JSON.
func translateGeminiResponse(body []byte, requestedModel string) ([]byte, error) {
	if errMsg, _, _, err := jsonparser.Get(body, "error", "message"); err == nil && len(errMsg) > 0 {
		return nil, fmt.Errorf("gemini returned error: %s", string(errMsg))
	}

	finishReasonRaw, _, _, _ := jsonparser.Get(body, "candidates", "[0]", "finishReason")
	finishReason := mapGeminiFinishReasonNS(string(finishReasonRaw))

	var textContent strings.Builder
	var reasoningContent strings.Builder
	var toolCalls []openAIToolCallResp

	// Iterate all parts in candidates[0].content.parts
	partsRaw, _, _, err := jsonparser.Get(body, "candidates", "[0]", "content", "parts")
	if err == nil && len(partsRaw) > 0 {
		_, _ = jsonparser.ArrayEach(partsRaw, func(partData []byte, _ jsonparser.ValueType, _ int, parseErr error) {
			if parseErr != nil {
				return
			}
			isThought, _ := jsonparser.GetBoolean(partData, "thought")
			if text, _, _, tErr := jsonparser.Get(partData, "text"); tErr == nil && len(text) > 0 {
				if isThought {
					reasoningContent.Write(text)
				} else {
					textContent.Write(text)
				}
				return
			}
			// Function call parts
			fnName, fnErr := jsonparser.GetString(partData, "functionCall", "name")
			if fnErr == nil && fnName != "" {
				argsRaw, _, _, _ := jsonparser.Get(partData, "functionCall", "args")
				if len(argsRaw) == 0 {
					argsRaw = []byte("{}")
				}
				toolCalls = append(toolCalls, openAIToolCallResp{
					ID:   fmt.Sprintf("call_%d", len(toolCalls)),
					Type: "function",
					Function: openAIFuncCallResp{
						Name:      fnName,
						Arguments: string(argsRaw),
					},
				})
			}
		})
	}

	promptTokens, _ := jsonparser.GetInt(body, "usageMetadata", "promptTokenCount")
	candidateTokens, _ := jsonparser.GetInt(body, "usageMetadata", "candidatesTokenCount")
	thoughtsTokens, _ := jsonparser.GetInt(body, "usageMetadata", "thoughtsTokenCount")
	totalTokens := promptTokens + candidateTokens + thoughtsTokens

	type oaMessage struct {
		Role             string               `json:"role"`
		Content          string               `json:"content"`
		ReasoningContent string               `json:"reasoning_content,omitempty"`
		ToolCalls        []openAIToolCallResp `json:"tool_calls,omitempty"`
	}
	type oaChoice struct {
		Index        int       `json:"index"`
		Message      oaMessage `json:"message"`
		FinishReason string    `json:"finish_reason"`
	}
	type oaUsage struct {
		PromptTokens     int64 `json:"prompt_tokens"`
		CompletionTokens int64 `json:"completion_tokens"`
		TotalTokens      int64 `json:"total_tokens"`
	}
	type oaCompletion struct {
		ID      string     `json:"id"`
		Object  string     `json:"object"`
		Created int64      `json:"created"`
		Model   string     `json:"model"`
		Choices []oaChoice `json:"choices"`
		Usage   oaUsage    `json:"usage"`
	}

	msg := oaMessage{
		Role:    "assistant",
		Content: textContent.String(),
	}
	if reasoningContent.Len() > 0 {
		msg.ReasoningContent = reasoningContent.String()
	}
	if len(toolCalls) > 0 {
		msg.ToolCalls = toolCalls
		if finishReason == "stop" {
			finishReason = "tool_calls"
		}
	}

	result := oaCompletion{
		ID:      "chatcmpl-gate",
		Object:  "chat.completion",
		Created: 0,
		Model:   requestedModel,
		Choices: []oaChoice{{Index: 0, Message: msg, FinishReason: finishReason}},
		Usage: oaUsage{
			PromptTokens:     promptTokens,
			CompletionTokens: candidateTokens + thoughtsTokens,
			TotalTokens:      totalTokens,
		},
	}
	return json.Marshal(result)
}

// openAIToolCallResp is the response shape for tool calls in non-streaming responses.
type openAIToolCallResp struct {
	ID       string             `json:"id"`
	Type     string             `json:"type"`
	Function openAIFuncCallResp `json:"function"`
}

type openAIFuncCallResp struct {
	Name      string `json:"name"`
	Arguments string `json:"arguments"`
}

// mapGeminiFinishReasonNS maps Gemini finish reasons to OpenAI finish reasons
// for non-streaming responses.
func mapGeminiFinishReasonNS(reason string) string {
	switch reason {
	case "STOP":
		return "stop"
	case "MAX_TOKENS":
		return "length"
	case "SAFETY", "RECITATION", "PROHIBITED_CONTENT", "SPII":
		return "content_filter"
	case "MALFORMED_FUNCTION_CALL":
		return "stop"
	default:
		return "stop"
	}
}

// ─── Error Normalization ──────────────────────────────────────────────────────

// geminiNormalizedError holds the translated error for downstream OpenAI clients.
type geminiNormalizedError struct {
	HTTPStatus int
	Body       []byte
}

// normalizeGeminiError converts a Gemini API error payload into an OpenAI-compatible
// error response, mapping Gemini gRPC status codes to OpenAI error types.
func normalizeGeminiError(httpStatus int, geminiBody []byte) geminiNormalizedError {
	message, _ := jsonparser.GetString(geminiBody, "error", "message")
	status, _ := jsonparser.GetString(geminiBody, "error", "status")
	if message == "" {
		message = fmt.Sprintf("upstream gemini error (http %d)", httpStatus)
	}

	var oaType, oaCode string
	var outHTTP int

	switch {
	case httpStatus == http.StatusUnauthorized || status == "UNAUTHENTICATED":
		outHTTP, oaType, oaCode = http.StatusUnauthorized, "authentication_error", "invalid_api_key"

	case httpStatus == http.StatusForbidden || status == "PERMISSION_DENIED":
		outHTTP, oaType, oaCode = http.StatusForbidden, "permission_error", "permission_denied"

	case httpStatus == http.StatusNotFound || status == "NOT_FOUND":
		outHTTP, oaType, oaCode = http.StatusNotFound, "invalid_request_error", "model_not_found"
		message = "The requested Gemini model was not found: " + message

	case httpStatus == http.StatusTooManyRequests || status == "RESOURCE_EXHAUSTED":
		outHTTP, oaType = http.StatusTooManyRequests, "rate_limit_error"
		if strings.Contains(strings.ToLower(message), "quota") {
			oaCode = "quota_exceeded"
		} else {
			oaCode = "rate_limit_exceeded"
		}

	case httpStatus == http.StatusBadRequest || status == "INVALID_ARGUMENT":
		outHTTP, oaType = http.StatusBadRequest, "invalid_request_error"
		if isGeminiSafetyBlock(geminiBody) {
			oaCode = "content_policy_violation"
			message = "The request was blocked by Gemini content safety filters. Please rephrase your message."
		} else {
			oaCode = "invalid_request"
		}

	case status == "UNAVAILABLE":
		outHTTP, oaType, oaCode = http.StatusServiceUnavailable, "api_error", "service_unavailable"

	case status == "DEADLINE_EXCEEDED":
		outHTTP, oaType, oaCode = http.StatusGatewayTimeout, "api_error", "timeout"

	case httpStatus >= 500 || status == "INTERNAL":
		outHTTP, oaType, oaCode = http.StatusBadGateway, "api_error", "internal_error"

	default:
		outHTTP, oaType, oaCode = http.StatusBadGateway, "api_error", "upstream_error"
	}

	type oaErr struct {
		Message string `json:"message"`
		Type    string `json:"type"`
		Code    string `json:"code,omitempty"`
	}
	type oaEnvelope struct {
		Error oaErr `json:"error"`
	}
	body, _ := json.Marshal(oaEnvelope{Error: oaErr{Message: message, Type: oaType, Code: oaCode}})
	return geminiNormalizedError{HTTPStatus: outHTTP, Body: body}
}

// isGeminiSafetyBlock returns true when the Gemini response indicates that the
// request was rejected by content safety filters.
func isGeminiSafetyBlock(body []byte) bool {
	finishReason, _ := jsonparser.GetString(body, "candidates", "[0]", "finishReason")
	if finishReason == "SAFETY" || finishReason == "PROHIBITED_CONTENT" || finishReason == "SPII" {
		return true
	}
	blockReason, _ := jsonparser.GetString(body, "promptFeedback", "blockReason")
	if blockReason != "" && blockReason != "BLOCK_REASON_UNSPECIFIED" {
		return true
	}
	status, _ := jsonparser.GetString(body, "error", "status")
	if status == "PROHIBITED_CONTENT" {
		return true
	}
	errMsg, _ := jsonparser.GetString(body, "error", "message")
	lower := strings.ToLower(errMsg)
	return strings.Contains(lower, "safety") && strings.Contains(lower, "block")
}

// ─── Streaming Tool-Call Delta Builder ───────────────────────────────────────

// buildGeminiToolCallDelta builds an OpenAI-compatible SSE chunk for a tool call
// function call event in streaming mode.
func buildGeminiToolCallDelta(index int, callID, funcName, argsJSON string, finishReason string) []byte {
	buf := bytes.NewBuffer(make([]byte, 0, 256))
	buf.WriteString(`{"id":"chatcmpl-gate","object":"chat.completion.chunk","choices":[{"index":0,"delta":{"tool_calls":[{"index":`)
	writeIntGemini(buf, index)
	if callID != "" {
		buf.WriteString(`,"id":`)
		writeJSONStringGemini(buf, callID)
	}
	buf.WriteString(`,"type":"function","function":{"name":`)
	writeJSONStringGemini(buf, funcName)
	buf.WriteString(`,"arguments":`)
	writeJSONStringGemini(buf, argsJSON)
	buf.WriteString(`}}]}`)
	if finishReason != "" {
		buf.WriteString(`,"finish_reason":"`)
		buf.WriteString(finishReason)
		buf.WriteByte('"')
	} else {
		buf.WriteString(`,"finish_reason":null`)
	}
	buf.WriteString(`}]}`)
	result := make([]byte, buf.Len())
	copy(result, buf.Bytes())
	return result
}

// writeIntGemini writes an integer to a bytes.Buffer without fmt allocation.
func writeIntGemini(buf *bytes.Buffer, n int) {
	if n == 0 {
		buf.WriteByte('0')
		return
	}
	var digits [20]byte
	pos := len(digits)
	for n > 0 {
		pos--
		digits[pos] = byte('0' + n%10)
		n /= 10
	}
	buf.Write(digits[pos:])
}

// writeJSONStringGemini writes a JSON-encoded string (with quotes) to buf.
func writeJSONStringGemini(buf *bytes.Buffer, s string) {
	buf.WriteByte('"')
	for i := 0; i < len(s); {
		r, size := utf8.DecodeRuneInString(s[i:])
		i += size
		switch r {
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
			if r < 0x20 {
				buf.WriteString(fmt.Sprintf(`\u%04x`, r))
			} else {
				buf.WriteRune(r)
			}
		}
	}
	buf.WriteByte('"')
}

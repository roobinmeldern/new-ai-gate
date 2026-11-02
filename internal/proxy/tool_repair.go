package proxy

import (
	"encoding/json"
	"strings"
	"unicode"
)

// RepairToolArguments fixes common malformed JSON issues emitted by models
// (especially GLM, DeepSeek, Qwen) when generating tool call arguments:
//  1. Strips markdown fences (```json ... ```)
//  2. Unescapes double-encoded JSON strings
//  3. Extracts balanced JSON from surrounding conversational text
//  4. Escapes literal newlines and control characters inside string literals
//  5. Removes trailing commas before } or ]
//  6. Normalizes read_files arguments (bare string or string array -> object array)
//  7. Ensures the result is a valid JSON object or array, defaulting to "{}"
func RepairToolArguments(toolName string, rawArgs string) string {
	trimmed := strings.TrimSpace(rawArgs)
	if trimmed == "" {
		return "{}"
	}

	// Step 1: Strip markdown code block fences if present
	trimmed = stripMarkdownFences(trimmed)

	// Step 2: If double-encoded as a JSON string literal (e.g. "{\"files\": ...}"), unquote it
	if strings.HasPrefix(trimmed, `"`) && strings.HasSuffix(trimmed, `"`) {
		var unquoted string
		if err := json.Unmarshal([]byte(trimmed), &unquoted); err == nil {
			unquotedTrimmed := strings.TrimSpace(unquoted)
			if (strings.HasPrefix(unquotedTrimmed, "{") && strings.HasSuffix(unquotedTrimmed, "}")) ||
				(strings.HasPrefix(unquotedTrimmed, "[") && strings.HasSuffix(unquotedTrimmed, "]")) {
				trimmed = unquotedTrimmed
			}
		}
	}

	// Step 3: Extract balanced JSON block if surrounded by conversational text
	trimmed = extractBalancedJSON(trimmed)

	// Step 4: Fix literal newlines and control characters inside string literals
	trimmed = escapeControlCharsInStrings(trimmed)

	// Step 5: Remove trailing commas before } or ]
	trimmed = removeTrailingCommas(trimmed)

	// Step 6: Check if already valid JSON object or array
	var parsed interface{}
	if err := json.Unmarshal([]byte(trimmed), &parsed); err == nil {
		switch val := parsed.(type) {
		case map[string]interface{}:
			// If tool is read_files and val["files"] is an array of strings, convert to array of objects
			if toolName == "read_files" {
				if filesArr, ok := val["files"].([]interface{}); ok {
					needNorm := false
					for _, item := range filesArr {
						if _, isStr := item.(string); isStr {
							needNorm = true
							break
						}
					}
					if needNorm {
						newArr := make([]interface{}, 0, len(filesArr))
						for _, item := range filesArr {
							if s, isStr := item.(string); isStr {
								newArr = append(newArr, map[string]interface{}{"path": s})
							} else {
								newArr = append(newArr, item)
							}
						}
						val["files"] = newArr
						if b, err := json.Marshal(val); err == nil {
							return string(b)
						}
					}
				}
			}
			return trimmed
		case []interface{}:
			return trimmed
		case string:
			// Model emitted a bare string like "internal/foo.go"
			if toolName == "read_files" {
				res := map[string]interface{}{
					"files": []map[string]interface{}{
						{"path": val},
					},
				}
				if b, err := json.Marshal(res); err == nil {
					return string(b)
				}
			}
		}
	}

	// Step 7: If not valid JSON, check if it's a bare string representing a file path for read_files
	if toolName == "read_files" && !strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[") {
		cleanPath := strings.Trim(trimmed, `"'`+" \t\r\n")
		if cleanPath != "" && !strings.ContainsAny(cleanPath, "\n\r{}[]") {
			res := map[string]interface{}{
				"files": []map[string]interface{}{
					{"path": cleanPath},
				},
			}
			if b, err := json.Marshal(res); err == nil {
				return string(b)
			}
		}
	}

	// Final validation: if still not valid JSON object/array, return "{}"
	if !json.Valid([]byte(trimmed)) || (!strings.HasPrefix(trimmed, "{") && !strings.HasPrefix(trimmed, "[")) {
		return "{}"
	}

	return trimmed
}

// stripMarkdownFences removes opening and closing markdown code fences (```json ... ```).
func stripMarkdownFences(s string) string {
	trimmed := strings.TrimSpace(s)
	if strings.HasPrefix(trimmed, "```") {
		// Drop opening ``` line
		if idx := strings.Index(trimmed, "\n"); idx != -1 {
			trimmed = strings.TrimSpace(trimmed[idx+1:])
		} else {
			trimmed = strings.TrimPrefix(trimmed, "```")
			trimmed = strings.TrimPrefix(trimmed, "json")
			trimmed = strings.TrimSpace(trimmed)
		}
	}
	if strings.HasSuffix(trimmed, "```") {
		lastIdx := strings.LastIndex(trimmed, "```")
		trimmed = strings.TrimSpace(trimmed[:lastIdx])
	}
	return trimmed
}

// extractBalancedJSON locates the first { or [ and extracts up to the matching } or ].
func extractBalancedJSON(s string) string {
	firstObj := strings.Index(s, "{")
	firstArr := strings.Index(s, "[")

	firstIdx := -1
	var openChar, closeChar byte

	if firstObj != -1 && (firstArr == -1 || firstObj < firstArr) {
		firstIdx = firstObj
		openChar = '{'
		closeChar = '}'
	} else if firstArr != -1 {
		firstIdx = firstArr
		openChar = '['
		closeChar = ']'
	}

	if firstIdx == -1 {
		return s
	}

	depth := 0
	inString := false
	escaped := false

	for i := firstIdx; i < len(s); i++ {
		c := s[i]

		if inString {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				inString = false
			}
			continue
		}

		if c == '"' {
			inString = true
			continue
		}

		if c == openChar {
			depth++
		} else if c == closeChar {
			depth--
			if depth == 0 {
				candidate := strings.TrimSpace(s[firstIdx : i+1])
				return candidate
			}
		}
	}

	return s
}

// escapeControlCharsInStrings replaces unescaped literal \n, \r, \t inside string literals.
func escapeControlCharsInStrings(s string) string {
	var sb strings.Builder
	sb.Grow(len(s))

	inString := false
	escaped := false

	for i := 0; i < len(s); i++ {
		c := s[i]

		if inString {
			if escaped {
				escaped = false
				sb.WriteByte(c)
				continue
			}
			if c == '\\' {
				escaped = true
				sb.WriteByte(c)
				continue
			}
			if c == '"' {
				inString = false
				sb.WriteByte(c)
				continue
			}
			// Escape illegal control characters inside string literal
			switch c {
			case '\n':
				sb.WriteString(`\n`)
			case '\r':
				sb.WriteString(`\r`)
			case '\t':
				sb.WriteString(`\t`)
			default:
				if c < 0x20 {
					// Skip or ignore other non-printable controls
					continue
				}
				sb.WriteByte(c)
			}
		} else {
			if c == '"' {
				inString = true
			}
			sb.WriteByte(c)
		}
	}

	return sb.String()
}

// removeTrailingCommas removes commas immediately preceding } or ] outside string literals.
func removeTrailingCommas(s string) string {
	var sb strings.Builder
	sb.Grow(len(s))

	inString := false
	escaped := false

	for i := 0; i < len(s); i++ {
		c := s[i]

		if inString {
			if escaped {
				escaped = false
			} else if c == '\\' {
				escaped = true
			} else if c == '"' {
				inString = false
			}
			sb.WriteByte(c)
			continue
		}

		if c == '"' {
			inString = true
			sb.WriteByte(c)
			continue
		}

		if c == ',' {
			// Look ahead for next non-whitespace character
			j := i + 1
			for j < len(s) && unicode.IsSpace(rune(s[j])) {
				j++
			}
			if j < len(s) && (s[j] == '}' || s[j] == ']') {
				// Skip this trailing comma
				continue
			}
		}

		sb.WriteByte(c)
	}

	return sb.String()
}

package proxy

import (
	"encoding/json"
	"testing"
)

func TestRepairToolArguments(t *testing.T) {
	tests := []struct {
		name     string
		toolName string
		input    string
		expected string
		validate func(t *testing.T, res string)
	}{
		{
			name:     "valid json object preserved",
			toolName: "read_files",
			input:    `{"files":[{"path":"foo.go"}]}`,
			expected: `{"files":[{"path":"foo.go"}]}`,
		},
		{
			name:     "empty arguments defaults to empty object",
			toolName: "any_tool",
			input:    `   `,
			expected: `{}`,
		},
		{
			name:     "markdown code fences stripped",
			toolName: "read_files",
			input:    "```json\n{\"files\":[{\"path\":\"bar.go\"}]}\n```",
			expected: `{"files":[{"path":"bar.go"}]}`,
		},
		{
			name:     "double-encoded json unquoted",
			toolName: "read_files",
			input:    `"{\"files\":[{\"path\":\"bar.go\"}]}"`,
			expected: `{"files":[{"path":"bar.go"}]}`,
		},
		{
			name:     "trailing conversational text stripped",
			toolName: "read_files",
			input:    `{"files":[{"path":"main.go"}]} Now I will read other files.`,
			expected: `{"files":[{"path":"main.go"}]}`,
		},
		{
			name:     "leading conversational text stripped",
			toolName: "read_files",
			input:    `Calling tool read_files: {"files":[{"path":"main.go"}]}`,
			expected: `{"files":[{"path":"main.go"}]}`,
		},
		{
			name:     "trailing commas removed",
			toolName: "calculate",
			input:    `{"expression": "2+2",}`,
			expected: `{"expression": "2+2"}`,
		},
		{
			name:     "trailing commas in array removed",
			toolName: "read_files",
			input:    `{"files": [{"path": "a.go",},],}`,
			expected: `{"files": [{"path": "a.go"}]}`,
		},
		{
			name:     "read_files string array normalized to object array",
			toolName: "read_files",
			input:    `{"files":["internal/transmux/transmuxer.go","internal/transmux/anthropic.go"]}`,
			validate: func(t *testing.T, res string) {
				var parsed struct {
					Files []struct {
						Path string `json:"path"`
					} `json:"files"`
				}
				if err := json.Unmarshal([]byte(res), &parsed); err != nil {
					t.Fatalf("failed to unmarshal: %v", err)
				}
				if len(parsed.Files) != 2 || parsed.Files[0].Path != "internal/transmux/transmuxer.go" {
					t.Errorf("unexpected normalized files: %+v", parsed.Files)
				}
			},
		},
		{
			name:     "read_files bare string normalized to files object",
			toolName: "read_files",
			input:    `"internal/proxy/handler.go"`,
			validate: func(t *testing.T, res string) {
				var parsed struct {
					Files []struct {
						Path string `json:"path"`
					} `json:"files"`
				}
				if err := json.Unmarshal([]byte(res), &parsed); err != nil {
					t.Fatalf("failed to unmarshal: %v", err)
				}
				if len(parsed.Files) != 1 || parsed.Files[0].Path != "internal/proxy/handler.go" {
					t.Errorf("unexpected normalized files: %+v", parsed.Files)
				}
			},
		},
		{
			name:     "unescaped newline inside string literal escaped",
			toolName: "write_file",
			input:    "{\"content\": \"line 1\nline 2\"}",
			validate: func(t *testing.T, res string) {
				var parsed struct {
					Content string `json:"content"`
				}
				if err := json.Unmarshal([]byte(res), &parsed); err != nil {
					t.Fatalf("failed to unmarshal: %v", err)
				}
				if parsed.Content != "line 1\nline 2" {
					t.Errorf("unexpected content: %q", parsed.Content)
				}
			},
		},
	}

	for _, tt := range tests {
		t.Run(tt.name, func(t *testing.T) {
			res := RepairToolArguments(tt.toolName, tt.input)
			if tt.expected != "" && res != tt.expected {
				t.Errorf("expected %s, got %s", tt.expected, res)
			}
			if tt.validate != nil {
				tt.validate(t, res)
			}
			if !json.Valid([]byte(res)) {
				t.Errorf("repaired result is not valid JSON: %s", res)
			}
		})
	}
}

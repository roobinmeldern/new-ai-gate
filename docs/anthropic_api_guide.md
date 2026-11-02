# Anthropic-Compatible Front API & Claude Code CLI Guide

## 1. Overview & Architecture

Clever AI Gate provides a high-throughput, dual-front-door gateway architecture in Go:
1. **OpenAI-Compatible Front API**: `/v1/chat/completions`, `/v1/models`, `/v1/embeddings`, etc.
2. **Anthropic-Compatible Front API**: `/v1/messages`, `/messages`, `/v1/messages/count_tokens`, `/messages/count_tokens`, `/v1/models`, `/models`.

This allows standard Anthropic tools and SDKs—including **Claude Code CLI**, **Cursor**, **Cline**, **Roo Code**, and official `@anthropic-ai/sdk` libraries—to seamlessly connect to the gateway and route to hundreds of backend providers (including CleanAPIs, OpenRouter, Puter, Cloudflare, OpenAI, Gemini, NVIDIA NIM, and self-hosted Ollama models).

```
+-----------------------------------------------------------------------------------+
|                           Clients & AI Tools                                      |
|  (Claude Code CLI, Cursor, Cline, Roo Code, Official Anthropic SDKs)               |
+-----------------------------------------------------------------------------------+
                                       |
                                       v  HTTP (POST /v1/messages or /messages)
+-----------------------------------------------------------------------------------+
|                           Clever AI Gate Engine                                   |
|                                                                                   |
|  1. Middleware: Dual-Header Auth (x-api-key / Bearer), Rate Limit, CORS          |
|  2. Inbound Translator: Anthropic Messages JSON -> Canonical OpenAI Chat Payload   |
|  3. Routing & Failover: Prefix/Wildcard Model Pool Resolution & Health Monitoring |
|  4. Reverse Transmuxer: Upstream OpenAI SSE -> Anthropic SSE Stream Protocol      |
|     (message_start -> content_block_start -> content_block_delta ->               |
|      content_block_stop -> message_delta -> message_stop)                        |
+-----------------------------------------------------------------------------------+
                                       |
                   +-------------------+-------------------+
                   |                                       |
                   v                                       v
      +-------------------------+             +-------------------------+
      |  CleanAPIs / Cloudflare |             | Native Anthropic /      |
      |  OpenAI-Compatible APIs |             | Direct Passthrough      |
      +-------------------------+             +-------------------------+
```

---

## 2. API Endpoints & Specifications

The gateway exposes the following Anthropic-compatible endpoints:

| Endpoint | Method | Description |
| :--- | :--- | :--- |
| `/v1/messages` | `POST` | Anthropic Messages API (Streaming & Non-Streaming) |
| `/messages` | `POST` | Root Anthropic Messages API (for SDKs stripping `/v1`) |
| `/v1/messages/count_tokens` | `POST` | Anthropic Input Token Counter |
| `/messages/count_tokens` | `POST` | Root Anthropic Token Counter |
| `/v1/models` & `/models` | `GET` | Dual-Schema Model Catalog (OpenAI + Anthropic metadata) |
| `/v1/models/:model` | `GET` | Dual-Schema Single Model Lookup |

### Authentication
Both header patterns are accepted:
- `x-api-key: <ADMIN_KEY_OR_TENANT_KEY>`
- `Authorization: Bearer <ADMIN_KEY_OR_TENANT_KEY>`

CORS preflight requests (`OPTIONS`) and Anthropic version headers (`anthropic-version`, `anthropic-beta`) are automatically handled.

---

## 3. Configuring Claude Code CLI for CleanAPI Opus 5.5

### 3.1 Settings Configuration
Claude Code CLI stores its settings in `~/.claude/settings.json`. Update the `env` section to direct traffic to the Clever AI Gate instance and specify `cleanapis/claude-opus-5.5` (or `claude-opus-5.5`):

```json
{
  "forceLoginMethod": "console",
  "env": {
    "ANTHROPIC_API_KEY": "30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132",
    "ANTHROPIC_AUTH_TOKEN": "30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132",
    "ANTHROPIC_BASE_URL": "http://192.168.5.25:8080",
    "ANTHROPIC_MODEL": "cleanapis/claude-opus-5.5",
    "ANTHROPIC_DEFAULT_OPUS_MODEL": "cleanapis/claude-opus-5.5",
    "ANTHROPIC_DEFAULT_SONNET_MODEL": "cleanapis/claude-opus-5.5",
    "ANTHROPIC_DEFAULT_HAIKU_MODEL": "cleanapis/claude-opus-5.5",
    "ANTHROPIC_DEFAULT_FABLE_MODEL": "cleanapis/claude-opus-5.5",
    "CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC": "1"
  }
}
```

> **Note on Model Patterns**:
> The gateway matches models in the following priority order:
> 1. Exact pool match (e.g., `cleanapis/claude-opus-5.5`).
> 2. Wildcard or base pattern (e.g., `claude-opus-5.5`), which falls back across active providers.

### 3.2 Configuring Claude Desktop for Linux

Claude Desktop combines an Electron-based Web Chat view (`claude.ai`) with embedded **Claude Code / Cowork** local agent runners. By default, desktop applications launched by Wayland/GNOME do not inherit shell environment variables and fall back to the user's cached consumer account.

To enforce the gateway configuration across Claude Desktop:

1. **Set `forceLoginMethod: "console"`**:
   Instructs Claude Code inside the desktop app to authenticate using the Gateway API Key rather than the personal Claude.ai account.
2. **Desktop Launcher Wrapper (`~/.local/bin/claude-desktop`)**:
   An executable wrapper exports all necessary environment variables before executing `/usr/lib/claude-desktop/claude-desktop`:
   ```bash
   #!/usr/bin/env bash
   export ANTHROPIC_BASE_URL="http://192.168.5.25:8080"
   export ANTHROPIC_API_KEY="30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132"
   export ANTHROPIC_AUTH_TOKEN="30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132"
   export ANTHROPIC_MODEL="cleanapis/claude-opus-5.5"
   export ANTHROPIC_DEFAULT_OPUS_MODEL="cleanapis/claude-opus-5.5"
   export ANTHROPIC_DEFAULT_SONNET_MODEL="cleanapis/claude-opus-5.5"
   export ANTHROPIC_DEFAULT_HAIKU_MODEL="cleanapis/claude-opus-5.5"
   export ANTHROPIC_DEFAULT_FABLE_MODEL="cleanapis/claude-opus-5.5"
   export CLAUDE_CODE_DISABLE_NONESSENTIAL_TRAFFIC="1"

   exec /usr/lib/claude-desktop/claude-desktop "$@"
   ```
3. **Application Entry (`~/.local/share/applications/com.anthropic.Claude.desktop`)**:
   Updated `Exec` lines to target `/home/salman/.local/bin/claude-desktop`.
4. **Desktop Config (`~/.config/Claude/claude_desktop_config.json`)**:
   Injected `managedConfig.env` with gateway endpoints.
5. **Systemd User Environment (`~/.config/environment.d/claude.conf`)**:
   Ensures any background processes or GUI session managers have access to the gateway credentials.

---

## 4. Verification & Testing

### 4.1 Testing via `curl` (Non-Streaming)
```bash
curl -i http://192.168.5.25:8080/v1/messages \
  -H "x-api-key: 30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "cleanapis/claude-opus-5.5",
    "max_tokens": 1024,
    "messages": [{"role": "user", "content": "Respond with PONG"}]
  }'
```

### 4.2 Testing via `curl` (Streaming SSE)
```bash
curl -N -i http://192.168.5.25:8080/v1/messages \
  -H "x-api-key: 30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "cleanapis/claude-opus-5.5",
    "max_tokens": 1024,
    "stream": true,
    "messages": [{"role": "user", "content": "Respond with PONG"}]
  }'
```

Expected SSE event sequence:
```text
event: message_start
data: {"type":"message_start","message":{...}}

event: content_block_start
data: {"type":"content_block_start","index":0,"content_block":{"type":"text","text":""}}

event: content_block_delta
data: {"type":"content_block_delta","index":0,"delta":{"type":"text_delta","text":"PONG"}}

event: content_block_stop
data: {"type":"content_block_stop","index":0}

event: message_delta
data: {"type":"message_delta","delta":{"stop_reason":"end_turn",...}}

event: message_stop
data: {"type":"message_stop"}
```

### 4.3 Testing via Claude Code CLI
Run non-interactive verification:
```bash
claude -p "Respond with 'Claude CLI test successful'"
```

Run calculation or coding task:
```bash
claude -p "What is 123 * 456? Calculate and show the result."
```

---

## 5. Deployment & Server Operations

### Synchronize and Deploy to Home Server
```bash
# 1. Sync updated code
sshpass -p 136517 rsync -avz \
  --exclude '.git' --exclude 'dbbackup' --exclude 'data' --exclude 'server' \
  ./ salman@192.168.5.25:/home/salman/Projects/dockers/clever-ai-gate/

# 2. Rebuild and restart the container
sshpass -p 136517 ssh salman@192.168.5.25 \
  "cd /home/salman/Projects/dockers/clever-ai-gate && docker compose up -d --build app"

# 3. Check health and logs
sshpass -p 136517 ssh salman@192.168.5.25 "docker logs --tail 50 clever-ai-gate-app"
```

---

## 9. Parallel Key Hedging & Exact-Model Failover Architecture

To protect production workflows and Claude Code sessions against rate limits and individual key quota depletion:

1. **Parallel Hedged Execution (Batches of 5)**:
   - For multi-key providers (e.g., CleanAPIs with 30+ keys), requests are evaluated in parallel waves of 5 keys simultaneously.
   - The first worker to receive a valid 2xx response atomically claims victory via `sync.Once`.
   - The winner immediately streams or returns the response to the client (`c.Writer`), while all slower or non-winning workers have their contexts canceled immediately (`cancelOthers`), preventing wasted credits or upstream queue buildup.
   - Individual failed keys (402 balance exhausted, 429 rate limit, 500 error) are penalized individually in the in-memory ring with a 30-minute cooldown, without affecting healthy keys.

2. **Full Provider Key Evaluation Before Switching**:
   - The gateway guarantees that all credentials belonging to the currently selected provider are evaluated across parallel waves before switching providers.
   - Evaluates up to 100 credentials per pool without premature aborts.
   - If wave 1 keys report depleted balances, the gateway seamlessly advances to wave 2, wave 3, etc., ensuring that if even 1 key out of 30+ has available quota, the client request succeeds with sub-second latency.

3. **Prioritized Exact-Model Fallback Chaining**:
   - Only when **all credentials of the active provider have been exhausted** does the gateway switch to another provider hosting the exact same model.
   - Fallback candidates are ranked by reliability (`routerai` 100, `true-sota` 95, `tokenrouter` 90, `zenmux` 85).
   - Problematic/dead accounts (`puter`, `jiekou`, `ofox`, `aplikasinala`) are assigned negative scores (-50) and never chosen over healthy providers.
   - Chaining order on home server: `cleanapis/claude-opus-5.5` (id: 1019401) $\to$ `routerai/claude-opus-5-5` (id: 1006943) $\to$ `true-sota/claude-opus-5.5` (id: 1018255) $\to$ `tokenrouter/anthropic/claude-opus-5.5` (id: 1003878).


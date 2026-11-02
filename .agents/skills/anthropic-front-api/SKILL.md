---
name: anthropic-front-api
description: >-
  Use this skill when configuring, developing, debugging, or testing the Anthropic Messages API
  front door (POST /v1/messages, POST /messages, count_tokens, models) in clever-ai-gate, or when
  integrating Anthropic-protocol clients like Claude Code CLI, Cursor, Cline, or the Anthropic SDK.
---

# Anthropic Front API & Client Integration Skill

This skill documents the procedures, architecture, configuration, and troubleshooting steps for the Anthropic Messages API (`/v1/messages` and `/messages`) implemented in `clever-ai-gate`.

## 1. When to Use This Skill
- Configuring or troubleshooting Claude Code CLI, Cursor, Cline, or official Anthropic SDKs targeting `clever-ai-gate`.
- Routing requests to upstream providers such as `cleanapis`, `openrouter`, `puter`, `cloudflare`, `nvidia`, `openai`, or `ollama`.
- Debugging stream reverse-transmuxing issues (e.g. `message_start`, `content_block_delta`, `message_stop`).
- Deploying and updating the gateway on home servers or remote Docker hosts.

---

## 2. Key Architecture & File Reference

| Component | File Path | Role |
| :--- | :--- | :--- |
| **Front Door Handler** | [internal/proxy/anthropic_front.go](file:///home/salman/Projects/golang/clever-ai-gate/internal/proxy/anthropic_front.go) | Request validation, dispatching, token counting, and response translation |
| **Reverse SSE Transmuxer** | [internal/transmux/anthropic_encoder.go](file:///home/salman/Projects/golang/clever-ai-gate/internal/transmux/anthropic_encoder.go) | Translates upstream OpenAI/provider SSE deltas into Anthropic event stream |
| **Inbound Request Translator** | [internal/proxy/anthropic_translate.go](file:///home/salman/Projects/golang/clever-ai-gate/internal/proxy/anthropic_translate.go) | Translates Anthropic messages & tools into canonical OpenAI completions format |
| **Non-Streaming Translator** | [internal/proxy/anthropic_response.go](file:///home/salman/Projects/golang/clever-ai-gate/internal/proxy/anthropic_response.go) | Translates non-streaming OpenAI chat response into Anthropic message envelope |
| **Dual Auth Middleware** | [internal/middleware/auth.go](file:///home/salman/Projects/golang/clever-ai-gate/internal/middleware/auth.go) | Authenticates `x-api-key` and `Authorization: Bearer` with Anthropic error envelopes |
| **Router Engine** | [internal/router/engine.go](file:///home/salman/Projects/golang/clever-ai-gate/internal/router/engine.go) | Route mounting for `/v1/*proxyPath`, `/messages`, and `/models` |

---

## 3. Configuring Claude Code CLI

Claude Code CLI reads settings from `~/.claude/settings.json`.

### Recommended `~/.claude/settings.json` Configuration
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

### Claude Desktop Setup (Linux)
1. **Desktop Launcher Wrapper**:
   `/home/salman/.local/bin/claude-desktop` exports all `ANTHROPIC_*` environment variables before running `/usr/lib/claude-desktop/claude-desktop`.
2. **Force Console / API Key Authentication**:
   Setting `"forceLoginMethod": "console"` in `~/.claude/settings.json` prevents Claude Desktop's Claude Code / Cowork agent from reverting to personal OAuth accounts.
3. **Desktop Entry**:
   `/home/salman/.local/share/applications/com.anthropic.Claude.desktop` points `Exec` to the wrapper script.
4. **Desktop Config**:
   `~/.config/Claude/claude_desktop_config.json` injects `managedConfig.env`.

### Verification Commands
```bash
# Verify basic response
claude -p "Respond with 'Claude CLI test successful'"

# Verify reasoning & multi-step execution
claude -p "What is 123 * 456? Calculate and show the result."
```

---

## 4. Testing Endpoints via cURL

### 4.1 Test Streaming Messages
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

### 4.2 Test Token Counting
```bash
curl -i http://192.168.5.25:8080/v1/messages/count_tokens \
  -H "x-api-key: 30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132" \
  -H "anthropic-version: 2023-06-01" \
  -H "content-type: application/json" \
  -d '{
    "model": "cleanapis/claude-opus-5.5",
    "messages": [{"role": "user", "content": "Hello world"}]
  }'
```

### 4.3 Test Dual-Schema Model Lookup
```bash
curl -i http://192.168.5.25:8080/v1/models/cleanapis/claude-opus-5.5 \
  -H "x-api-key: 30a135aa79f0b839d8fdeecbe41d4506ef596875d6823132"
```

---

## 5. Deployment Workflow to Home Server

```bash
# 1. Sync code to remote server
sshpass -p 136517 rsync -avz \
  --exclude '.git' --exclude 'dbbackup' --exclude 'data' --exclude 'server' \
  /home/salman/Projects/golang/clever-ai-gate/ salman@192.168.5.25:/home/salman/Projects/dockers/clever-ai-gate/

# 2. Rebuild and restart container
sshpass -p 136517 ssh salman@192.168.5.25 \
  "cd /home/salman/Projects/dockers/clever-ai-gate && docker compose up -d --build app"

# 3. Verify server status
sshpass -p 136517 ssh salman@192.168.5.25 "docker logs --tail 25 clever-ai-gate-app"
```

---

## 6. Troubleshooting Tips
1. **Claude CLI hangs on startup**:
   - Check `~/.claude/settings.json` for dead ports (e.g. `127.0.0.1:20128`).
   - Ensure `ANTHROPIC_BASE_URL` points to `http://192.168.5.25:8080`.
2. **Empty assistant content in stream**:
   - Complex models (like Opus 5.5) generate reasoning/thinking tokens first.
   - If `max_tokens` is too low (e.g. 50), the budget may be exhausted by reasoning tokens before content is emitted.
   - Set `max_tokens >= 1024`.
3. **Route 404 on `/messages`**:
   - Some SDKs strip `/v1` from `ANTHROPIC_BASE_URL`.
   - The gateway router registers both `/v1/*proxyPath` and `/messages`, `/messages/*proxyPath`.
4. **CleanAPIs 402 Insufficient Balance or Large Payload Failures**:
   - When large prompts (50K+ tokens / 200KB body) trigger 402/insufficient balance on CleanAPIs:
   - The gateway's **consecutive provider depletion circuit breaker** immediately trips after 2 consecutive 402 responses.
   - All CleanAPIs credentials in the pool are penalized with a 30-minute cooldown, avoiding futile retries across 100+ keys.
   - The gateway immediately cascades to `routerai/claude-opus-5-5` (3.0s latency) and `true-sota/claude-opus-5.5` (6.0s latency).

---

## 7. Systematic High-Performance Failover & Circuit Breaker Architecture

To ensure zero console errors and sub-second failover for Claude Code:
1. **Provider-Level Circuit Breaker (`internal/proxy/handler.go`)**:
   - Consecutive quota/auth rejections (402 Payment Required, 403 Forbidden, 401 Unauthorized, Insufficient Balance) are tracked per provider.
   - After $\ge 2$ consecutive errors from the same provider, the entire provider is fast-failed in $<1.2\text{s}$ via `p.PenalizeByProvider(provider, 30*time.Minute)`.
2. **Deterministic Scored Candidate Selection (`internal/proxy/model_matcher.go`)**:
   - `FindExactModelFallbacks` ranks candidate pools using baseline reputation scores:
     - `routerai`: 100, `true-sota`: 95, `tokenrouter`: 90, `zenmux`: 85
     - Suspended or dead providers (`puter`, `jiekou`, `ofox`, `aplikasinala`): -50
   - Healthy key availability ratios add up to +20 points.
   - Candidates are sorted descending by score, preventing dead/suspended accounts from ever being prioritized.
3. **Database Fallback Chaining**:
   - Configured in `model_pools.fallback_pool_id`:
     `cleanapis/claude-opus-5.5` (1019401) $\to$ `routerai/claude-opus-5-5` (1006943) $\to$ `true-sota/claude-opus-5.5` (1018255) $\to$ `tokenrouter/anthropic/claude-opus-5.5` (1003878).
   - Database fallbacks are given top priority over automatic pattern matching.
4. **Capped Fallback Attempts**:
   - When switching to a fallback pool, attempts are capped at 3 keys. If a fallback provider is unhealthy, it quickly cascades to the next provider instead of stalling for 20 keys.


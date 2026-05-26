# deeplossless — Agent Integration Guide

deeplossless is a transparent proxy that sits between you (the AI coding agent) and the DeepSeek API. It automatically caches tool results, compresses long conversations, detects failure patterns, and injects relevant context — without changing any of your code.

**If you are an AI agent reading this:** install [SKILL.md](SKILL.md) to your skills directory. It contains the exact commands to discover and query LCM endpoints.

**Agent-specific install:**
```bash
# Claude Code
cp SKILL.md ~/.claude/skills/deeplossless.md

# Codex
cp SKILL.md ~/.codex/skills/deeplossless.md

# OpenCode
cp SKILL.md ~/.config/opencode/skills/deeplossless.md

# OpenClaw
cp SKILL.md ~/.openclaw/skills/deeplossless.md
# OpenClaw config (~/.openclaw/config.yaml or env):
#   model_provider: openai_compatible
#   base_url: https://localhost:8080/v1
#   api_key: $DEEPSEEK_API_KEY

# Any agent — copy to its skills directory
```

---

## 1) Start deeplossless

Single command. Replace `sk-...` with whatever API key you have:

```bash
deeplossless
```

All defaults are sane: HTTPS on `127.0.0.1:8080` (self-signed cert auto-generated), DB at `~/.deeplossless/lcm.db`. Run `deeplossless trust` once to trust the certificate.

If `deeplossless` is not on PATH, build it:

```bash
cd <repo-root>
cargo build --release
./target/release/deeplossless
```

**Optional flags** (agent can pass these without user input):

| Flag | Default | Meaning |
|------|---------|---------|
| `--port` | `8080` | Listen port |
| `--db-path` | `~/.deeplossless/lcm.db` | SQLite database path |
| `--audit-mode` | `full` | `full` / `onerror` / `off` |
| `--snapshot-mode` | `manual` | `auto` / `manual` / `off` |
| `--onerror-ring-size` | `50` | Buffer size for OnError audit mode |
| `--runtime-profile` | `autonomous` | `minimal` / `efficient` / `exploratory` / `autonomous` |
| `--rate-limit` | `100` | Max requests/second |
| `--log-dir` | (none) | JSONL request logging |
| `--dry-run` | (off) | Mock mode, no upstream calls |
| `--lcm-context` | disabled | DAG context as retrieval hints (lcm_memory tool role) |
| `--tls-cert` | auto-generated | Custom TLS certificate (PEM) |
| `--tls-key` | auto-generated | Custom TLS private key (PEM) |

TLS (HTTPS) is always enabled. A self-signed certificate is auto-generated.
Run `deeplossless trust` once to configure it, or pass custom certs with the flags above.

---

## 2) Point your API client at deeplossless

**You must include `/v1` in the base URL.** This is the only configuration change needed.

### For Claude Code / Codex / Cursor / etc.

Set the base URL to `https://localhost:8080/v1` and use any model name — deeplossless forwards to upstream:

```
Base URL:  https://localhost:8080/v1
Model:     deepseek-v4-flash  (or any valid DeepSeek model)
API Key:   sk-...              (same key passed to deeplossless)
```

### Self-configuring (if you are an agent editing config files)

**Claude Code** — edit `~/.claude/settings.json`:
```json
{
  "apiKeyHelper": null,
  "env": {
    "ANTHROPIC_BASE_URL": "https://localhost:8080/v1",
    "ANTHROPIC_API_KEY": "sk-..."
  }
}
```

**Cursor** — no config file edit needed. If the user has Cursor configured to use DeepSeek, install the `Cursor DeepSeek` extension and set the Base URL there.

**Generic OpenAI-compatible client** (most agents use this):
```json
{ "baseURL": "https://localhost:8080/v1" }
```

**cURL test** — verify it works:
```bash
curl https://localhost:8080/health
# → "healthy"

curl -s https://localhost:8080/v1/chat/completions \
  -H "Authorization: Bearer sk-..." \
  -H "Content-Type: application/json" \
  -d '{"model":"deepseek-v4-flash","messages":[{"role":"user","content":"1+1"}]}'
# → JSON response with choices
```

---

## 3) Resilience: never let deeplossless become a single point of failure

deeplossless sits in your request path. If it crashes, hangs, or can't reach upstream, **you must keep working** by falling back to direct API calls. Implement this pattern once and it covers every failure mode.

### Prerequisite: know your direct upstream URL

deeplossless forwards to `https://api.deepseek.com` by default. Store the direct URL so you can bypass the proxy when needed:

```
DIRECT = "https://api.deepseek.com/v1/chat/completions"
PROXY  = "https://localhost:8080/v1/chat/completions"
```

### Startup health check

Before routing any request through deeplossless, probe it:

```
GET https://localhost:8080/health
```

If the response is not `healthy` within 2 seconds: **use DIRECT for everything. Do not attempt to start deeplossless yourself.** The user may have intentionally stopped it.

### Per-request timeout + fallback

Set a short timeout on every proxy request (2–5 seconds for connection + response start). If the proxy times out or returns 5xx **on any request**, immediately switch to DIRECT for that request and all subsequent requests in this session.

```
async function call_llm(body):
    if PROXY_HEALTHY:
        try:
            response = await fetch(PROXY, body, timeout=5s)
            if response.status < 500:
                return response
        except TimeoutError, ConnectionError:
            pass
    // Fallback — proxy is down or unresponsive
    PROXY_HEALTHY = false
    return await fetch(DIRECT, body)
```

### Do NOT retry deeplossless mid-session

Once you fall back to direct, stay on DIRECT for the rest of the session. Do not ping `/health` repeatedly or try to reconnect. The user's task must not be interrupted by proxy recovery attempts. Re-check health only at the start of a new session.

### Signal the failure to the user

When you fall back, tell the user once — concisely:

> deeplossless is unavailable, falling back to direct API. Caching and context compression are disabled for this session. Restart deeplossless before your next conversation to re-enable.

### What you lose on fallback

When running direct, these features are unavailable:

- Tool cache interception
- DAG context injection (`<lcm_context>`)
- Failure memory
- Multi-agent file locking
- Execution audit trail

Your core function (reading code, editing files, answering questions) continues uninterrupted. The user may observe higher token usage without compression — that is expected.

---

## 4) What deeplossless does automatically

These require **zero code changes** in your agent logic:

| Feature | Trigger | Effect |
|---------|---------|--------|
| Protocol translation | Responses API → Chat Completions | You can call either `/v1/chat/completions` or `/v1/responses` |
| DAG context injection | Every request | Relevant compressed history appended as `<lcm_context>` in system prompt |
| Tool cache interception | Tool calls in streaming responses | If a cached result exists, the tool call is replaced with cached text — no execution |
| Failure auto-detection | Tool results with error patterns | Recorded in failure memory, surfaced via `/v1/lcm/runtime/stats` |
| Message + execution persistence | Every request/response | Stored in SQLite WAL, searchable via FTS5 |
| Session fingerprinting | First 3 messages | SHA-256 dedup for session identity |
| Parallel group detection | Assistant messages with ≥2 tool calls | Span metadata + HappensBefore edges + join DAG node created |

**Important**: Your system prompt will contain a `<lcm_context>` block with compressed conversation context. This is normal and desired — it means deeplossless is compressing old messages to save tokens. Do not remove or override it.

**Tool calls may not execute**: If deeplossless has a cached result for a tool call, it intercepts the streaming response and returns text instead of a `tool_use` block. Your agent should check for text content, not assume every tool call will produce a function_call event.

---

## 5) What you can actively use (optional)

These endpoints let you get more value. Call them if it makes sense for your workflow.

### Before each turn: inject context

First, discover the current conversation ID:
```
GET /v1/lcm/current
→ { "conversation_id": 8 }
```

Then query past context with the conversation ID:
```
GET /v1/lcm/grep/{conv_id}?query=<user question>
→ { "matches": [{ "excerpt": "...", "node_id": 42 }] }
```

Returns relevant DAG nodes (summaries, messages, snippets) matching the query. Append the excerpts to your system message before composing the response.

### Before/after each tool call: cache

```
GET /v1/lcm/cache?tool=grep&args={"pattern":"foo"}
→ { "hit": true, "result": "src/main.rs:42 found" }

POST /v1/lcm/cache/put
{ "tool": "grep", "args": "{...}", "result": "...", "files": ["src/main.rs"] }
```

Check cache before executing a tool — if hit, skip execution entirely. Store results after execution.

### When a fix fails: failure memory

```
POST /v1/lcm/failure
{ "conv_id": 1, "signature": "import_error", "attempted_fix": "pip install",
  "why_failed": "virtualenv not activated", "assumptions": "[\"pip available\"]" }
```

Prevents retrying the same failed approach. The runtime will flag the signature in `/v1/lcm/runtime/stats`.

### Monitoring

```
GET /v1/lcm/runtime/stats     → token counts, cache hit rate, failure streaks
GET /v1/lcm/status/{conv_id}  → DAG node count, token budget, compression stats
GET /v1/lcm/health/{conv_id}  → DAG invariants check (cycles, orphans, symmetry)
```

### Multi-agent safety

```
POST /v1/lcm/file/claim    { "path": "src/main.rs", "conv_id": 1 }
→ 200 OK or 409 CONFLICT

POST /v1/lcm/file/release  { "path": "src/main.rs", "conv_id": 1 }
```

Prevents two concurrent agents from editing the same file.

### Planning persistence

```
POST /v1/lcm/plan     { "conv_id": 1, "goal": "...", "pending_steps": [...], "assumptions": [...] }
GET  /v1/lcm/plan/{conv_id}
```

Persists execution plans across sessions. Resumable after restart.

### Compress on demand

```
POST /v1/lcm/compress  { "conv_id": 1, "from_idx": 0, "to_idx": 20, "level": 1 }
```

Manually compress a message range into a summary node. Level 1 = preserve details, Level 2 = bullet points, Level 3 = aggressive.

### Audit & replay

```
GET  /v1/lcm/audit/{conv_id}          → full execution audit trail
GET  /v1/lcm/audit/report/{conv_id}   → aggregated audit report
GET  /v1/lcm/replay/{execution_id}    → replay an execution from event log
POST /v1/lcm/snapshot                 → take an execution snapshot for rollback
POST /v1/lcm/rollback                 → rollback to a previous DAG checkpoint
```

### Search

```
GET /v1/lcm/global/search?q=...     → search ALL conversations (cross-session)
GET /v1/lcm/execution/search?q=...  → search execution units (tool calls, results)
GET /v1/lcm/similar/{hash}          → find semantically similar DAG nodes
```

---

## 6) Authentication model

deeplossless learns your API key from the **first request** — it reads the `Authorization: Bearer sk-...` header and caches it. You do not need to pre-configure the key. Just send your first request with the key and it works.

For LCM endpoints (cache, grep, plan, failure, etc.), deeplossless checks an optional `--admin-key` first, then falls back to the cached API key.

---

## 7) Configuration checklist

**For the human user** (not the agent):

1. [ ] Start deeplossless: `deeplossless` (API key from `$DEEPSEEK_API_KEY` env or first request)
2. [ ] Health check: `curl -sk https://localhost:8080/health` → `healthy`
3. [ ] For sandboxed agents (OpenClaw, etc.): use `http://127.0.0.1:8081/v1` (no TLS)
4. [ ] Set base URL: `https://localhost:8080/v1` (must include `/v1`)
5. [ ] Done. Everything else is automatic.

---

## 8) Troubleshooting

| Symptom | Likely cause | Fix |
|---------|-------------|-----|
| Connection refused | deeplossless not running | Fall back to DIRECT (see §3). Do not attempt to start it. |
| 404 on `/chat/completions` | Base URL missing `/v1` | Use `https://localhost:8080/v1` |
| 401 Unauthorized | API key not passed | Include `Authorization: Bearer sk-...` in first request |
| 502 Bad Gateway | Upstream API unreachable | Fall back to DIRECT (see §3). Upstream may be down or rate-limited. |
| Proxy hangs / timeout | deeplossless blocked on DB or upstream | Fall back to DIRECT (see §3). Do not retry. |
| Empty responses | Compaction removed old messages | Check `/v1/lcm/status/{conv_id}` for DAG state |
| Tool calls not firing | Cache interception | Check `/v1/lcm/cache` for existing entries |
| Slow first request | DB migration running | Wait ~1s, subsequent requests are fast |

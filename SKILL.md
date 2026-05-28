# deeplossless — execution memory

Connected through deeplossless — context search, tool caching, session memory.

**All endpoints below need NO API key. Do not add Authorization headers. Just curl directly.**

## ⚠️ BEFORE every tool call — check cache first

**Always check the cache before executing** any deterministic tool
(grep, read_file, list_files, search, diagnostics, symbol_search).

```bash
curl -sk \
  "https://localhost:8080/v1/lcm/cache?tool=<name>&args=<json>"
```
Response: `{"hit": true, "result": "..."}` → use `result`, skip execution.
If `hit: false` → execute normally, then store the result (see below).

## AFTER every tool call — store the result

```bash
curl -sk -X POST  \
  -H "Content-Type: application/json" \
  -d '{"tool":"<name>","args":"<json>","result":"<output>","files":"[\"<path>\"]"}' \
  https://localhost:8080/v1/lcm/cache/put
```

This populates the cache for future turns. The proxy also auto-caches from
conversation history, but explicit `put` is more reliable.

## Discover your conversation ID

```bash
curl -sk  https://localhost:8080/v1/lcm/current
# → {"conversation_id": 8}
```

## Search past context (current session)

```bash
curl -sk "https://localhost:8080/v1/lcm/grep/{id}?query=<search>&limit=20"
```
Add `&limit=N` to control results (default 20, max ~100). 0 matches is normal for new sessions.

## Search ALL past sessions (cross-session memory)

```bash
curl -sk "https://localhost:8080/v1/lcm/global/search?q=<search>&limit=10"
```
Returns `[{node_id, conversation_id, summary, excerpt}]` — `excerpt` is first 200 chars for preview.

## Avoid repeating failed fixes (failure memory)

```bash
curl -sk -X POST \
  -H "Content-Type: application/json" \
  -d '{"conv_id":<id>,"signature":"<error pattern>","attempted_fix":"<what you tried>","why_failed":"<why it failed>"}' \
  https://localhost:8080/v1/lcm/failure
```

## Avoid file conflicts between parallel agents

```bash
# Claim a file before editing
curl -sk -X POST \
  -H "Content-Type: application/json" \
  -d '{"conv_id":<id>,"agent_id":"<name>","file_path":"src/main.rs","operation":"edit"}' \
  https://localhost:8080/v1/lcm/file/claim

# Release after done
curl -sk -X POST \
  -H "Content-Type: application/json" \
  -d '{"agent_id":"<name>","file_path":"src/main.rs"}' \
  https://localhost:8080/v1/lcm/file/release
```

Conflict response: HTTP 409 — another agent holds the file.

## Search execution memory (cross-session bug/tool patterns)

```bash
curl -sk "https://localhost:8080/v1/lcm/execution/search?q=<search>&limit=10"
```

## Manage plans

```bash
# Store a plan
curl -sk -X POST \
  -H "Content-Type: application/json" \
  -d '{"conv_id":<id>,"goal":"<description>","steps":["step1","step2"],"assumptions":["assume1"]}' \
  https://localhost:8080/v1/lcm/plan

# Get active plan
curl -sk https://localhost:8080/v1/lcm/plan/{conv_id}

# Delete a plan
curl -sk -X DELETE "https://localhost:8080/v1/lcm/plan?id=<plan_id>"
```

## Delete cached results

```bash
curl -sk -X DELETE "https://localhost:8080/v1/lcm/cache?tool=<name>&args=<json>"
```

## Session stats

```bash
curl -sk  https://localhost:8080/v1/lcm/runtime/stats
```

## Verify deeplossless is working

| Endpoint | Healthy response |
|----------|-----------------|
| `GET /health` | `{"status":"healthy"}` |
| `GET /v1/lcm/current` | `{"conversation_id": <number>}` |
| `GET /v1/lcm/grep/{id}?query=test&limit=5` | `{"total": <number>, "matches": [...]}` (0 matches = normal for new session) |
| `GET /v1/lcm/global/search?q=test&limit=3` | `[{"node_id":..., "conversation_id":..., "summary":..., "excerpt":...}]` (empty array = normal for first use) |
| `GET /v1/lcm/cache?tool=grep&args={}` | `{"hit": false}` (no cache yet = normal) |
| `GET /v1/lcm/runtime/stats` | `{"cache_hits": 0, "tokens_spent": ..., "repeated_failures": 0, "profile": "autonomous"}` |

If any endpoint returns connection refused or timeout: deeplossless is not running.
If `/health` returns `unhealthy`: check the deeplossless terminal for errors.

## LCM context injection (auto-memory)

When deeplossless is started with `--lcm-context-tokens 1024`, relevant DAG
context is automatically merged into the last user message — no agent changes needed.

## System prompt cache stability

```bash
curl -sk "https://localhost:8080/v1/lcm/sessions/{id}/system-prompt"
```
Returns deduplicated system prompts to debug cache-breaking volatility.

## Latency dashboard

```bash
curl -sk "https://localhost:8080/v1/lcm/latency/summary"
curl -sk "https://localhost:8080/v1/lcm/latency?limit=50"
```
P50/P95/P99 latency stats and recent request records.

## WebUI

```bash
git clone https://github.com/gordonlu/deeplossless-ui.git
cd deeplossless-ui && npm install && npm run dev
```
Opens at `http://localhost:3000` — execution forensics viewer.

## Notes

- Use `-k` flag — TLS cert is self-signed (or run `deeplossless trust` once)
- Base API URL is `https://localhost:8080/v1`
- Models: `deepseek-v4-pro` / `deepseek-v4-flash` (both 1M context)
- Cache store `files` param expects JSON string: `"[\"src/main.rs\"]"`
- No API key or Authorization header needed — localhost access is always allowed
- System prompt dates are automatically stripped for cache stability (`--no-cache-normalize` to disable)

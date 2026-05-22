# deeplossless Agent Integration Guide

deeplossless is an inference-aware execution runtime for AI coding agents.
v0.4.0.

---

## Two layers: Automatic vs. Active

deeplossless operates in two tiers:

### Automatic (zero agent changes)

These work transparently through the proxy, no agent code required:

| Feature | How |
|---------|-----|
| Protocol translation | Responses API ↔ Chat Completions, canonical IR |
| DAG context injection | `<lcm_context>` appended to system messages |
| Tool cache interception | Stream-level: detects tool calls, returns cached results inline |
| Pipeline auto-caching | Tool results extracted from conversation history |
| Failure auto-detection | Pipeline detects error patterns in tool results |
| Message persistence | Every request/response stored in SQLite WAL |
| Execution unit extraction | Tool call chains grouped into `think → act → observe → reflect` |
| Session fingerprinting | SHA-256 over first 3 messages |
| Debug dump | `GET /v1/lcm/runtime/debug-dump` (counters only, no user content) |

### Active (agent must call these endpoints)

These require the agent to make explicit HTTP calls. They will NOT fire automatically.

| Endpoint | When to call | Why |
|----------|-------------|-----|
| `GET /v1/lcm/cache?tool=&args=` | Before executing a tool | Skip execution if result is cached |
| `POST /v1/lcm/cache/put` | After executing a tool | Store result for future reuse |
| `GET /v1/lcm/grep/{conv_id}?query=` | Before composing the next prompt | Inject relevant DAG context into system message |
| `GET /v1/lcm/status/{conv_id}` | Periodically | Monitor DAG health (tokens, node count, level) |
| `GET /v1/lcm/expand/{node_id}` | When the model asks about history | Expand a compressed summary back to original messages |
| `GET /v1/lcm/global/search?q=` | Cross-session context needed | Search ALL conversations for relevant patterns |
| `GET /v1/lcm/execution/search?q=` | Debugging or learning | Find similar bugs, tool chains, code edits |
| `POST /v1/lcm/plan` | After creating a plan | Persist execution state |
| `GET /v1/lcm/plan/{conv_id}` | Resuming work | Read active plan |
| `POST /v1/lcm/failure` | After a fix fails | Record why_failed + invalidated assumptions |
| `POST /v1/lcm/file/claim` | Before editing a file | Multi-agent safety (409 if conflicted) |
| `POST /v1/lcm/file/release` | After editing | Release lock |
| `POST /v1/lcm/compress` | Context getting too long | Compress a message range into a summary node |
| `POST /v1/lcm/rollback` | Need to undo | Rollback to a previous DAG checkpoint |
| `GET /v1/lcm/replay/{execution_id}` | Audit or debug | Reconstruct full StreamEvent sequence from event log |
| `POST /v1/lcm/snapshot` | Before important operations | Take an execution snapshot for later rollback |
| `GET /v1/lcm/versions` | Audit trail | List memory version history |

---

## Active integration pattern

The critical pattern for agents that want to maximize deeplossless:

```
Before each turn:
  1. GET /v1/lcm/grep/{conv_id}?query=<user's question>
     → Append returned DAG context to system message

Before each tool call:
  2. GET /v1/lcm/cache?tool=<name>&args=<json>
     → If hit: use cached result, skip execution
     → If miss: execute normally

After each tool call:
  3. POST /v1/lcm/cache/put  {tool, args, result, files}
     → Store for future interception

After each fix failure:
  4. POST /v1/lcm/failure  {signature, attempted_fix, why_failed, assumptions, files}
     → Prevent the agent from retrying known-bad approaches
```

Example hook (Python):

```python
import requests, json, os

BASE = "http://127.0.0.1:8080"
CONV_ID = os.getenv("CONV_ID", "1")
AUTH = {"Authorization": f"Bearer {os.getenv('ADMIN_KEY', '')}"}

def before_turn(user_query):
    """Inject DAG context into the prompt."""
    r = requests.get(f"{BASE}/v1/lcm/grep/{CONV_ID}", params={"query": user_query}, headers=AUTH)
    if r.ok:
        matches = r.json().get("matches", [])
        return "\n".join(m["preview"] for m in matches[:5])
    return ""

def before_tool(tool_name, args_json):
    """Check cache before executing a tool."""
    r = requests.get(f"{BASE}/v1/lcm/cache", params={"tool": tool_name, "args": args_json}, headers=AUTH)
    if r.ok and r.json().get("hit"):
        return r.json()["result"]  # Use cached result, skip execution
    return None

def after_tool(tool_name, args_json, result, files=None):
    """Store tool result in cache."""
    requests.post(f"{BASE}/v1/lcm/cache/put", json={
        "tool": tool_name, "args": args_json,
        "result": result,
        "files": json.dumps(files or [])
    }, headers=AUTH)

def after_failure(signature, fix_attempted, why_failed, assumptions=None, files=None):
    """Record a failed fix to prevent retrying it."""
    requests.post(f"{BASE}/v1/lcm/failure", json={
        "conv_id": CONV_ID, "signature": signature,
        "attempted_fix": fix_attempted, "why_failed": why_failed,
        "assumptions": json.dumps(assumptions or []),
        "files": json.dumps(files or [])
    }, headers=AUTH)
```

---

## What agents should expect

- The system prompt contains a structured `<lcm_context>` block (automatic)
- Stream-level tool cache interception may replace tool calls with cached results (automatic)
- Repeated operations on unchanged files may be short-circuited (automatic)
- **Active endpoints return real data but require the agent to call them** — they are not automatically injected

Agents should NOT expect:

- Every tool call executes physically (cache layer may intercept)
- The full conversation history is in every prompt (DAG compresses it)
- Failures are forgotten (failure memory persists across turns)

---

## How to evaluate

```bash
cargo test --test long_session_benchmark -- --nocapture
cargo test --test proxy_integration          # 11 integration tests incl. cache + LCM
python3 bench/run.py
```

Measure inference economics, not retrieval quality:
- Repeated tool calls avoided
- Cache hit rate across long sessions
- Failure loops broken
- Execution continuity across turns

# deeplossless Agent Integration Guide

deeplossless is an inference-aware runtime that sits as an OpenAI-compatible proxy
between AI coding agents and the LLM API. It reduces repeated reasoning, tool
execution, repo rereads, and planning during long coding sessions.

This document is accurate as of v0.2.0. Some runtime features (tool cache,
failure memory, plan persistence) have working storage backends but are not yet
exposed as HTTP endpoints.

---

## What happens automatically

When an agent connects to deeplossless as its API proxy, the following happens
without any extra work by the agent:

| Feature | Mechanism |
|---------|-----------|
| Message persistence | Every request/response stored verbatim in SQLite WAL |
| DAG context assembly | Summaries + recent messages injected into system prompt |
| Execution unit extraction | Tool call chains (`think → act → observe → reflect`) extracted from message history |
| Session fingerprinting | SHA-256 over first 3 messages for multi-turn conversation tracking |
| Context injection | Three-tier `<lcm_context>` block (Summaries → Snippets → Recent) |
| Code change auto-invalidation | Stored code changes trigger cache invalidation for affected files |

## What the agent can call directly

These features have HTTP endpoints. Agents can call them in a lightweight hook layer.

| Feature | Endpoint | When to call |
|---------|----------|--------------|
| Tool cache lookup | `GET /v1/lcm/cache?tool=&args=` | Before executing a tool |
| Tool cache store | `POST /v1/lcm/cache/put` | After executing a tool |
| Failure pattern store | `POST /v1/lcm/failure` | After a fix fails |
| Plan store | `POST /v1/lcm/plan` | After creating a plan |
| Plan read | `GET /v1/lcm/plan/{conv_id}` | To resume a plan |

All endpoints use JSON. Tool args are normalized via SHA-256 hash
to produce deterministic cache keys regardless of argument order.

---

## Architecture

```
Agent
  ↓  POST /v1/chat/completions
deeplossless (proxy)
  ├─ fingerprint → store messages → DAG assemble → inject context
  ↓  forwarded with context
Upstream LLM API (DeepSeek / OpenAI)
```

Optional hook layer (~10 lines per hook):

```python
def before_tool(tool, args):
    r = requests.get("http://127.0.0.1:8080/v1/lcm/cache",
                      params={"tool": tool, "args": json.dumps(args)})
    if r.json().get("hit"):
        return r.json()["result"]  # skip execution

def after_tool(tool, args, result, files):
    requests.post("http://127.0.0.1:8080/v1/lcm/cache/put", json={
        "tool": tool, "args": json.dumps(args),
        "result": result, "files": json.dumps(files)})

def after_failure(sig, fix, why, files):
    requests.post("http://127.0.0.1:8080/v1/lcm/failure", json={
        "signature": sig, "attempted_fix": fix,
        "why_failed": why, "files": json.dumps(files)})

def after_plan(goal, steps, assumptions):
    requests.post("http://127.0.0.1:8080/v1/lcm/plan", json={
        "goal": goal, "steps": json.dumps(steps),
        "assumptions": json.dumps(assumptions)})
```

---

## Installation

```bash
git clone https://github.com/gordonlu/deeplossless.git
cd deeplossless
cargo build --release
```

---

## Starting the runtime

```bash
DEEPSEEK_API_KEY=sk-... ./target/release/deeplossless
```

The proxy listens on `http://127.0.0.1:8080`. Point any OpenAI-compatible client
at this base URL.

```bash
# Optional: select strategy profile
RUNTIME_PROFILE=efficient ./target/release/deeplossless
```

---

## Runtime profiles

| Profile | Cache | Retries | Speculative | Context | Plan Freeze | Token Budget |
|---------|-------|---------|-------------|---------|-------------|-------------|
| `minimal` | 100% | 1 | No | 20% | Yes | 30% |
| `efficient` | 80% | 2 | No | 50% | No | 60% |
| `exploratory` | 50% | 3 | Yes | 80% | No | 80% |
| `autonomous` | 30% | 5 | Yes | 100% | No | 95% |
| `custom` | user-defined via `RUNTIME_CACHE=`, `RUNTIME_RETRIES=`, etc. |

Profiles control the runtime policy engine's aggressiveness, not agent behavior.
All policy decisions are advisory — the agent can accept, ignore, or override
each recommendation.

---

## API endpoints for agents

### Read-only (available without extra integration)

```
GET  /v1/lcm/grep/{conv_id}?query=      FTS5 full-text search
GET  /v1/lcm/status/{conv_id}           DAG health (tokens, leaves, level)
GET  /v1/lcm/expand/{node_id}           Expand summary to children
GET  /v1/lcm/trace/{node_id}            Sentence-level provenance
GET  /v1/lcm/snippets/{node_id}         Extracted precision-critical values
GET  /v1/lcm/global/search?q=&limit=    Cross-session semantic search
GET  /v1/lcm/execution/search?q=        Find similar bugs, tool chains, edits
GET  /v1/lcm/runtime/stats              JSON metrics (tokens, cache rate, failures)
GET  /v1/lcm/runtime/report?label=&format=md|svg   Shareable session recap
```

### Active operations (available without extra integration)

```
POST /v1/lcm/compress  {conv_id, from, to}   Compress message range
POST /v1/lcm/delete    {conv_id, id}         Soft-delete from context
POST /v1/lcm/rollback  {conv_id, id}         Rollback to checkpoint
```

### Multi-agent safety

```
POST /v1/lcm/file/claim    {agent_id, file_path}    Claim file (409 if held)
POST /v1/lcm/file/release  {agent_id, file_path}    Release claim
GET  /v1/lcm/file/conflicts                         List all active claims
```

---

## Recommended tool categories for caching

Highest reuse value (deterministic, frequently repeated):

- `grep` / `search_content` / `search_code`
- `read_file` / `read`
- `list_files` / `ls` / `tree`
- `symbol_search` / `workspace_symbol`
- `diagnostics`

Lower reuse value:

- Non-deterministic external APIs
- Timestamp-dependent operations
- Operations on rapidly-changing state (e.g., compile/test output)

---

## Failure memory

When populated via `store_failure_pattern()`, deeplossless records:

- **signature** — normalized error message
- **attempted_fix** — what the agent tried
- **why_failed** — why the fix didn't work (the critical field)
- **invalidated_assumptions** — assumptions that turned out wrong
- **related_files** — files involved

The runtime policy engine can then recommend `RetryWithFix` instead of
blind retry when the same failure pattern is detected.

This is per-pattern, not global — retrying "E0308 type mismatch" doesn't
count against retries for "KeyError column_name".

---

## Plan persistence

When populated via `store_plan_state()`, deeplossless stores:

```
goal, pending_steps, completed_steps, blocked_steps,
invalidated_steps, assumptions, is_active
```

Plans are deactivated when new plans are created. File changes mark active
plans as potentially stale (assumptions may have been invalidated).

---

## What agents should expect

Agents using deeplossless should expect:

- Tool results may be served from cache (`GET /v1/lcm/cache`)
- Execution state persists across turns (messages, DAG nodes, provenance)
- The system prompt contains a structured `<lcm_context>` block
- Repeated operations on unchanged files may be short-circuited

Agents should NOT expect:

- Every tool call executes physically (cache layer may intercept)
- The full conversation history is in every prompt (DAG compresses it)
- Failures are forgotten (failure memory persists across turns)

---

## What the runtime is NOT

deeplossless does NOT:

- Replace the agent's planning or reasoning loop
- Control which tools the agent can call
- Modify the agent's prompts or decisions
- Force optimization — all policy decisions are advisory by default

---

## How to evaluate

Measure what the runtime reduces, not what it compresses:

| Good metrics | Bad metrics |
|-------------|-------------|
| Repeated rereads avoided | Compression ratio |
| Repeated planning avoided | Retrieval quality |
| Repeated tool calls reused | Embedding similarity |
| Failure loops prevented | Prompt length |
| Execution continuity across long sessions | Raw QPS |

Benchmark:
```bash
cargo test --test long_session_benchmark -- --nocapture
cargo test --test perf_benchmark -- --nocapture
python3 bench/run.py
```

---

## Recommended workloads

Best evaluation workloads:

- Long coding sessions (50+ turns)
- Dependency debugging with layered failures
- Partial symbol renames (some imports stale)
- Configuration drift (keys renamed, not all files updated)
- Misleading error chains (stack trace points wrong direction)

Weak evaluation workloads:

- Single-turn chat
- Trivia QA
- Short RAG demos

---

## Design note

deeplossless measures its success by inference economics — how much repeated
work it prevents — not by context window size or retrieval quality.

The core thesis: long AI coding sessions waste more tokens on repeated
inference than on insufficient context.

# Technical Reference

## Architecture

DeepLossless operates in two layers, sitting as an OpenAI-compatible proxy
between your client and the DeepSeek API:

### Layer 1: Memory (store & organize)

| Component | Role |
|-----------|------|
| **Semantic DAG** | True shared graph with embedding-based dedup (cosine ≥0.85 auto-merge), BM25 retrieval, sentence-level provenance spans |
| **Tree-sitter AST extraction** | 8 languages (Rust, Python, TS, JS, Java, C/C++, C#, Go) — precise function/class/type signatures extracted before compression |
| **Entropy-aware compaction** | Trigram novelty scoring — novel content preserved, redundant content aggressively compressed |
| **Memory scoring** | Access count + recency + importance with decay-based GC. Three-tier retention |
| **Code diff memory** | Stores what *changed* (file, diff, symbols, errors), not full code blocks |

### Layer 2: Runtime (optimize execution)

| Component | Role |
|-----------|------|
| **Tool Result Cache** | Deterministic `hash(tool + args)` cache with partial file-based invalidation. Zero-token reuse for grep/read_file/search |
| **Failure Memory** | Stores failed reasoning paths (`why_failed` + `invalidated_assumptions`), not just error strings. Prevents error loop token waste |
| **Plan Persistence** | Execution state (goal, steps, assumptions), not plan text. Avoids repeated planning |
| **Execution Units** | Agent memory atoms: `think → act → observe → reflect` cycles with outcome inference |
| **Runtime Policy** | Advisory decisions with confidence scores + estimated token savings |
| **Event Sourcing** | Append-only execution_events table — every StreamEvent persisted for replay |
| **Replay Engine** | Deterministic reconstruction of execution sequences from event log |
| **Snapshot Isolation** | Copy-on-write memory versions with budget-aware retention tiers (L0–L3) |

## Design Principles

- **Reasoning is expensive.** Don't redo it.
- **Repeated inference is waste.** Cache it.
- **Context windows are not memory.** Execution state is.
- **Stable execution state beats repeated replanning.**
- **Runtime policy should optimize, not control.** Advisory, configurable, overrideable.
- **Compression alone is insufficient.** Need reuse, avoidance, and distillation.
- **Incremental reasoning is more scalable than ever-growing context.**
- **Inspired more by incremental compilation than traditional chat memory.**

## Runtime Strategies

The runtime policy layer is **advisory and configurable**: users can prioritize
token efficiency, exploratory reasoning, or autonomous execution depending on
workload. The agent/UI can accept, ignore, or override each recommendation.

| Profile | Cache | Retries | Speculative | Context | Freeze Plans | Token Budget |
|---------|-------|---------|-------------|---------|-------------|-------------|
| **Minimal** | 100% | 1 | No | 20% | Yes | 30% |
| **Efficient** | 80% | 2 | No | 50% | No | 60% |
| **Exploratory** | 50% | 3 | Yes | 80% | No | 80% |
| **Autonomous** | 30% | 5 | Yes | 100% | No | 95% |
| **Custom** | user-defined | user-defined | user-defined | user-defined | user-defined | user-defined |

Custom profile config via environment:
```
RUNTIME_CACHE=0-1
RUNTIME_RETRIES=0-10
RUNTIME_SPECULATIVE=true|false
RUNTIME_CONTEXT=0-1
RUNTIME_FREEZE=true|false
RUNTIME_BUDGET=0.1-1
```

## API Reference

### Proxy

```
POST /v1/chat/completions     — OpenAI-compatible proxy, SSE streaming, DAG context injected
POST /v1/responses            — Responses API → Chat Completions (enables Codex + DeepSeek)
```

Model names are auto-mapped: `gpt-5*` → `deepseek-v4-pro`, `gpt-*-mini` → `deepseek-v4-flash`.

### Memory

```
GET  /v1/lcm/grep/{conv_id}?query=     — FTS5 BM25 full-text search
GET  /v1/lcm/expand/{node_id}          — Expand summary to children
GET  /v1/lcm/status/{conv_id}          — DAG health (tokens, leaves, level)
GET  /v1/lcm/snippets/{node_id}        — View extracted precision-critical values
GET  /v1/lcm/trace/{node_id}           — Sentence-level provenance with source excerpts
GET  /v1/lcm/stream/{conv_id}?budget=  — Streaming DAG context (SSE incremental delivery)
```

### Runtime

```
GET  /v1/lcm/global/search?q=&limit=     — Cross-session semantic search
GET  /v1/lcm/execution/search?q=         — Execution memory: bugs, tool chains, code edits
GET  /v1/lcm/runtime/stats               — Runtime metrics (tokens, cache rate, failures)
GET  /v1/lcm/runtime/report?conv_id=&format= — Session report (markdown or SVG share card)
GET  /v1/lcm/runtime/debug-dump          — Structured dump for GitHub issues (no user content)
GET  /v1/lcm/sessions                    — List all conversations
GET  /v1/lcm/sessions/{id}/events        — Execution events for a session
GET  /v1/lcm/sessions/{id}/system-prompt — Deduplicated system prompt history
GET  /v1/lcm/latency                     — Recent upstream latency records
GET  /v1/lcm/latency/summary             — Aggregated P50/P95/P99 latency
GET  /v1/lcm/cache/stability             — System prompt cache stability
```

### Replay

```
GET  /v1/lcm/replay/{execution_id}         — Reconstruct execution from event log
POST /v1/lcm/snapshot                       — Take an execution snapshot
GET  /v1/lcm/versions                       — List memory version history
```

### Agent Hooks

```
GET  /v1/lcm/cache?tool=&args=           — Check tool cache before execution
POST /v1/lcm/cache/put                   — Store tool result after execution
POST /v1/lcm/failure                     — Record failure pattern
POST /v1/lcm/plan                        — Store execution plan
GET  /v1/lcm/plan/{conv_id}              — Read active plan
POST /v1/lcm/file/claim                  — Claim file ownership (409 if conflict)
POST /v1/lcm/file/release                — Release file ownership
GET  /v1/lcm/file/conflicts              — List active file claims
```

### Operations

```
POST /v1/lcm/compress  {conv_id, from, to}  — Compress node range
POST /v1/lcm/delete    {conv_id, id}        — Soft-delete from active context
POST /v1/lcm/rollback  {conv_id, id}        — Rollback to checkpoint
POST /health                                — Health check (DB, upstream, compactor)
GET  /metrics                               — Prometheus metrics
```

## Benchmarks

### Scope and methodology

Current benchmarks use a **deterministic local agent loop** to isolate
runtime-level reuse effects (cache reuse, reread avoidance, failure memory,
plan persistence) from model variance. Real-world reductions with external
LLMs are expected to be **lower but follow similar trends**.

### Inference redundancy benchmark (4 tasks)

```
                    Baseline tokens    Runtime tokens    Reduction
Dependency mismatch      11,602             2,626           ↓77%
Symbol rename             5,105             1,120           ↓78%
Config drift              9,583             2,225           ↓77%
Misleading error          7,074             1,092           ↓85%
──────────────────────────────────────────────────────────
TOTAL                    33,364             7,063           ↓79%
Cache hits: 205. Rereads avoided: 96.
```

Run: `python3 bench/run.py`

### Long session (3 tasks, 86 turns)

```
                    Vanilla Agent    Runtime    Reduction
Tokens / session        21,070       13,500       ↓36%
Repeated replans           14            5         ↓64%
Repeated failures           8            3         ↓62%
Repo rereads               11            2         ↓82%
Cache hit rate              —           28%           —
Failures broken             —            5            —
```

Run: `cargo test --test long_session_benchmark -- --nocapture`

### How to benchmark

```bash
cargo test --test long_session_benchmark -- --nocapture   # 86-turn punchline
cargo test --test simulated_session -- --nocapture         # 20-turn detailed log
cargo bench                                                # micro-benchmarks

# Live runtime metrics (requires proxy running)
curl http://127.0.0.1:8080/v1/lcm/runtime/stats | jq .
```

### Micro-benchmarks (criterion)

```
Token counting (8K lines):          7.8 ms
Snippet extraction (4K lines):      5.8 ms
DAG assembly (1K nodes):          483 μs
Session fingerprint:               124 ns
Runtime cache decision:        sub-microsecond
Runtime full decision cycle:   sub-microsecond
Reasoning distillation (20 calls):  microseconds
```

## Codex + DeepSeek (Detailed)

deeplossless translates OpenAI's Responses API to Chat Completions, enabling
Codex to work with DeepSeek. Model names are auto-mapped: `gpt-5*` →
`deepseek-v4-pro`, `gpt-*-mini` → `deepseek-v4-flash`.

### Limitations with Codex

Codex uses a **client-side execution model** — tool calls, retries, and plan
state are managed inside the Codex process. deeplossless operates at the
canonical IR layer between Codex and DeepSeek, so some features work
transparently while others require agent-side LCM API integration:

| Feature | Available via Codex? | How |
|---------|:--:|------|
| Protocol translation (Responses → Chat) | YES | Canonical IR bidirectional translation |
| Tool Cache Interception | YES | Stream-level: detects tool calls, returns cached results inline |
| DAG Context Injection | YES | Merged into last user message (preserves tool chains) |
| Pipeline Auto-Caching | YES | Tool results extracted from conversation history automatically |
| Failure Auto-Detection | YES | Pipeline detects error patterns from tool results |
| Tool Cache (manual) | NO | Codex doesn't query `GET /v1/lcm/cache` |
| Failure Memory (manual) | NO | Codex doesn't query failure endpoints |
| Plan Persistence | NO | Codex maintains its own plan state |
| File Ownership Tracking | NO | Unsupported |
| Runtime Policy | NO | Decisions made by Codex, not the proxy |

Features marked YES work **transparently** — Codex doesn't need to know they exist.
The canonical IR layer intercepts and optimizes at the protocol level.

## Attribution

- **LongSeeker** — *Context-ReAct: Elastic Context Orchestration for Long-Horizon Search Agents* (May 2026)
- **LCM Paper** — Ehrlich & Blackman. *LCM: Lossless Context Management* (2026)

# deeplossless

**Inference-aware coding runtime for DeepSeek agents.**

deeplossless turns repeated inference into reusable execution state — reducing
repeated reasoning, tool execution, repo rereads, and context reconstruction
in long AI coding sessions.

Instead of relying on ever-growing context windows, deeplossless incrementally
reuses execution state, plans, failures, and tool results.

```
Typical coding session (simulated, 20 turns):

Vanilla:  7440 tokens  |  3 repeated planning rounds  |  2 rereads  |  1 retry loop
Runtime:  4500 tokens  |  1 planning round            |  0 rereads  |  0 retry loops
          ↓40% tokens  |  ↓67% replanning             |  ↓100%      |  ↓100%

Run: cargo test --test simulated_session -- --nocapture
```

> **109 tests pass. CI: check → clippy → test → doc.**

## Why

Long coding sessions waste most tokens on repeated work:

- rereading unchanged files
- repeated grep/search/compile
- replanning the same tasks
- retrying known-bad fixes
- reconstructing prior reasoning

**Long context ≠ efficient reasoning.** Context windows store more, but
don't prevent inference recomputation. deeplossless treats reasoning as a
reusable runtime resource — caching tool results, remembering failed paths,
persisting plan state, and injecting only what changed.

## Design Principles

- **Reasoning is expensive.** Don't redo it.
- **Repeated inference is waste.** Cache it.
- **Context windows are not memory.** Execution state is.
- **Stable execution state beats repeated replanning.**
- **Runtime policy should optimize, not control.** Advisory, configurable, overrideable.
- **Compression alone is insufficient.** Need reuse, avoidance, and distillation.
- **Incremental reasoning is more scalable than ever-growing context.**

## Architecture

deeplossless sits as an OpenAI-compatible proxy between your client and the
DeepSeek API, operating in two layers:

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
| **Event Sourcing** | Append-only log for audit trail and rollback |

## Runtime Strategies

deeplossless does not force optimization. The runtime policy layer is **advisory
and configurable**: users can prioritize token efficiency, exploratory reasoning,
or autonomous execution depending on workload. The agent/UI can accept, ignore,
or override each recommendation.

| Profile | Cache | Retries | Speculative | Context | Freeze Plans | Token Budget |
|---------|-------|---------|-------------|---------|-------------|-------------|
| **Minimal** | 100% | 1 | No | 20% | Yes | 30% |
| **Efficient** | 80% | 2 | No | 50% | No | 60% |
| **Exploratory** | 50% | 3 | Yes | 80% | No | 80% |
| **Autonomous** | 30% | 5 | Yes | 100% | No | 95% |
| **Custom** | user-defined | user-defined | user-defined | user-defined | user-defined | user-defined |

Set via `RUNTIME_PROFILE=minimal|efficient|exploratory|autonomous|custom`.
Custom: `RUNTIME_CACHE=0-1 RUNTIME_RETRIES=0-10 RUNTIME_SPECULATIVE=true|false RUNTIME_CONTEXT=0-1 RUNTIME_FREEZE=true|false RUNTIME_BUDGET=0.1-1`

## Quick start

```bash
git clone https://github.com/gordonlu/deeplossless.git
cd deeplossless
cargo build --release

DEEPSEEK_API_KEY=sk-... ./target/release/deeplossless

# Point any OpenAI-compatible client to the proxy:
deepseek config set base_url http://127.0.0.1:8080/v1
```

## Configuration

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8080` | Listen port |
| `--upstream` | `https://api.deepseek.com` | Upstream API base URL |
| `--db-path` | `~/.deepseek/lcm/lcm.db` | SQLite database path |
| `--api-key` | `DEEPSEEK_API_KEY` | DeepSeek API key (also extracted from first request) |
| `--admin-key` | `ADMIN_KEY` | Admin key for LCM endpoints (falls back to API key) |
| `--rate-limit` | `100` | Max requests/second (0 disables) |
| `--summarizer-model` | `deepseek-v4-pro` | Model for background LLM summarization |

## API

### Proxy

```
POST /v1/chat/completions     — Transparent proxy, SSE streaming, DAG context injected
```

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
GET  /v1/lcm/global/search?q=&limit=   — Cross-session semantic search
GET  /v1/lcm/execution/search?q=       — Execution memory: bugs, tool chains, code edits
GET  /v1/lcm/runtime/stats             — Runtime metrics (tokens, cache rate, failures)
```

### Operations

```
POST /v1/lcm/compress  {conv_id, from, to}  — Compress node range (LLM summarization)
POST /v1/lcm/delete    {conv_id, id}        — Soft-delete from active context
POST /v1/lcm/rollback  {conv_id, id}        — Rollback to checkpoint
POST /health                                — Health check (DB, upstream, compactor)
GET  /metrics                               — Prometheus metrics
```

## Benchmarks

### Simulated session (3 conversations, 20 turns, 8 languages)

```
                    Without Runtime    With Runtime    Reduction
Token / session         7440              4500            ↓40%
Cache hit rate          —                 42%             —
Repeated reasoning      3 rounds           1 round         ↓67%
Rereads                 2                  0               ↓100%
Failure loops           1                  0               ↓100%

Languages: Rust, Python, TypeScript, JavaScript, Java, C++, C#, Go
Run: cargo test --test simulated_session -- --nocapture
```

### How to benchmark

```bash
# Simulated (no API key needed)
cargo test --test simulated_session -- --nocapture

# Micro-benchmarks
cargo bench

# Live runtime metrics
curl http://127.0.0.1:8080/v1/lcm/runtime/stats | jq .

# Real-world comparison
# 1. Run a coding session WITHOUT the proxy, count tokens from API response
# 2. Run the SAME session WITH the proxy, read /v1/lcm/runtime/stats
# 3. Compare: token_saved = vanilla_tokens - runtime_tokens_spent
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

## Context injection

Every proxied request gets an `<lcm_context>` block injected into the system
prompt, showing the model its current DAG state:

```
<lcm_context>
  ── Summaries ──
  [summary 42] L1 Fixed port binding error (120 tok, 2 sources)
    ← sources: msg_39, msg_40

  ── Snippets ──
  [path] src/main.rs (src: msg_39)
  [num] 8080

  ── Recent Messages ──
  [msg 41] Current: deployment failed (45 tok)
</lcm_context>
```

## Requirements

- Rust 1.80+
- SQLite (bundled)
- DeepSeek API key

## Attribution

- **LongSeeker** — *Context-ReAct: Elastic Context Orchestration for Long-Horizon Search Agents* (May 2026)
- **LCM Paper** — Ehrlich & Blackman. *LCM: Lossless Context Management* (2026)

## License

MIT

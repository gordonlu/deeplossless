# deeplossless

**Lossless Context Management proxy for DeepSeek API.** DAG-based conversation
summarization with zero-loss SQLite persistence, full-text search, and
Context-ReAct style active context operations.

Acts as a transparent proxy between your AI client (e.g. deepseek-tui) and
`api.deepseek.com`, storing every message verbatim while assembling lossless
context summaries from a hierarchical DAG engine.

> **Status: Inference-aware Memory Runtime.** Deterministic tool cache, failure memory,
> plan persistence, execution provenance, advisory runtime policy with configurable profiles,
> semantic DAG with embedding dedup, AST-aware code extraction.
> 107 tests pass. CI: check → clippy → test → doc.

## Quick start

```bash
# Build from source
git clone https://github.com/gordonlu/deeplossless.git
cd deeplossless
cargo build --release

# Run
DEEPSEEK_API_KEY=sk-... ./target/release/deeplossless

# Point deepseek-tui (or any OpenAI-compatible client) to the proxy:
deepseek config set base_url http://127.0.0.1:8080/v1
```

## Features

| Feature | Description |
|---------|-------------|
| **Transparent proxy** | Forwards `/v1/chat/completions` to DeepSeek API. SSE streaming pass-through. |
| **Runtime Policy** | Configurable profiles (Minimal / Efficient / Exploratory / Autonomous / Custom). Advisory decisions with confidence + estimated token savings. |
| **Tool Result Cache** | Deterministic `hash(tool + args)` cache with partial file-based invalidation. Zero-token reuse for grep/read_file/search. |
| **Failure Memory** | Stores failed reasoning paths (why_failed + invalidated_assumptions), not just error strings. Prevents error loop token waste. |
| **Plan Persistence** | Execution state tracked as PlanState (goal, steps, assumptions), not plan text. Avoids repeated planning. |
| **Semantic DAG** | True shared DAG with embedding-based dedup (cosine ≥0.85 auto-merge), BM25 retrieval scoring, sentence-level provenance spans. |
| **Memory scoring** | Access count + recency + importance scoring with decay-based GC. Three-tier retention: critical / normal / ephemeral. |
| **Execution units** | Agent memory atoms: `think → act → observe → reflect` cycles stored with tool chains and outcome inference. |
| **Code diff memory** | Stores what changed (file, diff, symbols, error_before/after) not full code blocks. 95% of coding tokens are repeats. |
| **Tree-sitter AST extraction** | Rust code parsed with tree-sitter for precise function/type/struct signature extraction. |
| **Entropy-aware compaction** | Trigram novelty scoring adjusts compaction thresholds — novel content preserved, redundant content aggressively compressed. |
| **Streaming DAG** | SSE endpoint for incremental context delivery. Sliding-window incremental compaction. |
| **Cross-session search** | Global semantic search across conversations. Auto-merges similar nodes via embedding similarity. |
| **Event sourcing** | Append-only `dag_events` log for audit trail and rollback. |
| **FTS5 BM25 search** | Full-text search with BM25 scoring for English, LIKE fallback for CJK. |
| **Readiness probe** | `GET /health` checks DB, upstream, and compactor — returns 200/503 with per-check JSON. |
| **Rate limiting** | Configurable requests/second (default 100) with global fixed-window counter. |
| **Prometheus metrics** | `GET /metrics` exposes request totals, active requests, status-code breakdown, rate-limit hits, upstream errors, and uptime. |
| **Structured JSON errors** | All API errors return uniform `{"error": {"code": "...", "message": "..."}}` envelopes. |

## Architecture

```
┌──────────────┐     ┌──────────────────────────┐     ┌──────────────────┐
│ deepseek-tui │────▶│  deeplossless (8080)      │────▶│ api.deepseek.com │
│   (client)   │     │                           │     │   (upstream)     │
└──────────────┘     │  ┌────────────────────┐   │     └──────────────────┘
                     │  │  DAG Engine         │   │
                     │  │  ┌─────┐ ┌─────┐   │   │
                     │  │  │ L2  │ │ L3  │   │   │
                     │  │  └──┬──┘ └──┬──┘   │   │
                     │  │     │       │      │   │
                     │  │  ┌──▼──┐ ┌──▼──┐  │   │
                     │  │  │ L1  │ │ L1  │  │   │
                     │  │  └──┬──┘ └──┬──┘  │   │
                     │  │     └───┬────┘     │   │
                     │  │     ┌──▼──┐        │   │
                     │  │     │ leaf│        │   │
                     │  │     │ msgs│        │   │
                     │  │     └─────┘        │   │
                     │  └────────────────────┘   │
                     │  ┌────────────────────┐   │
                     │  │  SQLite (WAL)       │   │
                     │  │  ├─ conversations   │   │
                     │  │  ├─ messages        │   │
                     │  │  ├─ dag_nodes       │   │
                     │  │  └─ messages_fts    │   │
                     │  └────────────────────┘   │
                     └──────────────────────────┘
```

### Request flow

```
1. Client sends POST /v1/chat/completions
2. deeplossless extracts session fingerprint (SHA-256 of first 3 messages)
3. Messages stored in SQLite (async, non-blocking)
4. Async DAG compaction checks thresholds (soft: 80%, hard: 95%)
5. DAG context assembled from summaries + recent messages (token-budgeted)
6. Context panel rendered as <lcm_context> block, injected into system prompt
7. Modified request forwarded to DeepSeek API
8. SSE response streamed back to client (chunk-by-chunk)
```

## Configuration

### CLI arguments

```bash
deeplossless \
  --host 127.0.0.1 \
  --port 8080 \
  --upstream https://api.deepseek.com \
  --db-path ~/.deepseek/lcm/lcm.db \
  --admin-key sk-admin \
  --rate-limit 200
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8080` | Listen port |
| `--upstream` | `https://api.deepseek.com` | Upstream API base URL |
| `--db-path` | `~/.deepseek/lcm/lcm.db` | SQLite database path (supports `~` and `$HOME`) |
| `--api-key` | `DEEPSEEK_API_KEY` | DeepSeek API key (optional — extracted from first request if omitted) |
| `--admin-key` | `ADMIN_KEY` | Separate admin key for LCM endpoint auth (falls back to `DEEPSEEK_API_KEY` if unset) |
| `--rate-limit` | `100` | Max requests/second (0 disables) |
| `--summarizer-model` | `deepseek-v4-pro` | Model for background LLM summarization |

### Environment

| Variable | Required | Default | Description |
|----------|----------|---------|-------------|
| `DEEPSEEK_API_KEY` | Yes* | — | DeepSeek API key (may also be provided via first request's `Authorization` header) |
| `ADMIN_KEY` | No | — | Separate admin key for LCM endpoint auth; takes priority over `DEEPSEEK_API_KEY` |
| `RATE_LIMIT` | No | `100` | Max requests/second (0 disables) |
| `SUMMARIZER_MODEL` | No | `deepseek-v4-pro` | Model for background LLM summarization |

\* Required unless the API key is provided at runtime via the first proxied request's `Authorization` header.

## API endpoints

### Core proxy

```
POST /v1/chat/completions     — Transparent proxy to DeepSeek API
  Supports both streaming (SSE) and non-streaming requests.
  DAG context automatically injected into system messages.
```

### LCM retrieval

All LCM endpoints require `Authorization: Bearer <admin-key|deepseek-key>`.

```
GET  /v1/lcm/grep/{conv_id}?query=            — FTS5 full-text search
  Returns matching message excerpts with relevance ranking.

GET  /v1/lcm/expand/{node_id}                 — Expand summary to children
  Returns original messages/summaries under a node.

GET  /v1/lcm/status/{conv_id}                 — DAG health
  Returns total_tokens, leaf_count, summary_level.

GET  /v1/lcm/snippets/{node_id}               — View snippets
  Returns precision-critical values extracted before compression.

GET  /v1/lcm/trace/{node_id}                  — Sentence-level provenance
  Returns each sentence mapped to source node with byte offsets.

GET  /v1/lcm/global/search?q=&limit=           — Cross-session search
  Searches summaries across all conversations by access count.

GET  /v1/lcm/execution/search?q=&limit=        — Execution memory search
  Finds similar tool chains, code edits, and failure patterns.

GET  /v1/lcm/stream/{conv_id}?budget=&q=       — Streaming DAG context (SSE)
  Delivers summaries first, then recent messages incrementally.
```

### Context-ReAct operations

```
POST /v1/lcm/compress  {conv_id, from, to}    — Compress message range
  LLM summarization of nodes in [from, to]. Creates new DAG node.
  Returns {node_id, summary, token_count, snippets}.

POST /v1/lcm/delete    {conv_id, id}          — Soft-delete from active context
  Removes node from active context assembly. Raw data still in messages table.

POST /v1/lcm/rollback  {conv_id, id}          — Rollback to checkpoint
  Returns summary + children of a node for context reconstruction.

POST /health                                  — Health check (200 OK, 503 on failure)
  Returns per-check JSON: database, upstream, compactor liveness.

GET  /metrics                                 — Prometheus metrics
  Returns request totals, active requests, 2xx/4xx/5xx breakdown,
  rate-limit hits, upstream errors, and uptime in Prometheus text format.
```

## Runtime Profiles

The runtime policy engine provides advisory optimization — cache reuse, delta injection,
failure avoidance, context compaction. The agent/UI can accept, ignore, or override each
recommendation.

| Profile | Cache | Retries | Speculative | Context | Freeze Plans | Token Budget |
|---------|-------|---------|-------------|---------|-------------|-------------|
| **Minimal** | 100% | 1 | No | 20% | Yes | 30% |
| **Efficient** | 80% | 2 | No | 50% | No | 60% |
| **Exploratory** | 50% | 3 | Yes | 80% | No | 80% |
| **Autonomous** | 30% | 5 | Yes | 100% | No | 95% |
| **Custom** | user-defined | user-defined | user-defined | user-defined | user-defined | user-defined |

Set via environment: `RUNTIME_PROFILE=minimal|efficient|exploratory|autonomous|custom`

Custom parameters (when `RUNTIME_PROFILE=custom`):
`RUNTIME_CACHE=0.0-1.0` `RUNTIME_RETRIES=0-10` `RUNTIME_SPECULATIVE=true|false`
`RUNTIME_CONTEXT=0.0-1.0` `RUNTIME_FREEZE=true|false` `RUNTIME_BUDGET=0.1-1.0`

### Context injection format

Every proxied request gets an `<lcm_context>` block injected into all system
messages. The panel shows the model its current DAG state and available
operations:

```
<lcm_context>
  [summary 42] L1 — Fixed port binding error (120 tok, 2 parents)
    ├ path: src/main.rs
    ├ num: 8080
    └ /lcm/rollback 42

  [msg 39] Current: deployment failed (45 tok)

  Operations:
    /lcm/compress conv_id=1 from=1 to=42 — compress node range
    /lcm/delete conv_id=1 id=<node_id>   — delete node from context
    /lcm/rollback conv_id=1 id=<node_id> — rollback to checkpoint
</lcm_context>
```

The model can invoke operations via tool calls to the above endpoints,
enabling Context-ReAct style active context management.

## DAG compression levels

| Level | Name | Method | Target |
|-------|------|--------|--------|
| L1 | Normal | LLM summarization (`preserve_details`) | 90% of original tokens |
| L2 | Aggressive | LLM summarization (`bullet_points`) | 50% of original tokens |
| L3 | Fallback | Deterministic truncation (head+tail) | 512 tokens, guaranteed convergence |

Before each compression, deeplossless automatically extracts **snippets**
(code blocks, file paths, numeric constants, error messages) from the source
text and stores them in the DAG node, so critical values are never lost even
after aggressive compression.

## Database

All data is stored in a single SQLite file (default `~/.deepseek/lcm/lcm.db`):

- **`conversations`** — session metadata (fingerprint, model)
- **`messages`** — verbatim message history with token counts
- **`messages_fts`** — FTS5 full-text index (porter tokenizer)
- **`dag_nodes`** — DAG nodes with level, parent/child links, snippets

The database uses WAL mode with automatic checkpointing every 100 writes.

## Requirements

- Rust 1.80+
- SQLite (bundled via `rusqlite` bundled feature)
- DeepSeek API key

## Benchmarks

```
📐  DAG Compression
     Original: 740 tokens → Compressed: 335 tokens (2.2×)
     Snippets preserved: 7 critical values extracted
     Compression levels: L1 (LLM detailed) → L2 (LLM bullet) → L3 (deterministic)

🔍  FTS5 Full-Text Search
     Porter tokenizer with Unicode61 support
     Sub-millisecond query latency

⚡  Processing (criterion benchmarks)
     Token counting (8K lines):     7.8 ms
     Snippet extraction (4K lines):  5.8 ms
     DAG assembly (1K nodes):      483 μs
     Session fingerprint:          124 ns

💾  KV Cache Safe
     Context injection:          system prompt only (never touches messages)
     Memory:                     messages stored verbatim in SQLite
```

## Attribution

This project is inspired by and references:

- **LongSeeker** — *Context-ReAct: Elastic Context Orchestration for Long-Horizon Search Agents* (May 2026). [arXiv 2605.05191](https://arxiv.org/abs/2605.05191)
- **LCM Paper** — Clint Ehrlich & Theodore Blackman. *LCM: Lossless Context Management* (2026). [https://papers.voltropy.com/LCM](https://papers.voltropy.com/LCM)
- **lossless-claw** — Josh Lehman / Martian Engineering. Lossless Context Management plugin for OpenClaw. [https://github.com/Martian-Engineering/lossless-claw](https://github.com/Martian-Engineering/lossless-claw)

## License

MIT

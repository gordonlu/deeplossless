# deeplossless

**Lossless Context Management proxy for DeepSeek API.** DAG-based conversation
summarization with zero-loss SQLite persistence, full-text search, and
Context-ReAct style active context operations.

Acts as a transparent proxy between your AI client (e.g. deepseek-tui) and
`api.deepseek.com`, storing every message verbatim while assembling lossless
context summaries from a hierarchical DAG engine.

> **Status: v0.1.0 — Proof of concept.** Core features work and are tested,
> but the project is early-stage. Expect bugs, missing error handling, and
> breaking changes. Not recommended for production use without review.

## Quick start

```bash
# Install
cargo install deeplossless

# Run
DEEPSEEK_API_KEY=sk-... deeplossless

# Point deepseek-tui (or any OpenAI-compatible client) to the proxy:
deepseek config set base_url http://127.0.0.1:8080/v1
```

## Features

| Feature | Description |
|---------|-------------|
| **Transparent proxy** | Forwards `/v1/chat/completions` to DeepSeek API. SSE streaming pass-through. |
| **Lossless persistence** | Every message stored verbatim in SQLite WAL-mode database. |
| **Hierarchical DAG** | Multi-level summarization with lossless pointers. Three-level escalation (LLM → LLM aggressive → deterministic truncate). |
| **Session tracking** | SHA-256 fingerprint over first 3 messages identifies multi-turn conversations. |
| **Snippet extraction** | Before compression, extracts code blocks, file paths, numeric constants, percentages, proper nouns, and error messages. Preserved as structured metadata. |
| **FTS5 full-text search** | SQLite FTS5 with porter tokenizer for high-speed `lcm_grep`. |
| **Context-ReAct operations** | Model-accessible endpoints for `compress`, `delete`, and `rollback` — active context management without data loss. |
| **Structured context panel** | `<lcm_context>` block injected into system prompt shows summaries, snippets, and available operations per node. |

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
  --db-path ~/.deepseek/lcm/lcm.db
```

| Argument | Default | Description |
|----------|---------|-------------|
| `--host` | `127.0.0.1` | Listen address |
| `--port` | `8080` | Listen port |
| `--upstream` | `https://api.deepseek.com` | Upstream API base URL |
| `--db-path` | `~/.deepseek/lcm/lcm.db` | SQLite database path (supports `~` and `$HOME`) |

### Environment

| Variable | Required | Description |
|----------|----------|-------------|
| `DEEPSEEK_API_KEY` | Yes | DeepSeek API key |

## API endpoints

### Core proxy

```
POST /v1/chat/completions     — Transparent proxy to DeepSeek API
  Supports both streaming (SSE) and non-streaming requests.
  DAG context automatically injected into system messages.
```

### LCM retrieval

```
GET  /v1/lcm/grep/{conv_id}?query=            — FTS5 full-text search
  Returns matching message excerpts with relevance ranking.

GET  /v1/lcm/expand/{node_id}                 — Expand summary to children
  Returns original messages/summaries under a node.

GET  /v1/lcm/status/{conv_id}                 — DAG health
  Returns total_tokens, leaf_count, summary_level.

GET  /v1/lcm/snippets/{node_id}               — View snippets
  Returns precision-critical values extracted before compression.
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

POST /health                                  — Health check (200 OK)
```

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

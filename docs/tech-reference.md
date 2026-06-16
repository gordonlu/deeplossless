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

## Protocol Compatibility Test

The torture module provides deterministic protocol-level testing of the
deeplossless API endpoints. It uses a 0-LLM state machine to drive agent
scenarios across three protocol formats without requiring an API key.

### Methodology

Torture uses a **deterministic local driver** to verify that
deeplossless correctly handles all three supported protocol formats
(OpenAI Chat Completions, Claude Code, Codex CLI) and their
tool-call parameter conventions. Each scenario's YAML state machine
defines expected tool calls, file edits, and verification steps.

### Scenarios

The full suite discovers base scenarios at runtime with
`Scenario::list_base()`. Base scenarios are YAML files in `scenarios/`
whose file stem has no per-agent suffix such as `.claude_code` or
`.codex`; per-agent variants are selected by `Scenario::load_with_format()`.
The table below summarizes the current logical scenarios rather than
serving as a fixed count contract.

```
Scenario            Description
───────────         ───────────
fix_test_failure    Fix off-by-one bug
add_feature         Add new functionality
refactor_rename     Rename across files
search_to_fix       Search then repair
multi_file_edit     Coordinated multi-file changes
debug_from_logs     Root-cause from log output
security_fix        Patch a vulnerability
hidden_bug          Subtle logic error
reuse_existing      Reuse an existing utility instead of duplicating logic
```

### How to run

```bash
# Run all protocol compatibility tests
cargo test --all-targets

# Drive scenarios through the mock API
cargo run -- --torture-aces hidden_bug --agent-format claude_code

# Run the full suite
cargo run -- --torture-aces all

# Equivalent full-suite form; useful when scripts pass an empty value explicitly
cargo run -- --torture-aces=""

# Long-session stress test
cargo test --test long_session_benchmark -- --nocapture

# Live runtime metrics (requires proxy running)
curl http://127.0.0.1:8080/v1/lcm/runtime/stats | jq .
```

### Windows console encoding

Repository files and documentation are UTF-8. On Windows, mojibake in
PowerShell or `cmd.exe` usually means the console code page is not UTF-8;
it does not by itself indicate that source files are corrupted.

For local development, prefer Windows Terminal with a UTF-8 capable font.
If using classic PowerShell or `cmd.exe`, run `chcp 65001` before commands
that print non-ASCII documentation, scenario names, or rustdoc output.
PowerShell 7 generally handles UTF-8 better than Windows PowerShell 5.1.

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

# deeplossless — Agent Instructions

## Commands
- Build/test/lint (CI order): `cargo check --all-targets` → `cargo clippy --all-targets -- -D warnings` → `cargo test --all-targets` → `cargo doc --no-deps`
- Run: `DEEPSEEK_API_KEY=sk-... cargo run`
- Run with custom port/db: `cargo run -- --port 8080 --db-path ~/.deepseek/lcm/lcm.db`
- Benchmarks: `cargo bench`
- 54 tests (53 unit + 1 integration), all under `src/*.rs` + `tests/proxy_integration.rs`

## Architecture
- **Transparent proxy**: axum 0.8, forwards `/v1/chat/completions` to `api.deepseek.com` with SSE streaming
- **Modules** (11 source files under `src/`):
  - `proxy.rs` — HTTP routes (8 endpoints: proxy + LCM retrieval + Context-ReAct)
  - `pipeline.rs` — `ChatPipeline`: fingerprint → store → compact → assemble → inject
  - `dag.rs` — DAG engine (soft 80%, hard 95% thresholds, max 3 levels, recent 20 leaves)
  - `db.rs` — SQLite WAL, `Mutex<Connection>` (single-writer invariant)
  - `compactor.rs` — Async compaction via mpsc channel
  - `summarizer.rs` — Three-level escalation: LLM (L1) → LLM aggressive (L2) → deterministic truncate (L3)
  - `session.rs` — Unified message normalization (OpenAI/DeepSeek/Claude tool schemas)
  - `snippet.rs` — Precision-critical value extraction before compression
  - `tokenizer.rs` — tiktoken counting
- **DAG invariant**: `summary.child_ids = source_ids` (forward edge), `source.parent_ids = [summary]` (back-link)

## Key gotchas
- `insert_summary_atomic` uses `conn.unchecked_transaction()` — safe because `Mutex<Connection>` already serialises all DB access; nested `transaction()` would panic
- FTS5 for search, but CJK+English falls back to `LIKE` (verified: `unicode61` tokenizer doesn't reliably segment mixed CJK/English)
- No `openai`/`async-openai` crate — manual `reqwest` calls to DeepSeek API
- Platform: Rust edition 2024, `x86_64-pc-windows-gnu` toolchain (MSVC broken on this machine); `tokio::signal::ctrl_c()` for Windows graceful shutdown
- `AppState.api_key` is `Arc<Mutex<Option<String>>>` — populated lazily from first request's `Authorization` header
- Context injected only into the **first** system message (`break` after first match in `inject_context`)
- `collect_summary_chain` uses `seen: HashSet` to prevent recursive duplication
- DB tests use `tempfile::tempdir()` — no external services needed
- Integration test (`tests/proxy_integration.rs`) starts its own mock upstream axum server

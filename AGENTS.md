# deeplossless — Agent Instructions

## Thinking Rules (Semantic Integrity)

**Compilation is not correctness. Passing tests is not correctness. Demo success is not correctness.**

A system is only complete when semantics remain correct, failures remain observable, state remains consistent, and behavior survives real execution conditions.

### Core principles
- Do not stop at first working output — continue reasoning about failure paths, rollback, restart, concurrency, cleanup, ordering, persistence
- Avoid premature convergence — do not assume "good enough", "probably correct", "works in practice"
- Reason about systems, not isolated functions — local correctness can still produce global corruption
- Distinguish appearance from semantics — compilation, green tests, matching output do NOT prove correctness
- Explicitly identify unverified assumptions — what breaks after restart? under concurrency? under partial failure?

### Behavioral rules
- Never fake behavior — no fake retries, async, streaming, persistence, validation, concurrency, recovery, cleanup
- Do not hide failure — silent catch, ignored errors, swallowed exceptions, invisible degradation are all bugs
- Unsupported behavior must fail loudly — `Err("NOT_IMPLEMENTED")` over `Ok(default)`
- Do not confuse generated structure with real architecture — large code volume often hides semantic weakness; prefer smaller correct systems

### Verification rules
- Do not claim functionality without verification — failure-path tests, restart/recovery tests, persistence checks, concurrency verification are mandatory
- Unverified behavior must be labeled explicitly — assumptions, estimates, likely behavior are NOT verified guarantees
- Test failure paths, not only success paths — most corruption occurs during timeout, cancellation, restart, partial write, concurrent execution
- Persistence claims must survive restart — in-memory success is not persistence
- Streaming must be incremental — buffering everything before output is not streaming

### Semantic integrity rules
- All execution paths must preserve semantics — prevent semantic drift between dev/prod, test/live, mock/real
- State mutations must respect failure boundaries — never mutate visible state then fail later
- Ordering guarantees must be real — FIFO must actually be FIFO, exactly-once must actually be exactly-once
- Validation must enforce real constraints — validation that never rejects invalid state is fake validation
- Cleanup behavior is part of correctness — unbounded growth is semantic failure

### Reliability rules
- Avoid duplicated semantic logic — duplicated logic drifts over time
- Runtime behavior is more important than interface shape — clean APIs can hide race conditions, persistence failure, hidden sync bugs
- Recovery behavior must be designed explicitly — what survives crash, retry, replay, reconnect, partial completion?
- Temporal correctness matters — systems must remain correct across retries, restart, reconnection, delayed execution, concurrent mutation

### AI-specific rules
- AI agents optimize for narrative completion — appearing productive and declaring completion early conflicts with semantic correctness
- Generated confidence is not proof — confident explanations do not imply runtime correctness
- Continue reasoning after "success" — many failures only appear after scaling, restart, concurrency, persistence, retries, long-running execution

## Commands
- Build/test/lint (CI order): `cargo check --all-targets` → `cargo clippy --all-targets -- -D warnings` → `cargo test --all-targets` → `cargo doc --no-deps`
- Run: `DEEPSEEK_API_KEY=sk-... cargo run`
- Run with custom port/db: `cargo run -- --port 8080 --db-path ~/.deepseek/lcm/lcm.db`
- Benchmarks: `cargo bench`
- **287 tests** (249 unit + 18 integration + 12 soak + 5 benchmark + 1 long-session + 2 simulated), under `src/*.rs` + `tests/*.rs`

## Architecture (v0.4.0+)

- **Transparent proxy**: axum 0.8, forwards `/v1/chat/completions` and `/v1/responses` to upstream with SSE streaming
- **22 source files** under `src/`:
  - `proxy.rs` — HTTP routes (20+ endpoints: proxy + LCM retrieval + Context-ReAct + runtime inspection)
  - `pipeline.rs` — `ChatPipeline`: fingerprint → store → compact → assemble → inject
  - `dag.rs` — DAG engine, semantic dedup, compaction, provenance, validation, merge_nodes, GC
  - `db.rs` — SQLite WAL, `Mutex<Connection>` read-pool + single-writer, FTS5, embeddings, migrations
  - `compactor.rs` — Async compaction via mpsc channel with `CompactionPlanner`, scoring model, dirty-region tracking, metrics
  - `summarizer.rs` — Three-level escalation with `RetryClass`, jittered backoff, token-aware truncation, typed response, adaptive timeout, cooperative cancellation
  - `runtime.rs` — `RuntimeEvent`-driven execution cycle, `PipelineStage` trait, `RetryClass`, `RetryBackoff`, `RuntimeMode`, `RuntimePolicy`, `BackgroundTasks`
  - `runtime_events.rs` — Append-only `RuntimeEvent` enum (9 variants, execution/retry/cancellation lifecycle)
  - `runtime_state_view.rs` — Derived state from event log: `inspect()`, `explain()`, `dump_events()`, parity checks
  - `runtime_invariants.rs` — Runtime assertions: monotonic seq, cancellation well-formed, retry ordering
  - `dependency_kind.rs` — `DependencyKind` taxonomy (11 variants, 6 active, 5 reserved)
  - `dependency_view.rs` — Unified interpretation over dag_edges + lineage_edges + tool_cache
  - `provider.rs` — `LlmProvider` trait + `OpenAiCompatibleProvider`
  - `execution.rs` — `ExecutionUnit`, `ExecutionOutcome`, `LineageEdge`, scoring, `next_logical_seq()`
  - `parallel.rs` — `ForkJoinTracker`, spanning, parallel group detection, `HappensBeforeEdge`
  - `mutation.rs` — `MutationEngine` with background cycle, decay/strengthen/invalidate/motif mutations
  - `motif.rs` — Execution motif extraction with n-gram analysis, recency weighting, canonical tool names
  - `file_observation.rs` — AST-based file observation with `canonicalize_path`, `ObservationKind`, `is_stale()`
  - `tool_cache.rs` — L1 hot cache + L2 SQLite, `ToolKind` with `dependency_kind()` mapping
  - `session.rs` — Unified message normalization (OpenAI/DeepSeek/Claude/Gemini tool schemas)
  - `snippet.rs` — Precision-critical value extraction before compression
  - `tokenizer.rs` — tiktoken counting with `Mutex<HashMap>` encoding cache
  - `embeddings.rs` — Embedding-based similarity for semantic dedup

## Key gotchas
- `insert_summary_atomic` uses `conn.unchecked_transaction()` — safe because `Mutex<Connection>` already serialises all DB access; nested `transaction()` would panic
- FTS5 for search, but CJK+English falls back to `LIKE` (verified: `unicode61` tokenizer doesn't reliably segment mixed CJK/English)
- No `openai`/`async-openai` crate — manual `reqwest` calls to DeepSeek API
- Platform: Rust edition 2024, `x86_64-pc-windows-gnu` toolchain; `tokio::signal::ctrl_c()` for Windows graceful shutdown
- `AppState.api_key` is `Arc<StdMutex<Option<String>>>` — populated lazily from first request's `Authorization` header
- `AppState.admin_key` takes priority for LCM auth, with fallback to `api_key`
- Context injected only into the **first** system message (`break` after first match in `inject_context`)
- `compaction_id` is SHA256(conv_id, sorted(source_ids), level) for idempotent compaction
- `next_logical_seq()` (AtomicI64) replaces wall-clock for deterministic execution ordering
- `dag_edges` is the canonical topology authority; `child_ids`/`parent_ids` JSON arrays are legacy mirrors
- `model_map` is canonical in `protocol::ModelRegistry`; proxy.rs delegates to it (no more duplication)
- `shutdown_notify: Arc<Notify>` on `RuntimeServices` — 3 proxy spawn sites check it on entry
- Compactor worker handle registered in `BackgroundTasks` (no more orphan worker)
- `RuntimeStateView` computes state from events; mutable metrics are projections, not source of truth
- DB tests use `tempfile::tempdir()` — no external services needed
- Integration test (`tests/proxy_integration.rs`) starts its own mock upstream axum server
- Soak tests (`tests/runtime_event_soak.rs`) cover 1000-cycle parity, 500-retry storm, cross-conv isolation

## Architecture Docs (11 files under `docs/architecture/`)
- `dag-invariants.md` — 7 DAG invariants (acyclicity, symmetry, edge consistency, level ordering, idempotent compaction, revision monotonicity, deletion consistency)
- `frozen-invariants.md` — 9 cross-subsystem iron laws (single-writer, append-only events, ExecutionUnit immutability, logical seq, hard-deletion non-cascade, channel capacity, architecture cadence rule)
- `replay-model.md` — Replay semantics, guarantees, gaps, replay authority rules
- `runtime-lifecycle.md` — Startup/shutdown order, ownership model, restart behavior, forbidden patterns
- `dependency-authority.md` — 6 subsystem authority boundaries, forbidden direct-field-mutation patterns
- `dependency-model.md` — Dependency definition, taxonomy, authority table, forbidden inference patterns
- `runtime-events.md` — Frozen RuntimeEvent contract: allowed categories, forbidden payloads, ordering guarantees, cooperative cancellation contract, schema evolution rules
- `authority-boundary.md` — Per-DependencyKind authority assignment, 4 forbidden cross-boundary patterns

## Stabilization State (Phase 5)
- Runtime core is in stabilization window — no new core concepts without architecture cadence review
- Schema version constants frozen: `RUNTIME_EVENT_SCHEMA_VERSION`, `CANCELLATION_SOURCE_SCHEMA_VERSION`
- `docs/tech-reference.md` — technical reference (architecture, API, benchmarks, Codex integration)
- `REVIEW.md` — remaining architectural future-work items (~17, all subsystem-level redesign)

# Implementation Plan — Self-Evolving Execution Runtime

Status: **Phase 1 70% complete**. Codebase at v0.3.2.

---

## Phase 1 — Deterministic Execution Runtime

### 1.1 Canonical Stream/Event IR ✅
- `src/protocol/canonical.rs` — `StreamEvent`, `ContentPart`, `CanonicalRequest`
- Provider-neutral, replay-safe, deterministic ordering
- All tests pass

### 1.2 StreamAssembler ✅
- `src/protocol/streaming.rs`
- Parallel tool call buffering, flush/abort_flush distinction
- Streaming text passthrough with accumulated lifecycle content

### 1.3 Immutable Event Store 🔴
**Target: v0.4.0 files: `src/db.rs`**

Current state: `execution_units` table exists but writes are lossy (spawn_blocking fire-and-forget). Not event-sourced.

Changes needed:
- `execution_events` table: append-only, `(id, execution_id, event_kind, payload_json, seq_no, created_at)`
- Every `StreamEvent` flowing through the proxy gets persisted before emitting to client
- `tool_calls` and `tool_results` already have tables — add `execution_id` FK
- `store_execution_event()` — single-row insert, no updates

### 1.4 Deterministic Replay Engine 🔴
**Target: v0.4.0 files: `src/replay.rs`**

Current state: `CanonicalExecutionKey` can compute replay keys. No actual replay logic.

Changes needed:
- `ReplaySnapshot { execution_id, memory_version_id, planner_state, tool_state, stream_state }`
- `replay_execution(execution_id, until_event_seq) -> Vec<StreamEvent>`
- Replay reads from `execution_events` in seq_no order
- Snapshot restore: seek to `memory_version_id`, replay events from that point

### 1.5 Snapshot Isolation 🟡
**Target: v0.4.0 files: `src/snapshot.rs`**

Current state: `memory_versions` and `execution_snapshots` tables created. No CRUD.

Changes needed:
- `take_snapshot(execution_id, memory_version_id, tier)` — append-only insert
- `restore_snapshot(id) -> ReplaySnapshot`
- `enforce_budget()` — L0 ring-buffer eviction, L2 soft limit, hard size cap
- Semantic boundary triggers: tool completion, planner decision, motif boundary

---

## Phase 2 — Execution Memory Layer

### 2.1 Execution Graph ✅
- `src/dag.rs` — full DAG with embedding dedup, BM25 retrieval, provenance spans

### 2.2 Semantic Tool Reuse ✅
- `src/tool_cache.rs` — normalized args hash, partial invalidation, intercept-only tools

### 2.3 Execution Outcome Scoring 🔴
**Target: v0.4.0 files: `src/execution.rs`**

Current state: `ExecutionOutcome` enum exists (Success, RecoveredFailure, Blocked, CacheHit, Stale, Replayed). No scoring pipeline.

Changes needed:
- `score_execution(execution_id) -> ExecutionScore`
- Scoring dimensions: success_rate, retry_count, downstream_reuse, latency_cost, hallucination_risk
- Score stored as metadata on execution_units
- Used for memory weighting and motif extraction decisions

### 2.4 Memory Versioning 🔴
**Target: v0.4.0 files: `src/snapshot.rs`**

Current state: `memory_versions` table exists. No mutation engine.

Changes needed:
- `create_memory_version(parent_id, mutation_kind, mutation_desc)` — bump version
- Version bound to snapshot at creation time
- Executions bind to a specific version (no shared mutable graph during active execution)
- `list_versions() -> Vec<MemoryVersion>` for audit trail

### 2.5 Auditability Layer 🔴
**Target: v0.5.0 files: `src/audit.rs`**

Not started.

Changes needed:
- `execution_timeline(execution_id) -> Vec<TimelineEvent>` — every event with timestamps
- `tool_access_audit(agent_id, file_path) -> Vec<AccessRecord>`
- `reasoning_provenance(execution_id) -> Vec<ProvenanceNode>`
- All queries read from append-only `execution_events` table (Phase 1.3)

---

## Phase 3 — Self-Evolving Runtime

### 3.1 Mutation Engine 🔴
**Target: v0.5.0 files: `src/mutation.rs`**

Not started. Background task consuming execution logs, replay metrics, failure stats.

Changes needed:
- `MutationEngine` struct with configurable mutation policy
- Consumes: execution logs, replay metrics, failure patterns, reuse stats
- Produces: new memory versions with evolved topology
- Mutations: edge_strengthen, edge_decay, motif_split, motif_merge, invalidate_belief

### 3.2 Execution Motif Extraction 🔴
**Target: v0.5.0 files: `src/motif.rs`**

Not started.

Changes needed:
- `extract_motifs(execution_id) -> Vec<ExecutionMotif>`
- Pattern detection: repeated tool chains (grep→cat→summarize→patch)
- Motif metadata: confidence, reuse_count, success_rate, token_efficiency
- Motif-level snapshots (Phase 1.5 semantic boundaries)

### 3.3 Structural Reflection 🔴
**Target: v0.6.0 files: `src/reflection.rs`**

Not started.

Changes needed:
- Repeated retry loop → unstable execution edge (direct topology mutation)
- Hallucination detection: tool result contradicts known facts → contradiction edge
- Downstream effectiveness tracking across execution chains

### 3.4 Contradiction & Unlearning Engine 🔴
**Target: v0.6.0 files: `src/unlearning.rs`**

Not started.

Changes needed:
- Confidence decay over time for unused edges
- Dependency invalidation: file changed → dependent motifs lose confidence
- Stale execution pruning: old motifs with zero reuse → archive
- Hallucination edge suppression: contradiction score > threshold → remove

### 3.5 Replay-Verified Mutation 🔴
**Target: v0.6.0 files: `src/mutation.rs`**

Not started.

Changes needed:
- Replay historical executions against old AND evolved memory topology
- Measure: success improvement, token reduction, retry reduction
- Auto-rollback harmful mutations
- Mutation audit log: who/what/when/why for every graph change

---

## Phase 4 — Runtime Self-Optimization

### 4.1 Learned Planner Bias 🔴
**Target: v0.7.0**

Planner adapts using successful historical motifs. Prioritize stable workflows, low-cost paths, high-confidence subgraphs.

### 4.2 Execution Compilation 🔴
**Target: v0.7.0**

Frequently successful execution motifs become compiled runtime primitives. CodeInspectionMotif → reusable runtime macro with cached intermediate results.

### 4.3 Memory Compaction Engine 🔴
**Target: v0.8.0**

Convert large execution history into compact learned abstractions. Hot/Warm/Cold/Archived tiers. Already have DAG summarization (Level 1/2/3) — extend to execution motifs.

### 4.4 Counterfactual Replay 🔴
**Target: v0.8.0**

Simulate alternate execution paths. "What if tool X returned different result?" Used for robustness analysis, hallucination detection.

### 4.5 Organization-Level Learning 🔴
**Target: v0.9.0**

Aggregate execution intelligence across teams/agents. Reusable workflows, policy learning, security motif detection.

---

## Immediate Next Steps (v0.4.0)

Priority order:

| # | Task | Effort | Impact |
|---|------|--------|--------|
| 1 | `execution_events` table + append-only writes | 2h | Foundation for replay/audit |
| 2 | Snapshot CRUD + budget enforcement | 2h | Phase 1 completion |
| 3 | Execution outcome scoring pipeline | 3h | Feeds mutation engine |
| 4 | Memory version create/list API | 1h | Version binding foundation |
| 5 | Replay engine (event log reader) | 3h | Core replay capability |

Total v0.4.0: ~11h. Would bring Phase 1 to ~90% and Phase 2 to ~55%.

# Frozen Invariants

## Purpose
Define cross-subsystem invariants that are NOT negotiable and
MUST NOT be violated by any future refactor. These are the
contracts that all other subsystems rely on.

## Single-Writer Invariant

**The Database MUST have exactly one writer connection.**

`Database.writer: Mutex<Connection>` serializes all writes.
`conn.unchecked_transaction()` is safe ONLY because `Mutex`
guarantees at most one thread holds the connection. If the
write model changes (e.g., WAL2, multiple writers, sharding),
all `unchecked_transaction()` sites MUST be audited and
replaced with proper `transaction()` calls.

## Append-Only Event Tables

**`execution_events` and `dag_events` tables are append-only.**

NEVER execute UPDATE or DELETE on these tables in production.
Schema migrations MAY add columns (ALTER TABLE ADD COLUMN)
but MUST NOT drop or modify existing columns.

## Execution Unit Immutability

**`ExecutionUnit` is immutable after construction.**

All fields are set at `new()` / `new_with_span()` time. The
`id` is 0 until persisted (`store_execution_unit` assigns the
auto-increment ID). After store, the struct is NOT mutated.
`ExecutionOutcome` is set at construction, never changed.

## Logical Sequence Ordering

**`next_logical_seq()` provides deterministic ordering within a
process lifetime.**

It is an `AtomicI64` with `fetch_add(1, Relaxed)`. It replaces
`chrono::Utc::now().timestamp_millis()` for `epoch_ms` in
execution events and units. The counter is monotonic but NOT
persisted across process restarts.

## Graph Revision Monotonicity

**`DagEngine.revision_counter` MUST only increase.**

Incremented by `post_mutation()` on every DAG write. Provides
a lower bound on graph freshness. NOT reliable as an exact
version number (Relaxed ordering, no persistence guarantee).

## Compaction Idempotency

**Compaction of the same `(conv_id, source_ids, level)` MUST
produce the same `compaction_id` and reuse the existing node.**

`compaction_id` is `SHA256(conv_id, sorted(source_ids), level)`.
`compress_group_with_snippets` checks `find_by_compaction_id`
before inserting.

## No Duplicate Edge Insertion

**Edge insertion uses `INSERT OR IGNORE`.**

The unique index `idx_edges_unique ON dag_edges(from_id, to_id, kind)`
prevents duplicate edges. The `INSERT OR IGNORE` semantics mean
duplicate attempts are silently ignored, not errors.

## API Key Layering

**`admin_key` takes priority over `api_key` for LCM endpoints.
`api_key` is lazily populated from the first request's
Authorization header.**

Both are `Arc<StdMutex<Option<String>>>`. The lock duration is
short (read-only for key extraction). NEVER hold the lock across
await points — use `StdMutex`, not `tokio::sync::Mutex`.

## Channel Capacity

**Compactor mpsc channels have capacity 32.**

Senders use `send(cmd).await` — they will backpressure if the
worker is slow. Drainers use `try_recv()` — non-blocking.
The event channel MUST be drained after each command to prevent
event loss.

## Hard-Deletion Non-Cascade

**`purge_dag_node` does NOT cascade to edges, FTS, semantic
index, or provenance.**

Any code calling `purge_dag_node` MUST explicitly clean these
tables. Soft-delete (`delete_dag_node`, `deleted = 1`) is
preferred when cascade cleanup is not guaranteed.

## Forbidden Patterns

### NEVER call `tokio::sync::Mutex::blocking_read()` in async context
It panics. Use `StdMutex` for brief sync locks.

### NEVER call `rusqlite::Connection::transaction()` inside a
`Mutex<Connection>` guard
It panics (nested transactions not supported). Use
`unchecked_transaction()`.

### NEVER swallow errors from DAG write operations
All `INSERT`/`UPDATE`/`DELETE` on `dag_nodes`, `dag_edges`,
`provenance`, `semantic_index`, `snippets_fts` MUST either
propagate the error or log it at `warn!` level.

### NEVER mutate `execution_units` after storage
The `id` field transitions from 0 to N exactly once. All other
fields are set at construction. UPDATE on `execution_units`
is forbidden in production (migrations excepted).

## Architecture Cadence Rule (Phase 4 frozen)

Every new subsystem or module MUST define before implementation:

1. **Authority** — which subsystem owns this data/behavior
2. **Invariants** — what MUST always hold, enforced at runtime
3. **Forbidden patterns** — what callers MUST NOT do
4. **Lifecycle semantics** — how it starts, runs, shuts down

This rule applies to additions AND modifications. A change that
alters an existing invariant MUST update the corresponding
architecture document in `docs/architecture/`.

No new manager/controller/engine/bus/coordinator/reactive-graph
may be introduced without satisfying this rule.

## Stabilization Freeze (Phase 5 effective)

The runtime core is now in a stabilization window. Schema version
constants (`RUNTIME_EVENT_SCHEMA_VERSION`, etc.) are frozen.
New variants may be ADDED but existing variant shapes MUST NOT
change. Architecture documents reflect current reality, not
aspirations. The system's most valuable asset is now its semantic
trustworthiness, not its feature surface.

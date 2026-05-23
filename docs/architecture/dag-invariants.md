# DAG Invariants

## Purpose
Define the structural invariants that the DAG MUST maintain at
all times. Violations of these invariants are bugs, not warnings.

## Core Invariants

### I1: Acyclicity
The DAG MUST NOT contain cycles. For every edge `parent → child`,
there MUST NOT exist any path from `child` back to `parent`.

Enforced at mutation time by `check_no_cycle()` (depth 50) and
`merge_nodes` (depth 50). Detected during validation by
`validate_dag()` (depth 20).

### I2: Parent-Child Symmetry
For every node N, the set of nodes in `N.child_ids` MUST
have `N.id` in their `parent_ids` array. Conversely, for every
node M in `N.parent_ids`, `N.id` MUST appear in `M.child_ids`.

JSON arrays (`child_ids`, `parent_ids`) are the legacy mirror.
Typed `dag_edges` rows are the newer representation. **Both
representations MUST be consistent.**

**Long-term direction:** `dag_edges` is the canonical topology
authority. JSON arrays are compatibility mirrors and may
eventually be removed. New code MUST write to `dag_edges`;
MUST NOT rely on JSON arrays as the sole source of truth.

### I3: Edge Consistency
For every `summarizes` edge from `summary → source`, the source
node's `parent_ids` array MUST contain the summary node's ID.

For every `refines` edge, the same symmetry holds. `reuses` edges
cross conversation boundaries and do NOT modify `parent_ids`.

### I4: Level Ordering
For every edge `parent → child`: `parent.level >= child.level`.

Summary nodes (level > 0) MUST NOT be children of raw leaves
(level 0). Level 3 (deterministic fallback) is distinct from
Level 2 (LLM bullet points) in storage.

### I5: Idempotent Compaction
A compaction operation on source_ids `[S1, S2, ...]` at level `L`
MUST produce the same `compaction_id` every time. The
`compaction_id` is `SHA256(conv_id, sorted(source_ids), level)`.

A second compaction of the same sources MUST reuse the existing
summary node, not create a duplicate.

### I6: Revision Monotonicity
`DagEngine.revision_counter` MUST only increase. It is incremented
on every mutation via `post_mutation()`. The counter is
non-transactional (Relaxed ordering); it provides a lower-bound
on graph freshness, not an exact version.

### I7: Deletion Consistency
`delete_dag_node` sets `deleted = 1` (soft delete). The node is
excluded from `get_node` and `get_leaves` queries (WHERE
deleted = 0). Hard-delete (`purge_dag_node`) removes the row
entirely.

Hard-delete does NOT cascade to:
- `dag_edges` rows referencing the deleted node
- `snippets_fts` entries for the deleted node
- `semantic_index` entries
- `provenance` rows

These are known storage leaks. Code that calls `purge_dag_node`
MUST also clean these tables.

## Mutation Semantics

### Atomic insert
`insert_summary_atomic` wraps node creation, edge creation,
snippet index, semantic index, provenance, and parent back-link
update in a single `unchecked_transaction()`. If any mandatory
step fails (node insert, back-link update), the transaction
rolls back. Auxiliary steps (edge insert, FTS, semantic index,
provenance, audit event) are best-effort and do NOT cause
rollback on failure.

### Post-commit validation
After `insert_summary_atomic` returns:
1. `check_no_cycle()` — if cycle detected, `purge_dag_node` as
   manual rollback.
2. `verify_compaction_integrity()` — checks I1-I3. Warns on
   failure.
3. `compute_provenance()` — best-effort provenance spans.

### Edge replacement in merge_nodes
`merge_nodes` replaces auto-generated `summarizes` edges with
`refines` edges. This replacement is NOT transactional: edges
are deleted and re-inserted in separate writer-lock scopes.
A crash between deletion and re-insertion leaves a partial state.

## Forbidden Patterns

### MUST NOT insert a DAG node without edge
Every summary node MUST have corresponding `summarizes` or
`refines` edges. `insert_summary_atomic` enforces this
automatically. Manual `insert_dag_node` does NOT.

### MUST NOT leave dangling parent_ids
When a node is deleted (soft or hard), its children's `parent_ids`
arrays MUST be updated to remove the reference. This is NOT
enforced automatically by the schema (no foreign key cascade).

### MUST NOT call unchecked_transaction without Mutex
`conn.unchecked_transaction()` is ONLY safe because
`Mutex<Connection>` serializes all access. If the writer lock
model changes, all `unchecked_transaction()` calls MUST be
replaced with `transaction()`.

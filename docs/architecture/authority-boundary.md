# Authority Boundary Audit (Phase 3.2)

## Purpose
Define which subsystem is the single authority for each dependency
kind. This prevents silent divergence where two subsystems maintain
independent — and potentially conflicting — dependency records.

## Authority Table

For each `DependencyKind`, exactly ONE subsystem is the authority.

| DependencyKind | Authority | Storage | CAN create | CAN read | FORBIDDEN |
|---|---|---|---|---|---|
| `Coverage` | `DagEngine` | `dag_edges` (summarizes) | Compression, leaf insertion | All | Heuristic compression; topology mutation outside DagEngine |
| `Refinement` | `DagEngine` | `dag_edges` (refines) | `merge_nodes` | All | Creating refines edges without deleting summarizes |
| `CrossSessionReuse` | `DagEngine` | `dag_edges` (reuses) | `dedup_and_reuse` | All | Cross-session dedup without embedding similarity check |
| `SequentialOrdering` | `ChatPipeline` | `lineage_edges` (depends_on) | Consecutive unit pairs | Auditors | Wall-clock ordering; inferring ordering from timestamps |
| `ParallelJoin` | `ForkJoinTracker` | `dag_edges` (happens_before) | Group completion | Auditors | Inferring join structure from alone |
| `ReadsFile` | `tool_cache` | In-memory + SQLite | `extract_dependent_files` during tool execution | Cache invalidation | Inventing files not in tool args; guessing dependencies |
| `ProducesFile` | NONE (inactive) | — | — | — | Guessing from tool name alone |
| `SearchesFile` | NONE (inactive) | — | — | — | — |
| `Derivation` | NONE (inactive) | — | — | — | — |
| `Invalidation` | NONE (inactive) | — | — | — | — |
| `FailureCorrection` | NONE (inactive) | — | — | — | — |

## Key Authority Boundaries

### The DAG is the sole topology authority
Topology dependencies (Coverage, Refinement, CrossSessionReuse) are
created ONLY by `DagEngine` methods. No other subsystem may create
or modify `dag_edges` rows of these kinds. The `mutation.rs` module
is the only exception — it creates Coverage edges via its own
`run_cycle` path, and MUST coordinate with the DAG.

### File observation is NOT a dependency authority
`file_observation.rs` detects filesystem changes. It is a detection
mechanism, not an authority. The dependency downstream of file
changes is cache invalidation (ReadsFile), which is owned by
`tool_cache`. File observation feeds into cache invalidation; it
does NOT independently create dependencies.

### Execution ordering is derivable, not discoverable
SequentialOrdering (depends_on) is declared by the pipeline at
creation time. It CANNOT be inferred retroactively from timestamps
or any other source. During replay, ordering is reconstructed
from `logical_seq`, not from `depends_on` edges.

## Forbidden Cross-Boundary Patterns

1. **Cache invalidation from observation**: `tool_cache_invalidate`
   MUST read file paths from `on_files_changed`, which in turn
   MUST receive paths from `store_file_observation`. Cache
   invalidation MUST NOT independently probe filesystem state.

2. **Topology from execution**: DAG edges MUST NOT be created
   from execution unit metadata alone. `related_nodes` on
   `ExecutionUnit` is a structural convenience, not a topology
   authority.

3. **Dependency from motif**: Motif detection (`motif.rs`) identifies
   recurring patterns. It MUST NOT create any dependency edges.
   Motifs are observations, not authorities.

4. **Multiple authorities for the same dep**: No dependency
   relationship may be independently tracked by two subsystems.
   If duplication is discovered, one MUST be designated canonical.

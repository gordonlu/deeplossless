# Dependency Model (Frozen)

## Purpose
Define what "dependency" means in this runtime, which subsystem is
the authority for each kind, and what MAY NEVER be inferred.

## What Is a Dependency

A dependency is a **directed relationship** where entity B's
correctness or relevance depends on entity A's state.

Dependencies are declared by the authority that creates them.
Dependencies are NEVER inferred heuristically.

## Dependency Taxonomy

### Topology dependencies (DAG)
The DAG is the authoritative source for the conversation topology.

| Kind | Location | Created by | Meaning |
|------|----------|-----------|---------|
| `summarizes` | `dag_edges` | `compress_group_with_snippets`, `insert_leaf` | Summary node covers source nodes |
| `refines` | `dag_edges` | `merge_nodes` | Higher-level summary refines lower-level summaries |
| `reuses` | `dag_edges` | `dedup_and_reuse` | Cross-conversation semantic reuse |
| `happens_before` | `dag_edges` | `ForkJoinTracker::complete()` | Parallel execution ordering |

### Execution ordering dependencies
Ordering dependencies track the sequence of tool executions.

| Kind | Location | Created by | Meaning |
|------|----------|-----------|---------|
| `depends_on` | `lineage_edges` | `ChatPipeline` (consecutive units) | Unit B followed unit A |

MUST NOT use wall-clock for ordering. Use `logical_seq`.

### Artifact dependencies (cache invalidation)
Cache invalidation depends on file content changes.

| Kind | Location | Created by | Meaning |
|------|----------|-----------|---------|
| `dependent_files` | `tool_cache` (in-memory + SQLite) | Pipeline during tool execution | Cache entry X depends on file Y |

### FUTURE — not yet active
These are defined in the codebase but have no active producers:

| Kind | Defined in | Status |
|------|-----------|--------|
| `InvalidatedBy` | `execution.rs LineageEdge` | Enum only, no producer |
| `SuggestedBy` | `execution.rs LineageEdge` | Enum only, no producer |
| `CorrectedBy` | `execution.rs LineageEdge` | Enum only, no producer |
| `compare_observations` | `file_observation.rs` | Test-only, no producer |
| `DependencyIndex` | `artifacts.rs` | Full design, zero integration |

## Authority Table

| Data | Authority | CAN create | CAN read | FORBIDDEN |
|------|-----------|-----------|----------|-----------|
| Topology deps (summarizes, refines, reuses) | `DagEngine` | Yes, via compression/merge/dedup | All subsystems | Heuristic inference of topology |
| Execution ordering (depends_on) | `ChatPipeline` | Yes, consecutive unit pairs | `audit.rs` | Wall-clock ordering for replay |
| Parallel ordering (happens_before) | `ForkJoinTracker` | Yes, group completion | `pipeline.rs` | Infer from timestamp |
| Cache dependencies (dependent_files) | `tool_cache` module | Yes, via `extract_dependent_files` | `tool_cache_invalidate` | Invent files not extracted from args |
| File observation (compare) | NOT YET ACTIVE | No producer | No consumer | Mutating DAG from observation |

## Forbidden Patterns

### MUST NOT infer dependencies
No subsystem may guess that A depends on B. Dependencies are
explicitly declared by the creating authority.

### MUST NOT use multiple sources for the same dependency class
File change detection flows through `store_file_observation` →
`tool_cache_invalidate`. There is no separate "observation →
topology" path. If such a path is needed, it MUST use the
`file_observations` table as the single change-detection source.

### MUST NOT mutate topology based on observation
File observations detect filesystem changes. Cache invalidation
is a downstream effect of observation. DAG topology is independent
of filesystem state. These are separate dependency classes.

### MUST NOT record dependencies that are derivable
`DependsOn` edges between consecutive execution units are derivable
from `logical_seq` ordering. They are stored for audit convenience,
not as a source of truth. If they drift from logical_seq, the
logical_seq takes priority.

## Replay Semantics for Dependencies

During replay:
- Topology dependencies (dag_edges) reflect the topology AT REPLAY
  TIME, which may differ from execution time.
- Execution ordering (depends_on, logical_seq) reflects the
  ORDERING AT EXECUTION TIME, reconstructed from the event log.
- Cache dependencies are NOT replayed — cache state during replay
  is irrelevant.
- File observations are NOT replayed.

Replay MUST NOT reconstruct dependency state from live system
state (current DAG, current cache, current filesystem).

## Notes

- `artifacts.rs` defines a complete dependency tracking framework
  (`DependencyIndex`, `DependencyEdge`, `artifact_versions`) with
  no production wiring. Before wiring, MUST decide whether this
  replaces or supplements the `tool_cache` dependent_files system.
- `compare_observations` and `ObservationDiff` have no consumers.
  Before removing, verify no planned integration path depends on
  them.
- `InvalidatedBy` / `SuggestedBy` / `CorrectedBy` LineageEdge
  variants are currently dead code. MAY be removed or MAY be
  wired if Phase 3 identifies clear producers.

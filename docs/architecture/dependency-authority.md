# Dependency Authority

## Purpose
Define which subsystem is the single authority for each class of
data mutation and invalidation decision.

## Responsibilities
This document defines ownership boundaries. A subsystem MUST NOT
mutate data owned by another subsystem without going through the
owner's public interface.

## Authority Table

| Data | Authority | Mutators | Invalidation trigger |
|------|-----------|----------|---------------------|
| DAG topology (nodes, edges, parent_ids, child_ids) | `DagEngine` | `compress_group_with_snippets`, `merge_nodes`, `insert_leaf`, `dedup_and_reuse` | Post-mutation: revision_counter++, optional integrity check |
| Execution events (tool call outcomes) | `Database` (via `store_execution_event`) | proxy forwarding path (`db.store_execution_event`), `lcm_runtime_report` | Append-only. NEVER delete execution events. |
| Runtime metrics (tokens_spent, cache_hits, failure_streak) | `ExecutionCycle` | `record_cache_hit`, `record_failure`, `record_tokens`, `set_session_metrics` | Metrics MUST be updated through `record_*` methods, not direct field mutation. |
| Tool cache entries | `tool_cache` module (L1HotCache + DB) | `tool_cache_get`, `tool_cache_set`, `invalidate_affected_by_file` | File observation change triggers invalidation. |
| File observations | `file_observation` module (via `store_file_observation`) | `observe_file` creates observation; DB stores it atomically. | TTL-based staleness via `FileObservation::is_stale()`. |
| Compaction decisions | `CompactionPlanner` | `plan()`, `plan_slide_window()` | Planner produces a plan; worker executes via `do_compress()` → `compress_group_with_snippets()`. |

## Forbidden Patterns

### Direct field mutation on ExecutionCycle
MUST NOT write `cycle.metrics.cache_hits += 1`. MUST use
`cycle.record_cache_hit(name)`. The record methods are the only
entry point for metrics updates and are the future hook for
event-sourced replay.

### Bypassing the writer lock for DAG writes
MUST NOT call SQLite write operations without acquiring
`Database.writer` lock. All write methods on `Database` MUST
lock `self.writer` before executing SQL.

### Compaction without planner
`CompressGroup` command bypasses the planner (caller-specified
nodes). This is intentional — the caller assumes full
responsibility for correctness. `ReviewAndCompact` and
`SlideAndCompact` MUST go through the planner.

## Dependency Chain

```
RuntimeMetrics ← Execution events (read-only aggregation)
DAG topology ← Compaction (compress/merge) + Mutation engine
Tool cache ← File observations (invalidation)
File observations ← observe_file (atomic via store_file_observation)
Compaction plan ← DAG state (leaves + tokens) read by planner
```

## Overlap Conflicts

File observations and tool cache BOTH track file identity and
change detection. The overlapping authority is:
- Tool cache invalidates by file path + content hash
- File observations detect changes by content_hash + semantic_hash

Resolution: `store_file_observation` is the atomic write point.
Tool cache invalidation MUST query `file_observations` for
change detection, not independently probe filesystem state.
(Currently NOT implemented — both modules maintain independent
notions of file identity. This is a known gap.)

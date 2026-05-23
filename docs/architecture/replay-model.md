# Replay Model

## Purpose
Define what is required for deterministic replay of a conversation
execution trace, and what the current system guarantees.

## Replay Semantics

### What replay MUST produce
Given a `conversation_id` and a `seq_no` upper bound, replay MUST
reconstruct the same sequence of tool calls and LLM responses as
the original live execution, WITHOUT making real LLM API calls.

### What replay MUST NOT do
Replay MUST NOT execute file mutations, network calls, or any
side effects. Cache hits during replay MUST be served from
stored results, not from fresh tool cache lookups.

### Replay authority
Replay reconstructs state exclusively from persisted events
(`execution_events`, `dag_events`). During replay, the runtime
MUST NOT infer execution state from:

- wall-clock timing (`Instant::now()`, `SystemTime::now()`)
- current DAG topology (topology may have changed since execution)
- current tool cache state (cache entries may have been evicted)
- filesystem state (files may have been modified or deleted)
- live API responses (LLM calls, network requests)

## Current Guarantees

### Deterministic ordering
`execution_units.epoch_ms` is populated from
`crate::execution::next_logical_seq()` — a monotonic atomic
counter. This guarantees deterministic ordering within a single
process lifetime. Audit queries ORDER BY `epoch_ms DESC, id DESC`
provide stable sort order.

### Immutable event log
`execution_events` table is append-only (INSERT only, no UPDATE
or DELETE on existing rows). Each event records:
- `event_kind`, `event_payload`
- `span_id`, `parent_span_id`, `span_mode`, `parallel_group`
- `epoch_ms`, `replay_session_id`

### RuntimeMode
`ExecutionCycle.mode` supports three variants:
- `Live` — normal operation
- `Replay { session_id, up_to_seq }` — deterministic replay mode
  defined but NOT wired into the proxy execution path
- `DryRun` — policies evaluated, no execution

### ExecutionResult (side-effect marker)
`ExecutionResult.has_side_effects` and `.is_replay` flags allow
the execution layer to skip side effects during replay.

## Gaps

### Replay not wired
`RuntimeMode::Replay` is defined but no code in proxy.rs or
pipeline.rs checks `cycle.mode.is_replay()` to short-circuit
LLM calls. The summarizer has `shutdown_notify` for early exit
but has no replay short-circuit.

### No replay reader
`dag_events` and `execution_events` contain the raw event log,
but no function exists to replay events sequentially and
reconstruct `RuntimeState` from them. This is a prerequisite
for `RuntimeMode::Replay`.

### Cancellation during replay
The summarizer's `shutdown_notify` polls a `Notify` via
`now_or_never()`. The proxy's spawned SSE tasks have no
`AbortHandle` or `CancellationToken`. Replay tasks have
no way to be cancelled mid-execution.

### Parallel execution in replay
`ForkJoinTracker` uses `Instant::now()` for deadline tracking
(`deadline` field, `#[serde(skip)]`). During replay, wall-clock
deadlines are meaningless. Replay MUST use the recorded
`completed_at` timestamp or logical sequence numbers instead.

## Requirements for Complete Replay

1. A `replay_session_id` column in `execution_events` for
   grouping events by replay run.
2. A `replay_execution()` function that reads `execution_events`
   in `epoch_ms` order and feeds results to the proxy path.
3. All `tokio::spawn` call sites MUST accept an optional
   `CancellationToken` or `AbortHandle`.
4. `ForkJoinTracker.deadline` MUST use logical sequence
   comparison during replay, not `Instant::now()`.

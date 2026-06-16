# Runtime Event Schema (Frozen)

## Purpose
Define the contract for `RuntimeEvent` — the append-only runtime
lifecycle event stream that is the single source of truth for
`ExecutionCycle` lifecycle state.

This contract does **not** cover every row in the `execution_events`
table. Protocol replay rows may store canonical `StreamEvent` payloads,
and audit rows may store compact execution-unit lifecycle records. Those
rows share storage, but they are not `RuntimeEvent` unless explicitly
serialized with `RUNTIME_EVENT_SCHEMA_VERSION`.

## Allowed Event Categories

ONLY these categories of events may exist:

1. **Execution lifecycle**: `ExecutionStarted`, `ToolCallScheduled`,
   `ToolCallCompleted`, `ToolCallFailed`
2. **Retry lifecycle**: `RetryScheduled`, `RetryAborted`
3. **Cancellation lifecycle**: `CancellationRequested`,
   `CancellationAcknowledged`, `CancellationCompleted`

MUST NOT add events for: DAG topology changes, compaction results,
cache entry mutations, file observation snapshots, summarizer output,
or any other subsystem.

## Forbidden Payloads

Events carry lifecycle metadata, NOT data payloads. Specifically:

MUST NOT include:
- Full tool results (`Value` or `String` longer than 256 chars)
- Full LLM response bodies
- File contents
- DAG node summaries
- Cache entry contents
- Stack traces

MAY include:
- Token counts (u64)
- Error signatures (short strings, e.g., "ENOENT", "HTTP 429")
- Tool call IDs, span IDs (reference strings)
- Attempt numbers (u32)
- Boolean status flags (cache_hit, retryable, clean)

## Ordering Guarantees

`logical_seq` is the canonical ordering key. It is:

- Monotonic (AtomicI64, never decreases)
- Append-order (first event appended = lowest seq)
- NOT wall-clock
- NOT UUID-sortable
- Deterministic within a single process lifetime

Event ordering within a conversation is determined by `logical_seq`.
Cross-conversation ordering is NOT guaranteed.

## Cooperative Cancellation Contract

`CancellationRequested` signals the INTENT to cancel. It does NOT
imply immediate termination of in-flight work.

After a `CancellationRequested`, each in-flight execution MUST emit
one of:

1. `CancellationAcknowledged` — the execution stopped cleanly
2. `ToolCallCompleted` — the execution finished normally before the
   cancel signal was processed
3. `ToolCallFailed` — the execution failed (cancel or not)

The runtime MUST eventually emit `CancellationCompleted` after all
in-flight work has stopped. A `cancelled` state is considered
resolved when `CancellationCompleted` is emitted.

A `clean: true` completion means all in-flight work acknowledged
the cancel. `clean: false` means some work was abandoned.

## Event Payload Policy

Events MUST be lightweight. The event stream is lifecycle truth,
not a storage dump. If an event field exceeds 256 bytes, the
event is carrying a payload that belongs in storage, not in the
lifecycle event log.

## Projection Relationship

Mutable `ExecutionCycle.metrics` fields (`tokens_spent`, `cache_hits`,
`failure_streak`, etc.) are projections derived from the `RuntimeEvent`
lifecycle substream.
They are NOT sources of truth.

During the Phase 2 transition, both events and projections coexist.
Projections are the safety net. The transition sequence is:

1. `record_*` methods append events, then update projections
2. `RuntimeStateView` computes state from `RuntimeEvent` values
3. `projection_parity_check()` validates runtime events vs. projections
4. ONLY after sustained parity (long-run soak tests pass):
   projections may be removed

## Schema Evolution

MUST NOT remove existing event variants. MAY add new variants ONLY
within the allowed categories. MAY NOT change field types on
existing variants — this breaks replay compatibility.

New fields on existing variants require:
1. A `#[serde(default)]` or `Option` field with a zero-value default
2. Backward-compatible replay of old event streams

## Debug Dump Format

`RuntimeStateView::dump_events()` produces a human-readable log:

```
[0001] ExecutionStarted conv=1 seq=1 profile=efficient
[0002] ToolCallScheduled conv=1 tool=grep tcid=tc_1 span=sp_1 attempt=1
[0003] ToolCallCompleted conv=1 tool=grep tcid=tc_1 attempt=1 tokens=100 unit=10
[0004] ToolCallFailed conv=2 tool=read_file tcid=tc_2 attempt=1 error=ENOENT retryable
[0005] RetryScheduled conv=2 tcid=tc_2 attempt=2 fix="check path"
```

This format is for debugging only. It is not a contract — it MAY
change without notice.

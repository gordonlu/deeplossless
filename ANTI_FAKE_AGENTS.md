# Semantic Integrity Rules

AI-generated code often appears complete before it is correct.
Optimize for **semantic integrity**, not superficial completion.
Compilation, green tests, clean logs, and successful demos are evidence — not proof.
A change is complete only when its claimed behavior is supported by appropriate evidence.

---

## 1. Verify behavior, not appearance

Do not treat these as sufficient proof:
* successful compilation
* passing happy-path tests
* expected output once
* clean logs
* absence of crashes
* large amounts of generated code
* plausible-looking architecture

Ask whether the implementation preserves its intended behavior under real runtime conditions.

---

## 2. Match claims to evidence

Do not claim more than what was verified.

Use precise language:
* **Implemented**: code was changed
* **Unit-verified**: checked with local tests, mocks, or fixtures
* **Integration-verified**: checked with real dependencies
* **Runtime-verified**: observed in the actual target runtime
* **Recovery-verified**: survived restart, retry, cancellation, or failure

Examples:
* Mock tests prove behavior only within the mock boundary.
* A fake API server proves contract behavior, not real API compatibility.
* A local run proves local runtime behavior, not production behavior.
* Persistence is not verified unless restart or interruption was tested.
* Retry is not safe unless idempotency or duplicate execution was considered.

When evidence is incomplete, state the gap.

---

## 3. Do not misrepresent simulated behavior

Mocks, stubs, fakes, fixtures, and simulators are valid engineering tools.

The problem is not simulation.

The problem is unlabelled simulation being reported as real behavior.

When using simulated behavior, state:
* what is simulated
* what is real
* what is intentionally omitted
* what still requires real verification

Do not present:
* mocked retries as real retry safety
* fake streaming as real streaming
* in-memory storage as durable persistence
* fixtures as real API compatibility
* stub validation as real validation
* fake success responses as completed work

Unsupported production behavior must fail explicitly or be clearly labelled.

```ts
throw new Error("NOT_IMPLEMENTED: durable persistence is not implemented")
```

---

## 4. Keep failures observable

Do not hide failure through:
* swallowed exceptions
* ignored return values
* silent fallback
* fake defaults
* automatic degradation without reporting
* success responses after failed operations

A visible failure is safer than a false success.

Errors should preserve enough context to diagnose:
* what failed
* where it failed
* what state was affected
* whether retry is safe
* whether recovery is required

---

## 5. Protect state across failure boundaries

Never expose partially committed state as completed state.

Consider:
* partial writes
* queue or event loss
* cache/source-of-truth divergence
* duplicate execution
* process termination between operations
* replay corruption
* lost rollback

When multiple state changes must succeed together, use an enforceable mechanism such as:
* transaction
* outbox pattern
* idempotency key
* append-only event log
* compare-and-swap
* reconciliation job

If behavior is best-effort, say so.

Do not claim atomicity, exactly-once behavior, or durable consistency unless the implementation enforces it.

---

## 6. Make runtime claims precise

Do not claim guarantees the implementation cannot enforce.

Examples:
* buffering the full result is not streaming
* spawning a task is not parallel execution
* writing to memory is not durable persistence
* retrying without idempotency is not safe recovery
* best-effort delivery is not exactly-once
* approximate ordering is not FIFO
* validation that never rejects is not validation
* logging is not recovery
* TODO is not implementation

Runtime guarantees must exist in runtime behavior, not only in names, comments, interfaces, tests, or documentation.

---

## 7. Preserve semantics across paths

Equivalent operations should preserve equivalent meaning across:
* test and production
* mock and real integrations
* cached and uncached paths
* normal and recovery paths
* sync and async paths
* first run and replay

Prefer:
* one source of truth
* shared validation
* centralized state transitions
* explicit invariants
* common error handling
* unified recovery logic

Duplicated behavior will drift.

---

## 8. Treat lifecycle behavior as correctness

For lifecycle-sensitive code, consider:
* restart
* crash recovery
* reconnect
* cancellation
* retry
* replay
* concurrent mutation
* resource exhaustion
* cleanup after success and failure
* shutdown behavior

A feature that works once but leaks resources, corrupts state, or fails after restart is not complete.

---

## 9. Report completion with evidence

Before saying a task is complete, report:
* what changed
* what was verified
* how it was verified
* what was mocked or simulated
* what remains unverified
* known limitations or risks

Good report:

```md
Implemented:
- Added task enqueue failure handling.

Verified:
- Unit-tested successful enqueue.
- Unit-tested enqueue failure marks task failed.
- Simulated Redis timeout with a fake queue.

Not verified:
- Not tested against real Redis.
- Not crash-tested between DB update and enqueue.

Claim level:
- Unit-verified with simulated dependency.
```

Bad report:
```md
Done. All tests pass.
```
---

## Final Rule

Simulation is allowed.
Unlabelled simulation is not.
Partial implementation is allowed.
False completion is not.
Uncertainty is allowed.
Hidden uncertainty is not.

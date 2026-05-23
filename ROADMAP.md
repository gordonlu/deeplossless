# Roadmap — deeplossless

Status: **Phase 5 — Stabilization Window**. 287 tests, clippy clean.

---

## Completed (v0.4.0+)

### Phase 1 — Deterministic Execution Runtime ✅
| Item | Status |
|------|--------|
| Canonical Stream/Event IR (`StreamEvent`, `CanonicalRequest`) | ✅ |
| StreamAssembler (parallel tool buffering, text passthrough) | ✅ |
| Immutable Event Store (`execution_events` table, append-only) | ✅ |
| Deterministic Replay Engine (`replay_execution`, `GET /v1/lcm/replay/{id}`) | ✅ |
| Snapshot Isolation (tiered L0-L3, budget enforcement) | ✅ |

### Phase 2 — Execution Memory Layer ✅
| Item | Status |
|------|--------|
| DAG engine (embedding dedup, BM25 retrieval, provenance, GC) | ✅ |
| Semantic Tool Reuse (cache key, partial invalidation, stream interception) | ✅ |
| Execution Outcome Scoring (`score_execution`) | ✅ |
| Heap-aware Memory Metrics (outcome weights, cache hit tracking) | ✅ |
| Auditability Layer (`src/audit.rs`) | ✅ |
| Failure Pattern Memory (why_failed + invalidated_assumptions) | ✅ |
| Running Event Log (snapshots, mem-optimized encoder) | ✅ |

### Phase 2.5 — Event Runtime ✅
| Item | Status |
|------|--------|
| `RuntimeEvent` enum (9 variants: execution/retry/cancellation) | ✅ |
| `ExecutionCycle` lifecycle methods (event-first, projection-second) | ✅ |
| `RuntimeStateView` (derived state from event log: inspect/explain/dump/parity) | ✅ |
| `RuntimeInvariants` (monotonic seq, cancellation well-formed, retry ordering) | ✅ |
| Long-run soak tests (1000 cycles, 500-retry storm, cross-conv isolation) | ✅ |

### Phase 3 — Dependency Semantics ✅
| Item | Status |
|------|--------|
| `DependencyKind` taxonomy (11 variants: Coverage, Refinement, ReadsFile, etc.) | ✅ |
| `DependencyView` (unified interpretation: topology_descendants, execution_predecessors) | ✅ |
| `LlmProvider` trait + `OpenAiCompatibleProvider` | ✅ |
| `dependency-model.md` (authority table, forbidden inference patterns) | ✅ |
| `authority-boundary.md` (per-kind authority assignment) | ✅ |

### Phase 4 — Runtime Hardening ✅
| Item | Status |
|------|--------|
| Compactor shutdown gap (handle registered in `BackgroundTasks`) | ✅ |
| CancellationToken-style `shutdown_notify` on 3 proxy spawn sites | ✅ |
| `RetryClass` formal classification + `RetryBackoff` discipline | ✅ |
| `RetryClass::classify(error, http_status)` + per-class delay | ✅ |
| Summarizer: jittered backoff, token-aware truncation, typed response, adaptive timeout | ✅ |
| `CompactionPlanner` with configurable scoring weights | ✅ |
| Motif: unit_map hoist, HashSet dedup, confidence fix, canonical tool names | ✅ |
| Legacy lifecycle methods `#[deprecated]` with migration guidance | ✅ |
| Duplicated model mapping eliminated (proxy.rs → `ModelRegistry`) | ✅ |
| Execution event storage errors logged (no longer silently dropped) | ✅ |

### Phase 5 — Stabilization ✅
| Item | Status |
|------|--------|
| Schema version constants (`RUNTIME_EVENT_SCHEMA_VERSION`, etc.) | ✅ |
| Architecture cadence rule (authority/invariants/forbidden/lifecycle) | ✅ |
| `REVIEW.md` cleanup — all actionable items fixed, ~17 architectural items remaining | ✅ |
| Dependency taxonomy for cache quality (`ToolKind::dependency_kind()`) | ✅ |
| `RuntimeStateView::inspect()` comprehensive runtime report | ✅ |
| 11 architecture docs in `docs/architecture/` | ✅ |
| README restructured: user-facing (110 lines), tech details in `docs/tech-reference.md` | ✅ |

---

## Architecture Docs (11 files, `docs/architecture/`)

| Document | Scope |
|----------|-------|
| `dag-invariants.md` | 7 DAG invariants |
| `frozen-invariants.md` | 9 cross-subsystem iron laws + architecture cadence rule |
| `replay-model.md` | Replay semantics, guarantees, authority rules |
| `runtime-lifecycle.md` | Startup/shutdown order, ownership model |
| `dependency-authority.md` | 6 subsystem authority boundaries |
| `dependency-model.md` | Dependency definition, taxonomy, authority table |
| `runtime-events.md` | Frozen RuntimeEvent contract |
| `authority-boundary.md` | Per-DependencyKind authority |

---

## Future Work (subsystem-level redesign, ~17 items in REVIEW.md)

- **Runtime state decoupling**: StateView computing RuntimeState from event log
- **Explicit replay wiring**: `RuntimeMode::Replay` wired into proxy execution path
- **Cooperative cancellation**: CancellationToken for all spawned tasks
- **Observation pipeline**: Split detect/cache/invalidate, add revision history
- **Dead code removal**: 5 inactive LineageEdge variants, `artifacts.rs` unintegrated framework
- **Dependency consolidation**: `artifacts.rs` vs `tool_cache` — unified invalidation path
- **Motif dependency/context awareness**: Typed execution motifs beyond tool sequence

---

## Current Metrics

| Metric | Value |
|--------|-------|
| Tests | 287 (249 unit + 18 integration + 12 soak + 5 benchmark + 1 long-session + 2 simulated) |
| Source files | 22 under `src/` |
| Architecture docs | 11 |
| API endpoints | 20+ |
| Lint | clippy `-D warnings` clean |
| Edition | Rust 2024 |

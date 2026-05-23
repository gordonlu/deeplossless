# Runtime Lifecycle

## Purpose
Define the startup order, running semantics, shutdown order, and
restart behavior of all runtime services.

## Service Startup Order (runtime_coordinator::build)

1. **Database** — `Database::builder().path().build().await`
   Opens SQLite in WAL mode, runs migrations, creates read pool
   (8 connections) and writer connection.

2. **DAG Engine** — `DagEngine::builder().build(db.clone())`
   Wraps the Database. Auto-validation OFF in production.

3. **Compactor** — `Compactor::new(db, config)`
   Internally spawns `tokio::spawn(compactor_supervisor(...))`.
   The supervisor wraps `compact_worker` in `catch_unwind`.

4. **BackgroundTasks** — `Arc::new(BackgroundTasks::new())`
   Holds `Vec<JoinHandle<()>>` and `Arc<AtomicBool>` shutdown flag.

5. **Mutation Engine** — `MutationEngine::new(...)`
   Spawns `spawn_mutation_cycle(engine, interval, Some(shutdown_flag))`.
   Handle registered via `tasks.register_handle()`.

6. **HTTP client** — `reqwest::Client::builder().connect_timeout(10s)`

7. **Rate limiter** — `RateLimiter::new(rate_limit)`

8. **AppState assembly** — All fields populated at build time.
   `api_key` and `admin_key` are `Arc<StdMutex<Option<String>>>`,
   populated from CLI/env if provided, otherwise `None` (lazily set
   from first request's Authorization header).

## Service Running Semantics

### Compactor worker
- Single-threaded async task receiving `CompactCommand` via mpsc.
- All DB writes go through this single task (single-writer model).
- Responds with one `CompactEvent` per `command()` call.
- Additional events drained via `drain_events()`.
- Health check via `health_ping()` (Ping → Pong round-trip).

### Mutation engine
- Background loop: `spawn_mutation_cycle` calls `run_cycle` on
  interval, which produces mutation proposals.
- Checks `shutdown_flag` via `AtomicBool` each iteration.

### Proxy request handling
- Each HTTP request is handled by an axum handler.
- SSE streaming spawns a `tokio::spawn` task for each
  streaming response (3 sites: response storage, streaming proxy,
  Context-ReAct).
- Spawned tasks have NO AbortHandle, NO CancellationToken.
  They exit when the SSE channel closes (client disconnect) or
  when the upstream stream ends.

### Rate limiter
- Global fixed-window counter (100 req/s default).
- Background reset task updates `window_start` via `AtomicU64`.

## Shutdown Order

1. `RuntimeCoordinator::shutdown(timeout)` is called.
2. `BackgroundTasks::shutdown()` sets `AtomicBool` to true.
3. All registered `JoinHandle`s are drained and awaited with
   `tokio::time::timeout`.
4. The mutation engine loop exits when it polls the shutdown flag.
5. BackgroundTasks shutdown returns.

### What is NOT shut down
- The compactor worker: `_worker_handle` is stored but never
  registered in BackgroundTasks. No `CompactCommand::Shutdown` is
  sent. The worker continues running until the tokio runtime drops.
- The spawned SSE tasks: these self-terminate (channel close or
  upstream end), but have no forced cancellation path.
- The summarizer's `llm_summarize` retry loop: if retrying with
  extended backoff during shutdown, it will complete its retries
  before returning (unless `shutdown_notify` is configured).

## Restart Behavior

### Compactor
- Panic in worker: caught by `compactor_supervisor`'s
  `catch_unwind`, logged as `error!`. Worker is NOT restarted
  because `mpsc::Receiver` is not Clone.
- Graceful exit: `CompactCommand::Shutdown` → worker loop breaks.

### Mutation engine
- Panic: NOT caught. The `spawn_mutation_cycle` tokio task will
  terminate. No automatic restart.
- Graceful exit: polls `shutdown_flag`, loop exits.

### Rate limiter reset task
- Panic: NOT caught. Rate limiting silently stops resetting.
  The `AtomicU64` counter continues incrementing.

## Ownership Model

| Owner | Owns | Lifecycle |
|-------|------|-----------|
| `RuntimeCoordinator` | Service startup sequence, shutdown orchestration, `BackgroundTasks` supervision | Created at process start, destroyed on process exit |
| `Compactor` | Compaction worker lifecycle (`_worker_handle`), compaction command queue (mpsc channels) | Worker spawned in `Compactor::new()`, terminates on `Shutdown` command or process exit |
| `MutationEngine` | Mutation scheduling loop (`spawn_mutation_cycle`), proposal generation | Spawned in `build()`, polls `shutdown_flag` each iteration |
| `RateLimiter` | Rate-limit window counter, periodic reset task | Spawned in `RuntimeServices`, reset task self-terminates with runtime |
| `Database` | Writer connection, read pool (8 connections), WAL checkpointing | Created at process start, connections closed on `Drop` |
| `ExecutionCycle` | Runtime metrics, decisions log | Owned by `Arc<StdMutex<ExecutionCycle>>`, shared across proxy handlers |

Each owner is the SOLE authority for its lifecycle. No external
code may spawn or abort a task owned by another component without
going through the owner's public interface.

## Forbidden Patterns

### MUST NOT block the runtime thread
All DB writes go through `spawn_blocking` or the compactor's
dedicated async task. Synchronous DB calls in axum handlers
MUST be wrapped in `tokio::task::spawn_blocking`.

### MUST NOT spawn without lifecycle management
Every `tokio::spawn` MUST either:
- Register its handle in `BackgroundTasks` for graceful shutdown, OR
- Accept a `CancellationToken`/`AbortHandle` for forced cancellation, OR
- Document why self-termination is guaranteed.

### MUST NOT hold locks across await points
`StdMutex` (sync Mutex) guards are NOT `Send`. They MUST be
dropped before `.await`. `tokio::sync::Mutex` is used for the
compactor (lock held across `command()` which contains `.await`).

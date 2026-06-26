# DS4 Infrastructure Optimization

Date: 2026-06-17

## Scope

Seven infrastructure-level optimizations to align deeplossless with DeepSeek-V4's
native capabilities. All items are "底层基础设施优化" — protocol, streaming,
error handling, context assembly. No policy/strategy/routing decisions.

| ID | Item | Area |
|---|---|---|
| DS4-05 | `<think>` tag fallback | Streaming normalization |
| DS4-08/09 | DSML ↔ Canonical ToolCall parser | Protocol |
| DS4-14/15 | Stable prefix support | Request analysis |
| DS4-19 | Importance-based context ordering | Context assembly |
| DS4-32 | Stream robustness | Streaming normalization |
| DS4-33 | Structured error types | Cross-cutting error layer |
| DS4-40 | DeepSeek native API compatibility | Protocol + CLI |

---

## Architecture

```
                    Upstream SSE Stream
                           │ raw bytes
                   ┌───────▼────────┐
                   │ PartialLineBuf │  ← DS4-32
                   └───────┬────────┘
                           │ complete data: line
                   ┌───────▼────────┐
                   │ SseEventParser │  ← DS4-32
                   │ (unknown field │
                   │  tolerant)     │
                   └───────┬────────┘
                           │ raw JSON event
                   ┌───────▼────────────┐
                   │ DeepSeekNormalizer │  ← DS4-32
                   │ (前置 normalizer;  │
                   │  不改 existing     │
                   │  StreamAssembler)  │
                   └───────┬────────────┘
                           │ canonical event
              ┌────────────┼────────────┐
              │            │            │
       ┌──────▼───┐ ┌─────▼────┐ ┌────▼──────┐
       │structure │ │structure │ │ DSML      │  ← DS4-08/09
       │reasoning │ │tool_calls│ │ parser    │
       └──────────┘ └──────────┘ └────┬──────┘
                               [priority: structured > DSML]
                                      │
                               ┌──────▼──────┐
                               │ThinkTagFbk  │  ← DS4-05
                               │(content only│
                               │no reasoning │
                               │ received)   │
                               └──────┬──────┘
                                      │ canonical StreamEvent
                                      ▼

     ┌─ Cross-cutting Error Layer ─────────────┐
     │  ModelError (DS4-33)                     │
     │  ← upstream HTTP / SSE / JSON / DSML    │
     │  ← think / tool-call / context / prefix │
     └──────────────────────────────────────────┘

     ┌─ Before Send (upstream request path) ────┐
     │  ContextPack + Importance Ordering        │  ← DS4-19
     │    → historical DAG tiering               │
     │    → recent causal tail preserved         │
     │    → latest user always last              │
     │  Tool Schema Canonicalization             │  ← DS4-14/15
     │  PrefixStabilityChecker                   │  ← DS4-14/15
     │    (checks final CanonicalRequest)        │
     │  DeepSeek Native Options                  │  ← DS4-40
     │    reasoning_effort                       │
     │    dsml_mode (parse / emit / off / test)  │
     │    quick_instruction (experimental)        │
     └──────────────────────────────────────────┘
                        │
                        ▼
                   Upstream API
```

### Principles

- Each module is additive — existing paths unchanged when its feature is off.
- `ModelError` is cross-cutting: any module can produce an error, not a linear layer.
- Structured data (reasoning_content, tool_calls) always takes priority over
  content parsing fallbacks.
- Stream normalizers are **prepended**, not refactored into existing
  `StreamAssembler`.

---

## DS4-05: Think Tag Fallback

**File**: `src/think_tag.rs`

### Behavior

Only activates when the upstream assistant response **does not** contain
structured `reasoning_content`. Parses inline `<think>...</think>` tags from
assistant text content.

### Rules

1. Only `role=assistant` content from upstream — never user/tool/web/shell.
2. If any `StreamEvent::ReasoningDelta` was received this turn, skip fallback
   entirely (structured reasoning wins).
3. `<think>` and `</think>` may span arbitrary stream chunk boundaries.
4. Content inside `<think>...</think>` → `StreamEvent::ReasoningDelta`.
5. Content after `</think>` → `StreamEvent::TextDelta`.
6. If stream ends with unclosed `<think>`, store reasoning trace with
   `complete=false`.
7. Raw assistant content is preserved unchanged — fallback affects only the
   normalized event stream.

### State Machine

```
Idle ──<think>──► InThinkTag ──</think>──► InAnswer
  │                  │                        │
  └──(no tag)──► TextDelta          (end)────┘
                     │                         │
                 TextDelta              TextDelta
                                      (until stream end)
```

### Boundary Cases

- `<think>` split across chunks: buffer partial tag, detect on next chunk.
- `</think>` split across chunks: similar buffer + detect.
- `<think>` inside code block: only trigger if content starts with `<think>`,
  not if tag appears mid-content.
- No `<think>` tag at all: pass through as normal `TextDelta`.

### Tests

- structured reasoning arrives → fallback not invoked
- full `<think>...</think>` in single chunk
- opening tag split across two chunks
- closing tag split across two chunks
- no think tag → normal TextDelta only
- think tag inside code block (mid-content) → not triggered
- stream ends with unclosed `<think>` → incomplete reasoning

---

## DS4-08/09: DSML Parser

**File**: `src/protocol/dsml.rs`

### Priority

```
structured tool_calls > DSML parser > normal text
```

If the upstream response already includes a `tool_calls` array in the delta,
DSML parsing is skipped entirely for that turn. This prevents duplicate tool
calls from parsing both the structured field and the XML inline content.

### Input / Output

```
raw assistant content text
  │
  ▼
parse_dsml_tool_calls(raw: &str) -> Result<Vec<CanonicalToolCall>, DsmlError>
  │
  ├── CanonicalToolCall (for downstream protocol emitters)
  └── Raw DSML preserved for replay/debug
```

### Internal Types

```rust
struct DsmlDocument {
    invokes: Vec<DsmlInvoke>,
}

struct DsmlInvoke {
    name: String,
    parameters: Vec<DsmlParameter>,
}

struct DsmlParameter {
    name: String,
    value: DsmlValue,         // parsed from string="true|false"
    raw_string: String,       // original text value
}

enum DsmlValue {
    String(String),
    Number(f64),
    Bool(bool),
    Null,
    Array(Vec<DsmlValue>),
    Object(HashMap<String, DsmlValue>),
}

enum DsmlError {
    MalformedSyntax { position: usize, detail: String },
    UnclosedTag { tag: String },
    InvalidParameterType { name: String, raw: String },
    NestedTooDeep,
}
```

### DSML Format

```
<|DSML|tool_calls|
<|DSML|invoke| name="read_file">
<|DSML|parameter| name="path" string="true">/foo/bar.txt</|DSML|parameter|>
</|DSML|invoke|>
</|DSML|tool_calls|>
```

- `string="true"`: value is a JSON string
- `string="false"`: value must be parseable as JSON (number, bool, null,
  object, array)

### Streaming Parser

`StreamingDsmlParser` buffers partial DSML chunks until closing
`</|DSML|tool_calls|>` is received, then emits the complete set of
`CanonicalToolCall`s. Partial/truncated DSML at stream end is stored as
`DsmlIncomplete` error.

### Emitter (optional, test-only)

`emit_dsml_tool_calls(tool_calls: &[CanonicalToolCall]) -> String` — produces
DSML text. Used only for roundtrip tests and future native DeepSeek mode. Not
connected to the default request path.

### Raw Preservation

`RawToolCall::DeepSeekDsml(String)` variant preserves the raw DSML text for
replay and debugging. The normalized `CanonicalToolCall` is used for all
downstream protocol emission.

---

## DS4-14/15: Stable Prefix

**File**: `src/prefix_stability.rs` + `docs/deepseek-v4-prefix-stability.md`

### Purpose

Ensure that the prefix of every upstream request (system prompt, tool schemas,
profile) remains as stable as possible, enabling DeepSeek-V4's shared-prefix
and on-disk KV cache to reduce TTFT.

### Placement

```
ContextPack → Tool Schema Canonicalization → CanonicalRequest → PrefixStabilityChecker → upstream
```

The checker analyzes the **final** `CanonicalRequest` after all assembly is
done.

### PrefixStabilityChecker

```rust
struct PrefixStabilityChecker {
    mode: StabilityMode,     // Audit | Warn | Strict
}

enum StabilityMode {
    Audit,   // log only
    Warn,    // emit warning
    Strict,  // return error
}

struct StabilityReport {
    stable_prefix_hash: String,      // system + tools + profile
    full_prompt_hash: String,        // entire messages + tools
    segment_hashes: Vec<PackedHash>, // per-segment
    volatile_detections: Vec<VolatileDetection>,
    tool_schema_stable: bool,
}

struct VolatileDetection {
    segment: String,     // "system" | "tools" | "developer"
    content_type: String, // "timestamp" | "uuid" | "random_id" | "counter"
    excerpt: String,     // first 80 chars of detected content
}

struct PackedHash {
    segment: String,
    hash: String,
    byte_count: usize,
}
```

### Volatile Content Detection

Scans for patterns in `system` / `developer` messages and tool schemas:

- Timestamps: regex for ISO 8601, Unix epoch, `[ts]` markers
- UUIDs: `[0-9a-f]{8}-...`
- Random IDs: `run_id`, `trace_id`, `request_id`, `session_id` patterns
- Incrementing counters: numeric values at message boundaries
- Dynamic memory: content tagged with `[memory]` / ephemeral markers

### Tool Schema Canonicalization

```rust
fn canonicalize_tools(tools: &[ToolDef]) -> Vec<ToolDef>
```

- Sort tools by `name` alphabetically
- Within each tool, sort `properties` keys alphabetically
- Sort `required` and `enum` arrays
- Strip `description` fields marked with `@volatile` annotation
- Strip custom metadata fields (non-standard JSON schema keys)

### Modes

| Mode | Behavior |
|---|---|
| `audit` | Log stability report at `info!`, no request impact |
| `warn` | Log report at `warn!`, attach to response headers |
| `strict` | Return `ContextError::StablePrefixTooLarge` on volatility |

Default: `audit`. Configurable via `prefix_stability.mode`.

### Tests

- Only user message changes → `stable_prefix_hash` unchanged
- Only tool observation changes → `stable_prefix_hash` unchanged
- System prompt changes → `stable_prefix_hash` changes
- Tools in different input order → `stable_prefix_hash` unchanged
- System prompt with timestamp → strict mode fails

### Documentation

`docs/deepseek-v4-prefix-stability.md` covers:
- What constitutes the stable prefix
- What goes in the dynamic suffix
- KV cache reuse expectations
- Tool schema best practices

---

## DS4-19: ContextPack Importance Ordering

**File**: `src/context_pack.rs`

### Placement

Integrated into the pipeline's message assembly, before the final
`CanonicalRequest` is built.

### Partition Model

The assembled context is divided into ordered partitions. Partitions are
ordered by priority. Within each partition, existing order is preserved (causal
chains intact).

```
Priority  Content                    Behavior
─────────────────────────────────────────────────────
P0        latest user message        always last, never truncated
P1        current task state         preserved, truncated last
P2        recent causal tail         recent turns + tool calls + tool results
          (last N tool chains)       kept together, order preserved
P3        high-value evidence        from DAG tiering + query scoring
P4        historical summaries       from DAG summary chain
P5        old raw leaves             oldest messages, truncated first
```

### Causal Tail Preservation

The "recent causal tail" includes the last K assistant-tool-result cycles.
These are identified by scanning the message list for alternating
`assistant(tool_calls)` → `tool` pairs. This contiguous block is kept as a
single partition, never broken or reordered.

### Truncation Policy

When estimated token count exceeds the limit:
1. Drop old raw leaves (P5) first
2. Drop or condense historical summaries (P4)
3. Reduce evidence count (P3)
4. Prune old turns from causal tail (P2, preserving most recent N)
5. Current task (P1) and latest user (P0) are never truncated

### Integration with Existing DAG Tiering

```
   DAG assemble_context
        │
        ▼
   summary chain (L3 → L2 → L1)
   recent leaves
   query-scored results
        │
        ▼
   ContextPack::partition(messages, dag_nodes, token_budget)
        │
        ├── older history → DAG tiering (P3-P5)
        ├── recent causal tail → preserve order (P2)
        └── latest user → always last (P0)
        │
        ▼
   final ordered Vec<Message>
```

---

## DS4-32: Stream Robustness

**File**: `src/streaming.rs` additions (prepended normalizers)

### Approach

Add processing layers **before** the existing `StreamAssembler`. Do not refactor
`StreamAssembler` internals.

### PartialLineBuffer

```rust
struct PartialLineBuffer {
    buffer: String,
}

impl PartialLineBuffer {
    fn push_chunk(&mut self, chunk: &str) -> Vec<String>;
    // Returns complete data: lines assembled from chunks.
    // Partial lines stay in buffer until completed.
}
```

- Gathers incoming network chunks
- Detects SSE line boundaries (`\n`, `\r\n`)
- Returns only complete `data: ...` lines
- Incomplete last line stays buffered

### SseEventParser

```rust
struct SseEventParser;

impl SseEventParser {
    fn parse_line(line: &str) -> Result<Option<SseEvent>, StreamError>;
    // Returns None for non-data lines (event:, :comment, empty)
}

struct SseEvent {
    event_type: Option<String>,    // from "event: ..." line
    data: String,                  // from "data: ..." line
    is_done: bool,                 // "[DONE]"
}
```

- Unknown fields: log at `debug!`, continue
- `data: [DONE]`: mark as done
- Invalid JSON in data: return `StreamError::InvalidJsonEvent`
- Non-UTF-8 bytes: return `StreamError::InvalidSseLine`

### DeepSeekNormalizer

```rust
struct DeepSeekNormalizer;

impl DeepSeekNormalizer {
    fn normalize(event: SseEvent) -> Result<Vec<CanonicalEvent>, StreamError>;
}
```

- Parses `choices[0].delta` from Chat Completions SSE
- Separates `reasoning_content`, `content`, `tool_calls` into distinct
  canonical events
- Tolerates unknown fields in the JSON delta
- Detects DSML content and routes to DSML parser if enabled
- Routes to ThinkTagFallback if no structured reasoning

### Finish Reason Handling

```rust
enum FinishReason {
    Stop,
    Length,
    ToolCalls,
    ContentFilter,
    Unknown,
    DoneWithoutExplicitReason,
    StreamInterrupted,
}
```

- `finish_reason` from upstream is mapped directly (not fabricated)
- Missing `finish_reason + [DONE]`: `DoneWithoutExplicitReason`
- Missing `finish_reason + EOF no [DONE]`: `StreamInterrupted`
- Never fabricate `Stop` or `Length`

### Error / Cancellation Handling

- User cancellation: save partial raw stream + partial normalized events;
  emit `StreamInterrupted` instead of success `Done`
- Premature EOF: save partial state, emit `StreamInterrupted`
- Internal error (persistence, cache): emit error, do not erase partial stream

### Tests / Fixtures

- Partial line across chunks
- Multiple `data:` lines in one chunk
- Tool call arguments split across multiple chunks
- Unknown fields in upstream response
- Stream interrupted mid-response (EOF without `[DONE]`)
- Empty content with reasoning content
- `[DONE]` without `finish_reason`

---

## DS4-33: Structured Error Types

**File**: `src/model_error.rs`

### Design

Cross-cutting error enum. Any module can produce a `ModelError`. It is not a
linear layer in the pipeline.

```rust
#[derive(Debug, Clone, PartialEq)]
pub enum ModelError {
    // Upstream API errors
    Upstream(UpstreamError),

    // Protocol/parsing errors
    Protocol(ProtocolError),

    // SSE stream errors
    Stream(StreamError),

    // Reasoning/thinking errors
    Reasoning(ReasoningError),

    // Tool call errors
    ToolCall(ToolCallError),

    // Context errors
    Context(ContextError),

    // Routing errors
    Routing(RoutingError),

    // Internal runtime errors
    Runtime(RuntimeError),
}
```

### Sub-Enums

```rust
pub enum UpstreamError {
    RateLimited,
    AuthenticationFailed,
    PermissionDenied,
    ModelNotFound,
    QuotaExceeded,
    ServerError,
    Timeout,
    ConnectionFailed,
}

pub enum ProtocolError {
    InvalidJson,
    MissingChoices,
    MissingDelta,
    UnsupportedResponseShape,
    MixedProtocolShape,
}

pub enum StreamError {
    InvalidSseLine,
    InvalidJsonEvent,
    StreamInterrupted,
    DoneWithoutFinishReason,
    EventAfterDone,
}

pub enum ReasoningError {
    StructuredReasoningMalformed,
    ReasoningMixedIntoContent,
    ThinkTagUnclosed,
    ThinkTagInFinalAnswer,
    ReasoningStateMissingForToolChain,
}

pub enum ToolCallError {
    ToolCallsMalformed,
    DsmlMalformed,
    DsmlIncomplete,
    UnknownTool,
    MissingRequiredArgument,
    UnexpectedArgument,
    ArgumentTypeMismatch,
    ArgumentsJsonInvalid,
    ToolCallIdMissing,
    DuplicateToolCallId,
}

pub enum ContextError {
    ContextTooLong,
    StablePrefixTooLarge,
    DynamicSuffixTooLarge,
    EvidencePackTooLarge,
    ToolSchemaTooLarge,
}

pub enum RoutingError {
    ModelRouteUnavailable,
    InvalidReasoningEffort,
    ReasoningEffortNotSupported,
    FallbackExhausted,
}

pub enum RuntimeError {
    CacheReadFailed,
    CacheWriteFailed,
    PersistenceFailed,
    Cancellation,
}
```

### Metadata Per Error

```rust
struct ErrorMeta {
    severity: Severity,           // Critical | Error | Warning | Info
    recoverability: Recoverability, // Retryable | Degradable | Terminal
    retry_hint: Option<RetryHint>,
    phase: ErrorPhase,            // Receive | Parse | Route | Emit | Persist
}

enum Severity { Critical, Error, Warning, Info }
enum Recoverability { Retryable, Degradable, Terminal }

enum RetryHint {
    Backoff { base_ms: u64 },
    DowngradeModel,
    ReduceReasoningEffort,
    CompressContext,
    RequestModelFix,
}

enum ErrorPhase { Receive, Parse, Route, Emit, Persist }
```

### Error Mapping

```rust
fn classify_upstream_error(status: StatusCode, body: &str) -> ModelError;
fn classify_stream_error(sse_event: &SseEvent) -> ModelError;
fn classify_dsml_error(dsml_err: DsmlError) -> ModelError;
```

### Behavior Table

| Error | Severity | Recoverability | Action |
|---|---|---|---|
| RateLimited | Warning | Retryable | Backoff retry |
| AuthenticationFailed | Critical | Terminal | Don't retry |
| ContextTooLong | Error | Degradable | Compress + retry |
| DsmlMalformed | Error | Degradable | Don't execute, return schema error to model |
| StreamInterrupted | Error | Retryable | Retry with partial state |
| CacheWriteFailed | Warning | Degradable | Log, don't block main flow |

---

## DS4-40: DeepSeek Native API Compatibility

**Files**: `src/protocol/canonical.rs` (extensions), `src/main.rs` (CLI),
`src/protocol/mod.rs`

### Additions to ProviderCapabilities

```rust
struct ProviderCapabilities {
    // existing fields...
    pub deepseek_native: DeepSeekNativeCapabilities,
}

struct DeepSeekNativeCapabilities {
    pub reasoning_effort: bool,        // none / high / max
    pub dsml_parse: bool,              // parse DSML from upstream
    pub dsml_emit: bool,               // emit DSML to upstream
    pub quick_instruction: bool,       // experimental
}
```

### CLI Arguments

| Flag | Values | Default | Scope |
|---|---|---|---|
| `--reasoning-effort` | `none`, `high`, `max` | not set (upstream default) | Public |
| `--dsml-mode` | `off`, `parse`, `emit`, `roundtrip-test` | `parse` | Public |
| `--quick-instruction` | `on`, `off` | `off` | Experimental, hidden from --help |

### Body Override

The request body may include a `reasoning_effort` field at the top level.
If present, it overrides the CLI default for that request.

### Request Mapping

```json
// CanonicalRequest with reasoning_effort=high
{
  "model": "deepseek-v4-pro",
  "messages": [...],
  "reasoning_effort": "high"
}
```

When `dsml_mode=emit`, assistant tool calls in the request's conversation
history are serialized as DSML instead of JSON tool_calls. The existing
structured `tool_calls` field continues to work for non-DeepSeek endpoints.

### Default Path

All changes are default-off or default-pass-through:

- No `reasoning_effort` → upstream decides
- `dsml_mode=parse` (default) → parse DSML from upstream, never emit
- `quick_instruction=off` → no special token injection

Users who don't set any flag see no change.

---

## Integration Points

### Proxy: Response Path

```
upstream SSE → PartialLineBuffer → SseEventParser → DeepSeekNormalizer
  ├── structured tool_calls → StreamAssembler → protocol emitter
  ├── structured reasoning  → StreamAssembler → protocol emitter
  ├── no structured tool_calls → DSML parser → CanonicalToolCall
  ├── no structured reasoning → ThinkTagFallback → ReasoningDelta/TextDelta
  └── any error → ModelError → json_error response + event log
```

### Proxy: Request Path

```
incoming request → protocol adapter → canonical IR
  → ContextPack (importance ordering)
  → Tool Schema Canonicalization
  → PrefixStabilityChecker
  → DeepSeek native options injection
  → upstream API call
```

---

## Testing Strategy

| Item | Test approach |
|---|---|
| DS4-05 | Unit tests for ThinkTagParser with chunked inputs |
| DS4-08/09 | Unit tests for DsmlParser with well-formed/malformed DSML; roundtrip test |
| DS4-14/15 | Unit tests for PrefixStabilityChecker with volatile content detection; integration with ContextPack |
| DS4-19 | Unit tests for ContextPack partition ordering and truncation boundary |
| DS4-32 | Fixture-based tests for PartialLineBuffer, SseEventParser, DeepSeekNormalizer |
| DS4-33 | Unit tests for error classification and metadata mapping |
| DS4-40 | Integration test: request with reasoning_effort reaches upstream in correct field |

---

## Implementation Order

| Step | Item | Depends on |
|---|---|---|
| 1 | DS4-33: ModelError types | Nothing (standalone enum) |
| 2 | DS4-32: PartialLineBuffer + SseEventParser | Nothing (standalone normalizers) |
| 3 | DS4-05: ThinkTagFallback | DS4-32 (needs normalized events) |
| 4 | DS4-08/09: DSML parser | DS4-32 + DS4-33 (needs normalizer + error types) |
| 5 | DS4-14/15: PrefixStabilityChecker | DS4-19 (needs final CanonicalRequest) |
| 6 | DS4-19: ContextPack | Nothing (standalone partition logic) |
| 7 | DS4-40: DeepSeek native options | DS4-08/09 (DSML mode), DS4-19 (context assembly) |

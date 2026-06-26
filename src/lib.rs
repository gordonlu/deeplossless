#![allow(clippy::too_many_arguments)]
//! deeplossless — Lossless Context Management proxy for DeepSeek API.
//!
//! # Architecture Invariants
//!
//! ## DAG Direction (P0)
//!
//! A summary node stores its source (raw) node IDs in `child_ids`:
//!
//! ```text
//! summary.child_ids = [raw_1, raw_2, ...]
//! ```
//!
//! Conversely, each source node stores the summary ID in `parent_ids`:
//!
//! ```text
//! raw_n.parent_ids = [summary]
//! ```
//!
//! Edge direction: **raw → summary** (summarization flows from sources upward).
//! Nodes may have multiple parents via `Refines` and `Reuses` edges, forming a
//! true DAG. Cross-conversation sharing is enabled when embedding similarity
//! exceeds threshold, creating `Reuses` edges across conversation boundaries.
//!
//! - `get_children(node)` returns the nodes in `node.child_ids` (forward lookup)
//! - `get_parents(node)` returns nodes whose `child_ids` contains `node.id`
//!
//! ## Coverage Invariant (P0)
//!
//! A summary's coverage = the set of raw message IDs in its `child_ids`.
//! Context assembly (`assemble_context`) MUST NOT inject a raw leaf that
//! is already covered by any selected summary, preventing double injection.
//!
//! ## Level Invariant (P0)
//!
//! | Level | Name       | is_leaf | Content               |
//! |-------|------------|---------|-----------------------|
//! | 0     | Raw leaf   | true    | Original message text |
//! | 1     | L1 summary | false   | LLM preserve_details  |
//! | 2     | L2 summary | false   | LLM bullet_points     |
//! | 3+    | L3+ summary| false   | Deterministic/aggr.   |
//!
//! ## Snippet Authority (P1)
//!
//! Snippets extracted before compression are stored on the summary node
//! and take precedence over the summary text for precision-critical values
//! (code blocks, file paths, numeric constants, error messages).
//!
//! ## GC Reachability (P1)
//!
//! A node is reachable iff traversable from any tip or leaf via BFS along
//! BOTH `parent_ids` (raw → summary) and `child_ids` (summary → raw).
//! Unreachable nodes ("ghosts") are candidates for deletion.
//!
//! ## Assembly Invariant (P1)
//!
//! Context assembly output must stay within `token_budget`. No leaf
//! covered by a selected summary shall appear as a raw leaf.
//!
//! ## Transaction Invariant (P2)
//!
//! DAG mutations (insert node + back-link to sources) must be atomic.
//! A database transaction wraps both operations.

pub mod compactor;
pub mod dag;
pub mod db;
pub mod dependency_kind;
pub mod dependency_view;
pub mod diff_events;
pub mod embeddings;
pub mod artifacts;
pub mod assistant_validation;
pub mod audit;
pub mod event_store;
pub mod execution;
pub mod file_observation;
pub mod parallel;
pub mod provider;
pub mod runtime;
pub mod runtime_events;
pub mod runtime_invariants;
pub mod runtime_state_view;
pub mod tool_cache;
pub mod metrics;
pub mod motif;
pub mod mutation;
pub mod pipeline;
pub mod protocol;
pub mod proxy;
pub mod replay;
pub mod response_store;
pub mod session_store;
pub mod runtime_coordinator;
pub mod session;
pub mod context_pack;
pub mod model_error;
pub mod prefix_stability;
pub mod think_tag;
pub mod snapshot;
pub mod snippet;
pub mod summarizer;
pub mod tokenizer;
pub mod torture;

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::{Mutex, Notify};

/// Runtime execution services.
#[derive(Clone)]
pub struct RuntimeServices {
    pub client: reqwest::Client,
    pub cycle: Arc<StdMutex<runtime::ExecutionCycle>>,
    pub rate_limiter: Arc<runtime::RateLimiter>,
    /// Shutdown signal — notified when the runtime is stopping.
    /// Background tasks MUST select on this to avoid orphan mutations.
    pub shutdown_notify: Arc<Notify>,
}

/// Storage and persistence services.
#[derive(Clone)]
pub struct StorageServices {
    pub db: Arc<db::Database>,
    pub dag: Arc<dag::DagEngine>,
    pub response_store: response_store::ResponseStore,
    pub session_store: session_store::SessionStore,
}

/// Shared application state — split along service boundaries.
#[derive(Clone)]
pub struct AppState {
    // ── Upstream / Network ───────────────────────────────────────────
    pub upstream: String,
    /// API key extracted from the first incoming request's Authorization header.
    pub api_key: Arc<StdMutex<Option<String>>>,
    /// Separate admin key for LCM endpoint authentication.
    pub admin_key: Arc<StdMutex<Option<String>>>,

    /// Cache stability tracker — records system prompt hashes to compute
    /// prompt cache stability metrics.  Keyed by conversation ID.
    pub cache_stability: Arc<StdMutex<std::collections::HashMap<i64, Vec<String>>>>,
    /// Reasoning content cache — stores `reasoning_content` from DeepSeek responses
    /// keyed by fingerprint, so it can be injected into the next turn's request.
    /// Required by DeepSeek thinking mode: reasoning_content must be passed back.
    pub reasoning_cache: Arc<StdMutex<std::collections::HashMap<String, String>>>,

    // ── Storage ──────────────────────────────────────────────────────
    pub storage: StorageServices,

    // ── Execution ────────────────────────────────────────────────────
    pub compactor: Arc<Mutex<compactor::Compactor>>,
    pub runtime: RuntimeServices,

    // ── Config ───────────────────────────────────────────────────────
    pub summarizer_model: String,
    pub dry_run: bool,
    pub log_dir: Option<String>,
    pub record: Option<String>,
    pub passthrough: bool,
    pub no_pipeline: bool,
    pub no_header_mod: bool,
    pub lcm_context: bool,
    /// Normalize system prompts for cache-friendliness. Opt-in via
    /// `--cache-normalize`. Replaces timestamps/UUIDs with stable markers.
    pub cache_normalize: bool,
    /// LCM context injection default budget in tokens (0 = off).
    pub lcm_context_tokens: u64,

    /// Workspace root for stable conversation identity.
    pub workspace: Option<String>,
}

impl AppState {
    pub fn summarizer_model(&self) -> &str {
        &self.summarizer_model
    }
}

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

#![allow(clippy::too_many_arguments)]

pub mod compactor;
pub mod dag;
pub mod db;
pub mod pipeline;
pub mod proxy;
pub mod session;
pub mod snippet;
pub mod summarizer;
pub mod tokenizer;

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub upstream: String,
    /// API key extracted from the first incoming request's Authorization header.
    /// `None` until the first request arrives. The compactor/spawner reads this
    /// for background summarization calls.
    pub api_key: Arc<StdMutex<Option<String>>>,
    pub db: Arc<db::Database>,
    pub dag: Arc<dag::DagEngine>,
    pub compactor: Arc<Mutex<compactor::Compactor>>,
    pub client: reqwest::Client,
}

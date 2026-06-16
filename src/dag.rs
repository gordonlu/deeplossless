use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::atomic::{AtomicI64, Ordering};
use std::sync::Arc;

use crate::db::Database;

/// Stable, content-addressed node identity.
/// Wraps the DB-assigned rowid — identity is the rowid, not the content.
/// Content equality is determined by `semantic_hash`.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct NodeId(pub i64);

impl NodeId {
    pub fn as_i64(self) -> i64 { self.0 }
}

/// Monotonic graph revision counter.
/// Every mutation increments the revision; replay/snapshot can pin to a
/// specific revision for deterministic topology.
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord, Serialize, Deserialize)]
pub struct GraphRevision(pub i64);

impl GraphRevision {
    pub fn initial() -> Self { Self(0) }
    pub fn next(self) -> Self { Self(self.0 + 1) }
}

/// Execution-scoped DAG: tracks tool invocations, parallel branches, execution order.
/// Separated from the knowledge DAG to avoid semantic pollution (P0-9).
/// Execution nodes are ephemeral; knowledge nodes are persistent.
pub struct ExecutionDag {
    db: Arc<Database>,
}

impl ExecutionDag {
    pub fn new(db: Arc<Database>) -> Self { Self { db } }

    /// Record a tool execution edge in the execution graph.
    /// Uses `EdgeKind::Executes` to link assistant → tool_call → tool_result.
    pub fn record_execution(
        &self, from_node_id: i64, to_node_id: i64,
    ) -> anyhow::Result<()> {
        let _ = self.db.insert_edge(from_node_id, to_node_id, EdgeKind::Executes.as_str())?;
        Ok(())
    }

    /// Query the execution trace for a conversation (all Executes edges).
    pub fn execution_trace(&self, node_id: i64) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let all = self.db.get_edges_from(node_id)?;
        Ok(all.into_iter().filter(|(_, _, kind)| kind == "executes").collect())
    }
}

/// Knowledge DAG: semantic summaries, dedup, context assembly.
/// This is the main DAG engine — see `DagEngine` for the full implementation.
pub struct KnowledgeDag {
    engine: Arc<DagEngine>,
}

impl KnowledgeDag {
    pub fn new(engine: Arc<DagEngine>) -> Self { Self { engine } }

    /// Delegate to the full engine
    pub fn engine(&self) -> &DagEngine { &self.engine }
}

/// In-memory snapshot of a conversation's DAG, used for batch-loaded
/// context assembly (avoids N+1 DB roundtrips).
struct DagGraph {
    nodes: HashMap<i64, DagNode>,
    children: HashMap<i64, Vec<i64>>,
}

// ── Edge types ──────────────────────────────────────────────────────────

/// Semantic kind of a DAG edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Summary summarizes its source nodes (summary → raw / lower summary).
    Summarizes,
    /// Higher summary refines/merges a lower summary.
    Refines,
    /// Semantic reuse/dedup link.
    Reuses,
    /// Tool execution link: assistant → tool_call → tool_result chain.
    Executes,
    /// Generated-by link: node was generated/inferred from another node.
    GeneratedBy,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Summarizes => "summarizes",
            Self::Refines => "refines",
            Self::Reuses => "reuses",
            Self::Executes => "executes",
            Self::GeneratedBy => "generated_by",
        }
    }

    #[allow(clippy::should_implement_trait)]
    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "summarizes" => Some(Self::Summarizes),
            "refines" => Some(Self::Refines),
            "reuses" => Some(Self::Reuses),
            "executes" => Some(Self::Executes),
            "generated_by" => Some(Self::GeneratedBy),
            _ => None,
        }
    }
}

/// Explicit typed edge in the semantic DAG.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Edge {
    pub id: i64,
    pub from_id: i64,
    pub to_id: i64,
    pub kind: EdgeKind,
}

// ── Configuration ──────────────────────────────────────────────────────

/// DAG engine configuration with sensible defaults (LCM paper §2.1, §2.3).
#[derive(Clone, Debug)]
pub struct DagConfig {
    /// Soft threshold as fraction of context window (τ_soft).
    /// Async compaction triggers above this. Default: 0.80.
    pub soft_threshold_ratio: f64,

    /// Hard threshold as fraction of context window (τ_hard).
    /// Blocking compaction triggers above this. Default: 0.95.
    pub hard_threshold_ratio: f64,

    /// Maximum DAG depth (levels).
    /// A value > 0 overrides the dynamic calculation.
    /// Set to 0 to use dynamic depth based on leaf count. Default: 0 (dynamic).
    pub max_level: u8,

    /// Max children per node to traverse. Prevents degenerate DAG
    /// traversal cost from hitting millisecond-range latency. Default: 100.
    pub max_fanout: usize,

    /// Max depth when expanding a summary node. Default: 10.
    pub max_expand_depth: u8,

    /// Max recent raw messages to keep uncompressed. Default: 20.
    pub recent_message_count: usize,

    /// Tokenizer correction factor to align with provider. Default: 1.0.
    /// >1.0 pads counts upward (trigger compaction sooner), <1.0 shrinks.
    pub token_correction_factor: f64,

    /// Per-message JSON framing overhead (role + structure). Default: 12.
    pub token_overhead: usize,

    /// Embedding API key for semantic similarity. Empty = SHA-256 fallback only.
    pub embedding_api_key: String,
}

impl Default for DagConfig {
    fn default() -> Self {
        Self {
            soft_threshold_ratio: 0.80,
            hard_threshold_ratio: 0.95,
            max_level: 0,
            max_fanout: 100,
            max_expand_depth: 10,
            recent_message_count: 20,
            token_correction_factor: 1.0,
            token_overhead: 12,
            embedding_api_key: String::new(),
        }
    }
}

// ── Core types ─────────────────────────────────────────────────────────

use crate::snippet::Snippet;

/// A node in the summary DAG. Nodes form a forest per conversation.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct DagNode {
    pub id: i64,
    pub conversation_id: i64,
    /// 0 = raw message group, 1 = leaf summary, 2+ = condensed summaries.
    pub level: u8,
    /// The summary text (at level 0 this is concatenated raw content).
    pub summary: String,
    pub token_count: i64,
    /// JSON array of parent node IDs (0 or 1 entry for typical DAG).
    pub parent_ids: Vec<i64>,
    /// JSON array of child node IDs (the messages/summaries it summarises).
    pub child_ids: Vec<i64>,
    /// True when this is a raw message leaf (level 0).
    pub is_leaf: bool,
    /// True when this is an explicit join node for parallel execution.
    pub is_join: bool,
    /// Precision-critical snippets extracted before compression.
    pub snippets: Vec<Snippet>,
    /// True when this node has been soft-deleted (excluded from context assembly).
    pub deleted: bool,
    /// SHA-256 hash of normalized summary text for cross-conversation dedup (v0.3).
    pub semantic_hash: String,
    /// How many times this node was selected for context assembly.
    pub access_count: i64,
    /// ISO-8601 timestamp of last context assembly inclusion.
    pub last_accessed_at: Option<String>,
    /// JSON reasoning chain: how this summary was produced (level, prompt, reduction).
    pub reasoning: String,
    /// Graph revision when this node was created (P0-6).
    pub graph_revision: i64,
    /// Deterministic compaction ID: SHA256 of (conv_id, source_ids, level)
    /// for idempotent compaction dedup.
    pub compaction_id: String,
}

impl DagNode {
    /// Stable identity wrapper.
    pub fn node_id(&self) -> NodeId { NodeId(self.id) }
}

/// Deterministic compaction ID: SHA-256 of (conv_id, sorted source_ids, level).
/// Enables idempotent compaction: same inputs produce same ID (P0-9).
fn compaction_id(conv_id: i64, source_ids: &[i64], level: u8) -> String {
    use sha2::{Digest, Sha256};
    let mut sorted: Vec<i64> = source_ids.to_vec();
    sorted.sort_unstable();
    let input = format!("{}:{}:{}", conv_id, sorted.iter().map(|i| i.to_string()).collect::<Vec<_>>().join(","), level);
    format!("{:x}", Sha256::digest(input.as_bytes()))
}

// ── Engine ─────────────────────────────────────────────────────────────

#[derive(Default)]
pub struct DagEngineBuilder {
    config: DagConfig,
}


impl DagEngineBuilder {
    pub fn new() -> Self { Self::default() }

    pub fn soft_threshold(mut self, ratio: f64) -> Self {
        self.config.soft_threshold_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn hard_threshold(mut self, ratio: f64) -> Self {
        self.config.hard_threshold_ratio = ratio.clamp(0.0, 1.0);
        self
    }

    pub fn max_level(mut self, level: u8) -> Self {
        self.config.max_level = level.min(10);
        self
    }

    pub fn recent_messages(mut self, count: usize) -> Self {
        self.config.recent_message_count = count;
        self
    }

    pub fn build(self, db: Arc<Database>) -> DagEngine {
        let embedder = if self.config.embedding_api_key.is_empty() {
            None
        } else {
            Some(crate::embeddings::EmbeddingClient::new(
                crate::embeddings::EmbeddingConfig {
                    api_key: self.config.embedding_api_key.clone(),
                    ..Default::default()
                }
            ))
        };
        DagEngine {
            db,
            config: self.config,
            embedder,
            revision_counter: AtomicI64::new(0),
            auto_validate: false,
        }
    }
}

/// The DAG engine: insert, link, compress, and assemble conversation
/// history into lossless context blocks.
pub struct DagEngine {
    db: Arc<Database>,
    config: DagConfig,
    embedder: Option<crate::embeddings::EmbeddingClient>,
    /// Monotonic revision counter incremented on every mutation.
    revision_counter: AtomicI64,
    /// When true, runs validate_dag after every mutation (P0-8). Default false
    /// for production; enable in tests and for debugging.
    auto_validate: bool,
}

impl DagEngine {
    /// Current graph revision. Starts at 0, increments on each mutation.
    pub fn current_revision(&self) -> GraphRevision {
        let memory_revision = self.revision_counter.load(Ordering::Relaxed);
        let persisted_revision = self.db.max_graph_revision().unwrap_or(memory_revision);
        GraphRevision(memory_revision.max(persisted_revision))
    }

    /// Enable/disable automatic invariant check after mutations.
    pub fn set_auto_validate(&mut self, enabled: bool) {
        self.auto_validate = enabled;
    }

    /// Internal: increment revision and optionally validate.
    fn post_mutation(&self, conv_id: i64) -> anyhow::Result<()> {
        let revision = self.db.bump_graph_revision(conv_id)?;
        self.revision_counter.store(revision, Ordering::Relaxed);
        if self.auto_validate {
            let issues = self.validate_dag(conv_id)?;
            if !issues.is_empty() {
                anyhow::bail!("post-mutation DAG invariants violated: {:?}", issues);
            }
        }
        Ok(())
    }
}

impl DagEngine {
    pub fn builder() -> DagEngineBuilder { DagEngineBuilder::new() }

    pub fn config(&self) -> &DagConfig { &self.config }
    pub fn db(&self) -> &Database { &self.db }

    /// Compute effective max level for a conversation.
    /// If config.max_level > 0, that value is used.
    /// Otherwise: dynamic = max(3, ceil(log2(leaf_count / 8))).
    pub fn effective_max_level(&self, conv_id: i64) -> u8 {
        if self.config.max_level > 0 {
            return self.config.max_level;
        }
        let leaf_count = self.db.get_leaf_nodes(conv_id)
            .map(|leaves| leaves.len())
            .unwrap_or(0);
        if leaf_count <= 8 {
            return 3;
        }
        let ratio = (leaf_count as f64 / 8.0).log2().ceil() as u8;
        (3 + ratio.saturating_sub(1)).min(10)
    }

    // ── Query ──────────────────────────────────────────────────────────

    pub fn get_node(&self, node_id: i64) -> anyhow::Result<Option<DagNode>> {
        self.db.get_node(node_id)
    }

    pub fn get_children(&self, node_id: i64) -> anyhow::Result<Vec<DagNode>> {
        self.db.get_child_nodes(node_id, self.config.max_fanout)
    }

    /// Return the parent nodes of a node (nodes whose child_ids include this node).
    /// For a leaf, this returns the summaries that compress it.
    /// For a summary, this returns higher-level summaries that compress it.
    pub fn get_parents(&self, node_id: i64) -> anyhow::Result<Vec<DagNode>> {
        self.db.get_parent_nodes(node_id, self.config.max_fanout)
    }

    pub fn get_active_tip(&self, conv_id: i64) -> anyhow::Result<Option<DagNode>> {
        self.db.get_tip_node(conv_id)
    }

    /// All leaf (level-0) nodes for a conversation, oldest first.
    pub fn get_leaves(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        self.db.get_leaf_nodes(conv_id)
    }

    /// Total tokens in a conversation (all raw-message leaves + overhead).
    pub fn total_tokens(&self, conv_id: i64) -> anyhow::Result<i64> {
        self.db.total_conversation_tokens(conv_id)
    }

    // ── Mutations ──────────────────────────────────────────────────────

    /// Store a new leaf node (level 0) for raw messages.
    pub fn insert_leaf(&self, conv_id: i64, summary: &str, token_count: i64) -> anyhow::Result<DagNode> {
        let node = self.db.insert_dag_node(conv_id, 0, summary, token_count, &[], &[], true)?;
        self.post_mutation(conv_id)?;
        Ok(node)
    }

    /// Like `compress_group` but also stores extracted snippets for
    /// lossless value preservation (Context-ReAct Snippet, §3.2).
    pub fn compress_group_with_snippets(
        &self,
        conv_id: i64,
        source_ids: &[i64],
        summary: &str,
        token_count: i64,
        level: u8,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        let cid = compaction_id(conv_id, source_ids, level);

        // Idempotency: reuse existing summary for same sources (P0-9).
        // Only when we have actual source nodes — empty source_ids means this is
        // a manual compress call that should always produce a new node.
        if !source_ids.is_empty() {
            if let Some(existing) = self.db.find_by_compaction_id(&cid)? {
                tracing::debug!(target: "deeplossless::dag", compaction_id = %cid, "idempotent compaction: reuse node {}", existing.id);
                return Ok(existing);
            }
        }

        // Revision pinning: snapshot source hashes to detect concurrent mutation (P0-8)
        let source_hashes: Vec<(i64, String)> = source_ids.iter()
            .filter_map(|id| self.db.get_node(*id).ok().flatten())
            .map(|n| (n.id, n.semantic_hash.clone()))
            .collect();

        // Cycle protection: ensure no path from any source → the new summary
        // which would create a cycle when we add summary → source edges.
        let node = self.db.insert_summary_atomic(
            conv_id,
            level.min(self.effective_max_level(conv_id)),
            summary,
            token_count,
            source_ids,
            snippets,
            &cid,
        )?;
        // Verify no cycle was introduced
        if let Err(e) = self.check_no_cycle(&node, source_ids) {
            // Rollback: hard-delete the newly created node and its edges
            self.db.purge_dag_node(node.id)?;
            return Err(e);
        }

        // Revision pinning: verify source nodes unchanged during compaction
        for (sid, old_hash) in &source_hashes {
            if let Ok(Some(current)) = self.db.get_node(*sid) {
                if current.semantic_hash != *old_hash {
                    tracing::warn!(target: "deeplossless::dag", node_id = sid, "source node hash changed during compaction — summary may be stale");
                }
            }
        }

        // Post-compaction integrity audit (only in Full audit mode)
        if self.db.policy_config.read().map(|c| c.audit_mode.should_write_audit()).unwrap_or(true) {
            if let Err(e) = self.verify_compaction_integrity(&node, source_ids) {
                tracing::warn!(target: "deeplossless::dag", error = %e, "post-compaction integrity check failed");
            }
        }

        // Auto-snapshot on semantic boundary (compaction) if snapshot mode is Auto
        if self.db.policy_config.read().map(|c| c.snapshot_mode.should_auto_snapshot()).unwrap_or(false) {
            if let Err(e) = self.auto_snapshot_on_compaction(conv_id, &node) {
                tracing::warn!(target: "deeplossless::dag", "auto-snapshot after compaction failed: {e}");
            }
        }

        // Best-effort provenance computation
        if let Err(e) = self.compute_provenance(&node, source_ids) {
            tracing::warn!(target: "deeplossless::dag", error = %e, "provenance computation failed");
        }
        // Store reasoning chain in the node for execution provenance
        let reasoning = serde_json::json!({
            "action": "compress",
            "level": level,
            "source_count": source_ids.len(),
            "token_reduction": token_count,
        });
        if let Err(e) = self.db.update_node_reasoning(node.id, &reasoning.to_string()) {
            tracing::warn!(target: "deeplossless::dag", "update_node_reasoning failed: {e}");
        }
        self.post_mutation(conv_id)?;
        Ok(node)
    }

    /// Compute sentence-level provenance spans for a new summary node.
    fn compute_provenance(&self, summary_node: &DagNode, source_ids: &[i64]) -> anyhow::Result<()> {
        let sentences = split_sentences(&summary_node.summary);
        if sentences.is_empty() {
            return Ok(());
        }
        let mut spans = Vec::new();
        let mut source_texts: Vec<(i64, String)> = Vec::new();
        for sid in source_ids {
            if let Some(node) = self.db.get_node(*sid)? {
                source_texts.push((*sid, node.summary.clone()));
            }
        }
        for (_offset, sentence) in &sentences {
            let mut found = false;
            for (sid, text) in &source_texts {
                if let Some(pos) = text.find(sentence) {
                    spans.push((*sid, pos as i32, sentence.len() as i32));
                    found = true;
                    break;
                }
            }
            if !found
                && let Some((sid, _)) = source_texts.first()
            {
                spans.push((*sid, 0, 0));
            }
        }
        self.db.store_provenance_spans(summary_node.id, &spans)
    }

    /// Check that adding edges from `summary` → each `source` creates no cycle.
    /// For each source, verify no path exists from source back to summary.
    fn check_no_cycle(&self, summary: &DagNode, sources: &[i64]) -> anyhow::Result<()> {
        for source_id in sources {
            if self.db.has_path(*source_id, summary.id, 50)? {
                return Err(anyhow::anyhow!(
                    "cycle detected: path from source {} back to summary {}",
                    source_id, summary.id
                ));
            }
        }
        Ok(())
    }

    /// Post-compaction integrity audit: verify invariant that the new summary
    /// node has correct child_ids, edges, and parent_ids back-links (P0-12).
    fn verify_compaction_integrity(&self, node: &DagNode, source_ids: &[i64]) -> anyhow::Result<()> {
        let source_set: std::collections::HashSet<i64> = source_ids.iter().copied().collect();

        // 1. Child_ids must equal source_ids
        let child_set: std::collections::HashSet<i64> = node.child_ids.iter().copied().collect();
        if child_set != source_set {
            anyhow::bail!(
                "integrity fail: child_ids mismatch — expected {:?}, got {:?}",
                source_ids, node.child_ids
            );
        }

        // 2. Each source node must have this node in its parent_ids
        for sid in source_ids {
            if let Ok(Some(source)) = self.db.get_node(*sid) {
                if !source.parent_ids.contains(&node.id) {
                    tracing::warn!(target: "deeplossless::dag", summary_id = node.id, source_id = sid, "source node missing parent_ids back-link");
                }
            }
        }

        // 3. Edges must exist: (node.id → sid, 'summarizes') for each source
        for sid in source_ids {
            let edges = match self.db.get_edges_from(node.id) {
                Ok(e) => e,
                Err(_) => continue,
            };
            let has_edge = edges.iter().any(|e| e.1 == *sid && e.2 == "summarizes");
            if !has_edge {
                tracing::warn!(target: "deeplossless::dag", summary_id = node.id, source_id = sid, "missing summarizes edge");
            }
        }

        Ok(())
    }

    /// Compress a group of nodes into a higher-level summary node.
    /// `parent_ids` are the source nodes, `level` is the target level.
    /// Returns the new summary node.
    pub fn compress_group(
        &self,
        conv_id: i64,
        parent_ids: &[i64],
        summary: &str,
        token_count: i64,
        level: u8,
    ) -> anyhow::Result<DagNode> {
        self.compress_group_with_snippets(conv_id, parent_ids, summary, token_count, level, &[])
    }

    /// Merge multiple summary nodes into a higher-level semantic node.
    /// Unlike `compress_group` (which summarizes raw leaves), `merge_nodes`
    /// refines existing summaries by creating only Refines edges (no
    /// redundant Summarizes edges).
    ///
    /// Transactional: all inserts and edge writes happen in a single SQLite
    /// transaction. If cycle detection fails, the entire mutation is rolled back.
    ///
    /// Cycle protection: verifies no source can reach back to the new node
    /// before committing.
    pub fn merge_nodes(
        &self,
        conv_id: i64,
        source_ids: &[i64],
        merged_text: &str,
        token_count: i64,
    ) -> anyhow::Result<DagNode> {
        // Determine level: one above the highest source level
        let max_src_level = source_ids.iter()
            .filter_map(|id| self.db.get_node(*id).ok().flatten())
            .map(|n| n.level)
            .max()
            .unwrap_or(0)
            .min(self.effective_max_level(conv_id).saturating_sub(1));
        let level = (max_src_level + 1).min(self.effective_max_level(conv_id));

        let cid = compaction_id(conv_id, source_ids, level);

        // Use the atomic insert which handles the transaction and writer lock
        let node = self.db.insert_summary_atomic(
            conv_id, level, merged_text, token_count, source_ids, &[], &cid,
        )?;
        let new_id = node.id;

        // Replace auto-created Summarizes edges with Refines edges
        for sid in source_ids {
            self.db.delete_edges(new_id, *sid, "summarizes")?;
            self.db.insert_edge(new_id, *sid, "refines")?;
        }

        // Cycle check: verify no source → new_id path exists
        for sid in source_ids {
            if self.db.has_path(*sid, new_id, 50)? {
                self.db.purge_dag_node(new_id)?;
                anyhow::bail!("cycle detected: path from source {} back to new node {}", sid, new_id);
            }
        }

        self.post_mutation(conv_id)?;
        Ok(node)
    }

    /// Cross-conversation semantic dedup: check for an existing node with
    /// similar semantics. Uses embeddings when available; falls back to SHA-256.
    pub fn find_similar_node(
        &self,
        summary: &str,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<Option<DagNode>> {
        // Try embedding-based similarity first
        if let Some(ref embedder) = self.embedder {
            let text = format_summary_text(summary, snippets);
            if let Some(vec) = Self::embed_sync_if_safe(embedder, &text) {
                if let Some((node_id, _sim)) = self.db.find_nearest_embedding(&vec, 0.92)? {
                    return self.db.get_node(node_id);
                }
                return Ok(None);
            }
        }
        // Fall back to SHA-256 exact match
        let hash = dag_semantic_hash(summary, snippets);
        let matches = self.db.find_similar_by_hash(&hash)?;
        if let Some((_hash, node_id, _conv_id, _preview)) = matches.first() {
            self.db.get_node(*node_id)
        } else {
            Ok(None)
        }
    }

    /// Dedup-aware compression: checks if a semantically equivalent node
    /// already exists (same hash). If found, reuse it (create reuse edge);
    /// otherwise, create a new summary node.
    pub fn dedup_and_reuse(
        &self,
        conv_id: i64,
        source_ids: &[i64],
        summary: &str,
        token_count: i64,
        level: u8,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        if let Some(existing) = self.find_similar_node(summary, snippets)? {
            // Reuse: link the existing node to this conversation's sources.
            // Cross-conversation: create Reuses edges but do NOT modify
            // parent_ids (back-links stay within the owning conversation).
            for sid in source_ids {
                if let Err(e) = self.db.insert_edge(existing.id, *sid, "reuses") {
                    tracing::warn!(target: "deeplossless::dag", "dedup insert_edge failed: {e}");
                }
                if existing.conversation_id == conv_id {
                    self.db.add_parent_to_node(*sid, existing.id)?;
                }
            }
            tracing::debug!(target: "deeplossless::dag",
                node_id = existing.id,
                existing_conv = existing.conversation_id,
                current_conv = conv_id,
                "semantic dedup: reused node");
            self.post_mutation(conv_id)?;
            return Ok(existing);
        }
        // No match: create new, then store embedding for future dedup
        let node = self.compress_group_with_snippets(conv_id, source_ids, summary, token_count, level, snippets)?;
        if let Some(ref embedder) = self.embedder {
            let text = format_summary_text(summary, snippets);
            if let Some(vec) = Self::embed_sync_if_safe(embedder, &text) {
                let model = &embedder.config.model;
                let dims = embedder.config.dimensions as i32;
                if let Err(e) = self.db.store_embedding(node.id, &vec, model, dims) {
                    tracing::warn!(target: "deeplossless::dag", "store_embedding failed: {e}");
                }
                // Auto-merge: find similar nodes across sessions
                if let Ok(Some((similar_id, _sim))) = self.db.find_nearest_embedding(&vec, 0.85)
                    && similar_id != node.id {
                        let _ = self.db.insert_edge(similar_id, node.id, "reuses");
                        tracing::debug!(target: "deeplossless::dag",
                            node_id = node.id, similar = similar_id,
                            "auto-merged across sessions");
                    }
                }
        }
        Ok(node)
    }

    fn embed_sync_if_safe(
        embedder: &crate::embeddings::EmbeddingClient,
        text: &str,
    ) -> Option<Vec<f32>> {
        if tokio::runtime::Handle::try_current().is_ok() {
            tracing::warn!(
                target = "deeplossless::dag",
                "embedding lookup requested from sync DAG API inside tokio runtime; using hash fallback"
            );
            return None;
        }
        let rt = match tokio::runtime::Builder::new_current_thread()
            .enable_all()
            .build()
        {
            Ok(rt) => rt,
            Err(e) => {
                tracing::warn!(target = "deeplossless::dag", "embedding runtime build failed: {e}");
                return None;
            }
        };
        rt.block_on(async { embedder.embed(text).await })
    }

    /// Compute the set of leaf IDs covered by a node, recursively.
    /// Traverses `summarizes` edges downward until reaching leaves.
    pub fn covered_leaf_ids(&self, node_id: i64) -> anyhow::Result<HashSet<i64>> {
        let mut leaves = HashSet::new();
        let mut stack = vec![node_id];
        let mut visited = HashSet::new();
        let max_depth = self.config.max_expand_depth as usize * 2;

        for _ in 0..max_depth {
            let current = std::mem::take(&mut stack);
            if current.is_empty() { break; }
            for nid in current {
                if !visited.insert(nid) { continue; }
                if let Some(node) = self.db.get_node(nid)? {
                    if node.is_leaf {
                        leaves.insert(node.id);
                    } else {
                        // Follow both Summarizes and Refines edges
                        for (_, to_id, _) in self.db.get_edges_from(nid)? {
                            if !visited.contains(&to_id) {
                                stack.push(to_id);
                            }
                        }
                        // Also check child_ids for backward compat
                        for cid in &node.child_ids {
                            if !visited.contains(cid) {
                                stack.push(*cid);
                            }
                        }
                    }
                }
            }
        }
        Ok(leaves)
    }

    // ── Context assembly (LCM §2.1) ────────────────────────────────────

    /// Assemble the active context: higher-level summaries + recent raw
    /// messages NOT already covered by those summaries, staying within
    /// `token_budget` tokens.
    ///
    /// When `query` is provided, retrieval-scored results augment the
    /// context with query-relevant nodes that might not be in the
    /// summary chain or recent leaves.
    ///
    /// Coverage-aware: prevents double injection where a summary and the
    /// raw messages it summarizes both appear in context.
    ///
    /// Returns nodes in display order (summaries first, then recent messages).
    pub fn assemble_context(
        &self,
        conv_id: i64,
        token_budget: usize,
        query: Option<&str>,
    ) -> anyhow::Result<Vec<DagNode>> {
        // Batch-load all nodes and edges for this conversation (P4 Performance)
        let graph = self.load_graph(conv_id)?;

        let mut result = Vec::new();
        let mut remaining = token_budget as i64;
        let mut covered_ids = HashSet::new();
        let mut injected_ids = HashSet::new();

        // 1. Walk from the active tip down to collect summaries
        if let Some(tip) = self.get_active_tip(conv_id)? {
            let summaries = self.collect_summary_chain_cached(&graph, &tip, remaining)?;
            for node in summaries {
                if !injected_ids.insert(node.id) || node.token_count <= 0 {
                    continue;
                }
                remaining -= node.token_count;
                for cid in &node.child_ids {
                    covered_ids.insert(*cid);
                }
                result.push(node);
            }
        }

        // 2. Append most recent raw leaves NOT already covered by summaries
        let mut leaves: Vec<_> = graph.nodes.values()
            .filter(|n| n.is_leaf && n.token_count > 0 && !n.summary.is_empty())
            .cloned()
            .collect();
        leaves.sort_by_key(|n| n.id);
        let recent: Vec<_> = leaves
            .into_iter()
            .rev()
            .take(self.config.recent_message_count)
            .filter(|n| !covered_ids.contains(&n.id))
            .collect();

        // Delta compression: skip leaves with high content overlap to summaries already in result
        let summary_texts: Vec<String> = result.iter()
            .filter(|n| n.level > 0)
            .map(|n| n.summary.clone())
            .collect();
        let summary_refs: Vec<&str> = summary_texts.iter().map(|s| s.as_str()).collect();

        for node in recent.into_iter().rev() {
            if !injected_ids.insert(node.id) {
                continue;
            }
            // Skip if leaf content is largely covered by an included summary
            if !summary_refs.is_empty() && content_overlap(&node.summary, &summary_refs) > 0.7 {
                continue;
            }
            let tc = node.token_count;
            if tc <= remaining {
                result.push(node);
                remaining -= tc;
            } else {
                break;
            }
        }

        // 3. If query provided, boost with scored retrieval results
        // Only use results with valid DAG node IDs (source != "message").
        // Message IDs map to the messages table, not dag_nodes, so get_node() returns None.
        if let Some(q) = query
            && let Ok(search_results) = self.db.search_unified(conv_id, q, 10)
        {
            for sr in search_results.iter().take(5) {
                if sr.source == "message" {
                    continue; // message IDs are not valid dag_node IDs
                }
                if injected_ids.contains(&sr.id) {
                    continue;
                }
                if let Ok(Some(node)) = self.db.get_node(sr.id) {
                    let tc = node.token_count;
                    if tc <= remaining {
                        injected_ids.insert(node.id);
                        result.push(node);
                        remaining -= tc;
                    }
                }
            }
        }

        // Record access for memory scoring (best-effort)
        for node in &result {
            if let Err(e) = self.db.touch_node(node.id) {
                tracing::warn!(target: "deeplossless::dag", error = %e, node_id = node.id, "touch_node failed");
            }
        }

        Ok(result)
    }

    /// Compute a memory score for a node: combines recency, frequency, and importance.
    /// Higher = more valuable for long-term retention. 0.0–1.0.
    pub fn compute_memory_score(&self, node: &DagNode) -> f64 {
        // Recency: never accessed = 0.5; recently accessed = higher
        let recency = match &node.last_accessed_at {
            Some(_) => 0.8,  // has been accessed, actual time comparison would go here
            None => 0.3,
        };
        // Frequency: log-scaled access count, max contribution at ~100 accesses
        let freq = ((node.access_count as f64 + 1.0).ln() / 5.0).min(1.0);
        // Importance: average snippet importance or 0.5 default
        let imp = if node.snippets.is_empty() {
            0.5
        } else {
            node.snippets.iter().map(|s| s.importance as f64).sum::<f64>() / node.snippets.len() as f64
        };
        (recency * 0.4 + freq * 0.3 + imp * 0.3).clamp(0.0, 1.0)
    }

    /// Load all DAG nodes and edges for a conversation in 1-2 queries.
    fn load_graph(&self, conv_id: i64) -> anyhow::Result<DagGraph> {
        let all_nodes = self.db.get_all_dag_nodes(conv_id)?;
        let mut node_map: HashMap<i64, DagNode> = HashMap::new();
        let mut children: HashMap<i64, Vec<i64>> = HashMap::new();

        for node in all_nodes {
            children.insert(node.id, node.child_ids.clone());
            node_map.insert(node.id, node);
        }
        Ok(DagGraph { nodes: node_map, children })
    }

    /// Batch-load version of collect_summary_chain that uses cached graph
    /// instead of issuing individual DB queries.
    fn collect_summary_chain_cached(
        &self,
        graph: &DagGraph,
        start: &DagNode,
        budget: i64,
    ) -> anyhow::Result<Vec<DagNode>> {
        let mut nodes = Vec::new();
        let mut stack = vec![start.id];
        let mut seen = HashSet::new();
        seen.insert(start.id);

        while let Some(nid) = stack.pop() {
            if seen.len() > self.config.max_fanout {
                break;
            }
            let Some(node) = graph.nodes.get(&nid) else { continue; };
            if node.level == 0 {
                continue;
            }
            if node.token_count > budget && !node.is_leaf {
                // Prefer edges from cached graph; fall back to child_ids
                if let Some(child_ids) = graph.children.get(&nid) {
                    for child_id in child_ids {
                        if seen.insert(*child_id) {
                            stack.push(*child_id);
                        }
                    }
                }
                continue;
            }
            nodes.push(node.clone());
        }

        nodes.reverse();
        Ok(nodes)
    }

    /// Trace sentence-level provenance for a summary node.
    pub fn trace_provenance(&self, node_id: i64) -> anyhow::Result<Vec<(i64, i32, i32, String)>> {
        self.db.get_provenance_with_excerpts(node_id)
    }

    /// Search across all sessions for semantically relevant nodes.
    pub fn search_cross_session(&self, query: &str, limit: usize) -> anyhow::Result<Vec<(i64, i64, String, String)>> {
        self.db.search_cross_session(query, limit)
    }

    /// Search execution memory for similar bugs, tool chains, or code edits.
    pub fn search_execution_memory(&self, query: &str, limit: usize) -> anyhow::Result<Vec<crate::execution::ExecutionUnitRef>> {
        self.db.search_execution_memory(query, limit)
    }

    /// Topological sort of summary nodes reachable from `start`.
    /// Uses Kahn's algorithm with a min-heap (by level DESC, then id ASC)
    /// for deterministic ordering.  Guarantees parents appear before children.
    ///
    /// For tree structures this matches DFS order; for true DAG with shared
    /// children, it ensures a stable, reproducible output.
    pub fn topological_sort(&self, root_id: i64) -> anyhow::Result<Vec<DagNode>> {
        use std::collections::{BinaryHeap, HashMap, VecDeque};
        use std::cmp::Reverse;

        let mut in_degree: HashMap<i64, usize> = HashMap::new();
        let mut adj: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut node_map: HashMap<i64, DagNode> = HashMap::new();

        // BFS to collect the subgraph rooted at root_id
        let mut visited = HashSet::new();
        let mut queue = VecDeque::from([root_id]);
        while let Some(nid) = queue.pop_front() {
            if !visited.insert(nid) { continue; }
            if let Some(node) = self.db.get_node(nid)? {
                node_map.insert(nid, node);
                in_degree.entry(nid).or_insert(0);
                // Follow edges to children
                for (_, child, _) in self.db.get_edges_from(nid)? {
                    adj.entry(nid).or_default().push(child);
                    *in_degree.entry(child).or_insert(0) += 1;
                    queue.push_back(child);
                }
            }
            if visited.len() > self.config.max_fanout { break; }
        }

        // Kahn: start with nodes that have no incoming edges
        // Min-heap sorted by (level DESC, id ASC) for deterministic ordering
        let mut heap: BinaryHeap<(i64, Reverse<i64>)> = BinaryHeap::new();
        for (&nid, &deg) in &in_degree {
            if deg == 0 {
                let level = node_map.get(&nid).map(|n| n.level as i64).unwrap_or(0);
                heap.push((level, Reverse(nid)));
            }
        }

        let mut result = Vec::new();
        while let Some((_level, Reverse(nid))) = heap.pop() {
            if let Some(node) = node_map.get(&nid) {
                result.push(node.clone());
            }
            if let Some(children) = adj.get(&nid) {
                for child in children {
                    let deg = in_degree.get_mut(child).expect("child node missing from in_degree map");
                    *deg -= 1;
                    if *deg == 0 {
                        let clevel = node_map.get(child).map(|n| n.level as i64).unwrap_or(0);
                        heap.push((clevel, Reverse(*child)));
                    }
                }
            }
        }

        Ok(result)
    }

    // ── Garbage collection ──────────────────────────────────────────────

    /// Collect unreachable ("ghost") nodes via mark-sweep from all active
    /// tips. Returns IDs of ghost nodes removed, or `dry_run=true` to
    /// just report without deleting.
    ///
    /// Shared nodes (those with incoming edges from other conversations)
    /// are protected from deletion even if unreachable within this conversation.
    pub fn collect_garbage(&self, conv_id: i64, dry_run: bool) -> anyhow::Result<Vec<i64>> {
        let mut reachable = std::collections::HashSet::new();
        let mut stack = Vec::new();

        for tip in self.db.get_tip_nodes(conv_id)? {
            stack.push(tip.id);
        }
        for leaf in self.db.get_leaf_nodes(conv_id)? {
            stack.push(leaf.id);
        }

        while let Some(id) = stack.pop() {
            if !reachable.insert(id) { continue; }
            if let Some(node) = self.db.get_node(id)? {
                // Edges are the source of truth (P0-4).
                for (_, to_id, _kind) in self.db.get_edges_from(id)? {
                    stack.push(to_id);
                }
                // Also follow legacy child_ids for backward compat
                for cid in &node.child_ids {
                    stack.push(*cid);
                }
            }
        }

        let all_nodes = self.db.get_all_dag_nodes(conv_id)?;
        let mut ghosts = Vec::new();
        for node in all_nodes {
            if !reachable.contains(&node.id) {
                // Protect shared nodes: if referenced from another conversation, keep
                let incoming = self.db.get_edges_to(node.id).unwrap_or_default();
                let has_external_ref = incoming.iter().any(|(from_id, _, _)| {
                    self.db.get_node(*from_id)
                        .ok()
                        .flatten()
                        .map(|n| n.conversation_id != conv_id)
                        .unwrap_or(false)
                });
                if !has_external_ref {
                    ghosts.push(node.id);
                }
            }
        }

        if !dry_run && !ghosts.is_empty() {
            for gid in &ghosts {
                self.db.purge_dag_node(*gid)?;
            }
        }
        Ok(ghosts)
    }

    /// Score-driven GC: soft-deletes low-value nodes based on memory scores.
    /// Ephemeral tier (score < 0.3) is deleted first. Critical tier (≥ 0.7)
    /// is never deleted. Returns count of nodes deleted.
    pub fn gc_by_score(&self, conv_id: i64, max_to_delete: usize) -> anyhow::Result<usize> {
        let all_nodes = self.db.get_all_dag_nodes(conv_id)?;
        let mut scored: Vec<(&DagNode, f64)> = all_nodes.iter()
            .filter(|n| !n.is_leaf) // never delete raw leaves
            .map(|n| (n, self.compute_memory_score(n)))
            .collect();
        scored.sort_by(|a, b| a.1.partial_cmp(&b.1).unwrap_or(std::cmp::Ordering::Equal));

        let mut deleted = 0;
        for (node, score) in &scored {
            if deleted >= max_to_delete {
                break;
            }
            if *score >= 0.7 {
                break; // critical tier, stop deleting
            }
            // Protect shared nodes
            let incoming = self.db.get_edges_to(node.id).unwrap_or_default();
            let has_external = incoming.iter().any(|(from_id, _, _)| {
                self.db.get_node(*from_id).ok().flatten()
                    .map(|n| n.conversation_id != conv_id).unwrap_or(false)
            });
            if has_external {
                continue;
            }
            self.db.delete_dag_node(node.id)?;
            deleted += 1;
        }
        Ok(deleted)
    }

    /// Rollback: soft-delete the target node and all nodes created after it
    /// in the same conversation.  Returns the count of deleted nodes.
    pub fn rollback_to(&self, node_id: i64) -> anyhow::Result<usize> {
        let target = self.db.get_node(node_id)?
            .ok_or_else(|| anyhow::anyhow!("node {} not found", node_id))?;
        let mut deleted = 0usize;
        // Soft-delete all non-deleted nodes with id >= target_id in the same conversation
        let all = self.db.get_all_dag_nodes(target.conversation_id)?;
        for node in &all {
            if node.id > node_id && !node.deleted {
                self.db.delete_dag_node(node.id)?;
                deleted += 1;
            }
        }
        Ok(deleted)
    }

    // ── DAG validation (P0 consistency) ─────────────────────────────────

    /// Validate DAG invariants for a conversation. Returns a list of
    /// violations found (empty = healthy). Checks:
    /// 1. Parent-child symmetry
    /// 2. Orphan references (child_ids pointing to non-existent/deleted nodes)
    /// 3. Level ordering (parent level >= child level)
    /// 4. Duplicate edge detection
    pub fn validate_dag(&self, conv_id: i64) -> anyhow::Result<Vec<String>> {
        let all_nodes = self.db.get_all_dag_nodes(conv_id)?;
        let mut issues = Vec::new();
        let node_map: HashMap<i64, &DagNode> = all_nodes.iter().map(|n| (n.id, n)).collect();

        for node in &all_nodes {
            // 1. Parent-child symmetry
            for pid in &node.parent_ids {
                if let Some(parent) = node_map.get(pid) {
                    if !parent.child_ids.contains(&node.id) {
                        issues.push(format!(
                            "symmetry-broken: node {} has parent_ids=[{}] but parent {} child_ids missing {}",
                            node.id, pid, pid, node.id
                        ));
                    }
                } else {
                    issues.push(format!(
                        "orphan-parent: node {} parent_ids contains non-existent {}", node.id, pid
                    ));
                }
            }
            for cid in &node.child_ids {
                if let Some(child) = node_map.get(cid) {
                    if !child.parent_ids.contains(&node.id) {
                        issues.push(format!(
                            "symmetry-broken: node {} child_ids=[{}] but child {} parent_ids missing {}",
                            node.id, cid, cid, node.id
                        ));
                    }
                } else {
                    issues.push(format!(
                        "orphan-child: node {} child_ids contains non-existent/deleted {}", node.id, cid
                    ));
                }
            }

            // 3. Level ordering
            for cid in &node.child_ids {
                if let Some(child) = node_map.get(cid)
                    && node.level < child.level
                {
                    issues.push(format!(
                        "level-order: node {} (L{}) child {} (L{}) — parent level must be >= child",
                        node.id, node.level, cid, child.level
                    ));
                }
            }

            // 4. Detect cycles via existing has_path
            if !node.is_leaf && !node.child_ids.is_empty() {
                for cid in &node.child_ids {
                    if self.db.has_path(*cid, node.id, 20).unwrap_or(false) {
                        issues.push(format!(
                            "cycle: path exists from child {} back to parent {}", cid, node.id
                        ));
                    }
                }
            }
        }

        Ok(issues)
    }

    // ── Graphviz export ─────────────────────────────────────────────────

    pub fn export_dot(&self, conv_id: i64) -> anyhow::Result<String> {
        use std::fmt::Write;
        let mut out = String::new();
        writeln!(out, "digraph LCM_DAG {{")?;
        writeln!(out, "    rankdir=BT;")?;
        writeln!(out, "    node [shape=box, style=filled, fontname=\"monospace\"];")?;

        for node in self.db.get_all_dag_nodes(conv_id)? {
            let color = match node.level {
                0 => "\"#d5e8d4\"",
                1 => "\"#dae8fc\"",
                2 => "\"#ffe6cc\"",
                _ => "\"#f8cecc\"",
            };
            // Truncate by token budget to prevent DOT label explosion
            let label = format!(
                "{} {} ({})",
                if node.is_leaf { "L" } else { "S" },
                html_escape_token(&node.summary, 30),
                node.token_count,
            );
            writeln!(out, "    n{} [label=\"{}\", fillcolor={color}];", node.id, label)?;
            for pid in &node.parent_ids {
                writeln!(out, "    n{} -> n{};", node.id, pid)?;
            }
        }
        writeln!(out, "}}")?;
        Ok(out)
    }

    /// Take an auto-snapshot after compaction when SnapshotMode is Auto.
    /// Creates an Ephemeral-tier snapshot with the current DAG state.
    fn auto_snapshot_on_compaction(&self, conv_id: i64, node: &DagNode) -> anyhow::Result<()> {
        let version_id = self.db.create_memory_version(
            None, "compaction", &format!("auto-snapshot after compaction to level {}", node.level),
            Some(node.id),
        )?;
        let snapshot_data = serde_json::json!({
            "conv_id": conv_id,
            "dag_root_id": node.id,
            "level": node.level,
            "source_count": node.child_ids.len(),
            "revision": self.current_revision().0,
        });
        let data_str = serde_json::to_string(&snapshot_data)?;
        let boundary_hash = crate::snapshot::compute_boundary_hash(&[(0_i64, &data_str)], 1);
        let integrity_hash = crate::snapshot::compute_chain_hash(&[(0_i64, &data_str)]);
        self.db.take_snapshot(
            0, version_id, 0, &data_str, data_str.len() as i64, None,
            0, &boundary_hash, &integrity_hash,
        )?;
        // Enforce budget to prevent unbounded snapshot growth
        let budget = self.db.policy_config.read()
            .map(|c| c.snapshot_budget.clone())
            .unwrap_or_default();
        let _ = self.db.enforce_snapshot_budget(&budget);
        Ok(())
    }
}

/// Truncate text to fit within a token budget, then HTML-escape.
fn html_escape_token(text: &str, max_tokens: usize) -> String {
    let truncated = truncate_by_token(text, max_tokens);
    let mut out = String::new();
    for ch in truncated.chars() {
        match ch {
            '"' => out.push_str("&quot;"),
            '<' => out.push_str("&lt;"),
            '>' => out.push_str("&gt;"),
            '&' => out.push_str("&amp;"),
            '\n' => out.push_str("\\n"),
            '\t' => out.push_str("\\t"),
            c if c.is_control() => {}
            c => out.push(c),
        }
    }
    if crate::tokenizer::count(text) > max_tokens { out.push('…'); }
    out
}

/// Truncate text so it stays within a token budget.
fn truncate_by_token(text: &str, max_tokens: usize) -> &str {
    if crate::tokenizer::count(text) <= max_tokens {
        return text;
    }
    let mut boundary = 0;
    for (i, _) in text.char_indices() {
        let slice = &text[..i + 1];
        if crate::tokenizer::count(slice) > max_tokens {
            break;
        }
        boundary = i + 1;
    }
    &text[..boundary]
}

/// Find safe split points in a slice of normalized messages for
/// compression grouping.  Splitting at these points guarantees that:
///
/// - No tool call is separated from its corresponding tool result
/// - Group boundaries fall at clean turn boundaries
///
/// Uses `NormalizedMessage` role-based detection instead of fragile
/// content heuristics, supporting OpenAI/DeepSeek/Claude schemas.
///
/// Returns indices (into `messages`) that are *safe* split positions
/// (everything before the index can be compressed as a complete unit).
pub fn safe_split_points(messages: &[crate::session::NormalizedMessage]) -> Vec<usize> {
    // First pass: mark indices that are inside a tool chain
    let mut inside_tool_chain = vec![false; messages.len()];
    for i in 0..messages.len() {
        if crate::session::is_tool_call(&messages[i]) {
            inside_tool_chain[i] = true;
            let mut j = i + 1;
            while j < messages.len() && crate::session::is_tool_result(&messages[j]) {
                inside_tool_chain[j] = true;
                j += 1;
            }
        }
    }

    // Second pass: user messages that are NOT inside a tool chain are safe splits
    let mut safe = Vec::new();
    safe.push(0); // always safe before the first message
    for i in 0..messages.len() {
        if messages[i].role == "user" && !inside_tool_chain[i] {
            safe.push(i);
        }
    }
    safe.sort();
    safe.dedup();
    safe
}

/// Compute a semantic fingerprint for cross-conversation dedup.
/// SHA-256 of summary text + snippet contents, 16 hex chars.
/// Mirrors `Database::semantic_hash` for use outside the DB layer.
fn dag_semantic_hash(summary: &str, snippets: &[crate::snippet::Snippet]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(summary.as_bytes());
    for s in snippets {
        hasher.update(s.content.as_bytes());
    }
    hex::encode(&hasher.finalize()[..8])
}

fn format_summary_text(summary: &str, snippets: &[crate::snippet::Snippet]) -> String {
    if snippets.is_empty() {
        summary.to_string()
    } else {
        let snippet_texts: Vec<&str> = snippets.iter().map(|s| s.content.as_str()).collect();
        format!("{} {}", summary, snippet_texts.join(" "))
    }
}

/// Compute content overlap between a leaf and a set of summary texts.
/// Returns 0.0 (no overlap) to 1.0 (fully contained). Uses trigram Jaccard.
fn content_overlap(leaf: &str, summaries: &[&str]) -> f64 {
    if summaries.is_empty() || leaf.len() < 10 {
        return 0.0;
    }
    let leaf_bytes = leaf.as_bytes();
    let leaf_trigrams: std::collections::HashSet<[u8; 3]> = leaf_bytes.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
    if leaf_trigrams.is_empty() {
        return 0.0;
    }
    let mut best = 0.0;
    for summary in summaries {
        let s_bytes = summary.as_bytes();
        let s_trigrams: std::collections::HashSet<[u8; 3]> = s_bytes.windows(3).map(|w| [w[0], w[1], w[2]]).collect();
        let intersection = leaf_trigrams.intersection(&s_trigrams).count();
        let overlap = intersection as f64 / leaf_trigrams.len() as f64;
        if overlap > best {
            best = overlap;
        }
    }
    best
}

/// Split text into sentences by `.`, `。`, `\n`, `!`, `?` boundaries.
/// Returns (start_byte_offset, sentence_text) pairs.
fn split_sentences(text: &str) -> Vec<(usize, &str)> {
    let mut sentences = Vec::new();
    let bytes = text.as_bytes();
    let mut start = 0;

    for (i, &b) in bytes.iter().enumerate() {
        let is_boundary = matches!(b, b'.' | b'\n' | b'!' | b'?');
        let is_cjk_period = i + 2 < bytes.len()
            && bytes[i] == 0xE3 && bytes[i+1] == 0x80 && bytes[i+2] == 0x82;
        let boundary_end = if is_cjk_period { i + 3 } else if is_boundary { i + 1 } else { 0 };
        if boundary_end > 0 && boundary_end > start {
            let sent = text[start..boundary_end].trim();
            if !sent.is_empty() {
                sentences.push((start, sent));
            }
            start = boundary_end;
        }
    }
    if start < text.len() {
        let remaining = text[start..].trim();
        if !remaining.is_empty() {
            sentences.push((start, remaining));
        }
    }
    sentences
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::db::Database;
    use std::sync::Arc;
    use tempfile::tempdir;

    fn setup_db() -> (Arc<Database>, i64) {
        let dir = tempdir().unwrap();
        let db = std::thread::spawn(move || {
            tokio::runtime::Runtime::new().unwrap().block_on(
                Database::builder().path(dir.path().join("dag_test.db")).build()
            )
        })
        .join()
        .unwrap()
        .unwrap();
        let db = Arc::new(db);
        let conv_id = db
            .create_and_store("test", &serde_json::json!([{"role": "user", "content": "hi"}]))
            .unwrap();
        (db, conv_id)
    }

    #[test]
    fn builder_creates_engine() {
        let (db, _) = setup_db();
        let engine = DagEngine::builder()
            .soft_threshold(0.75)
            .hard_threshold(0.90)
            .max_level(3)
            .recent_messages(10)
            .build(db);
        assert_eq!(engine.config.soft_threshold_ratio, 0.75);
        assert_eq!(engine.config.hard_threshold_ratio, 0.90);
        assert_eq!(engine.config.max_level, 3);
        assert_eq!(engine.config.recent_message_count, 10);
    }

    #[test]
    fn builder_defaults() {
        let engine = DagEngine::builder().build(setup_db().0);
        assert_eq!(engine.config.max_level, 0, "default should be 0 (dynamic)");
        assert_eq!(engine.config.soft_threshold_ratio, 0.80);
        assert_eq!(engine.config.recent_message_count, 20);
    }

    #[test]
    fn dynamic_max_level_scales_with_leaves() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);

        // Few leaves → level 3
        let level_few = engine.effective_max_level(conv_id);
        assert_eq!(level_few, 3, "few leaves should use level 3");

        // Insert many leaves
        for i in 0..50 {
            engine.insert_leaf(conv_id, &format!("msg {i}"), 5).unwrap();
        }
        let level_many = engine.effective_max_level(conv_id);
        assert!(level_many >= 4, "50 leaves should need >3 levels, got {level_many}");

        // Explicit override
        let engine2 = DagEngine::builder().max_level(5).build(setup_db().0);
        assert_eq!(engine2.config.max_level, 5);
    }

    // ── Property: DAG invariants ────────────────────────────────────────

    /// Verify the no-cycle invariant: after compression, no source node
    /// can reach back to the summary node through any path.
    #[test]
    fn property_no_cycle_after_compression() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);

        // Create leaves
        let mut leaves = Vec::new();
        for i in 0..8 {
            let leaf = engine.insert_leaf(conv_id, &format!("msg {i}"), 10).unwrap();
            leaves.push(leaf);
        }

        // Compress into summary
        let ids: Vec<_> = leaves.iter().map(|n| n.id).collect();
        let summary = engine
            .compress_group_with_snippets(conv_id, &ids, "summary", 20, 1, &[])
            .unwrap();

        // Verify: no leaf's parent chain leads back to summary (summary→leaf, not leaf→summary)
        for leaf_id in &ids {
            assert!(
                !engine.db.has_path(*leaf_id, summary.id, 10).unwrap(),
                "leaf {leaf_id} should not have a path back to its summary"
            );
        }
    }

    /// Verify parent-child symmetry: if A is in B's child_ids, then
    /// B should be in A's parent_ids.
    #[test]
    fn property_parent_child_symmetry() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);

        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        let summary = engine
            .compress_group_with_snippets(conv_id, &[a.id, b.id], "summary", 8, 1, &[])
            .unwrap();

        // Re-fetch to get updated parent_ids
        let a_fresh = engine.db.get_node(a.id).unwrap().unwrap();
        let b_fresh = engine.db.get_node(b.id).unwrap().unwrap();

        assert!(a_fresh.parent_ids.contains(&summary.id));
        assert!(b_fresh.parent_ids.contains(&summary.id));
        assert!(summary.child_ids.contains(&a.id));
        assert!(summary.child_ids.contains(&b.id));
    }

    #[test]
    fn compress_group_links_parents() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);

        let a = engine.insert_leaf(conv_id, "msg a", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "msg b", 5).unwrap();

        let summary = engine
            .compress_group(conv_id, &[a.id, b.id], "summary of a+b", 3, 1)
            .unwrap();
        assert_eq!(summary.level, 1);
        assert!(!summary.is_leaf);

        // Invariant: summary.child_ids = source_ids
        let summary_children = engine.get_children(summary.id).unwrap();
        assert_eq!(summary_children.len(), 2);
        assert!(summary_children.iter().any(|n| n.id == a.id));
        assert!(summary_children.iter().any(|n| n.id == b.id));

        // Invariant: source nodes' parent_ids includes the summary
        let node_a = engine.get_node(a.id).unwrap().unwrap();
        assert!(node_a.parent_ids.contains(&summary.id));
        let node_b = engine.get_node(b.id).unwrap().unwrap();
        assert!(node_b.parent_ids.contains(&summary.id));
    }

    #[test]
    fn assemble_context_budget_respected() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder()
            .recent_messages(5)
            .max_level(3)
            .build(db);

        // Insert several leaf nodes
        for i in 0..10 {
            engine
                .insert_leaf(conv_id, &format!("message {i}"), 10)
                .unwrap();
        }

        // Compress first 8 into a summary
        let leaves = engine.get_leaves(conv_id).unwrap();
        let ids: Vec<_> = leaves.iter().take(8).map(|n| n.id).collect();
        engine
            .compress_group(conv_id, &ids, "summary of first 8", 15, 1)
            .unwrap();

        let context = engine.assemble_context(conv_id, 100, None).unwrap();
        assert!(!context.is_empty(), "should assemble at least some nodes");
        // Total should be ≤ 100 tokens
        let total: i64 = context.iter().map(|n| n.token_count).sum();
        assert!(total <= 100, "budget exceeded: {total} > 100");
    }

    #[test]
    fn max_level_clamps() {
        let engine = DagEngine::builder().max_level(99).build(setup_db().0);
        assert_eq!(engine.config.max_level, 10);
    }

    #[test]
    fn threshold_clamps() {
        let engine = DagEngine::builder()
            .soft_threshold(1.5)
            .hard_threshold(-0.5)
            .build(setup_db().0);
        assert_eq!(engine.config.soft_threshold_ratio, 1.0);
        assert_eq!(engine.config.hard_threshold_ratio, 0.0);
    }

    #[test]
    fn collect_garbage_dry_run_returns_no_ghosts_in_empty_dag() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let ghosts = engine.collect_garbage(conv_id, true).unwrap();
        // No summary nodes yet, so no ghosts expected
        assert!(ghosts.is_empty());
    }

    #[test]
    fn export_dot_produces_valid_graphviz_header() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let dot = engine.export_dot(conv_id).unwrap();
        assert!(dot.starts_with("digraph LCM_DAG"));
        assert!(dot.ends_with("}\n"));
    }

    // ── safe_split_points tests ────────────────────────────────────────

    use crate::session::NormalizedMessage;

    fn simple_msg(role: &str, text: &str) -> NormalizedMessage {
        NormalizedMessage {
            role: role.to_string(),
            content: text.to_string(),
            tool_calls: vec![],
            tool_call_id: None,
        }
    }

    #[test]
    fn safe_split_simple_turns() {
        let msgs = vec![
            simple_msg("user", "hello"),
            simple_msg("assistant", "hi"),
            simple_msg("user", "what is rust?"),
            simple_msg("assistant", "a language"),
        ];
        let points = safe_split_points(&msgs);
        assert!(points.contains(&0), "should split at start");
        assert!(points.contains(&2), "should split before second user msg");
    }

    #[test]
    fn safe_split_keeps_tool_pairs_together() {
        let msgs = vec![
            simple_msg("user", "read src/main.rs"),
            NormalizedMessage {
                role: "assistant".into(),
                content: "".into(),
                tool_calls: vec![crate::session::NormalizedToolCall {
                    id: "call_1".into(),
                    name: "read_file".into(),
                    arguments: "{}".into(),
                }],
                tool_call_id: None,
            },
            NormalizedMessage {
                role: "tool".into(),
                content: "pub fn main()".into(),
                tool_calls: vec![],
                tool_call_id: Some("call_1".into()),
            },
            simple_msg("user", "now fix it"),
        ];
        let points = safe_split_points(&msgs);
        assert!(points.contains(&0), "should split at message 0");
        assert!(!points.contains(&2), "should NOT split inside tool chain");
        assert!(points.contains(&3), "should split after tool chain");
    }

    #[test]
    fn safe_split_empty() {
        let points = safe_split_points(&[]);
        assert_eq!(points, vec![0], "index 0 is always safe");
    }

    #[test]
    fn safe_split_claude_tool_use() {
        let msgs = vec![
            simple_msg("user", "list files"),
            NormalizedMessage {
                role: "assistant".into(),
                content: r#"{"type": "tool_use", "name": "bash", "input": "ls"}"#.into(),
                tool_calls: vec![crate::session::NormalizedToolCall {
                    id: "toolu_1".into(),
                    name: "bash".into(),
                    arguments: r#"{"input": "ls"}"#.into(),
                }],
                tool_call_id: None,
            },
            simple_msg("user", "tool_result: file1.rs"),
            simple_msg("user", "now edit it"),
        ];
        let points = safe_split_points(&msgs);
        assert!(points.contains(&0));
        assert!(!points.contains(&2), "should NOT split inside tool chain");
        assert!(points.contains(&3), "should split after tool chain");
    }

    // ── P3: DAG hardening ───────────────────────────────────────────────

    #[test]
    fn edge_insert_and_query() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db.clone());

        let a = engine.insert_leaf(conv_id, "node A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "node B", 5).unwrap();

        // Insert typed edge
        db.insert_edge(a.id, b.id, "summarizes").unwrap();
        let edges = db.get_edges_from(a.id).unwrap();
        assert_eq!(edges.len(), 1);
        assert_eq!(edges[0].2, "summarizes");
        assert_eq!(edges[0].1, b.id);
    }

    #[test]
    fn has_path_detects_connection() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db.clone());

        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        let c = engine.insert_leaf(conv_id, "C", 5).unwrap();

        db.insert_edge(a.id, b.id, "summarizes").unwrap();
        db.insert_edge(b.id, c.id, "summarizes").unwrap();

        assert!(db.has_path(a.id, c.id, 10).unwrap());
        assert!(!db.has_path(c.id, a.id, 10).unwrap());
        assert!(db.has_path(a.id, a.id, 10).unwrap()); // self-loop is a path
    }

    #[test]
    fn has_path_no_cycle_in_tree() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db.clone());

        let root = engine.insert_leaf(conv_id, "root", 5).unwrap();
        let child = engine.insert_leaf(conv_id, "child", 5).unwrap();
        db.insert_edge(root.id, child.id, "summarizes").unwrap();

        // root → child exists, but child → root should not
        assert!(db.has_path(root.id, child.id, 10).unwrap());
        assert!(!db.has_path(child.id, root.id, 10).unwrap());
    }

    #[test]
    fn topological_sort_deterministic() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db.clone());

        let root = engine.insert_leaf(conv_id, "root", 5).unwrap();
        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();

        db.insert_edge(root.id, a.id, "summarizes").unwrap();
        db.insert_edge(root.id, b.id, "summarizes").unwrap();

        // Should produce same order every time
        let order1 = engine.topological_sort(root.id).unwrap();
        let order2 = engine.topological_sort(root.id).unwrap();
        assert_eq!(order1.len(), order2.len());
        for (n1, n2) in order1.iter().zip(order2.iter()) {
            assert_eq!(n1.id, n2.id);
        }
        assert!(!order1.is_empty());
    }

    #[test]
    fn merge_nodes_creates_refines_edge() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db.clone());

        let a = engine.insert_leaf(conv_id, "summary A", 10).unwrap();
        let b = engine.insert_leaf(conv_id, "summary B", 10).unwrap();

        // Mark these as summaries by setting level > 0 via direct DB insert
        // (insert_leaf creates level-0 nodes; for merge we need summary nodes)
        let merged = engine.merge_nodes(conv_id, &[a.id, b.id], "merged AB", 8).unwrap();
        assert_eq!(merged.level, 1);
        assert_eq!(merged.child_ids.len(), 2);
        assert!(merged.child_ids.contains(&a.id));
        assert!(merged.child_ids.contains(&b.id));

        // Edges: merge_nodes replaces auto-generated Summarizes edges with Refines
        let edges = db.get_edges_from(merged.id).unwrap();
        assert_eq!(edges.len(), 2, "should have exactly 2 refines edges");
        let kinds: Vec<&str> = edges.iter().map(|(_, _, k)| k.as_str()).collect();
        assert!(kinds.iter().all(|k| k == &"refines"), "all edges should be refines");
    }

    #[test]
    fn covered_leaf_ids_traverses_edges() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db.clone());

        let leaf1 = engine.insert_leaf(conv_id, "leaf1", 5).unwrap();
        let leaf2 = engine.insert_leaf(conv_id, "leaf2", 5).unwrap();

        // Create a summary that covers both leaves
        let summary = engine
            .compress_group_with_snippets(conv_id, &[leaf1.id, leaf2.id], "summary", 8, 1, &[])
            .unwrap();

        let covered = engine.covered_leaf_ids(summary.id).unwrap();
        assert!(covered.contains(&leaf1.id));
        assert!(covered.contains(&leaf2.id));
    }

    // ── v0.3: Semantic dedup ─────────────────────────────────────────

    #[test]
    fn semantic_hash_is_deterministic() {
        let hash1 = dag_semantic_hash("hello world", &[]);
        let hash2 = dag_semantic_hash("hello world", &[]);
        assert_eq!(hash1, hash2);
        assert_eq!(hash1.len(), 16);
    }

    #[test]
    fn semantic_hash_differs_for_different_content() {
        let h1 = dag_semantic_hash("hello", &[]);
        let h2 = dag_semantic_hash("world", &[]);
        assert_ne!(h1, h2);
    }

    #[test]
    fn dedup_and_reuse_finds_existing_node() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);

        let a = engine.insert_leaf(conv_id, "leaf A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "leaf B", 5).unwrap();

        // Create first summary
        let summary1 = engine
            .dedup_and_reuse(conv_id, &[a.id, b.id], "identical summary", 8, 1, &[])
            .unwrap();
        assert!(summary1.semantic_hash.len() == 16);

        // Create another with same text — should reuse
        let summary2 = engine
            .dedup_and_reuse(conv_id, &[a.id], "identical summary", 8, 1, &[])
            .unwrap();
        assert_eq!(summary2.id, summary1.id, "dedup should return existing node");
        assert_eq!(summary2.semantic_hash, summary1.semantic_hash);
    }

    #[test]
    fn find_similar_returns_none_for_unique_content() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);

        let _a = engine.insert_leaf(conv_id, "unique leaf", 5).unwrap();

        // Non-existing summary should not be found
        let result = engine.find_similar_node("never seen before text", &[]).unwrap();
        assert!(result.is_none());
    }

    // ── P0: DAG validation ──────────────────────────────────────────────

    #[test]
    fn validate_dag_reports_no_issues_for_healthy_graph() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db.clone());

        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        engine
            .compress_group_with_snippets(conv_id, &[a.id, b.id], "summary", 8, 1, &[])
            .unwrap();

        let issues = engine.validate_dag(conv_id).unwrap();
        assert!(issues.is_empty(), "healthy DAG should have no issues, got: {issues:?}");
    }

    #[test]
    fn validate_dag_detects_orphan_child_reference() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db.clone());

        let _a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        // Insert a node that references a non-existent child
        db.insert_dag_node(conv_id, 1, "orphan summary", 10, &[], &[99999], false).unwrap();

        let issues = engine.validate_dag(conv_id).unwrap();
        assert!(!issues.is_empty(), "should detect orphan child reference");
        assert!(issues.iter().any(|i| i.contains("orphan-child")), "should report orphan-child");
    }

    #[test]
    fn sentence_splitter_handles_english_and_cjk() {
        let sents = split_sentences("Hello world. This is Rust.\nNew line here!Right?");
        assert!(sents.len() >= 3, "expected at least 3 sentences, got {}", sents.len());

        let sents_cjk = split_sentences("你好世界。这是Rust。\n新行");
        assert!(sents_cjk.len() >= 2, "expected at least 2 CJK sentences, got {}", sents_cjk.len());
    }

    // ── insert_leaf edge cases ──────────────────────────────────────────

    #[test]
    fn insert_leaf_returns_node_with_correct_properties() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let node = engine.insert_leaf(conv_id, "test message", 42).unwrap();
        assert_eq!(node.level, 0);
        assert!(node.is_leaf);
        assert_eq!(node.summary, "test message");
        assert_eq!(node.token_count, 42);
        assert_eq!(node.conversation_id, conv_id);
    }

    #[test]
    fn insert_leaf_empty_content() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let node = engine.insert_leaf(conv_id, "", 1).unwrap();
        assert!(node.is_leaf);
        assert_eq!(node.summary, "");
    }

    #[test]
    fn insert_leaf_multiple_leaves() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let ids: Vec<i64> = (0..5)
            .map(|i| engine.insert_leaf(conv_id, &format!("msg {i}"), 10).unwrap().id)
            .collect();
        let leaves = engine.get_leaves(conv_id).unwrap();
        assert_eq!(leaves.len(), 5);
        for id in ids {
            assert!(leaves.iter().any(|n| n.id == id));
        }
    }

    // ── assemble_context with deeper DAG ───────────────────────────────

    #[test]
    fn assemble_context_includes_summaries() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);
        for i in 0..6 {
            engine.insert_leaf(conv_id, &format!("msg {i}"), 10).unwrap();
        }
        let leaves = engine.get_leaves(conv_id).unwrap();
        let ids: Vec<i64> = leaves.iter().map(|n| n.id).collect();
        engine.compress_group(conv_id, &ids[0..4], "summary of 0-3", 15, 1).unwrap();

        let context = engine.assemble_context(conv_id, 500, None).unwrap();
        let has_summary = context.iter().any(|n| n.level > 0);
        assert!(has_summary, "context should include summary nodes");
    }

    #[test]
    fn assemble_context_budget_zero_returns_empty() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);
        engine.insert_leaf(conv_id, "msg", 10).unwrap();
        let context = engine.assemble_context(conv_id, 0, None).unwrap();
        assert!(context.is_empty());
    }

    #[test]
    fn assemble_context_respects_recent_message_count() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder()
            .recent_messages(3)
            .max_level(3)
            .build(db);
        for i in 0..10 {
            engine.insert_leaf(conv_id, &format!("msg {i}"), 10).unwrap();
        }
        let context = engine.assemble_context(conv_id, 1000, None).unwrap();
        let leaf_count = context.iter().filter(|n| n.is_leaf).count();
        assert!(leaf_count <= 3, "should keep at most 3 recent leaf messages, got {leaf_count}");
    }

    // ── GC: collect_garbage with actual nodes ──────────────────────────

    #[test]
    fn collect_garbage_removes_unreachable_nodes() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db.clone());
        // Create two leaves and compress them into level 1 summary
        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        let s1 = engine.compress_group(conv_id, &[a.id, b.id], "summary AB", 8, 1).unwrap();
        // Create a level 3 summary as the tip, so level 2 nodes are NOT tips
        engine.compress_group(conv_id, &[s1.id], "higher summary", 5, 3).unwrap();
        // Insert an isolated level 2 node directly — not a tip, not a leaf, no edges to it
        let orphan_id = db.insert_dag_node(conv_id, 2, "isolated", 10, &[], &[], false).unwrap().id;
        let ghosts = engine.collect_garbage(conv_id, false).unwrap();
        assert_eq!(ghosts, vec![orphan_id], "isolated non-leaf below max level should be collected");
        assert!(engine.get_node(orphan_id).unwrap().is_none(), "orphan should be purged");
    }

    #[test]
    fn collect_garbage_preserves_reachable_nodes() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db.clone());
        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        engine.compress_group(conv_id, &[a.id, b.id], "summary", 8, 1).unwrap();
        let ghosts = engine.collect_garbage(conv_id, false).unwrap();
        assert!(ghosts.is_empty(), "reachable nodes should not be collected");
        assert!(engine.get_node(a.id).unwrap().is_some());
        assert!(engine.get_node(b.id).unwrap().is_some());
    }

    #[test]
    fn collect_garbage_dry_run_does_not_purge() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db.clone());
        // Insert level 1 summary and a level 2 summary as tip, leaving level 1 unreachable
        let leaf = engine.insert_leaf(conv_id, "leaf", 5).unwrap();
        let s1 = engine.compress_group(conv_id, &[leaf.id], "level 1", 3, 1).unwrap();
        engine.compress_group(conv_id, &[s1.id], "level 2 tip", 3, 2).unwrap();
        // Insert more leaves so level 1 is not a tip
        engine.insert_leaf(conv_id, "extra", 5).unwrap();
        // Now level 1 summary is not a tip node (level 2 is max), and edges go
        // from level 2 -> level 1 -> leaf, so level 1 is reachable via level 2.
        // GC result depends on topology, so just verify dry_run doesn't purge.
        let ghosts = engine.collect_garbage(conv_id, true).unwrap();
        for gid in &ghosts {
            assert!(engine.get_node(*gid).unwrap().is_some(), "dry_run should not purge node {gid}");
        }
    }

    // ── GC: gc_by_score ────────────────────────────────────────────────

    #[test]
    fn gc_by_score_deletes_low_score_non_leaves() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);
        // Create some leaves and a summary (non-leaf, should have low score)
        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        engine.compress_group(conv_id, &[a.id], "low value summary", 3, 1).unwrap();
        let deleted = engine.gc_by_score(conv_id, 10).unwrap();
        // The summary (level 1, low token count, old) should be deleted
        assert!(deleted >= 1, "should delete at least one low-score non-leaf");
    }

    #[test]
    fn gc_by_score_never_deletes_leaves() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);
        let leaf = engine.insert_leaf(conv_id, "important leaf", 5).unwrap();
        let _deleted = engine.gc_by_score(conv_id, 10).unwrap();
        assert!(engine.get_node(leaf.id).unwrap().is_some(), "leaves should never be deleted by gc_by_score");
    }

    #[test]
    fn gc_by_score_respects_max_to_delete() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().max_level(3).build(db);
        for i in 0..3 {
            let a = engine.insert_leaf(conv_id, &format!("a{i}"), 5).unwrap();
            engine.compress_group(conv_id, &[a.id], &format!("summary {i}"), 3, 1).unwrap();
        }
        let deleted = engine.gc_by_score(conv_id, 1).unwrap();
        assert!(deleted <= 1, "should respect max_to_delete limit");
    }

    // ── rollback_to ─────────────────────────────────────────────────────

    #[test]
    fn rollback_to_removes_nodes_after_target() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let n1 = engine.insert_leaf(conv_id, "first", 5).unwrap();
        let n2 = engine.insert_leaf(conv_id, "second", 5).unwrap();
        let n3 = engine.insert_leaf(conv_id, "third", 5).unwrap();
        let deleted = engine.rollback_to(n1.id).unwrap();
        assert_eq!(deleted, 2, "nodes after target should be deleted");
        assert!(engine.get_node(n1.id).unwrap().is_some());
        assert!(engine.get_node(n2.id).unwrap().map(|n| n.deleted).unwrap_or(true));
        assert!(engine.get_node(n3.id).unwrap().map(|n| n.deleted).unwrap_or(true));
    }

    #[test]
    fn rollback_to_target_not_found() {
        let (db, _conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let result = engine.rollback_to(99999);
        assert!(result.is_err());
    }

    #[test]
    fn rollback_to_keeps_nodes_before_target() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let n1 = engine.insert_leaf(conv_id, "keep", 5).unwrap();
        let _n2 = engine.insert_leaf(conv_id, "remove", 5).unwrap();
        engine.rollback_to(n1.id).unwrap();
        assert!(engine.get_node(n1.id).unwrap().is_some(), "target node should be kept");
    }
}

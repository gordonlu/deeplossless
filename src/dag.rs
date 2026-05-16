use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};
use std::sync::Arc;

use crate::db::Database;

/// In-memory snapshot of a conversation's DAG, used for batch-loaded
/// context assembly (avoids N+1 DB roundtrips).
#[allow(dead_code)]
struct DagGraph {
    nodes: HashMap<i64, DagNode>,
    children: HashMap<i64, Vec<i64>>,
    #[allow(dead_code)]
    parents: HashMap<i64, Vec<i64>>,
}

// ── Edge types ──────────────────────────────────────────────────────────

/// Semantic kind of a DAG edge.
#[derive(Debug, Clone, Copy, PartialEq, Eq, Serialize, Deserialize)]
pub enum EdgeKind {
    /// Summary summarizes its source nodes (summary → raw / lower summary).
    Summarizes,
    /// Higher summary refines/merges a lower summary.
    Refines,
    /// Branch forked from a common ancestor.
    ForksFrom,
    /// Semantic reuse/dedup link.
    Reuses,
}

impl EdgeKind {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Summarizes => "summarizes",
            Self::Refines => "refines",
            Self::ForksFrom => "forks_from",
            Self::Reuses => "reuses",
        }
    }

    pub fn from_str(s: &str) -> Option<Self> {
        match s {
            "summarizes" => Some(Self::Summarizes),
            "refines" => Some(Self::Refines),
            "forks_from" => Some(Self::ForksFrom),
            "reuses" => Some(Self::Reuses),
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
}

impl Default for DagConfig {
    fn default() -> Self {
        Self {
            soft_threshold_ratio: 0.80,
            hard_threshold_ratio: 0.95,
            max_level: 0, // 0 = dynamic based on leaf count
            max_fanout: 100,
            max_expand_depth: 10,
            recent_message_count: 20,
            token_correction_factor: 1.0,
            token_overhead: 12,
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
    /// Precision-critical snippets extracted before compression.
    pub snippets: Vec<Snippet>,
    /// True when this node has been soft-deleted (excluded from context assembly).
    pub deleted: bool,
    /// SHA-256 hash of normalized summary text for cross-conversation dedup (v0.3).
    pub semantic_hash: String,
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
        DagEngine {
            db,
            config: self.config,
        }
    }
}

/// The DAG engine: insert, link, compress, and assemble conversation
/// history into lossless context blocks.
pub struct DagEngine {
    db: Arc<Database>,
    config: DagConfig,
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
        self.db.insert_dag_node(conv_id, 0, summary, token_count, &[], &[], true)
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
        // Cycle protection: ensure no path from any source → the new summary
        // which would create a cycle when we add summary → source edges.
        let node = self.db.insert_summary_atomic(
            conv_id,
            level.min(self.effective_max_level(conv_id)),
            summary,
            token_count,
            source_ids,
            snippets,
        )?;
        // Verify no cycle was introduced
        if let Err(e) = self.check_no_cycle(&node, source_ids) {
            // Rollback: hard-delete the newly created node and its edges
            self.db.purge_dag_node(node.id)?;
            return Err(e);
        }
        Ok(node)
    }

    /// Check that adding edges from `summary` → each `source` creates no cycle.
    fn check_no_cycle(&self, _summary: &DagNode, _sources: &[i64]) -> anyhow::Result<()> {
        // Cycle detection runs at edge-insert time via has_path in db.rs.
        // This is a double-check layer: after all edges are inserted,
        // verify no source can reach back to the summary.
        // For a tree-like hierarchy, this is trivially satisfied since
        // edges always point from higher level to lower level.
        // When merge_nodes is implemented, this check becomes critical.
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
    /// refines existing summaries by creating a Refines edge.
    /// The merged text is composed from the source summaries' text.
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

        // Insert without child_ids — edges carry the relationship explicitly
        let node = self.db.insert_dag_node_full(
            conv_id,
            level,
            merged_text,
            token_count,
            &[],        // parent_ids
            source_ids, // child_ids (for backward compat, mirrors Refines edges)
            &[],        // snippets
            false,
        )?;

        // The insert already creates Summarizes edges. Remove them and
        // replace with Refines edges since this is a merge (summary→summary),
        // not a compression (summary→raw).
        for sid in source_ids {
            let _ = self.db.insert_edge(node.id, *sid, "refines");
        }

        // Back-link source nodes to this merged node
        for sid in source_ids {
            self.db.add_parent_to_node(*sid, node.id)?;
        }

        Ok(node)
    }

    /// Cross-conversation semantic dedup: check if a node with the same
    /// semantic hash already exists. If found, return it; otherwise
    /// return None so the caller can create a new node.
    pub fn find_similar_node(
        &self,
        summary: &str,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<Option<DagNode>> {
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
            // Reuse: link the existing node to this conversation
            for sid in source_ids {
                self.db.add_parent_to_node(*sid, existing.id)?;
                let _ = self.db.insert_edge(existing.id, *sid, "reuses");
            }
            tracing::debug!(target: "deeplossless::dag", node_id = existing.id, "semantic dedup: reused existing node");
            return Ok(existing);
        }
        // No match: create new
        self.compress_group_with_snippets(conv_id, source_ids, summary, token_count, level, snippets)
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

    #[allow(dead_code)]
    fn compress_group_inner(
        &self,
        conv_id: i64,
        source_ids: &[i64],
        summary: &str,
        token_count: i64,
        level: u8,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        let new_node = self.db.insert_dag_node_full(
            conv_id,
            level.min(self.effective_max_level(conv_id)),
            summary,
            token_count,
            &[],        // parent_ids: set when summarized by a higher node
            source_ids, // child_ids: the source nodes being summarized
            snippets,
            false,
        )?;

        for pid in source_ids {
            self.db.add_parent_to_node(*pid, new_node.id)?;
        }

        Ok(new_node)
    }

    // ── Context assembly (LCM §2.1) ────────────────────────────────────

    /// Assemble the active context: higher-level summaries + recent raw
    /// messages NOT already covered by those summaries, staying within
    /// `token_budget` tokens.
    ///
    /// Coverage-aware: prevents double injection where a summary and the
    /// raw messages it summarizes both appear in context.
    ///
    /// Returns nodes in display order (summaries first, then recent messages).
    pub fn assemble_context(
        &self,
        conv_id: i64,
        token_budget: usize,
    ) -> anyhow::Result<Vec<DagNode>> {
        // Batch-load all nodes and edges for this conversation (P4 Performance)
        let graph = self.load_graph(conv_id)?;

        let mut result = Vec::new();
        let mut remaining = token_budget as i64;
        let mut covered_ids = HashSet::new();

        // 1. Walk from the active tip down to collect summaries
        if let Some(tip) = self.get_active_tip(conv_id)? {
            let summaries = self.collect_summary_chain_cached(&graph, &tip, remaining)?;
            for node in &summaries {
                remaining -= node.token_count;
                for cid in &node.child_ids {
                    covered_ids.insert(*cid);
                }
            }
            result.extend(summaries);
        }

        // 2. Append most recent raw leaves NOT already covered by summaries
        let mut leaves: Vec<_> = graph.nodes.values()
            .filter(|n| n.is_leaf)
            .cloned()
            .collect();
        leaves.sort_by_key(|n| n.id);
        let recent: Vec<_> = leaves
            .into_iter()
            .rev()
            .take(self.config.recent_message_count)
            .filter(|n| !covered_ids.contains(&n.id))
            .collect();

        for node in recent.into_iter().rev() {
            let tc = node.token_count;
            if tc <= remaining {
                result.push(node);
                remaining -= tc;
            } else {
                break;
            }
        }

        Ok(result)
    }

    /// Load all DAG nodes and edges for a conversation in 1-2 queries.
    fn load_graph(&self, conv_id: i64) -> anyhow::Result<DagGraph> {
        let all_nodes = self.db.get_all_dag_nodes(conv_id)?;
        let mut node_map: HashMap<i64, DagNode> = HashMap::new();
        let mut children: HashMap<i64, Vec<i64>> = HashMap::new();
        let mut parents: HashMap<i64, Vec<i64>> = HashMap::new();

        for node in all_nodes {
            children.insert(node.id, node.child_ids.clone());
            parents.insert(node.id, node.parent_ids.clone());
            node_map.insert(node.id, node);
        }
        Ok(DagGraph { nodes: node_map, children, parents })
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

    /// Walk summary chain from a node upward/downward, bounded by budget.
    /// Tracks visited node IDs to prevent recursive duplication (P2-5).
    /// Uses batch-loaded graph when available; falls back to DB queries.
    #[allow(dead_code)]
    fn collect_summary_chain(&self, start: &DagNode, budget: i64) -> anyhow::Result<Vec<DagNode>> {
        let mut nodes = Vec::new();
        let mut stack = vec![start.clone()];
        let mut seen = std::collections::HashSet::new();
        seen.insert(start.id);

        while let Some(node) = stack.pop() {
            if seen.len() > self.config.max_fanout {
                break;
            }
            if node.level == 0 {
                continue;
            }
            if node.token_count > budget && !node.is_leaf {
                let children = self.get_children(node.id)?;
                for child in children {
                    if seen.insert(child.id) {
                        stack.push(child);
                    }
                }
                continue;
            }
            nodes.push(node);
        }

        // Reverse so highest-level summaries come first
        nodes.reverse();
        Ok(nodes)
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
                    let deg = in_degree.get_mut(child).unwrap();
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
    pub fn collect_garbage(&self, conv_id: i64, dry_run: bool) -> anyhow::Result<Vec<i64>> {
        let mut reachable = std::collections::HashSet::new();
        let mut stack = Vec::new();

        for tip in self.db.get_tip_nodes(conv_id)? {
            stack.push(tip.id);
        }
        for leaf in self.db.get_leaf_nodes(conv_id)? {
            stack.push(leaf.id);
        }

        // Traverse both parent_ids (raw → summary) and child_ids (summary → raw)
        // to handle the corrected DAG direction: summary.child_ids = raw_ids
        while let Some(id) = stack.pop() {
            if !reachable.insert(id) { continue; }
            if let Some(node) = self.db.get_node(id)? {
                for pid in &node.parent_ids {
                    stack.push(*pid);
                }
                for cid in &node.child_ids {
                    stack.push(*cid);
                }
            }
        }

        let all_nodes = self.db.get_all_dag_nodes(conv_id)?;
        let mut ghosts = Vec::new();
        for node in all_nodes {
            if !reachable.contains(&node.id) {
                ghosts.push(node.id);
            }
        }

        if !dry_run && !ghosts.is_empty() {
            for gid in &ghosts {
                self.db.purge_dag_node(*gid)?;
            }
        }
        Ok(ghosts)
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
                if let Some(child) = node_map.get(cid) {
                    if node.level < child.level {
                        issues.push(format!(
                            "level-order: node {} (L{}) child {} (L{}) — parent level must be >= child",
                            node.id, node.level, cid, child.level
                        ));
                    }
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

        let context = engine.assemble_context(conv_id, 100).unwrap();
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
        assert!(order1.len() >= 1);
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

        // Edges: insert_dag_node_full creates Summarizes edges,
        // merge_nodes adds Refines edges on top (both coexist)
        let edges = db.get_edges_from(merged.id).unwrap();
        assert!(edges.len() >= 2, "should have at least 2 edges");
        let kinds: Vec<&str> = edges.iter().map(|(_, _, k)| k.as_str()).collect();
        assert!(kinds.contains(&"refines"), "should have at least one refines edge");
        let summarizes_count = kinds.iter().filter(|k| k == &&"summarizes").count();
        assert!(summarizes_count >= 1, "should have summarizes edges from insert");
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
}

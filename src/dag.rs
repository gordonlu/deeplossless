use serde::{Deserialize, Serialize};
use std::sync::Arc;

use crate::db::Database;

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

    /// Maximum DAG depth (levels). Default: 3 (LCM §2.3 escalation).
    pub max_level: u8,

    /// Max children per node to traverse. Prevents degenerate DAG
    /// traversal cost from hitting millisecond-range latency. Default: 100.
    pub max_fanout: usize,

    /// Max depth when expanding a summary node. Default: 10.
    pub max_expand_depth: u8,

    /// Max recent raw messages to keep uncompressed. Default: 20.
    pub recent_message_count: usize,
}

impl Default for DagConfig {
    fn default() -> Self {
        Self {
            soft_threshold_ratio: 0.80,
            hard_threshold_ratio: 0.95,
            max_level: 3,
            max_fanout: 100,
            max_expand_depth: 10,
            recent_message_count: 20,
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

    // ── Query ──────────────────────────────────────────────────────────

    pub fn get_node(&self, node_id: i64) -> anyhow::Result<Option<DagNode>> {
        self.db.get_node(node_id)
    }

    pub fn get_children(&self, node_id: i64) -> anyhow::Result<Vec<DagNode>> {
        self.db.get_child_nodes(node_id, self.config.max_fanout)
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
        parent_ids: &[i64],
        summary: &str,
        token_count: i64,
        level: u8,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        let new_node = self.db.insert_dag_node_full(
            conv_id,
            level.min(self.config.max_level),
            summary,
            token_count,
            parent_ids,
            &[],
            snippets,
            false,
        )?;
        for pid in parent_ids {
            self.db.add_child_to_node(*pid, new_node.id)?;
        }
        Ok(new_node)
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

    #[allow(dead_code)]
    fn compress_group_inner(
        &self,
        conv_id: i64,
        parent_ids: &[i64],
        summary: &str,
        token_count: i64,
        level: u8,
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        let new_node = self.db.insert_dag_node_full(
            conv_id,
            level.min(self.config.max_level),
            summary,
            token_count,
            parent_ids,
            &[],
            snippets,
            false,
        )?;

        // Back-link each parent to this new node
        for pid in parent_ids {
            self.db.add_child_to_node(*pid, new_node.id)?;
        }

        Ok(new_node)
    }

    // ── Context assembly (LCM §2.1) ────────────────────────────────────

    /// Assemble the active context: recent raw messages + higher-level
    /// summaries, staying within `token_budget` tokens.
    ///
    /// Returns nodes in display order (summaries first, then recent messages).
    pub fn assemble_context(
        &self,
        conv_id: i64,
        token_budget: usize,
    ) -> anyhow::Result<Vec<DagNode>> {
        let mut result = Vec::new();
        let mut remaining = token_budget as i64;

        // 1. Walk from the active tip down to collect summaries
        if let Some(tip) = self.get_active_tip(conv_id)? {
            let summaries = self.collect_summary_chain(&tip, remaining)?;
            for node in &summaries {
                remaining -= node.token_count;
            }
            result.extend(summaries);
        }

        // 2. Append most recent raw leaves (up to budget)
        let leaves = self.get_leaves(conv_id)?;
        let recent: Vec<_> = leaves
            .into_iter()
            .rev()
            .take(self.config.recent_message_count)
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

    /// Walk up from a node following parent links (breadth-first, bounded).
    fn collect_summary_chain(&self, start: &DagNode, budget: i64) -> anyhow::Result<Vec<DagNode>> {
        let mut nodes = Vec::new();
        let mut stack = vec![start.clone()];
        let mut visited = 0;

        while let Some(node) = stack.pop() {
            visited += 1;
            if visited > self.config.max_fanout {
                break;
            }
            if node.level == 0 {
                continue;
            }
            if node.token_count > budget && !node.is_leaf {
                // Too expensive — expand children instead
                let children = self.get_children(node.id)?;
                stack.extend(children);
                continue;
            }
            nodes.push(node);
        }

        // Reverse so highest-level summaries come first
        nodes.reverse();
        Ok(nodes)
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

        while let Some(id) = stack.pop() {
            if !reachable.insert(id) { continue; }
            if let Some(node) = self.db.get_node(id)? {
                for pid in node.parent_ids {
                    stack.push(pid);
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
                self.db.delete_dag_node(*gid)?;
            }
        }
        Ok(ghosts)
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
            let label = format!(
                "{} {} ({})",
                if node.is_leaf { "L" } else { "S" },
                html_escape(&node.summary, 80),
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

fn html_escape(text: &str, max_chars: usize) -> String {
    let mut out = String::new();
    for ch in text.chars().take(max_chars) {
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
    if text.len() > max_chars { out.push('…'); }
    out
}

/// Return true when a message content looks like a tool result.
fn is_tool_result(content: &str) -> bool {
    content.contains("tool_result") || content.contains("ToolResult")
}

/// Return true when a message content looks like a tool call.
fn is_tool_use(content: &str) -> bool {
    content.contains("tool_use") || content.contains("ToolUse")
}

/// Find safe split points in a slice of `(role, content)` pairs for
/// compression grouping.  Splitting at these points guarantees that:
///
/// - No `tool_use` is separated from its corresponding `tool_result`
/// - Group boundaries fall at clean turn boundaries
///
/// Returns indices (into `pairs`) that are *safe* split positions
/// (everything before the index can be compressed as a complete unit).
pub fn safe_split_points(pairs: &[(String, String)]) -> Vec<usize> {
    // First pass: mark indices that are inside a tool chain
    let mut inside_tool_chain = vec![false; pairs.len()];
    for i in 0..pairs.len() {
        if pairs[i].0 == "assistant" && is_tool_use(&pairs[i].1) {
            inside_tool_chain[i] = true;
            let mut j = i + 1;
            while j < pairs.len() && pairs[j].0 == "user" && is_tool_result(&pairs[j].1) {
                inside_tool_chain[j] = true;
                j += 1;
            }
            // Also mark the next non-tool user message if it's clearly
            // a continuation (starts the next turn)
        }
    }

    // Second pass: user messages that are NOT inside a tool chain are safe splits
    let mut safe = Vec::new();
    safe.push(0); // always safe before the first message
    for i in 0..pairs.len() {
        if pairs[i].0 == "user" && !inside_tool_chain[i] {
            safe.push(i);
        }
    }
    safe.sort();
    safe.dedup();
    safe
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
    fn insert_and_retrieve_leaf() {
        let (db, conv_id) = setup_db();
        let engine = DagEngine::builder().build(db);
        let leaf = engine
            .insert_leaf(conv_id, "user: hello\nassistant: hi", 12)
            .unwrap();
        assert_eq!(leaf.level, 0);
        assert!(leaf.is_leaf);
        assert_eq!(leaf.token_count, 12);

        let fetched = engine.get_node(leaf.id).unwrap().unwrap();
        assert_eq!(fetched.summary, leaf.summary);
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

        // Parents should be back-linked
        let children_a = engine.get_children(a.id).unwrap();
        assert_eq!(children_a.len(), 1);
        assert_eq!(children_a[0].id, summary.id);
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

    #[test]
    fn safe_split_simple_turns() {
        let pairs = vec![
            ("user".into(), "hello".into()),
            ("assistant".into(), "hi".into()),
            ("user".into(), "what is rust?".into()),
            ("assistant".into(), "a language".into()),
        ];
        let points = safe_split_points(&pairs);
        // Should split before each user message: index 0 and 2
        assert!(points.contains(&0), "should split at start");
        assert!(points.contains(&2), "should split before second user msg");
    }

    #[test]
    fn safe_split_keeps_tool_pairs_together() {
        let pairs = vec![
            ("user".into(), "read src/main.rs".into()),
            ("assistant".into(), "tool_use: read_file".into()),
            ("user".into(), "tool_result: pub fn main()".into()),
            ("user".into(), "now fix it".into()),
        ];
        let points = safe_split_points(&pairs);
        // 0: before first user ✓
        // after tool_use (index 1), fast-forward past tool_result (index 2)
        // → safe at index 3 (after the tool chain)
        assert!(points.contains(&0), "should split at message 0");
        assert!(!points.contains(&2), "should NOT split inside tool chain");
        assert!(points.contains(&3), "should split after tool chain");
    }

    #[test]
    fn safe_split_empty() {
        let points = safe_split_points(&[]);
        assert_eq!(points, vec![0], "index 0 is always safe");
    }
}

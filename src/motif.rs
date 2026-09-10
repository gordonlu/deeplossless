//! Execution Motif Extraction — detects recurring tool chain patterns
//! from execution histories.
//!
//! A motif is a frequently repeated sequence of tool calls (e.g.
//! `grep → read_file → edit_file → cargo check`).  Motifs enable
//! workflow reuse, planner bias, and execution compilation (Phase 4).

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet};

// ── Core types ─────────────────────────────────────────────────────────

/// A detected execution motif — a recurring tool chain pattern.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ExecutionMotif {
    /// Unique motif identifier (0 until persisted).
    pub id: i64,
    /// Ordered tool names forming the pattern.
    pub tool_chain: Vec<String>,
    /// Canonical key for dedup: dot-joined tool names.
    pub pattern_key: String,
    /// Number of times this motif was observed.
    pub occurrence_count: usize,
    /// Ratio of successful outcomes (0.0–1.0).
    pub success_rate: f64,
    /// Average token cost (sum of all unit tokens / count).
    pub avg_token_cost: f64,
    /// Confidence score (0.0–1.0), based on occurrence_count relative to total.
    pub confidence: f64,
    /// Execution unit IDs matching this motif.
    pub execution_unit_ids: Vec<i64>,
    /// Conversation IDs where this motif appeared.
    pub conversation_ids: Vec<i64>,
    /// ISO-8601 timestamp of most recent occurrence (P1: decay/aging).
    #[serde(default)]
    pub last_seen_at: String,
    /// Recency weight (0.0-1.0): newer motifs score higher. Computed as
    /// 1.0 / (1 + age_days) decay function.
    #[serde(default)]
    pub recency_weight: f64,
}

impl ExecutionMotif {
    /// Compute recency weight from the last seen timestamp and a reference
    /// "now" timestamp (both ISO-8601). Returns 1.0 if timestamps are invalid.
    pub fn compute_recency(&mut self, now: &str) {
        let age_days = parse_age_days(&self.last_seen_at, now);
        self.recency_weight = 1.0 / (1.0 + age_days);
    }
}

/// Parse the age in days between two ISO-8601 timestamps.
fn parse_age_days(seen: &str, now: &str) -> f64 {
    if seen.is_empty() { return 0.0; }
    // Simple parse: take the date portion (YYYY-MM-DD)
    let parse_date = |s: &str| -> Option<(i32, u32, u32)> {
        let parts: Vec<&str> = s.split('T').next()?.split('-').collect();
        if parts.len() != 3 { return None; }
        Some((parts[0].parse().ok()?, parts[1].parse().ok()?, parts[2].parse().ok()?))
    };
    let (sy, sm, sd) = match parse_date(seen) { Some(v) => v, None => return 0.0 };
    let (ny, nm, nd) = match parse_date(now) { Some(v) => v, None => return 0.0 };
    let seen_days = sy as f64 * 365.0 + sm as f64 * 30.0 + sd as f64;
    let now_days = ny as f64 * 365.0 + nm as f64 * 30.0 + nd as f64;
    (now_days - seen_days).max(0.0)
}

/// Configuration for motif extraction.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct MotifConfig {
    /// Minimum n-gram length (tool chain length).
    pub min_ngram: usize,
    /// Maximum n-gram length.
    pub max_ngram: usize,
    /// Minimum occurrences to qualify as a motif.
    pub min_occurrences: usize,
    /// Minimum success rate to keep a motif (0.0 = keep all).
    pub min_success_rate: f64,
    /// Deduplicate consecutive identical tool calls before pattern detection.
    pub dedup_consecutive: bool,
}

impl Default for MotifConfig {
    fn default() -> Self {
        Self {
            min_ngram: 2,
            max_ngram: 4,
            min_occurrences: 2,
            min_success_rate: 0.0,
            dedup_consecutive: true,
        }
    }
}

// ── Motif extractor ────────────────────────────────────────────────────

/// Detects execution motifs from a stream of execution units.
pub struct MotifExtractor {
    pub config: MotifConfig,
}

impl MotifExtractor {
    pub fn new(config: MotifConfig) -> Self {
        Self { config }
    }

    /// Extract motifs from execution units grouped by conversation.
    ///
    /// `unit_groups`: map from conversation_id to ordered list of execution units.
    /// Returns a list of unique motifs sorted by occurrence_count descending.
    pub fn extract(
        &self,
        unit_groups: &HashMap<i64, Vec<crate::execution::ExecutionUnit>>,
    ) -> Vec<ExecutionMotif> {
        if unit_groups.is_empty() {
            return vec![];
        }

        // Step 1: Build tool sequences per conversation (id, tool_name)
        let mut tool_sequences: HashMap<i64, Vec<(i64, String)>> = HashMap::new();

        for (&conv_id, units) in unit_groups {
            let mut seq = self.to_tool_sequence(units);
            if self.config.dedup_consecutive {
                seq = self.dedup_consecutive_seq(&seq);
            }
            tool_sequences.insert(conv_id, seq);
        }

        // Step 2: Generate n-grams and count occurrences.
        // Uses joined string key instead of Vec<String> to avoid per-window
        // allocation (P0-15). Null (\u{1}) separator prevents collision.
        let mut ngram_counts: HashMap<String, NgramStats> = HashMap::new();

        for (&conv_id, seq) in &tool_sequences {
            if seq.len() < self.config.min_ngram {
                continue;
            }
            for n in self.config.min_ngram..=self.config.max_ngram.min(seq.len()) {
                for window in seq.windows(n) {
                    let tool_names: Vec<&str> = window.iter().map(|(_, t)| t.as_str()).collect();
                    if tool_names.iter().any(|t| t.is_empty()) {
                        continue;
                    }
                    let key = tool_names.join("\u{1}");
                    let entry = ngram_counts.entry(key).or_insert_with(|| NgramStats {
                        occurrence_count: 0,
                        success_count: 0,
                        total_tokens: 0,
                        execution_unit_ids: HashSet::new(),
                        conversation_ids: HashSet::new(),
                    });
                    entry.occurrence_count += 1;
                    entry.conversation_ids.insert(conv_id);

                    // Collect unit IDs via HashSet to avoid duplicates (P0-11)
                    for (id, _) in window {
                        entry.execution_unit_ids.insert(*id);
                    }
                }
            }
        }

        // Step 3: Hoist unit_map — build once, not per-motif (P0-1)
        let unit_map: HashMap<i64, &crate::execution::ExecutionUnit> = unit_groups
            .values()
            .flatten()
            .map(|u| (u.id, u))
            .collect();

        for stats in ngram_counts.values_mut() {
            let mut success_count = 0usize;
            let mut total_tokens = 0i64;

            for &uid in &stats.execution_unit_ids {
                if let Some(u) = unit_map.get(&uid) {
                    if matches!(
                        u.outcome,
                        crate::execution::ExecutionOutcome::Success
                            | crate::execution::ExecutionOutcome::CacheHit
                    ) {
                        success_count += 1;
                    }
                    total_tokens += crate::tokenizer::count(&u.tool_result) as i64;
                }
            }
            stats.success_count = success_count;
            stats.total_tokens = total_tokens;
        }

        // Step 4: Filter and build motifs
        let total_occurrences: usize = ngram_counts.values().map(|s| s.occurrence_count).sum();
        let mut motifs = Vec::new();

        for (key, stats) in ngram_counts {
            if stats.occurrence_count < self.config.min_occurrences {
                continue;
            }
            // Success rate: ratio of successful units within motif matches (P0-10)
            let success_rate = if !stats.execution_unit_ids.is_empty() {
                stats.success_count as f64 / stats.execution_unit_ids.len() as f64
            } else {
                0.0
            };
            if success_rate < self.config.min_success_rate {
                continue;
            }

            let avg_token_cost = if stats.occurrence_count > 0 {
                stats.total_tokens as f64 / stats.occurrence_count as f64
            } else {
                0.0
            };

            // Confidence: normalized by total occurrences, not total units (P0-12)
            let confidence = (stats.occurrence_count as f64 / total_occurrences.max(1) as f64)
                .min(1.0);

            let mut conv_ids: Vec<i64> = stats.conversation_ids.into_iter().collect();
            conv_ids.sort();

            let mut eids: Vec<i64> = stats.execution_unit_ids.into_iter().collect();
            eids.sort();

            let (pattern_key, chain) = pattern_key_from_chain(&key);

            motifs.push(ExecutionMotif {
                id: 0,
                pattern_key,
                tool_chain: chain,
                occurrence_count: stats.occurrence_count,
                success_rate,
                avg_token_cost,
                confidence,
                execution_unit_ids: eids,
                conversation_ids: conv_ids,
                last_seen_at: chrono::Utc::now().to_rfc3339(),
                recency_weight: 1.0,
            });
        }

        // Sort by occurrence count descending
        motifs.sort_by_key(|b| std::cmp::Reverse(b.occurrence_count));
        motifs
    }

    /// Convert a slice of execution units to (id, tool_name) pairs.
    fn to_tool_sequence(&self, units: &[crate::execution::ExecutionUnit]) -> Vec<(i64, String)> {
        units.iter().map(|u| (u.id, canonicalize_tool_name(&u.tool_name))).collect()
    }

    /// Deduplicate consecutive identical tool names, keeping the first unit ID.
    fn dedup_consecutive_seq(&self, seq: &[(i64, String)]) -> Vec<(i64, String)> {
        let mut result = Vec::new();
        for item in seq {
            if result.last().map(|(_, t)| t) != Some(&item.1) {
                result.push(item.clone());
            }
        }
        result
    }
}

/// Convert a joined pattern key back to a tool chain Vec.
fn pattern_key_from_chain(key: &str) -> (String, Vec<String>) {
    let chain: Vec<String> = key.split('\u{1}').map(|s| s.to_string()).collect();
    (key.to_string(), chain)
}

/// Canonicalize tool name for stable motif identity across provider aliases (P1).
/// Collapses known synonyms to a single canonical form.
fn canonicalize_tool_name(name: &str) -> String {
    match name {
        "search_content" | "grep" | "rg" | "search" => "grep".into(),
        "read_file" | "read" | "cat" | "view" => "read_file".into(),
        "write_to_file" | "write_file" | "edit" | "edit_file" => "edit_file".into(),
        "execute_command" | "exec" | "bash" | "run" => "execute_command".into(),
        "list_files" | "ls" | "dir" | "glob" => "list_files".into(),
        other => other.to_string(),
    }
}

// ── Internal stats accumulator ─────────────────────────────────────────

struct NgramStats {
    occurrence_count: usize,
    success_count: usize,
    total_tokens: i64,
    execution_unit_ids: HashSet<i64>,
    conversation_ids: HashSet<i64>,
}

// ── Top-level extraction helpers ───────────────────────────────────────

/// Convenience: extract motifs from the database for a list of conversation IDs.
pub fn extract_motifs_from_db(
    db: &crate::db::Database,
    conv_ids: &[i64],
    config: Option<MotifConfig>,
) -> anyhow::Result<Vec<ExecutionMotif>> {
    let extractor = MotifExtractor::new(config.unwrap_or_default());
    let mut unit_groups: HashMap<i64, Vec<crate::execution::ExecutionUnit>> = HashMap::new();
    for &conv_id in conv_ids {
        if let Ok(units) = db.get_execution_units(conv_id, 500) {
            if !units.is_empty() {
                unit_groups.insert(conv_id, units);
            }
        }
    }
    Ok(extractor.extract(&unit_groups))
}

/// Find execution unit IDs that match a given motif pattern.
/// Returns unit IDs instead of cloned units to avoid large allocations (P1-13).
pub fn find_motif_matches(
    units: &[crate::execution::ExecutionUnit],
    motif: &ExecutionMotif,
) -> Vec<i64> {
    let extractor = MotifExtractor::new(MotifConfig::default());
    let seq = extractor.to_tool_sequence(units);

    if seq.len() < motif.tool_chain.len() {
        return vec![];
    }

    let chain = &motif.tool_chain;
    let seq_tools: Vec<&str> = seq.iter().map(|(_, t)| t.as_str()).collect();
    let chain_strs: Vec<&str> = chain.iter().map(|s| s.as_str()).collect();
    let mut matched_ids = HashSet::new();

    for start in 0..=seq_tools.len().saturating_sub(chain.len()) {
        if &seq_tools[start..start + chain.len()] == chain_strs.as_slice() {
            for i in start..start + chain.len() {
                if let Some(unit) = units.get(i) {
                    matched_ids.insert(unit.id);
                }
            }
        }
    }

    let mut result: Vec<i64> = matched_ids.into_iter().collect();
    result.sort();
    result
}

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use crate::execution::{ExecutionOutcome, ExecutionUnit};

    fn make_unit(id: i64, conv_id: i64, tool: &str, outcome: ExecutionOutcome, tokens: usize) -> ExecutionUnit {
        ExecutionUnit {
            id,
            conversation_id: conv_id,
            reasoning_before: String::new(),
            tool_name: tool.into(),
            tool_args: "{}".into(),
            tool_result: "x".repeat(tokens * 4),
            reasoning_after: String::new(),
            outcome,
            related_nodes: vec![],
            created_at: String::new(),
            tool_args_json: None,
            reasoning_steps: vec![],
            ..Default::default()
        }
    }

    #[test]
    fn empty_units_produces_empty_motifs() {
        let extractor = MotifExtractor::new(MotifConfig::default());
        let groups = HashMap::new();
        let motifs = extractor.extract(&groups);
        assert!(motifs.is_empty());
    }

    #[test]
    fn single_chain_no_motif() {
        let extractor = MotifExtractor::new(MotifConfig::default());
        let units = vec![
            make_unit(1, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(2, 1, "read_file", ExecutionOutcome::Success, 20),
        ];
        let mut groups = HashMap::new();
        groups.insert(1, units);
        let motifs = extractor.extract(&groups);
        // Single occurrence of any n-gram, min_occurrences=2 → none
        assert!(motifs.is_empty());
    }

    #[test]
    fn repeated_pattern_detected() {
        let extractor = MotifExtractor::new(MotifConfig {
            min_occurrences: 2,
            dedup_consecutive: true,
            ..Default::default()
        });

        // Pattern: grep → read_file appears twice
        let mut groups = HashMap::new();
        groups.insert(1, vec![
            make_unit(1, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(2, 1, "read_file", ExecutionOutcome::Success, 20),
            make_unit(3, 1, "edit_file", ExecutionOutcome::Success, 30),
            make_unit(4, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(5, 1, "read_file", ExecutionOutcome::Success, 20),
            make_unit(6, 1, "build", ExecutionOutcome::Success, 40),
        ]);

        let motifs = extractor.extract(&groups);
        let grep_read: Vec<&ExecutionMotif> = motifs.iter()
            .filter(|m| m.tool_chain == vec!["grep", "read_file"])
            .collect();
        assert!(!grep_read.is_empty(), "grep→read_file should be detected");
        if let Some(m) = grep_read.first() {
            assert!(m.occurrence_count >= 2);
            assert!(m.confidence > 0.0);
        }
    }

    #[test]
    fn motif_success_rate_computed() {
        let extractor = MotifExtractor::new(MotifConfig {
            min_occurrences: 1,
            ..Default::default()
        });

        let mut groups = HashMap::new();
        groups.insert(1, vec![
            make_unit(1, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(2, 1, "build", ExecutionOutcome::Blocked, 100),
        ]);

        let motifs = extractor.extract(&groups);
        // grep→build appears once
        let gb: Vec<&ExecutionMotif> = motifs.iter()
            .filter(|m| m.tool_chain == vec!["grep", "build"])
            .collect();
        if let Some(m) = gb.first() {
            assert!(m.success_rate < 1.0, "half the chain failed");
        }
    }

    #[test]
    fn dedup_compresses_consecutive_identical() {
        let extractor = MotifExtractor::new(MotifConfig::default());
        let seq: Vec<(i64, String)> = vec![
            (1, "grep".into()), (2, "grep".into()), (3, "read_file".into()),
        ];
        let deduped = extractor.dedup_consecutive_seq(&seq);
        assert_eq!(deduped.len(), 2);
        assert_eq!(deduped[0].1, "grep");
        assert_eq!(deduped[1].1, "read_file");
    }

    #[test]
    fn across_conversations_aggregated() {
        let extractor = MotifExtractor::new(MotifConfig {
            min_occurrences: 2,
            ..Default::default()
        });

        // Same pattern in two conversations
        let mut groups = HashMap::new();
        groups.insert(1, vec![
            make_unit(1, 1, "list_files", ExecutionOutcome::Success, 5),
            make_unit(2, 1, "grep", ExecutionOutcome::Success, 10),
        ]);
        groups.insert(2, vec![
            make_unit(3, 2, "list_files", ExecutionOutcome::Success, 5),
            make_unit(4, 2, "grep", ExecutionOutcome::Success, 10),
        ]);

        let motifs = extractor.extract(&groups);
        let lf: Vec<&ExecutionMotif> = motifs.iter()
            .filter(|m| m.tool_chain == vec!["list_files", "grep"])
            .collect();
        assert!(!lf.is_empty(), "cross-conversation pattern should be detected");
        if let Some(m) = lf.first() {
            assert_eq!(m.occurrence_count, 2);
            assert_eq!(m.conversation_ids.len(), 2);
        }
    }

    #[test]
    fn find_motif_matches_returns_correct_units() {
        let motif = ExecutionMotif {
            id: 1,
            tool_chain: vec!["grep".into(), "read_file".into()],
            pattern_key: "grep.read_file".into(),
            occurrence_count: 2,
            success_rate: 1.0,
            avg_token_cost: 15.0,
            confidence: 0.5,
            execution_unit_ids: vec![],
            conversation_ids: vec![1],
            last_seen_at: String::new(),
            recency_weight: 1.0,
        };

        let units = vec![
            make_unit(1, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(2, 1, "read_file", ExecutionOutcome::Success, 20),
            make_unit(3, 1, "edit_file", ExecutionOutcome::Success, 30),
            make_unit(4, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(5, 1, "read_file", ExecutionOutcome::Success, 20), // second match
        ];

        let matched = find_motif_matches(&units, &motif);
        assert_eq!(matched.len(), 4, "two motif matches × 2 unique unit IDs each = 4");
    }

    #[test]
    fn motif_config_defaults_are_reasonable() {
        let cfg = MotifConfig::default();
        assert_eq!(cfg.min_ngram, 2);
        assert_eq!(cfg.max_ngram, 4);
        assert_eq!(cfg.min_occurrences, 2);
        assert!(cfg.dedup_consecutive);
    }

    #[test]
    fn ngram_filter_respects_min_occurrences() {
        let extractor = MotifExtractor::new(MotifConfig {
            min_occurrences: 5,
            ..Default::default()
        });

        let mut groups = HashMap::new();
        groups.insert(1, vec![
            make_unit(1, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(2, 1, "build", ExecutionOutcome::Success, 10),
        ]);

        let motifs = extractor.extract(&groups);
        assert!(motifs.is_empty(), "only 1 occurrence, min is 5");
    }
}

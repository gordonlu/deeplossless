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

        // Step 2: Generate n-grams and count occurrences
        let mut ngram_counts: HashMap<Vec<String>, NgramStats> = HashMap::new();

        for (&conv_id, seq) in &tool_sequences {
            if seq.len() < self.config.min_ngram {
                continue;
            }
            for n in self.config.min_ngram..=self.config.max_ngram.min(seq.len()) {
                for window in seq.windows(n) {
                    let key: Vec<String> = window.iter().map(|(_, t)| t.clone()).collect();
                    if key.iter().any(|t| t.is_empty()) {
                        continue;
                    }
                    let entry = ngram_counts.entry(key).or_insert_with(|| NgramStats {
                        occurrence_count: 0,
                        success_count: 0,
                        total_tokens: 0,
                        execution_unit_ids: vec![],
                        conversation_ids: HashSet::new(),
                    });
                    entry.occurrence_count += 1;
                    entry.conversation_ids.insert(conv_id);

                    // Take unit IDs directly from the current window
                    let matched_ids: Vec<i64> = window.iter().map(|(id, _)| *id).collect();
                    entry.execution_unit_ids.extend(matched_ids);
                }
            }
        }

        // Step 3: Count successes and tokens per motif
        for stats in ngram_counts.values_mut() {
            // Get unit outcomes from the IDs
            let mut success_count = 0usize;
            let mut total_tokens = 0i64;

            // Build a map of unit_id → unit for quick lookup
            let unit_map: HashMap<i64, &crate::execution::ExecutionUnit> = unit_groups
                .values()
                .flatten()
                .map(|u| (u.id, u))
                .collect();

            for &uid in &stats.execution_unit_ids {
                if let Some(u) = unit_map.get(&uid) {
                    if matches!(
                        u.outcome,
                        crate::execution::ExecutionOutcome::Success
                            | crate::execution::ExecutionOutcome::CacheHit
                    ) {
                        success_count += 1;
                    }
                    // Rough token estimate from tool result length
                    total_tokens += (u.tool_result.len() / 4) as i64;
                }
            }
            stats.success_count = success_count;
            stats.total_tokens = total_tokens;
        }

        // Step 4: Filter and build motifs
        let total_units: usize = unit_groups.values().map(|v| v.len()).sum();
        let mut motifs = Vec::new();

        for (key, stats) in ngram_counts {
            if stats.occurrence_count < self.config.min_occurrences {
                continue;
            }
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

            let confidence = (stats.occurrence_count as f64 / total_units.max(1) as f64)
                .min(1.0);

            let mut conv_ids: Vec<i64> = stats.conversation_ids.into_iter().collect();
            conv_ids.sort();

            let mut eids = stats.execution_unit_ids.clone();
            eids.sort();
            eids.dedup();

            motifs.push(ExecutionMotif {
                id: 0,
                pattern_key: key.join("."),
                tool_chain: key,
                occurrence_count: stats.occurrence_count,
                success_rate,
                avg_token_cost,
                confidence,
                execution_unit_ids: eids,
                conversation_ids: conv_ids,
            });
        }

        // Sort by occurrence count descending
        motifs.sort_by_key(|b| std::cmp::Reverse(b.occurrence_count));
        motifs
    }

    /// Convert a slice of execution units to (id, tool_name) pairs.
    fn to_tool_sequence(&self, units: &[crate::execution::ExecutionUnit]) -> Vec<(i64, String)> {
        units.iter().map(|u| (u.id, u.tool_name.clone())).collect()
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

// ── Internal stats accumulator ─────────────────────────────────────────

struct NgramStats {
    occurrence_count: usize,
    success_count: usize,
    total_tokens: i64,
    execution_unit_ids: Vec<i64>,
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

/// Find execution units that match a given motif pattern.
pub fn find_motif_matches(
    units: &[crate::execution::ExecutionUnit],
    motif: &ExecutionMotif,
) -> Vec<crate::execution::ExecutionUnit> {
    let extractor = MotifExtractor::new(MotifConfig::default());
    let seq = extractor.to_tool_sequence(units);

    if seq.len() < motif.tool_chain.len() {
        return vec![];
    }

    let chain = &motif.tool_chain;
    let seq_tools: Vec<&str> = seq.iter().map(|(_, t)| t.as_str()).collect();
    let chain_strs: Vec<&str> = chain.iter().map(|s| s.as_str()).collect();
    let mut matches = Vec::new();

    for start in 0..=seq_tools.len().saturating_sub(chain.len()) {
        if &seq_tools[start..start + chain.len()] == chain_strs.as_slice() {
            for i in start..start + chain.len() {
                if i < units.len() {
                    matches.push(units[i].clone());
                }
            }
        }
    }

    matches
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
        };

        let units = vec![
            make_unit(1, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(2, 1, "read_file", ExecutionOutcome::Success, 20),
            make_unit(3, 1, "edit_file", ExecutionOutcome::Success, 30),
            make_unit(4, 1, "grep", ExecutionOutcome::Success, 10),
            make_unit(5, 1, "read_file", ExecutionOutcome::Success, 20), // second match
        ];

        let matched = find_motif_matches(&units, &motif);
        assert_eq!(matched.len(), 4, "two motif matches × 2 units each = 4");
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

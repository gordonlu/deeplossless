/// Context working-view selection.
///
/// The base score remains cheap and transport-neutral. `DynamicSignals` lets
/// execution state re-rank old context as the task changes — the useful part of
/// AttnCompress — without a proxy-attention/PPL model.
pub fn knapsack_select(costs: &[u64], values: &[f64], capacity: u64) -> Vec<bool> {
    debug_assert_eq!(costs.len(), values.len(), "costs and values must have equal length");
    let n = costs.len();
    if n == 0 || capacity == 0 {
        return vec![false; n];
    }
    let cap = capacity as usize;
    let mut dp = vec![0.0f64; cap + 1];
    let mut keep = vec![vec![false; cap + 1]; n];
    for i in 0..n {
        let c = costs[i] as usize;
        let v = values[i];
        for w in (c..=cap).rev() {
            let include = dp[w - c] + v;
            if include > dp[w] {
                dp[w] = include;
                keep[i][w] = true;
            }
        }
    }
    let mut selected = vec![false; n];
    let mut w = cap;
    for i in (0..n).rev() {
        if keep[i][w] {
            selected[i] = true;
            w -= costs[i] as usize;
        }
    }
    selected
}

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportanceOrdering {
    Preserve,
    ReverseChronological,
    ByImportance,
    Knapsack,
}

/// Dynamic execution-state evidence used to re-rank a context item.
/// All fields are normalized to 0..=1.
#[derive(Debug, Clone, Copy, Default, PartialEq)]
pub struct DynamicSignals {
    /// Direct dependency of the active plan/current step.
    pub plan_dependency: f64,
    /// Refers to a file/symbol/resource changed in the current execution.
    pub changed_resource: f64,
    /// Participates in the current failure/retry lineage.
    pub failure_lineage: f64,
    /// Contains an exact precision-critical snippet/value.
    pub precision: f64,
    /// Semantic similarity to the current goal/query.
    pub semantic_similarity: f64,
}

impl DynamicSignals {
    pub fn normalized(self) -> Self {
        Self {
            plan_dependency: self.plan_dependency.clamp(0.0, 1.0),
            changed_resource: self.changed_resource.clamp(0.0, 1.0),
            failure_lineage: self.failure_lineage.clamp(0.0, 1.0),
            precision: self.precision.clamp(0.0, 1.0),
            semantic_similarity: self.semantic_similarity.clamp(0.0, 1.0),
        }
    }

    /// State-aware boost. Plan dependency is strongest; stale-looking old
    /// context can therefore become important again when the plan returns to it.
    pub fn boost(self) -> f64 {
        let s = self.normalized();
        0.35 * s.plan_dependency
            + 0.20 * s.changed_resource
            + 0.20 * s.failure_lineage
            + 0.15 * s.precision
            + 0.10 * s.semantic_similarity
    }
}

#[derive(Debug, Clone)]
pub struct ImportanceScore {
    pub recency: f64,
    pub role_weight: f64,
    pub token_count: f64,
    pub dynamic: f64,
    pub total: f64,
}

impl ImportanceScore {
    fn compute(index: usize, total_count: usize, role: &str, text: &str) -> Self {
        let recency = if total_count <= 1 { 1.0 } else { index as f64 / (total_count - 1) as f64 };
        let role_weight = match role {
            "system" => 0.5,
            "user" => 0.8,
            "assistant" => 1.0,
            "tool" => 0.3,
            _ => 0.6,
        };
        let token_est = (text.len() as f64 / 4.0).ceil();
        let token_count = (token_est / 4096.0).min(1.0);
        let base = recency * 0.4 + role_weight * 0.4 + token_count * 0.2;
        Self { recency, role_weight, token_count, dynamic: 0.0, total: base }
    }

    fn apply_dynamic(&mut self, signals: DynamicSignals) {
        self.dynamic = signals.boost();
        // Preserve 40% of the cheap base score and devote 60% to task state.
        // This permits deliberate recall of old but newly relevant evidence.
        let base = self.recency * 0.4 + self.role_weight * 0.4 + self.token_count * 0.2;
        self.total = 0.4 * base + 0.6 * self.dynamic;
    }
}

#[derive(Debug, Clone)]
pub struct ContextMessage {
    pub role: String,
    pub raw: serde_json::Value,
    pub importance: ImportanceScore,
    pub original_index: usize,
}

#[derive(Debug, Clone)]
pub struct ContextPack {
    messages: Vec<ContextMessage>,
}

impl ContextPack {
    pub fn new(raw_messages: &[serde_json::Value]) -> Self {
        let messages = raw_messages.iter().enumerate().map(|(i, msg)| {
            let role = msg["role"].as_str().unwrap_or("user").to_string();
            let content = msg["content"].as_str().unwrap_or("");
            ContextMessage {
                role: role.clone(),
                raw: msg.clone(),
                importance: ImportanceScore::compute(i, raw_messages.len(), &role, content),
                original_index: i,
            }
        }).collect();
        Self { messages }
    }

    /// Re-rank the working view from current execution state. Missing indices
    /// receive no dynamic boost. This can be called again after a tool result,
    /// file mutation, failure, or plan transition.
    pub fn rescore_dynamic(&mut self, signals: &std::collections::HashMap<usize, DynamicSignals>) {
        for message in &mut self.messages {
            message.importance.apply_dynamic(
                signals.get(&message.original_index).copied().unwrap_or_default(),
            );
        }
    }

    pub fn knapsack_select(&self, budget: u64) -> Vec<ContextMessage> {
        let costs: Vec<u64> = self.messages.iter().map(|m| {
            let text = m.raw["content"].as_str().unwrap_or("");
            (text.len() as u64 / 4).max(1)
        }).collect();
        let values: Vec<f64> = self.messages.iter().map(|m| m.importance.total).collect();
        let selected_mask = knapsack_select(&costs, &values, budget);
        self.messages.iter()
            .enumerate()
            .filter(|(i, _)| selected_mask[*i])
            .map(|(_, m)| m.clone())
            .collect()
    }

    pub fn reorder(&mut self, strategy: ImportanceOrdering) {
        match strategy {
            ImportanceOrdering::Preserve => {}
            ImportanceOrdering::ReverseChronological => self.messages.reverse(),
            ImportanceOrdering::ByImportance => {
                self.messages.sort_by(|a, b| {
                    b.importance.total.partial_cmp(&a.importance.total)
                        .unwrap_or(std::cmp::Ordering::Equal)
                });
            }
            ImportanceOrdering::Knapsack => {}
        }
    }

    pub fn into_messages(self) -> Vec<serde_json::Value> {
        self.messages.into_iter().map(|m| m.raw).collect()
    }

    pub fn len(&self) -> usize { self.messages.len() }
    pub fn is_empty(&self) -> bool { self.messages.is_empty() }
}

#[cfg(test)]
mod tests {
    use super::*;
    use std::collections::HashMap;

    fn make_msg(role: &str, content: &str) -> serde_json::Value {
        serde_json::json!({"role": role, "content": content})
    }

    #[test]
    fn preserve_order() {
        let msgs = vec![make_msg("user", "hi"), make_msg("assistant", "hello")];
        let mut pack = ContextPack::new(&msgs);
        pack.reorder(ImportanceOrdering::Preserve);
        let out = pack.into_messages();
        assert_eq!(out[0]["role"], "user");
        assert_eq!(out[1]["role"], "assistant");
    }

    #[test]
    fn reverse_chronological() {
        let msgs = vec![make_msg("user", "first"), make_msg("assistant", "second")];
        let mut pack = ContextPack::new(&msgs);
        pack.reorder(ImportanceOrdering::ReverseChronological);
        let out = pack.into_messages();
        assert_eq!(out[0]["role"], "assistant");
    }

    #[test]
    fn dynamic_plan_dependency_recalls_old_context() {
        let msgs = vec![
            make_msg("tool", "old config observation"),
            make_msg("assistant", "recent generic chatter"),
        ];
        let mut pack = ContextPack::new(&msgs);
        let signals = HashMap::from([(
            0,
            DynamicSignals {
                plan_dependency: 1.0,
                precision: 1.0,
                ..Default::default()
            },
        )]);
        pack.rescore_dynamic(&signals);
        pack.reorder(ImportanceOrdering::ByImportance);
        let out = pack.into_messages();
        assert_eq!(out[0]["content"], "old config observation");
    }

    #[test]
    fn dynamic_signals_are_clamped() {
        let signals = DynamicSignals {
            plan_dependency: 5.0,
            changed_resource: -1.0,
            ..Default::default()
        };
        let normalized = signals.normalized();
        assert_eq!(normalized.plan_dependency, 1.0);
        assert_eq!(normalized.changed_resource, 0.0);
    }

    #[test]
    fn knapsack_empty_input() {
        assert!(knapsack_select(&[], &[], 100).is_empty());
    }

    #[test]
    fn knapsack_selects_highest_value_within_budget() {
        let selected = knapsack_select(&[10, 20, 15], &[5.0, 10.0, 6.0], 25);
        let total_val: f64 = selected.iter().enumerate()
            .filter(|(_, selected)| **selected)
            .map(|(i, _)| [5.0, 10.0, 6.0][i])
            .sum();
        assert!((total_val - 11.0).abs() < 1e-9);
    }

    #[test]
    fn context_pack_knapsack_respects_budget() {
        let msgs = vec![
            make_msg("user", "Fix the build error in Cargo.toml"),
            make_msg("assistant", "Looking at Cargo.toml"),
            make_msg("tool", "serde_json = 1.0"),
        ];
        let pack = ContextPack::new(&msgs);
        assert!(pack.knapsack_select(5).len() < msgs.len());
        assert_eq!(pack.knapsack_select(500).len(), msgs.len());
    }
}

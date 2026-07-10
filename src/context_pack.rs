/// DS4-14: ContextPack importance ordering.
/// Wraps messages with computed importance scores and supports
/// Preserve, ReverseChronological, or ByImportance ordering.

/// 0/1 knapsack selector: given items with value and cost, select the subset
/// within `capacity` that maximizes total value.
///
/// Returns a boolean mask: `true` at index `i` means item `i` is selected.
/// Uses standard DP O(n*capacity) — suitable for token budgets up to ~100k.
pub fn knapsack_select(costs: &[u64], values: &[f64], capacity: u64) -> Vec<bool> {
    debug_assert_eq!(costs.len(), values.len(), "costs and values must have equal length");
    let n = costs.len();
    if n == 0 || capacity == 0 {
        return vec![false; n];
    }
    let cap = capacity as usize;
    // dp[w] = max value achievable with weight ≤ w
    let mut dp = vec![0.0f64; cap + 1];
    // keep[i][w] = whether item i was selected to achieve dp[w] at step i
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

    // Trace back to find which items were selected
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

#[derive(Debug, Clone)]
pub struct ImportanceScore {
    pub recency: f64,
    pub role_weight: f64,
    pub token_count: f64,
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
        let total = recency * 0.4 + role_weight * 0.4 + token_count * 0.2;
        Self { recency, role_weight, token_count, total }
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
        let messages: Vec<ContextMessage> = raw_messages.iter().enumerate().map(|(i, msg)| {
            let role = msg["role"].as_str().unwrap_or("user").to_string();
            let content = msg["content"].as_str().unwrap_or("");
            let importance = ImportanceScore::compute(i, raw_messages.len(), &role, content);
            ContextMessage {
                role,
                raw: msg.clone(),
                importance,
                original_index: i,
            }
        }).collect();
        Self { messages }
    }

    /// Select messages using 0/1 knapsack within a token budget.
    /// Returns the selected messages (subset of self.messages).
    pub fn knapsack_select(&self, budget: u64) -> Vec<ContextMessage> {
        let costs: Vec<u64> = self.messages.iter().map(|m| {
            let text = m.raw["content"].as_str().unwrap_or("");
            (text.len() as u64 / 4).max(1) // rough token estimate
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
            ImportanceOrdering::ReverseChronological => {
                self.messages.reverse();
            }
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
        assert_eq!(out[1]["role"], "user");
    }

    #[test]
    fn by_importance() {
        let msgs = vec![
            make_msg("user", "short"),
            make_msg("assistant", "a detailed response with several words"),
            make_msg("tool", ""),
        ];
        let mut pack = ContextPack::new(&msgs);
        pack.reorder(ImportanceOrdering::ByImportance);
        let out = pack.into_messages();
        // assistant (1.0 weight, mid recency) > tool (0.3 weight, high recency) > user (0.8 weight, low recency)
        assert_eq!(out[0]["role"], "assistant");
        assert_eq!(out[out.len() - 1]["role"], "user");
    }

    #[test]
    fn empty() {
        let pack = ContextPack::new(&[]);
        assert!(pack.is_empty());
    }

    #[test]
    fn single_message() {
        let msgs = vec![make_msg("user", "hi")];
        let mut pack = ContextPack::new(&msgs);
        pack.reorder(ImportanceOrdering::ByImportance);
        assert_eq!(pack.len(), 1);
    }

    // ── Knapsack tests ───────────────────────────────────────────────

    #[test]
    fn knapsack_empty_input() {
        let selected = knapsack_select(&[], &[], 100);
        assert!(selected.is_empty());
    }

    #[test]
    fn knapsack_zero_capacity() {
        let selected = knapsack_select(&[10, 20], &[1.0, 2.0], 0);
        assert_eq!(selected, vec![false, false]);
    }

    #[test]
    fn knapsack_selects_highest_value_within_budget() {
        // Items: (cost, value)
        // A: (10, 5.0), B: (20, 10.0), C: (15, 6.0)
        // Budget 25 → optimal is A+C = 11.0 > B alone = 10.0
        let selected = knapsack_select(&[10, 20, 15], &[5.0, 10.0, 6.0], 25);
        let total_val: f64 = selected.iter().enumerate()
            .filter(|(_, s)| **s).map(|(i, _)| [5.0, 10.0, 6.0][i]).sum();
        assert!((total_val - 11.0).abs() < 1e-9, "optimal value should be 11 (A+C)");
    }

    #[test]
    fn knapsack_prefers_two_items_over_one() {
        // Items: (cost, value)
        // A: (10, 5.0), B: (10, 5.0), C: (20, 9.0)
        // Budget 20 → A+B=10.0 > C=9.0, should pick A+B
        let selected = knapsack_select(&[10, 10, 20], &[5.0, 5.0, 9.0], 20);
        let total_val: f64 = selected.iter().enumerate()
            .filter(|(_, s)| **s).map(|(i, _)| [5.0, 5.0, 9.0][i]).sum();
        assert!((total_val - 10.0).abs() < 1e-9, "optimal value should be 10 (A+B)");
    }

    #[test]
    fn knapsack_fits_exact_capacity() {
        // Items: (cost, value) = [(5,5), (10,10), (15,15)], capacity=15
        // Optimal: either A+B (value 15) or C (value 15)
        let selected = knapsack_select(&[5, 10, 15], &[5.0, 10.0, 15.0], 15);
        let total_val: f64 = selected.iter().enumerate()
            .filter(|(_, s)| **s).map(|(i, _)| [5.0, 10.0, 15.0][i]).sum();
        assert!((total_val - 15.0).abs() < 1e-9, "optimal value should be 15");
        // Must not exceed capacity
        let total_cost: u64 = selected.iter().enumerate()
            .filter(|(_, s)| **s).map(|(i, _)| [5u64, 10, 15][i]).sum();
        assert!(total_cost <= 15, "must not exceed capacity");
    }

    #[test]
    fn knapsack_context_pack_within_budget() {
        let msgs = vec![
            make_msg("system", "You are a coding assistant."),
            make_msg("user", "Fix the build error in Cargo.toml"),
            make_msg("assistant", "Looking at the Cargo.toml, I see the issue."),
            make_msg("tool", "grep result: serde_json = \"1.0\""),
            make_msg("user", "Yes, that's the file."),
        ];
        let pack = ContextPack::new(&msgs);

        // Budget 5 tokens — only the most important messages fit
        let selected = pack.knapsack_select(5);
        assert!(!selected.is_empty(), "should select at least some messages");
        assert!(selected.len() < msgs.len(), "should not select all messages with tight budget");

        // Budget 500 tokens — should select all
        let all = pack.knapsack_select(500);
        assert_eq!(all.len(), msgs.len(), "generous budget should select all messages");
    }

    #[test]
    fn knapsack_selects_higher_value_message_with_tight_budget() {
        // Two messages of equal token cost (~1 token each)
        // "hi" assistant (0 recency, 1.0 role): total = 0*0.4 + 1.0*0.4 + small = 0.4
        // "ok" tool (1.0 recency, 0.3 role): total = 1.0*0.4 + 0.3*0.4 + small = 0.52
        // tool message has higher total due to recency
        let msgs = vec![
            make_msg("assistant", "hi"),
            make_msg("tool", "ok"),
        ];
        let pack = ContextPack::new(&msgs);
        let selected = pack.knapsack_select(1);
        assert_eq!(selected.len(), 1, "tight budget should pick exactly one");
        assert_eq!(selected[0].role, "tool", "should pick higher-value message");
    }
}

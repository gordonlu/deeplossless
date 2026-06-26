/// DS4-14: ContextPack importance ordering.
/// Wraps messages with computed importance scores and supports
/// Preserve, ReverseChronological, or ByImportance ordering.

#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ImportanceOrdering {
    Preserve,
    ReverseChronological,
    ByImportance,
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
}

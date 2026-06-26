/// DS4-34: Prefix stability checker.
/// Compares successive generated text outputs to detect stable prefixes.
/// Uses manual string scanning — no regex dependency.

#[derive(Debug, Clone)]
pub struct StabilityConfig {
    /// How many characters of prefix to examine for stability.
    pub prefix_char_count: usize,
}

impl Default for StabilityConfig {
    fn default() -> Self {
        Self { prefix_char_count: 100 }
    }
}

#[derive(Debug, Clone, PartialEq)]
pub enum DiffOp {
    Equal { text: String },
    Insert { text: String },
    Delete { text: String },
    Replace { old: String, new: String },
}

/// Checks whether `old` and `new` share a stable prefix.
/// A prefix is stable when the first `prefix_char_count` characters
/// are identical between successive generations.
pub struct PrefixStabilityChecker {
    config: StabilityConfig,
}

impl PrefixStabilityChecker {
    pub fn new(config: StabilityConfig) -> Self {
        Self { config }
    }

    /// Check if new text has a stable prefix relative to old text.
    /// Returns (is_stable, stable_prefix).
    pub fn check(&self, old: &str, new: &str) -> (bool, String) {
        let limit = self.config.prefix_char_count.min(old.len()).min(new.len());
        if limit == 0 {
            return (false, String::new());
        }
        let mut stable_end = 0;
        for (i, (a, b)) in old.bytes().zip(new.bytes()).enumerate() {
            if i >= limit || a != b {
                break;
            }
            stable_end = i + 1;
        }
        if stable_end > 0 {
            (stable_end == limit, new[..stable_end].to_string())
        } else {
            (false, String::new())
        }
    }

    /// Compute a token-level diff between old and new text.
    pub fn diff(&self, old: &str, new: &str) -> Vec<DiffOp> {
        if old == new {
            return vec![DiffOp::Equal { text: old.to_string() }];
        }
        if old.is_empty() {
            return vec![DiffOp::Insert { text: new.to_string() }];
        }
        if new.is_empty() {
            return vec![DiffOp::Delete { text: old.to_string() }];
        }

        // Find common prefix
        let prefix_len = old.bytes().zip(new.bytes()).take_while(|(a, b)| a == b).count();
        let prefix = &old[..prefix_len];

        // Find common suffix (after differing section)
        let old_suffix = &old[prefix_len..];
        let new_suffix = &new[prefix_len..];
        let suffix_len = old_suffix.bytes().rev().zip(new_suffix.bytes().rev())
            .take_while(|(a, b)| a == b).count();

        let mut ops = Vec::new();
        if !prefix.is_empty() {
            ops.push(DiffOp::Equal { text: prefix.to_string() });
        }

        let old_mid = &old[prefix_len..old.len() - suffix_len];
        let new_mid = &new[prefix_len..new.len() - suffix_len];

        if old_mid.is_empty() && !new_mid.is_empty() {
            ops.push(DiffOp::Insert { text: new_mid.to_string() });
        } else if new_mid.is_empty() && !old_mid.is_empty() {
            ops.push(DiffOp::Delete { text: old_mid.to_string() });
        } else if !old_mid.is_empty() && !new_mid.is_empty() {
            ops.push(DiffOp::Replace { old: old_mid.to_string(), new: new_mid.to_string() });
        }

        if suffix_len > 0 {
            ops.push(DiffOp::Equal { text: new[new.len() - suffix_len..].to_string() });
        }

        ops
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn identical_prefix() {
        let checker = PrefixStabilityChecker::new(StabilityConfig { prefix_char_count: 10 });
        let (stable, prefix) = checker.check("hello world", "hello there");
        assert!(!stable, "prefix diverges at index 6: 'w' != 't'");
        assert_eq!(prefix, "hello ");
    }

    #[test]
    fn diverging_prefix() {
        let checker = PrefixStabilityChecker::new(StabilityConfig { prefix_char_count: 10 });
        let (stable, prefix) = checker.check("abcdefghij", "abcxyzmnop");
        assert!(!stable);
        assert_eq!(prefix, "abc");
    }

    #[test]
    fn empty_input() {
        let checker = PrefixStabilityChecker::new(StabilityConfig::default());
        let (stable, prefix) = checker.check("", "hello");
        assert!(!stable);
        assert!(prefix.is_empty());
    }

    #[test]
    fn both_empty() {
        let checker = PrefixStabilityChecker::new(StabilityConfig::default());
        let (stable, prefix) = checker.check("", "");
        assert!(!stable);
        assert!(prefix.is_empty());
    }

    #[test]
    fn short_text() {
        let checker = PrefixStabilityChecker::new(StabilityConfig { prefix_char_count: 100 });
        let (stable, _) = checker.check("hi", "hi");
        assert!(stable);
    }

    #[test]
    fn diff_equal() {
        let checker = PrefixStabilityChecker::new(StabilityConfig::default());
        let ops = checker.diff("hello", "hello");
        assert_eq!(ops, vec![DiffOp::Equal { text: "hello".to_string() }]);
    }

    #[test]
    fn diff_insert() {
        let checker = PrefixStabilityChecker::new(StabilityConfig::default());
        let ops = checker.diff("", "hello");
        assert_eq!(ops, vec![DiffOp::Insert { text: "hello".to_string() }]);
    }

    #[test]
    fn diff_delete() {
        let checker = PrefixStabilityChecker::new(StabilityConfig::default());
        let ops = checker.diff("hello", "");
        assert_eq!(ops, vec![DiffOp::Delete { text: "hello".to_string() }]);
    }

    #[test]
    fn diff_replace() {
        let checker = PrefixStabilityChecker::new(StabilityConfig::default());
        let ops = checker.diff("abcdef", "abcxyz");
        assert_eq!(ops.len(), 2);
        assert_eq!(ops[0], DiffOp::Equal { text: "abc".to_string() });
        assert_eq!(ops[1], DiffOp::Replace { old: "def".to_string(), new: "xyz".to_string() });
    }
}

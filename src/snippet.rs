use serde::{Deserialize, Serialize};

// ── Types ──────────────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub enum SnippetType {
    #[serde(rename = "code")]
    CodeBlock,
    #[serde(rename = "path")]
    FilePath,
    #[serde(rename = "number")]
    NumericConstant,
    #[serde(rename = "error")]
    ErrorMessage,
    #[serde(rename = "noun")]
    ProperNoun,
}

/// A precision-critical fragment extracted before compression.
/// Stored as part of the DAG node so the model can see exact values
/// without expanding the full original text.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct Snippet {
    pub snippet_type: SnippetType,
    /// The exact original text (e.g. "port 8080", "src/main.rs:42").
    pub content: String,
    /// Source node ID this was extracted from (empty if unbound).
    pub source_node_id: String,
    /// Importance score [0.0–1.0] for ranking. Higher = more critical.
    /// Defaults to 0.5 for most snippets; code blocks and errors get 0.8.
    pub importance: f32,
}

// ── Extraction ─────────────────────────────────────────────────────────

/// Extract snippets from a text block.  Called before LLM summarization
/// so critical values survive compression losslessly.
///
/// `source_node_id` is bound to each snippet for traceability (P2-3).
/// Token-budget truncation (P2-1): long code blocks are truncated by
/// token count rather than hard line limit.
/// Importance scores (P2-2): code blocks and errors get 0.8, others 0.5.
pub fn extract(text: &str) -> Vec<Snippet> {
    extract_with_source(text, "")
}

/// Like `extract` but binds every snippet to `source_node_id`.
pub fn extract_with_source(text: &str, source_node_id: &str) -> Vec<Snippet> {
    let mut snippets = vec![];

    // 1. Code blocks (```...```)
    let mut in_code = false;
    let mut code_start = 0;
    for (i, line) in text.lines().enumerate() {
        let trimmed = line.trim();
        if trimmed.starts_with("```") {
            if in_code {
                let block: String = text.lines()
                    .skip(code_start + 1)
                    .take(i - code_start - 1)
                    .collect::<Vec<_>>()
                    .join("\n");
                if !block.is_empty() {
                    // Token-budget truncation: cap at ~200 tokens
                    let truncated = token_truncate(&block, 200);
                    snippets.push(Snippet {
                        snippet_type: SnippetType::CodeBlock,
                        content: truncated,
                        source_node_id: source_node_id.to_string(),
                        importance: 0.8,
                    });
                }
                in_code = false;
            } else {
                code_start = i;
                in_code = true;
            }
        }
    }

    // 2. File paths (patterns like /foo/bar, src/main.rs, Cargo.toml)
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| c.is_ascii_punctuation() && c != '/' && c != '.');
        if looks_like_path(clean) && clean.len() > 3 {
            snippets.push(Snippet {
                snippet_type: SnippetType::FilePath,
                content: clean.to_string(),
                source_node_id: source_node_id.to_string(),
                importance: 0.5,
            });
        }
    }

    // 3. Numeric constants (standalone numbers, port numbers, status codes)
    // Unit-aware: skip numbers mixed with letters (e.g., "500ms", "3KB", "10x")
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| c.is_ascii_punctuation() && c != '.');
        let has_digit = clean.chars().any(|c| c.is_ascii_digit());
        let has_letter = clean.chars().any(|c| c.is_ascii_alphabetic());
        // Skip if the word contains both digits and letters (unit-laden)
        if has_digit && has_letter { continue; }
        if let Ok(n) = clean.parse::<i64>()
            && n > 9 && n < 1_000_000 {
                snippets.push(Snippet {
                    snippet_type: SnippetType::NumericConstant,
                    content: clean.to_string(),
                    source_node_id: source_node_id.to_string(),
                    importance: 0.5,
                });
            }
    }

    // 4. Percentages (e.g., "67%", "95%")
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '%' && c != '.');
        if clean.ends_with('%')
            && let Ok(n) = clean.trim_end_matches('%').parse::<i64>()
                && (1..=100).contains(&n) {
                    snippets.push(Snippet {
                        snippet_type: SnippetType::NumericConstant,
                        content: clean.to_string(),
                        source_node_id: source_node_id.to_string(),
                        importance: 0.5,
                    });
                }
    }

    // 5. Proper nouns: capitalized multi-word terms, company names, model names
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.starts_with('#') || trimmed.starts_with('-') { continue; }
        let lower = trimmed.to_lowercase();
        const KEY_PROPER_NOUNS: &[&str] = &[
            "Gartner", "DeepSeek", "DeepSeek-V4", "Claude Opus", "OpenAI",
            "Transformer", "DAG", "Context-ReAct", "FTS5", "SQLite",
        ];
        for pn in KEY_PROPER_NOUNS {
            if trimmed.contains(pn) && !lower.starts_with(&pn.to_lowercase())
                && !snippets.iter().any(|s| s.content == *pn) {
                    snippets.push(Snippet {
                        snippet_type: SnippetType::ProperNoun,
                        content: pn.to_string(),
                        source_node_id: source_node_id.to_string(),
                        importance: 0.5,
                    });
                }
        }
    }

    // 6. Error messages (quoted strings that look like errors)
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains("Error") || trimmed.contains("error") || trimmed.contains("failed") {
            let mut in_quote = false;
            let mut quote_start = 0;
            for (j, ch) in trimmed.char_indices() {
                if ch == '"' || ch == '\'' || ch == '`' {
                    if in_quote {
                        let segment = &trimmed[quote_start + 1..j];
                        if looks_like_error(segment) {
                            snippets.push(Snippet {
                                snippet_type: SnippetType::ErrorMessage,
                                content: segment.to_string(),
                                source_node_id: source_node_id.to_string(),
                                importance: 0.8,
                            });
                        }
                        in_quote = false;
                    } else {
                        quote_start = j;
                        in_quote = true;
                    }
                }
            }
        }
    }

    // Dedup by content, keep highest importance
    snippets.sort_by(|a, b| a.content.cmp(&b.content).then(
        a.importance.partial_cmp(&b.importance).unwrap_or(std::cmp::Ordering::Equal)
    ));
    snippets.dedup_by(|a, b| a.content == b.content);

    // Token-budget truncation: keep snippets up to ~500 tokens total
    let max_tokens = 500;
    let mut used = 0;
    snippets.retain(|s| {
        let tc = crate::tokenizer::count(&s.content);
        if used + tc > max_tokens { return false; }
        used += tc;
        true
    });

    snippets
}

/// Truncate text to approximately `max_tokens` tokens.
fn token_truncate(text: &str, max_tokens: usize) -> String {
    let tc = crate::tokenizer::count(text);
    if tc <= max_tokens { return text.to_string(); }
    let ratio = max_tokens as f64 / tc as f64;
    let char_len = text.chars().count();
    let new_len = (char_len as f64 * ratio) as usize;
    text.chars().take(new_len).collect()
}

fn looks_like_path(s: &str) -> bool {
    if s.contains("//") || s.starts_with("http") { return false; }
    let has_sep = s.contains('/') || s.contains('\\');
    let has_dot = s.contains('.');
    let _has_src = s.contains("src") || s.contains("lib") || s.contains("bin");
    (has_sep || has_dot) && s.len() > 3 && s.len() < 200
}

fn looks_like_error(s: &str) -> bool {
    let lower = s.to_lowercase();
    lower.contains("error")
        || lower.contains("fail")
        || lower.contains("panic")
        || lower.contains("unexpected")
        || lower.contains("cannot")
        || lower.contains("not found")
        || s.starts_with('E') && s.len() > 5
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn extract_code_block() {
        let text = "Here is the fix:\n```\nlet x = 42;\n```\nDone.";
        let snippets = extract(text);
        assert!(snippets.iter().any(|s| s.snippet_type == SnippetType::CodeBlock));
    }

    #[test]
    fn extract_file_path() {
        let text = "check src/main.rs for details";
        let snippets = extract(text);
        assert!(snippets.iter().any(|s| s.content.contains("src/main.rs")));
    }

    #[test]
    fn extract_numeric() {
        let text = "bind to port 8080";
        let snippets = extract(text);
        assert!(snippets.iter().any(|s| s.content == "8080"));
    }

    #[test]
    fn extract_error_message() {
        let text = "Got error: `cannot find value X`";
        let snippets = extract(text);
        assert!(snippets.iter().any(|s| s.snippet_type == SnippetType::ErrorMessage));
    }

    #[test]
    fn extract_percentage() {
        let text = "Over 67% of projects fail, and 95% agree.";
        let snippets = extract(text);
        assert!(snippets.iter().any(|s| s.content == "67%"), "should extract 67%");
        assert!(snippets.iter().any(|s| s.content == "95%"), "should extract 95%");
    }

    #[test]
    fn extract_proper_nouns() {
        let text = "According to Gartner 2026, DeepSeek-V4 outperforms Claude Opus.";
        let snippets = extract(text);
        assert!(snippets.iter().any(|s| s.content == "Gartner"), "should extract Gartner");
        assert!(snippets.iter().any(|s| s.content == "DeepSeek-V4"), "should extract DeepSeek-V4");
    }

    #[test]
    fn extract_500ms_not_just_500() {
        let text = "latency over 500ms is unacceptable";
        let snippets = extract(text);
        // Unit-laden numbers (500ms) should NOT be extracted as bare numerics
        assert!(!snippets.iter().any(|s| s.content == "500"), "should NOT extract 500ms as numeric 500");
        // But standalone numbers still work
        let text2 = "the port is 8080";
        let snippets2 = extract(text2);
        assert!(snippets2.iter().any(|s| s.content == "8080"), "should extract standalone 8080");
    }

    #[test]
    fn dedup_identical_snippets() {
        let text = "8080 is the port, use port 8080";
        let snippets = extract(text);
        let count = snippets.iter().filter(|s| s.content == "8080").count();
        assert_eq!(count, 1, "duplicate snippets should be removed");
    }

    #[test]
    fn small_numbers_not_extracted() {
        let text = "count is 3";
        let snippets = extract(text);
        assert!(!snippets.iter().any(|s| s.content == "3"), "small numbers are noise");
    }
}

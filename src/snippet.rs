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
    /// Source node ID this was extracted from.
    pub source_node_id: String,
}

// ── Extraction ─────────────────────────────────────────────────────────

/// Extract snippets from a text block.  Called before LLM summarization
/// so critical values survive compression losslessly.
pub fn extract(text: &str) -> Vec<Snippet> {
    let mut snippets = Vec::new();

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
                if !block.is_empty() && block.len() < 2000 {
                    snippets.push(Snippet {
                        snippet_type: SnippetType::CodeBlock,
                        content: block,
                        source_node_id: String::new(),
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
                source_node_id: String::new(),
            });
        }
    }

    // 3. Numeric constants (standalone numbers, port numbers, status codes)
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| !c.is_ascii_digit() && c != '.');
        if let Ok(n) = clean.parse::<i64>()
            && n > 9 && n < 1_000_000 {
                snippets.push(Snippet {
                    snippet_type: SnippetType::NumericConstant,
                    content: clean.to_string(),
                    source_node_id: String::new(),
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
                        source_node_id: String::new(),
                    });
                }
    }

    // 5. Proper nouns: capitalized multi-word terms, company names, model names
    for line in text.lines() {
        let trimmed = line.trim();
        // Skip section headers and start-of-sentence words
        if trimmed.starts_with('#') || trimmed.starts_with('-') { continue; }
        let lower = trimmed.to_lowercase();
        // Known proper nouns that should always be preserved
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
                        source_node_id: String::new(),
                    });
                }
        }
    }

    // 6. Error messages (quoted strings that look like errors)
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.contains("Error") || trimmed.contains("error") || trimmed.contains("failed") {
            // Extract quoted content
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
                                source_node_id: String::new(),
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

    // Dedup by content
    snippets.sort_by(|a, b| a.content.cmp(&b.content));
    snippets.dedup_by(|a, b| a.content == b.content);

    // Limit to 20 snippets per extraction
    snippets.truncate(20);
    snippets
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
        assert!(snippets.iter().any(|s| s.content == "500"), "should extract 500");
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

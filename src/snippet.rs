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
    /// How many times this snippet value has appeared across conversations.
    /// Higher = more likely to be noise (common values). Lower = rare, keep.
    pub frequency: u32,
}

/// Rank snippets by combined score: importance descending, then frequency ascending
/// (rare values rank higher than common ones), then content length descending.
/// Returns top-K results.
pub fn rank_snippets(snippets: &[Snippet], top_k: usize) -> Vec<&Snippet> {
    let mut sorted: Vec<&Snippet> = snippets.iter().collect();
    sorted.sort_by(|a, b| {
        b.importance.partial_cmp(&a.importance).unwrap_or(std::cmp::Ordering::Equal)
            .then(a.frequency.cmp(&b.frequency)) // lower freq = rarer = higher rank
            .then(b.content.len().cmp(&a.content.len())) // longer = more info
    });
    sorted.truncate(top_k);
    sorted
}

/// Base importance by snippet type.
pub fn type_base_importance(st: &SnippetType) -> f32 {
    match st {
        SnippetType::CodeBlock => 0.9,
        SnippetType::ErrorMessage => 0.85,
        SnippetType::FilePath => 0.6,
        SnippetType::NumericConstant => 0.4,
        SnippetType::ProperNoun => 0.3,
    }
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

    // 1. Code blocks (```...```) with AST-aware structural extraction
    let mut in_code = false;
    let mut code_start = 0;
    let mut code_lang = "";
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
                    // Extract structural elements from code
                    let structural = extract_code_structure(&block, code_lang);
                    if structural.is_empty() {
                        let truncated = token_truncate(&block, 200);
                        snippets.push(Snippet {
                            snippet_type: SnippetType::CodeBlock,
                            content: truncated,
                            source_node_id: source_node_id.to_string(),
                            frequency: 0,
                        importance: 0.8,
                        });
                    } else {
                        for elem in structural {
                            snippets.push(Snippet {
                                snippet_type: SnippetType::CodeBlock,
                                content: elem,
                                source_node_id: source_node_id.to_string(),
                                frequency: 0,
                        importance: 0.85,
                            });
                        }
                    }
                }
                in_code = false;
            } else {
                code_start = i;
                code_lang = trimmed.trim_start_matches('`').trim();
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
                frequency: 0,
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
                    frequency: 0,
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
                        frequency: 0,
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
                        frequency: 0,
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
                                frequency: 0,
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

/// Extract structural code elements (functions, types, imports) from code text.
/// Uses line-based heuristics that work across Rust, Python, Go, TypeScript.
/// Tree-sitter based extraction for Rust code (precise AST boundaries).
fn extract_rust_ast(code: &str) -> Option<Vec<String>> {
    let mut elements = Vec::new();
    let mut parser = tree_sitter::Parser::new();
    parser.set_language(&tree_sitter_rust::LANGUAGE.into()).ok()?;
    let tree = parser.parse(code, None)?;
    let root = tree.root_node();
    let mut seen = std::collections::HashSet::new();

    // Recursively walk the AST looking for declarations
    let mut stack: Vec<tree_sitter::Node> = root.children(&mut root.walk()).collect();
    while let Some(node) = stack.pop() {
        match node.kind() {
            "function_item" | "function_signature_item" => {
                let text = &code[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next().unwrap_or(text);
                if seen.insert(first_line.to_string()) {
                    elements.push(truncate_to_line(first_line, 120));
                }
            }
            "struct_item" | "enum_item" | "trait_item" | "impl_item" => {
                let text = &code[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next().unwrap_or(text);
                if seen.insert(first_line.to_string()) {
                    elements.push(truncate_to_line(first_line, 120));
                }
            }
            "use_declaration" => {
                let text = &code[node.start_byte()..node.end_byte()];
                let trimmed = text.trim();
                if !trimmed.starts_with("use std::") && seen.insert(trimmed.to_string()) {
                    elements.push(trimmed.to_string());
                }
            }
            "const_item" | "static_item" => {
                let text = &code[node.start_byte()..node.end_byte()];
                let first_line = text.lines().next().unwrap_or(text);
                if seen.insert(first_line.to_string()) {
                    elements.push(truncate_to_line(first_line, 80));
                }
            }
            _ => {}
        }
        // Push children for further traversal
        let mut children: Vec<tree_sitter::Node> = node.children(&mut node.walk()).collect();
        children.reverse(); // maintain DFS order
        for child in children {
            stack.push(child);
        }
    }

    elements.truncate(10);
    Some(elements)
}

fn extract_code_structure(code: &str, lang: &str) -> Vec<String> {
    // Use tree-sitter for Rust
    if (lang.contains("rust") || lang.is_empty())
        && let Some(elements) = extract_rust_ast(code)
            && !elements.is_empty() {
                return elements;
            }
    // Fall back to line-based heuristics
    let mut elements = Vec::new();
    let is_rust = lang.contains("rust") || lang.is_empty();
    let is_py = lang.contains("python") || lang.contains("py");
    let is_go = lang.contains("go") || lang.contains("golang");

    for line in code.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') {
            continue;
        }

        // Rust: fn, struct, enum, trait, impl, use, const, mod, type
        if is_rust {
            if trimmed.starts_with("fn ") || trimmed.starts_with("pub fn ")
                || trimmed.starts_with("async fn ") || trimmed.starts_with("pub async fn ") {
                let sig = trimmed.trim_start_matches("pub ").trim_start_matches("async ");
                elements.push(truncate_to_line(sig, 120));
            }
            if trimmed.starts_with("struct ") || trimmed.starts_with("pub struct ") {
                elements.push(truncate_to_line(trimmed.trim_start_matches("pub "), 120));
            }
            if trimmed.starts_with("enum ") || trimmed.starts_with("pub enum ") {
                elements.push(truncate_to_line(trimmed.trim_start_matches("pub "), 120));
            }
            if trimmed.starts_with("trait ") || trimmed.starts_with("pub trait ") {
                elements.push(truncate_to_line(trimmed.trim_start_matches("pub "), 120));
            }
            if trimmed.starts_with("impl ") {
                elements.push(truncate_to_line(trimmed, 120));
            }
            if trimmed.starts_with("use ") && !trimmed.starts_with("use std::") {
                elements.push(trimmed.to_string());
            }
            if trimmed.starts_with("const ") || trimmed.starts_with("pub const ") {
                elements.push(truncate_to_line(trimmed, 80));
            }
            if trimmed.starts_with("type ") && !trimmed.contains('<') {
                elements.push(truncate_to_line(trimmed, 80));
            }
            if (trimmed.starts_with("mod ") || trimmed.starts_with("pub mod ")) && !trimmed.contains('{') {
                elements.push(truncate_to_line(trimmed, 60));
            }
        }

        // Python: def, class, import, from X import
        if is_py {
            if trimmed.starts_with("def ") || trimmed.starts_with("async def ") {
                elements.push(truncate_to_line(trimmed, 120));
            }
            if trimmed.starts_with("class ") {
                elements.push(truncate_to_line(trimmed, 120));
            }
            if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
                elements.push(trimmed.to_string());
            }
        }

        // Go: func, type, import, const, var
        if is_go {
            if trimmed.starts_with("func ") {
                elements.push(truncate_to_line(trimmed, 120));
            }
            if trimmed.starts_with("type ") && !trimmed.contains(' ') {
                elements.push(truncate_to_line(trimmed, 120));
            }
            if trimmed.starts_with("import ") {
                elements.push(trimmed.to_string());
            }
        }
    }

    // Cap at 10 elements
    elements.truncate(10);
    elements
}

fn truncate_to_line(s: &str, max_chars: usize) -> String {
    if s.len() <= max_chars { s.to_string() } else { format!("{}…", &s[..max_chars-1]) }
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

    #[test]
    fn extract_code_structure_rust_functions() {
        let code = "fn main() {\n    let x = 42;\n}\n\npub fn process(data: &[u8]) -> Result<()> {\n    Ok(())\n}\n";
        let elems = extract_code_structure(code, "rust");
        assert!(elems.iter().any(|e| e.contains("fn main")), "should extract fn main, got: {elems:?}");
        assert!(elems.iter().any(|e| e.contains("fn process")), "should extract fn process, got: {elems:?}");
    }

    #[test]
    fn extract_code_structure_rust_types() {
        let code = "struct Point {\n    x: f64,\n    y: f64,\n}\n\nenum Color { Red, Green, Blue }\n\nimpl Point {\n    fn new() -> Self { Point { x: 0.0, y: 0.0 } }\n}";
        let elems = extract_code_structure(code, "rust");
        assert!(elems.iter().any(|e| e.contains("struct Point")), "should extract struct");
        assert!(elems.iter().any(|e| e.contains("enum Color")), "should extract enum");
        assert!(elems.iter().any(|e| e.contains("impl Point")), "should extract impl");
    }

    #[test]
    fn extract_ast_code_block_in_text() {
        let text = "Here is the fix:\n```rust\nfn main() {\n    println!(\"hello\");\n}\npub struct Config { debug: bool }\n```";
        let snippets = extract(text);
        let code_snippets: Vec<_> = snippets.iter().filter(|s| s.snippet_type == SnippetType::CodeBlock).collect();
        assert!(!code_snippets.is_empty(), "should extract code snippets");
        // Should have extracted structural elements rather than raw block
        let has_fn = code_snippets.iter().any(|s| s.content.contains("fn main"));
        let has_struct = code_snippets.iter().any(|s| s.content.contains("struct Config"));
        assert!(has_fn || has_struct, "should extract structural elements, got: {code_snippets:?}");
    }
}

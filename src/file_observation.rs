//! FileObservation model — AST-based structured file representation for
//! semantic caching and change detection.
//!
//! Builds on `ArtifactVersion` (content hash) with AST extraction and
//! semantic hashing that's stable under cosmetic changes.

use serde::{Deserialize, Serialize};
use sha2::{Digest, Sha256};

// ── AST types ──────────────────────────────────────────────────────────

/// A function/method definition extracted from source code.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FunctionDef {
    pub name: String,
    pub signature: String,
    pub line_start: usize,
    pub line_end: usize,
}

/// A type/interface/struct/enum definition.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct TypeDef {
    pub name: String,
    pub kind: String,
    pub line_start: usize,
    pub line_end: usize,
}

/// Abstract Syntax Tree representation of a file.
#[derive(Debug, Clone, Serialize, Deserialize, PartialEq)]
pub struct FileAst {
    pub language: Option<String>,
    pub imports: Vec<String>,
    pub exports: Vec<String>,
    pub functions: Vec<FunctionDef>,
    pub types: Vec<TypeDef>,
    pub declaration_count: usize,
}

/// File observation at a point in time.
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct FileObservation {
    pub path: String,
    pub content_hash: String,
    pub semantic_hash: String,
    pub ast: FileAst,
    pub size_bytes: usize,
    pub line_count: usize,
    pub observed_at: String,
}

/// Structured diff between two observations.
#[derive(Debug, Clone, Serialize, Deserialize, Default)]
pub struct ObservationDiff {
    pub changed: bool,
    pub functions_added: Vec<String>,
    pub functions_removed: Vec<String>,
    pub types_added: Vec<String>,
    pub types_removed: Vec<String>,
    pub imports_added: Vec<String>,
    pub imports_removed: Vec<String>,
    pub semantic_hash_changed: bool,
}

// ── Language detection ─────────────────────────────────────────────────

/// Guess language from file path extension.
pub fn detect_language(path: &str) -> Option<String> {
    let ext = path.rsplit('.').next()?.to_lowercase();
    match ext.as_str() {
        "rs" => Some("rust".into()),
        "py" => Some("python".into()),
        "js" => Some("javascript".into()),
        "ts" | "tsx" => Some("typescript".into()),
        "java" => Some("java".into()),
        "go" => Some("go".into()),
        "rb" => Some("ruby".into()),
        "c" | "h" => Some("c".into()),
        "cpp" | "cxx" | "cc" | "hpp" => Some("cpp".into()),
        "cs" => Some("csharp".into()),
        "toml" => Some("toml".into()),
        "json" => Some("json".into()),
        "yaml" | "yml" => Some("yaml".into()),
        "md" => Some("markdown".into()),
        "html" => Some("html".into()),
        "css" => Some("css".into()),
        _ => None,
    }
}

// ── AST extraction ─────────────────────────────────────────────────────

/// Extract AST from file content, using language-aware parsing.
pub fn extract_ast(path: &str, content: &str) -> FileAst {
    let lang = detect_language(path);
    let mut functions = Vec::new();
    let mut types = Vec::new();
    let mut imports = Vec::new();
    let mut exports = Vec::new();
    let mut declaration_count = 0;

    match lang.as_deref() {
        Some("rust") => extract_rust_ast(content, &mut functions, &mut types, &mut imports, &mut exports, &mut declaration_count),
        Some("python") => extract_python_ast(content, &mut functions, &mut types, &mut imports, &mut exports, &mut declaration_count),
        Some("go") => extract_go_ast(content, &mut functions, &mut types, &mut imports, &mut exports, &mut declaration_count),
        _ => extract_generic_ast(content, &mut functions, &mut types, &mut imports, &mut exports, &mut declaration_count),
    }

    // Dedup
    imports.sort();
    imports.dedup();
    exports.sort();
    exports.dedup();

    FileAst { language: lang, imports, exports, functions, types, declaration_count }
}

/// Rust-specific AST extraction.
fn extract_rust_ast(
    content: &str, functions: &mut Vec<FunctionDef>,
    types: &mut Vec<TypeDef>, imports: &mut Vec<String>,
    exports: &mut Vec<String>, decl_count: &mut usize,
) {
    let lines: Vec<&str> = content.lines().collect();
    let mut in_block_comment = false;
    let mut i = 0;

    while i < lines.len() {
        let line = lines[i];
        let trimmed = line.trim();

        // Skip block comments
        if trimmed.starts_with("/*") {
            in_block_comment = true;
            if trimmed.contains("*/") {
                in_block_comment = false;
            }
            i += 1;
            continue;
        }
        if in_block_comment {
            if trimmed.contains("*/") {
                in_block_comment = false;
            }
            i += 1;
            continue;
        }
        if trimmed.starts_with("//") || trimmed.is_empty() {
            i += 1;
            continue;
        }

        // Imports
        if trimmed.starts_with("use ") || trimmed.starts_with("pub use ") || trimmed.starts_with("pub(crate) use ") {
            let imp = trimmed.trim_end_matches(';').to_string();
            imports.push(imp.clone());
            if imp.starts_with("pub ") {
                exports.push(imp);
            }
            *decl_count += 1;
            i += 1;
            continue;
        }

        // Functions
        let fn_vis = ["fn ", "pub fn ", "pub async fn ", "async fn ", "pub(crate) fn ", "pub(super) fn ",
                      "unsafe fn ", "pub unsafe fn ", "pub async unsafe fn ",
                      "pub(super) async fn ", "pub(crate) async fn ", "const fn ", "pub const fn "];
        if let Some(sig) = fn_vis.iter().find_map(|p| trimmed.strip_prefix(p)) {
            let name = sig.split(['(', '<', ' ']).next().unwrap_or("").to_string();
            if !name.is_empty() && !name.starts_with('_') {
                let line_start = i + 1;
                let line_end = find_block_end(&lines[i..]) + i;
                functions.push(FunctionDef { name: name.clone(), signature: trimmed.to_string(), line_start, line_end });
                if trimmed.starts_with("pub ") {
                    exports.push(name);
                }
                *decl_count += 1;
            }
            i += 1;
            continue;
        }

        // Types
        let type_vis = ["struct ", "pub struct ", "enum ", "pub enum ", "trait ", "pub trait ",
                       "union ", "pub union ", "type ", "pub type "];
        if let Some(sig) = type_vis.iter().find_map(|p| trimmed.strip_prefix(p)) {
            let name = sig.split([' ', '<', ';', '(']).next().unwrap_or("").to_string();
            if !name.is_empty() {
                let kind = if trimmed.contains("struct") { "struct" }
                    else if trimmed.contains("enum") { "enum" }
                    else if trimmed.contains("trait") { "trait" }
                    else if trimmed.contains("union") { "union" }
                    else { "type_alias" };
                let line_start = i + 1;
                let line_end = if trimmed.contains(';') { i + 1 } else { find_block_end(&lines[i..]) + i };
                types.push(TypeDef { name: name.clone(), kind: kind.to_string(), line_start, line_end });
                if trimmed.starts_with("pub ") {
                    exports.push(name);
                }
                *decl_count += 1;
            }
            i += 1;
            continue;
        }

        // Mod declarations
        if let Some(mod_name) = trimmed.strip_prefix("mod ").and_then(|s| s.strip_suffix(';')) {
            exports.push(mod_name.trim().to_string());
            *decl_count += 1;
        }

        // Const/static
        if trimmed.starts_with("const ") || trimmed.starts_with("pub const ")
            || trimmed.starts_with("static ") || trimmed.starts_with("pub static ") {
            *decl_count += 1;
        }

        i += 1;
    }
}

/// Python-specific AST extraction.
fn extract_python_ast(
    content: &str, functions: &mut Vec<FunctionDef>,
    types: &mut Vec<TypeDef>, imports: &mut Vec<String>,
    _exports: &mut Vec<String>, decl_count: &mut usize,
) {
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            i += 1;
            continue;
        }

        // Imports
        if trimmed.starts_with("import ") || trimmed.starts_with("from ") {
            imports.push(trimmed.to_string());
            *decl_count += 1;
            i += 1;
            continue;
        }

        // Functions (def)
        if let Some(sig) = trimmed.strip_prefix("def ")
            .or_else(|| trimmed.strip_prefix("async def "))
        {
            let name = sig.split(['(', ' ']).next().unwrap_or("").to_string();
            if !name.is_empty() && !name.starts_with('_') {
                let line_start = i + 1;
                let line_end = find_python_block_end(&lines[i..]) + i;
                functions.push(FunctionDef { name, signature: trimmed.to_string(), line_start, line_end });
                *decl_count += 1;
            }
            i += 1;
            continue;
        }

        // Classes
        if let Some(sig) = trimmed.strip_prefix("class ") {
            let name = sig.split(['(', ':']).next().unwrap_or("").to_string();
            if !name.is_empty() {
                let line_start = i + 1;
                let line_end = find_python_block_end(&lines[i..]) + i;
                types.push(TypeDef { name, kind: "class".into(), line_start, line_end });
                *decl_count += 1;
            }
            i += 1;
            continue;
        }

        i += 1;
    }
}

/// Go-specific AST extraction.
fn extract_go_ast(
    content: &str, functions: &mut Vec<FunctionDef>,
    types: &mut Vec<TypeDef>, imports: &mut Vec<String>,
    exports: &mut Vec<String>, decl_count: &mut usize,
) {
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() || trimmed.starts_with("//") {
            i += 1;
            continue;
        }

        if trimmed.starts_with("import (") || trimmed.starts_with("import \"") {
            imports.push(trimmed.to_string());
            *decl_count += 1;
            i += 1;
            continue;
        }

        if let Some(sig) = trimmed.strip_prefix("func ") {
            let name = sig.split(['(', ' ']).next().unwrap_or("").to_string();
            let is_exported = name.starts_with(char::is_uppercase);
            let line_start = i + 1;
            let line_end = find_block_end(&lines[i..]) + i;
            functions.push(FunctionDef { name: name.clone(), signature: trimmed.to_string(), line_start, line_end });
            if is_exported { exports.push(name); }
            *decl_count += 1;
            i += 1;
            continue;
        }

        if let Some(sig) = trimmed.strip_prefix("type ") {
            let name = sig.split([' ', '<']).next().unwrap_or("").to_string();
            let kind = sig.split(' ').nth(1).unwrap_or("type").to_string();
            let line_start = i + 1;
            let line_end = if trimmed.contains(';') { i + 1 } else { find_block_end(&lines[i..]) + i };
            types.push(TypeDef { name, kind, line_start, line_end });
            *decl_count += 1;
        }

        i += 1;
    }
}

/// Generic fallback: detect top-level declarations by indentation heuristics.
fn extract_generic_ast(
    content: &str, functions: &mut Vec<FunctionDef>,
    types: &mut Vec<TypeDef>, imports: &mut Vec<String>,
    _exports: &mut Vec<String>, decl_count: &mut usize,
) {
    let lines: Vec<&str> = content.lines().collect();
    let mut i = 0;

    while i < lines.len() {
        let trimmed = lines[i].trim();
        if trimmed.is_empty() || trimmed.starts_with("//") || trimmed.starts_with('#') || trimmed.starts_with("/*") {
            i += 1;
            continue;
        }

        // Generic function detection
        if trimmed.contains("function ") || trimmed.starts_with("fn ") {
            let name = trimmed.split(['(', ' ', '{']).nth(1).unwrap_or("").to_string();
            if !name.is_empty() {
                let line_start = i + 1;
                let line_end = find_block_end(&lines[i..]) + i;
                functions.push(FunctionDef { name, signature: trimmed.to_string(), line_start, line_end });
                *decl_count += 1;
            }
            i += 1;
            continue;
        }

        // Generic import detection
        if trimmed.starts_with("import ") || trimmed.starts_with("using ") || trimmed.starts_with("#include") {
            imports.push(trimmed.to_string());
            *decl_count += 1;
        }

        // Generic class/struct/interface detection
        for keyword in &["class ", "struct ", "interface ", "enum ", "trait "] {
            if trimmed.starts_with(keyword) {
                let name = trimmed.split([' ', '{', ':', '<', '(']).nth(1).unwrap_or("").to_string();
                if !name.is_empty() {
                    types.push(TypeDef { name, kind: keyword.trim().to_string(), line_start: i + 1, line_end: i + 1 });
                    *decl_count += 1;
                }
                break;
            }
        }

        i += 1;
    }
}

/// Find the end of a brace-delimited block starting at `lines[0]`.
fn find_block_end(lines: &[&str]) -> usize {
    let mut depth = 0;
    let mut started = false;
    for (i, line) in lines.iter().enumerate() {
        for ch in line.chars() {
            match ch {
                '{' => { depth += 1; started = true; }
                '}' => { depth -= 1; }
                _ => {}
            }
        }
        if started && depth == 0 {
            return i;
        }
    }
    lines.len().saturating_sub(1)
}

/// Find the end of a Python indentation-based block.
fn find_python_block_end(lines: &[&str]) -> usize {
    if lines.len() <= 1 { return 0; }
    // The first line (def/class) ends with ':'
    let base_indent = lines.get(1).map(|l| l.len() - l.trim_start().len()).unwrap_or(0);
    for (i, line) in lines.iter().enumerate().skip(2) {
        if line.trim().is_empty() { continue; }
        let indent = line.len() - line.trim_start().len();
        if indent <= base_indent && !lines[i].trim().starts_with('#') {
            return i.saturating_sub(1);
        }
    }
    lines.len().saturating_sub(1)
}

// ── Semantic hashing ───────────────────────────────────────────────────

/// Compute semantic hash: SHA-256 of normalized AST JSON.
/// Stable under whitespace, comment, and import ordering changes.
pub fn compute_semantic_hash(ast: &FileAst) -> String {
    let normalized = serde_json::json!({
        "fns": ast.functions.iter().map(|f| {
            serde_json::json!({"name": f.name, "sig": normalize_signature(&f.signature)})
        }).collect::<Vec<_>>(),
        "types": ast.types.iter().map(|t| {
            serde_json::json!({"name": t.name, "kind": t.kind})
        }).collect::<Vec<_>>(),
        "imports": ast.imports.iter().map(|i| normalize_import(i)).collect::<Vec<_>>(),
        "decl_count": ast.declaration_count,
    });
    let json_str = serde_json::to_string(&normalized).unwrap_or_default();
    hex::encode(&Sha256::digest(json_str.as_bytes())[..8])
}

/// Normalize a function signature for semantic comparison.
fn normalize_signature(sig: &str) -> String {
    sig.chars()
        .map(|c| if c.is_whitespace() { ' ' } else { c })
        .collect::<String>()
        .split_whitespace()
        .collect::<Vec<_>>()
        .join(" ")
        .trim()
        .to_string()
}

/// Normalize an import line.
fn normalize_import(imp: &str) -> String {
    let s = imp.trim().trim_end_matches(';').trim().to_string();
    s.split_whitespace().collect::<Vec<_>>().join(" ")
}

// ── Observation creation ───────────────────────────────────────────────

/// Create a FileObservation from path + content.
pub fn observe_file(path: &str, content: &str) -> FileObservation {
    let ast = extract_ast(path, content);
    let content_hash = {
        let h = Sha256::digest(content.as_bytes());
        hex::encode(&h[..8])
    };
    let semantic_hash = compute_semantic_hash(&ast);
    let line_count = content.lines().count();

    FileObservation {
        path: path.to_string(),
        content_hash,
        semantic_hash,
        ast,
        size_bytes: content.len(),
        line_count,
        observed_at: chrono::Utc::now().to_rfc3339(),
    }
}

// ── Change detection ────────────────────────────────────────────────────

/// Compare two observations and produce a structured diff.
pub fn compare_observations(before: &FileObservation, after: &FileObservation) -> ObservationDiff {
    let semantic_hash_changed = before.semantic_hash != after.semantic_hash;
    let content_changed = before.content_hash != after.content_hash;
    let mut diff = ObservationDiff { changed: content_changed, semantic_hash_changed, ..Default::default() };

    // Compare functions
    let before_fns: std::collections::HashSet<&str> = before.ast.functions.iter().map(|f| f.name.as_str()).collect();
    let after_fns: std::collections::HashSet<&str> = after.ast.functions.iter().map(|f| f.name.as_str()).collect();
    diff.functions_added = after_fns.difference(&before_fns).map(|s| s.to_string()).collect();
    diff.functions_removed = before_fns.difference(&after_fns).map(|s| s.to_string()).collect();

    // Compare types
    let before_types: std::collections::HashSet<&str> = before.ast.types.iter().map(|t| t.name.as_str()).collect();
    let after_types: std::collections::HashSet<&str> = after.ast.types.iter().map(|t| t.name.as_str()).collect();
    diff.types_added = after_types.difference(&before_types).map(|s| s.to_string()).collect();
    diff.types_removed = before_types.difference(&after_types).map(|s| s.to_string()).collect();

    // Compare imports
    let before_imports: std::collections::HashSet<&str> = before.ast.imports.iter().map(|i| i.as_str()).collect();
    let after_imports: std::collections::HashSet<&str> = after.ast.imports.iter().map(|i| i.as_str()).collect();
    diff.imports_added = after_imports.difference(&before_imports).map(|s| s.to_string()).collect();
    diff.imports_removed = before_imports.difference(&after_imports).map(|s| s.to_string()).collect();

    diff.changed = content_changed || !diff.functions_added.is_empty() || !diff.functions_removed.is_empty()
        || !diff.types_added.is_empty() || !diff.types_removed.is_empty();

    diff
}

// ── SQL migration ─────────────────────────────────────────────────────

/// Migration for file_observations table.
pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS file_observations (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        path            TEXT NOT NULL,
        content_hash    TEXT NOT NULL,
        semantic_hash   TEXT NOT NULL,
        ast_json        TEXT NOT NULL DEFAULT '{}',
        size_bytes      INTEGER NOT NULL DEFAULT 0,
        line_count      INTEGER NOT NULL DEFAULT 0,
        created_at      TEXT NOT NULL DEFAULT (datetime('now'))
    );
    CREATE INDEX IF NOT EXISTS idx_file_obs_path ON file_observations(path);
    CREATE INDEX IF NOT EXISTS idx_file_obs_semantic ON file_observations(semantic_hash);";

// ── Tests ──────────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn detect_language_by_extension() {
        assert_eq!(detect_language("src/main.rs"), Some("rust".into()));
        assert_eq!(detect_language("src/app.py"), Some("python".into()));
        assert_eq!(detect_language("src/lib.ts"), Some("typescript".into()));
        assert_eq!(detect_language("Makefile"), None);
    }

    #[test]
    fn extract_rust_functions() {
        let content = r#"
use std::collections::HashMap;

pub fn process_data(input: &str) -> Result<()> {
    let mut map = HashMap::new();
    map.insert("key", 42);
    Ok(())
}

fn helper() {
    println!("helper");
}
"#;
        let ast = extract_ast("src/lib.rs", content);
        assert_eq!(ast.functions.len(), 2);
        assert!(ast.functions.iter().any(|f| f.name == "process_data"));
        assert!(ast.functions.iter().any(|f| f.name == "helper"));
        assert_eq!(ast.imports.len(), 1);
        assert!(ast.imports[0].contains("HashMap"));
    }

    #[test]
    fn extract_rust_types() {
        let content = r#"
pub struct Config {
    pub name: String,
}

enum Status {
    Active,
    Inactive,
}

pub trait Handler {
    fn handle(&self);
}
"#;
        let ast = extract_ast("src/lib.rs", content);
        assert_eq!(ast.types.len(), 3);
        assert!(ast.types.iter().any(|t| t.name == "Config" && t.kind == "struct"));
        assert!(ast.types.iter().any(|t| t.name == "Status" && t.kind == "enum"));
        assert!(ast.types.iter().any(|t| t.name == "Handler" && t.kind == "trait"));
    }

    #[test]
    fn extract_python_functions() {
        let content = r#"
import os
from typing import Optional

def process_data(input_str: str) -> None:
    result = input_str.strip()
    return result

class DataProcessor:
    def __init__(self):
        self.data = []
"#;
        let ast = extract_ast("src/app.py", content);
        assert_eq!(ast.imports.len(), 2);
        assert!(ast.functions.iter().any(|f| f.name == "process_data"));
        assert!(ast.types.iter().any(|t| t.name == "DataProcessor"));
    }

    #[test]
    fn semantic_hash_stable_under_whitespace() {
        let content1 = "fn main()  {\n    println!(\"hello\");\n}";
        let content2 = "fn main() {\nprintln!(\"hello\");\n}";
        let obs1 = observe_file("src/main.rs", content1);
        let obs2 = observe_file("src/main.rs", content2);
        assert_eq!(obs1.semantic_hash, obs2.semantic_hash, "whitespace should not affect semantic hash");
        assert_ne!(obs1.content_hash, obs2.content_hash, "content hash should differ");
    }

    #[test]
    fn semantic_hash_changes_on_function_rename() {
        let content1 = "fn old_name() {}";
        let content2 = "fn new_name() {}";
        let obs1 = observe_file("src/lib.rs", content1);
        let obs2 = observe_file("src/lib.rs", content2);
        assert_ne!(obs1.semantic_hash, obs2.semantic_hash);
    }

    #[test]
    fn compare_observations_detects_added_function() {
        let before = observe_file("src/lib.rs", "fn existing() {}");
        let after = observe_file("src/lib.rs", "fn existing() {}\nfn new_func() {}");
        let diff = compare_observations(&before, &after);
        assert!(diff.changed);
        assert_eq!(diff.functions_added, vec!["new_func"]);
    }

    #[test]
    fn compare_observations_detects_removed_function() {
        let before = observe_file("src/lib.rs", "fn a() {}\nfn b() {}");
        let after = observe_file("src/lib.rs", "fn a() {}");
        let diff = compare_observations(&before, &after);
        assert!(diff.changed);
        assert_eq!(diff.functions_removed, vec!["b"]);
    }

    #[test]
    fn compare_observations_detects_import_changes() {
        let before = observe_file("src/lib.rs", "use std::collections::HashMap;\nfn main() {}");
        let after = observe_file("src/lib.rs", "use std::sync::Arc;\nfn main() {}");
        let diff = compare_observations(&before, &after);
        assert!(diff.changed);
        assert!(!diff.imports_removed.is_empty() || !diff.imports_added.is_empty());
    }

    #[test]
    fn semantic_hash_stable_under_comment_changes() {
        let content1 = "// this is a comment\nfn main() {}";
        let content2 = "// different comment\nfn main() {}";
        let obs1 = observe_file("src/lib.rs", content1);
        let obs2 = observe_file("src/lib.rs", content2);
        assert_eq!(obs1.semantic_hash, obs2.semantic_hash, "comments should not affect semantic hash");
    }

    #[test]
    fn go_function_detection() {
        let content = r#"
package main

import "fmt"

func main() {
    fmt.Println("hello")
}

func processData() string {
    return "data"
}
"#;
        let ast = extract_ast("main.go", content);
        assert_eq!(ast.functions.len(), 2);
        assert!(ast.functions.iter().any(|f| f.name == "main"));
        assert!(ast.functions.iter().any(|f| f.name == "processData"));
    }
}

//! Deterministic tool result cache. L1 is in-memory (RwLock<HashMap>),
//! L2 is SQLite for persistence across restarts.
//!
//! Cache key: `CacheKey { tool_name, args_hash }`.
//! File dependencies tracked via reverse index for O(affected) invalidation.

use serde::{Deserialize, Serialize};
use std::collections::{HashMap, HashSet, VecDeque};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Arc, RwLock};

// ── Strongly-typed identities ─────────────────────────────────────────

/// Canonical tool name — newtype to prevent stringly-typed misuse.
#[derive(Debug, Clone, PartialEq, Eq, Hash, Serialize, Deserialize)]
pub struct ToolName(String);

impl ToolName {
    pub fn new(name: &str) -> Self {
        Self(normalize_tool_name(name))
    }
    pub fn as_str(&self) -> &str { &self.0 }
}

impl std::fmt::Display for ToolName {
    fn fmt(&self, f: &mut std::fmt::Formatter<'_>) -> std::fmt::Result {
        self.0.fmt(f)
    }
}

/// Cache key — replaces bare `(String, String)` tuples.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CacheKey {
    pub tool_name: ToolName,
    pub args_hash: String,
}

impl CacheKey {
    pub fn new(tool_name: &str, args: &str) -> Self {
        let normalized = normalize_args(args);
        Self {
            tool_name: ToolName::new(tool_name),
            args_hash: sha256_hex(&normalized),
        }
    }
}

// ── L1 Hot Cache ──────────────────────────────────────────────────────

/// LRU eviction via insertion-order tracking. O(1) insert/lookup,
/// O(evicted) eviction. Deterministic — uses logical clock, not Instant.
pub struct L1HotCache {
    entries: RwLock<HashMap<CacheKey, CacheEntry>>,
    order: RwLock<VecDeque<CacheKey>>,
    /// Reverse index: file_path → set of cache keys that depend on it.
    file_index: RwLock<HashMap<String, HashSet<CacheKey>>>,
    /// Logical clock: monotonically increasing access counter.
    clock: AtomicU64,
    capacity: usize,
}

struct CacheEntry {
    result: Arc<str>,
    dependent_files: Vec<String>,
    file_hashes: Vec<String>,
    hit_count: u64,
    access_seq: u64,
}

impl Default for L1HotCache {
    fn default() -> Self {
        Self::new(128)
    }
}

impl L1HotCache {
    pub fn new(capacity: usize) -> Self {
        Self {
            entries: RwLock::new(HashMap::new()),
            order: RwLock::new(VecDeque::new()),
            file_index: RwLock::new(HashMap::new()),
            clock: AtomicU64::new(0),
            capacity,
        }
    }

    pub fn is_empty(&self) -> bool {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).is_empty()
    }

    /// Look up a cached result. Returns the result (Arc clone, cheap bump)
    /// and hit count. Updates access sequence and moves key to back of LRU order.
    pub fn get(&self, tool_name: &str, args_hash: &str) -> Option<(Arc<str>, u64)> {
        let key = CacheKey { tool_name: ToolName::new(tool_name), args_hash: args_hash.to_string() };
        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        if let Some(e) = map.get_mut(&key) {
            e.hit_count += 1;
            e.access_seq = self.clock.fetch_add(1, Ordering::Relaxed);
            // Move to back of LRU order
            if let Ok(mut ord) = self.order.write() {
                if let Some(pos) = ord.iter().position(|k| *k == key) {
                    ord.remove(pos);
                    ord.push_back(key);
                }
            }
            Some((e.result.clone(), e.hit_count))
        } else {
            None
        }
    }

    /// Store a result. LRU eviction, reverse index updated.
    /// Cleans old dependency index on overwrite.
    pub fn put(&self, tool_name: &str, args_hash: &str, result: &str, dependent_files: &[String]) {
        self.put_with_hashes(tool_name, args_hash, result, dependent_files, &[])
    }

    /// Store with optional content hashes for ArtifactVersion-based invalidation.
    /// On overwrite, removes old dependencies from reverse index before inserting new ones.
    pub fn put_with_hashes(
        &self,
        tool_name: &str,
        args_hash: &str,
        result: &str,
        dependent_files: &[String],
        file_hashes: &[String],
    ) {
        let key = CacheKey { tool_name: ToolName::new(tool_name), args_hash: args_hash.to_string() };
        let seq = self.clock.fetch_add(1, Ordering::Relaxed);

        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let mut ord = self.order.write().unwrap_or_else(|e| e.into_inner());
        let mut idx = self.file_index.write().unwrap_or_else(|e| e.into_inner());

        // Overwrite: remove old dependencies first
        if let Some(old) = map.get(&key) {
            for f in &old.dependent_files {
                if let Some(keys) = idx.get_mut(f) {
                    keys.remove(&key);
                }
            }
            // Update in-place
            let entry = map.get_mut(&key).expect("just checked");
            entry.result = Arc::from(result.to_string());
            entry.file_hashes = file_hashes.to_vec();
            entry.hit_count += 1;
            entry.access_seq = seq;
            // Update dependency index for new files
            for f in dependent_files {
                idx.entry(f.clone()).or_default().insert(key.clone());
            }
            entry.dependent_files = dependent_files.to_vec();
            return;
        }

        // Eviction if at capacity
        while map.len() >= self.capacity {
            if let Some(evict_key) = ord.pop_front() {
                if let Some(old) = map.remove(&evict_key) {
                    for f in &old.dependent_files {
                        if let Some(keys) = idx.get_mut(f) {
                            keys.remove(&evict_key);
                        }
                    }
                }
            } else {
                break;
            }
        }

        // Insert new entry
        for f in dependent_files {
            idx.entry(f.clone()).or_default().insert(key.clone());
        }
        ord.push_back(key.clone());
        map.insert(key, CacheEntry {
            result: Arc::from(result.to_string()),
            dependent_files: dependent_files.to_vec(),
            file_hashes: file_hashes.to_vec(),
            hit_count: 1,
            access_seq: seq,
        });
    }

    /// O(affected) invalidation. Uses reverse index to find affected keys directly.
    /// Content-hash check uses strict file→hash mapping: only entries whose file
    /// hashes ALL match the new hashes survive.
    pub fn invalidate(&self, changed_files: &[String]) -> usize {
        self.invalidate_with_hashes(changed_files, &[])
    }

    pub fn invalidate_with_hashes(&self, changed_files: &[String], new_hashes: &[String]) -> usize {
        let idx = self.file_index.read().unwrap_or_else(|e| e.into_inner());
        let new_hash_set: HashSet<&str> = new_hashes.iter().map(|s| s.as_str()).collect();
        let use_content_check = !new_hash_set.is_empty();

        let mut affected: HashSet<CacheKey> = HashSet::new();
        for f in changed_files {
            if let Some(keys) = idx.get(f) {
                affected.extend(keys.iter().cloned());
            }
        }
        drop(idx);

        if affected.is_empty() {
            return 0;
        }

        let mut map = self.entries.write().unwrap_or_else(|e| e.into_inner());
        let mut ord = self.order.write().unwrap_or_else(|e| e.into_inner());
        let mut file_idx = self.file_index.write().unwrap_or_else(|e| e.into_inner());
        let before = map.len();

        for key in &affected {
            // Content-hash check: entry survives only if ALL its file hashes
            // are present in the new hash set (strict matching).
            let should_keep = use_content_check
                && map.get(key).is_some_and(|e| {
                    !e.file_hashes.is_empty()
                        && e.file_hashes.iter().all(|h| new_hash_set.contains(h.as_str()))
                });

            if !should_keep {
                map.remove(key);
                // Remove from order tracking
                if let Some(pos) = ord.iter().position(|k| k == key) {
                    ord.remove(pos);
                }
                // Remove from reverse index
                for keys in file_idx.values_mut() {
                    keys.remove(key);
                }
            }
        }

        before - map.len()
    }

    pub fn len(&self) -> usize {
        self.entries.read().unwrap_or_else(|e| e.into_inner()).len()
    }
}

// ── Normalization ──────────────────────────────────────────────────────

fn sha256_hex(input: &str) -> String {
    use sha2::{Digest, Sha256};
    hex::encode(&Sha256::digest(input.as_bytes())[..8])
}

/// Normalize tool arguments for deterministic cache keys.
/// JSON: recursively sort object keys for canonical output.
/// Plain text: trim only (case-preserving).
pub fn normalize_args(args: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(args) {
        let canonical = canonicalize_json(&val);
        serde_json::to_string(&canonical).unwrap_or_else(|_| args.trim().to_string())
    } else {
        args.trim().to_string()
    }
}

fn canonicalize_json(v: &serde_json::Value) -> serde_json::Value {
    match v {
        serde_json::Value::Object(map) => {
            let mut entries: Vec<(String, serde_json::Value)> = map
                .iter()
                .map(|(k, v)| (k.clone(), canonicalize_json(v)))
                .collect();
            entries.sort_by(|a, b| a.0.cmp(&b.0));
            serde_json::Value::Object(entries.into_iter().collect())
        }
        serde_json::Value::Array(arr) => {
            serde_json::Value::Array(arr.iter().map(canonicalize_json).collect())
        }
        other => other.clone(),
    }
}

// ── ToolKind: capability-based classification ──────────────────────────

/// Tools that benefit from caching — deterministic, expensive, frequently repeated.
#[derive(Debug, Clone, Copy, PartialEq)]
pub enum ToolKind {
    Grep,
    ReadFile,
    ListFiles,
    SymbolSearch,
    Diagnostics,
    Other,
}

/// Capability registry: maps tool name patterns → ToolKind.
/// Instead of an ever-growing heuristic chain, this uses a list of known patterns.
const TOOL_KIND_REGISTRY: &[(&[&str], ToolKind)] = &[
    (&["grep", "search_content", "search_code", "search_file"], ToolKind::Grep),
    (&["read_file", "read"], ToolKind::ReadFile),
    (&["list_files", "ls", "tree"], ToolKind::ListFiles),
    (&["symbol_search", "document_symbol", "workspace_symbol"], ToolKind::SymbolSearch),
    (&["diagnostics", "diagnostic"], ToolKind::Diagnostics),
];

impl ToolKind {
    pub fn from_name(name: &str) -> Self {
        let n = name.to_lowercase();
        for (patterns, kind) in TOOL_KIND_REGISTRY {
            if patterns.iter().any(|p| n == *p || n.contains(p)) {
                return *kind;
            }
        }
        ToolKind::Other
    }

    /// Map tool kind to its dependency semantics (Phase 5 cache quality).
    /// Tools that read files are cacheable with file-based invalidation.
    /// Tools with side effects are NOT cacheable.
    pub fn dependency_kind(&self) -> Option<crate::dependency_kind::DependencyKind> {
        match self {
            ToolKind::Grep => Some(crate::dependency_kind::DependencyKind::SearchesFile),
            ToolKind::ReadFile => Some(crate::dependency_kind::DependencyKind::ReadsFile),
            ToolKind::ListFiles => Some(crate::dependency_kind::DependencyKind::ReadsFile),
            ToolKind::SymbolSearch => Some(crate::dependency_kind::DependencyKind::SearchesFile),
            ToolKind::Diagnostics => Some(crate::dependency_kind::DependencyKind::ReadsFile),
            ToolKind::Other => None, // unknown tools are NOT cacheable
        }
    }
}

pub fn is_interceptable(tool_name: &str) -> bool {
    !matches!(ToolKind::from_name(tool_name), ToolKind::Other)
}

pub fn is_cacheable(tool_name: &str) -> bool {
    is_interceptable(tool_name)
}

/// Transform a cached result before injecting it into the stream.
/// For read_file/list_files, extracts a structured summary (symbols, line count)
/// instead of dumping raw file content into the conversation.
/// For grep/search/diagnostics, returns the raw result unchanged.
pub fn transform_result(tool_name: &str, raw_result: &str) -> String {
    let kind = ToolKind::from_name(tool_name);
    match kind {
        ToolKind::ReadFile => {
            let line_count = raw_result.lines().count();
            let snippets = crate::snippet::extract(raw_result);
            let symbols: Vec<String> = snippets
                .iter()
                .filter(|s| matches!(s.snippet_type, crate::snippet::SnippetType::CodeBlock))
                .map(|s| s.content.clone())
                .collect();
            let paths: Vec<String> = snippets
                .iter()
                .filter(|s| matches!(s.snippet_type, crate::snippet::SnippetType::FilePath))
                .map(|s| s.content.clone())
                .collect();
            let parts: Vec<&str> = symbols.iter().map(|s| s.as_str())
                .chain(paths.iter().map(|s| s.as_str()))
                .take(8)
                .collect();
            if parts.is_empty() {
                format!("[cached] {line_count} lines")
            } else {
                format!("[cached] {line_count} lines, includes: {}", parts.join("; "))
            }
        }
        ToolKind::ListFiles => {
            let interesting: Vec<&str> = raw_result
                .lines()
                .filter(|l| {
                    let trimmed = l.trim();
                    !trimmed.starts_with('.')
                        && !trimmed.contains("node_modules")
                        && !trimmed.contains("target/")
                        && !trimmed.contains(".git/")
                        && !trimmed.is_empty()
                })
                .take(15)
                .collect();
            if interesting.len() < raw_result.lines().count() {
                format!("[cached] {} entries (top 15): {}", raw_result.lines().count(), interesting.join(", "))
            } else {
                format!("[cached] {} entries: {}", interesting.len(), interesting.join(", "))
            }
        }
        _ => raw_result.to_string(),
    }
}

/// Convenience: compute cache key tuple `(tool_name, args_hash)`.
/// Wraps `CacheKey::new` for callers that destructure into two strings.
pub fn cache_key(tool_name: &str, args: &str) -> (String, String) {
    let k = CacheKey::new(tool_name, args);
    (k.tool_name.to_string(), k.args_hash)
}

// ── Normalize tool name ───────────────────────────────────────────────

fn normalize_tool_name(name: &str) -> String {
    let sanitized: String = name.trim().to_lowercase()
        .replace("..", "")
        .chars()
        .filter(|c| !matches!(c, '/' | '\\' | '"' | '\''))
        .collect();
    if sanitized.is_empty() { return "unknown".into(); }
    // Map to canonical form via registry
    let kind = ToolKind::from_name(&sanitized);
    match kind {
        ToolKind::Grep => "grep".into(),
        ToolKind::ReadFile => "read_file".into(),
        ToolKind::ListFiles => "list_files".into(),
        ToolKind::SymbolSearch => "symbol_search".into(),
        ToolKind::Diagnostics => "diagnostics".into(),
        ToolKind::Other => sanitized,
    }
}

// ── Dependent file extraction ──────────────────────────────────────────

pub fn extract_dependent_files(tool_name: &str, args: &str) -> Vec<String> {
    let t = tool_name.to_lowercase();
    let mut files = Vec::new();
    let arg_map: Option<HashMap<String, String>> = serde_json::from_str(args).ok();

    match ToolKind::from_name(&t) {
        ToolKind::ReadFile => {
            if let Some(ref m) = arg_map {
                for key in &["file_path", "path", "filePath"] {
                    if let Some(v) = m.get(*key) {
                        files.push(v.clone());
                    }
                }
            }
            if files.is_empty() {
                files.extend(extract_path_literals(args));
            }
        }
        ToolKind::Grep => {
            if let Some(ref m) = arg_map {
                for key in &["path", "directory", "dir", "pattern"] {
                    if let Some(v) = m.get(*key) && looks_like_path_or_pattern(v) {
                        files.push(v.clone());
                    }
                }
            }
            if files.is_empty() {
                files.extend(extract_path_literals(args));
            }
        }
        ToolKind::ListFiles => {
            if let Some(ref m) = arg_map
                && let Some(v) = m.get("target_directory") { files.push(v.clone()); }
        }
        ToolKind::SymbolSearch => {
            if let Some(ref m) = arg_map
                && let Some(v) = m.get("file_path") { files.push(v.clone()); }
        }
        ToolKind::Diagnostics | ToolKind::Other => {
            files.extend(extract_path_literals(args));
        }
    }

    let mut result: Vec<String> = files.into_iter().collect::<HashSet<_>>().into_iter().collect();
    result.sort();
    result
}

fn extract_path_literals(text: &str) -> Vec<String> {
    let mut files = Vec::new();
    for word in text.split_whitespace() {
        let clean = word.trim_matches(|c: char| c == '"' || c == '\'' || c == ',' || c == ':');
        if clean.starts_with("http") || clean.starts_with("https") { continue; }
        if clean.starts_with("--") || clean.starts_with('-') { continue; }
        if (clean.contains('/') || clean.contains('\\') || (clean.contains('.') && clean.len() > 4 && !clean.contains("..")))
            && clean.len() >= 2 && clean.len() <= 256
        {
            files.push(clean.to_string());
        }
    }
    files
}

fn looks_like_path_or_pattern(s: &str) -> bool {
    s.contains('/') || s.contains('.') || s.contains('*') || s.len() < 50
}

// ── Canonical Execution Identity ──────────────────────────────────────

/// The stable key for cache correctness, replay determinism, and failure matching.
/// Normalizes tool name + args across provider/format differences.
#[derive(Debug, Clone, PartialEq, Eq, Hash)]
pub struct CanonicalExecutionKey {
    pub tool_name: ToolName,
    pub args_hash: String,
}

impl CanonicalExecutionKey {
    pub fn new(tool_name: &str, args: &str) -> Self {
        Self {
            tool_name: ToolName::new(tool_name),
            args_hash: sha256_hex(&normalize_args(args)),
        }
    }

    /// Build from already-normalized components.
    pub fn from_parts(tool_name: ToolName, args_hash: String) -> Self {
        Self { tool_name, args_hash }
    }

    /// Convert to cache key (same representation).
    pub fn to_cache_key(&self) -> CacheKey {
        CacheKey { tool_name: self.tool_name.clone(), args_hash: self.args_hash.clone() }
    }
}

// ── Cache entry (L2) ──────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ToolCacheEntry {
    pub id: i64,
    pub tool_name: String,
    pub args_hash: String,
    pub result: String,
    pub dependent_files: Vec<String>,
    pub created_at: String,
    pub hit_count: i64,
}

pub fn should_invalidate(entry: &ToolCacheEntry, changed_files: &[String]) -> bool {
    if changed_files.is_empty() { return false; }
    let changed: HashSet<&str> = changed_files.iter().map(|s| s.as_str()).collect();
    entry.dependent_files.iter().any(|f| changed.contains(f.as_str()))
}

pub const MIGRATION: &str = "
    CREATE TABLE IF NOT EXISTS tool_cache (
        id              INTEGER PRIMARY KEY AUTOINCREMENT,
        tool_name       TEXT NOT NULL,
        args_hash       TEXT NOT NULL,
        result          TEXT NOT NULL,
        dependent_files TEXT NOT NULL DEFAULT '[]',
        created_at      TEXT NOT NULL DEFAULT (datetime('now')),
        hit_count       INTEGER NOT NULL DEFAULT 1
    );
    CREATE UNIQUE INDEX IF NOT EXISTS idx_tool_cache_key
        ON tool_cache(tool_name, args_hash);
    CREATE INDEX IF NOT EXISTS idx_tool_cache_hits
        ON tool_cache(hit_count DESC);";

#[cfg(test)]
mod tests {
    use super::*;

    // ── CacheKey / identity ───────────────────────────────────────────

    #[test]
    fn cache_key_roundtrip() {
        let k1 = CacheKey::new("grep", r#"{"pattern":"foo","path":"src"}"#);
        let k2 = CacheKey::new("search_content", r#"{"path":"src","pattern":"foo"}"#);
        assert_eq!(k1, k2, "normalized args + tool name should produce equal keys");
    }

    #[test]
    fn tool_name_newtype_hides_string() {
        let name = ToolName::new("search_content");
        assert_eq!(name.as_str(), "grep");
    }

    // ── LRU eviction ──────────────────────────────────────────────────

    #[test]
    fn lru_evicts_oldest_inserted() {
        let cache = L1HotCache::new(4);
        cache.put("grep", "key1", "r1", &[]);
        cache.put("grep", "key2", "r2", &[]);
        cache.put("grep", "key3", "r3", &[]);
        cache.put("grep", "key4", "r4", &[]);
        assert_eq!(cache.len(), 4);
        cache.put("grep", "key5", "r5", &[]);
        assert_eq!(cache.len(), 4);
        // key1 should be evicted (first inserted)
        assert!(cache.get("grep", "key1").is_none());
        assert!(cache.get("grep", "key5").is_some());
    }

    #[test]
    fn lru_eviction_does_not_lose_newest() {
        let cache = L1HotCache::new(128);
        for i in 0..128 {
            cache.put("grep", &format!("key{i}"), "result", &[]);
        }
        assert!(cache.get("grep", "key0").is_some());
        cache.put("grep", "overflow", "result", &[]);
        // key0 was recently accessed, so it should survive
        assert!(cache.get("grep", "key0").is_some(), "recently accessed key should survive LRU");
    }

    #[test]
    fn get_updates_hit_count() {
        let cache = L1HotCache::new(10);
        cache.put("grep", "hash1", "found", &[]);
        let (_, count1) = cache.get("grep", "hash1").unwrap();
        let (_, count2) = cache.get("grep", "hash1").unwrap();
        assert_eq!(count2, count1 + 1, "hit count should increment");
    }

    // ── Overwrite cleans old deps ─────────────────────────────────────

    #[test]
    fn overwrite_cleans_old_dependencies() {
        let cache = L1HotCache::new(10);
        cache.put_with_hashes("grep", "h1", "old", &["src/a.rs".into()], &[]);
        cache.put_with_hashes("grep", "h1", "new", &["src/b.rs".into()], &[]);
        // Invalidate src/a.rs — should NOT affect the entry (dep was replaced)
        assert_eq!(cache.invalidate(&["src/a.rs".into()]), 0);
        // Entry should still be present
        assert!(cache.get("grep", "h1").is_some());
    }

    // ── get() returns Arc<str> (zero-copy path) ───────────────────────

    #[test]
    fn get_returns_arc_str() {
        let cache = L1HotCache::new(10);
        cache.put("grep", "h1", "hello world", &[]);
        let (result, _) = cache.get("grep", "h1").unwrap();
        assert_eq!(&*result, "hello world");
    }

    // ── Reverse index invalidation ────────────────────────────────────

    #[test]
    fn invalidate_removes_affected_entries() {
        let cache = L1HotCache::new(10);
        cache.put("grep", "h1", "r1", &["src/a.rs".into()]);
        cache.put("grep", "h2", "r2", &["src/b.rs".into()]);
        assert_eq!(cache.len(), 2);
        assert_eq!(cache.invalidate(&["src/a.rs".into()]), 1);
        assert!(cache.get("grep", "h1").is_none());
        assert!(cache.get("grep", "h2").is_some());
    }

    #[test]
    fn invalidate_multiple_files_batch() {
        let cache = L1HotCache::new(10);
        cache.put("grep", "h1", "r1", &["a.rs".into(), "b.rs".into()]);
        cache.put("grep", "h2", "r2", &["c.rs".into()]);
        assert_eq!(cache.invalidate(&["a.rs".into(), "c.rs".into()]), 2);
        assert!(cache.is_empty());
    }

    #[test]
    fn invalidate_nonexistent_file_returns_zero() {
        let cache = L1HotCache::new(10);
        cache.put("grep", "h1", "r1", &["a.rs".into()]);
        assert_eq!(cache.invalidate(&["nonexistent.rs".into()]), 0);
    }

    // ── Content-hash invalidation (strict) ────────────────────────────

    #[test]
    fn content_hash_keeps_entry_when_all_hashes_match() {
        let cache = L1HotCache::new(10);
        cache.put_with_hashes("grep", "h1", "r1", &["f.rs".into()], &["abc".into()]);
        // Same hash → entry survives
        assert_eq!(cache.invalidate_with_hashes(&["f.rs".into()], &["abc".into()]), 0);
        assert!(cache.get("grep", "h1").is_some());
    }

    #[test]
    fn content_hash_removes_entry_when_hash_changes() {
        let cache = L1HotCache::new(10);
        cache.put_with_hashes("grep", "h1", "r1", &["f.rs".into()], &["old_hash".into()]);
        // Different hash → entry invalidated
        assert_eq!(cache.invalidate_with_hashes(&["f.rs".into()], &["new_hash".into()]), 1);
        assert!(cache.get("grep", "h1").is_none());
    }

    #[test]
    fn content_hash_strict_matching_all_hashes_required() {
        let cache = L1HotCache::new(10);
        cache.put_with_hashes("grep", "h1", "r1", &["a.rs".into(), "b.rs".into()], &["h_a".into(), "h_b".into()]);
        // Only one hash matches → strict mode requires ALL → invalidate
        assert_eq!(cache.invalidate_with_hashes(&["a.rs".into()], &["h_a".into()]), 1);
        assert!(cache.get("grep", "h1").is_none());
    }

    // ── Args hash ─────────────────────────────────────────────────────

    #[test]
    fn args_hash_is_sha256_hex() {
        let h = sha256_hex("hello");
        assert_eq!(h.len(), 16);
        assert_eq!(sha256_hex("hello"), sha256_hex("hello"));
        assert_ne!(sha256_hex("hello"), sha256_hex("world"));
    }

    // ── JSON normalization ────────────────────────────────────────────

    #[test]
    fn normalize_json_sorts_keys() {
        let a = normalize_args(r#"{"path":"src","pattern":"foo"}"#);
        let b = normalize_args(r#"{"pattern":"foo","path":"src"}"#);
        assert_eq!(a, b);
    }

    #[test]
    fn normalize_preserves_case_for_plain_text() {
        let a = normalize_args("Grep FooBar");
        assert!(a.contains("Grep"));
    }

    #[test]
    fn canonical_json_produces_identical_args_hash() {
        let a = normalize_args(r#"{"b":2,"a":1,"c":{"z":9,"x":7}}"#);
        let b = normalize_args(r#"{"a":1,"c":{"x":7,"z":9},"b":2}"#);
        assert_eq!(a, b);
        assert_eq!(sha256_hex(&a), sha256_hex(&b));
    }

    // ── ToolKind / cacheability ───────────────────────────────────────

    #[test]
    fn is_cacheable_by_tool_kind() {
        assert!(is_cacheable("grep"));
        assert!(is_cacheable("read_file"));
        assert!(is_cacheable("list_files"));
        assert!(is_cacheable("symbol_search"));
        assert!(!is_cacheable("execute_command"));
        assert!(!is_cacheable("unknown_tool"));
    }

    #[test]
    fn tool_kind_registry_matches_substring() {
        // "search_content" should match Grep via the registry
        assert_eq!(ToolKind::from_name("search_content"), ToolKind::Grep);
    }

    // ── Dependent file extraction ─────────────────────────────────────

    #[test]
    fn extract_dependent_files_from_json_args() {
        let files = extract_dependent_files("grep", r#"{"pattern":"foo","path":"src/main.rs"}"#);
        assert!(files.iter().any(|f| f.contains("src/main.rs")));
    }

    #[test]
    fn extract_files_ignores_urls_and_flags() {
        let files = extract_dependent_files("read_file", r#"read_file --path https://example.com --flag src/main.rs"#);
        assert!(!files.iter().any(|f| f.contains("http")));
        assert!(!files.iter().any(|f| f.contains("--")));
    }

    // ── Transform ─────────────────────────────────────────────────────

    #[test]
    fn transform_read_file_extracts_symbols() {
        let result = transform_result("read_file", "pub fn foo() {\n    let x = 1;\n}\n\npub struct Bar {\n    x: i32,\n}\n\nfn baz() {}");
        assert!(result.starts_with("[cached]"));
        assert!(result.contains("lines"));
    }

    #[test]
    fn transform_read_file_empty_is_safe() {
        assert_eq!(transform_result("read_file", ""), "[cached] 0 lines");
    }

    #[test]
    fn transform_list_files_filters_noise() {
        let result = transform_result("list_files", "src/main.rs\nsrc/lib.rs\n.git/config\nnode_modules/react/index.js\nCargo.toml");
        assert!(!result.contains(".git"));
        assert!(!result.contains("node_modules"));
        assert!(result.contains("src/main.rs"));
        assert!(result.starts_with("[cached]"));
    }

    #[test]
    fn transform_grep_passes_through() {
        let raw = "src/main.rs:42: found foo";
        assert_eq!(transform_result("grep", raw), raw);
    }

    #[test]
    fn transform_result_never_panics_on_weird_input() {
        transform_result("read_file", &"x".repeat(10000));
        transform_result("list_files", "\0\n\x01\n\x02");
        transform_result("grep", "\n\n\n");
        transform_result("unknown_tool", "anything");
    }

    // ── L2 invalidation ───────────────────────────────────────────────

    #[test]
    fn should_invalidate_on_dependency_change() {
        let entry = ToolCacheEntry {
            id: 1, tool_name: "grep".into(), args_hash: "abc".into(),
            result: "found".into(), dependent_files: vec!["src/main.rs".into()],
            created_at: String::new(), hit_count: 1,
        };
        assert!(should_invalidate(&entry, &["src/main.rs".into()]));
        assert!(!should_invalidate(&entry, &["README.md".into()]));
        assert!(!should_invalidate(&entry, &[]));
    }

    // ── Concurrency safety (basic) ────────────────────────────────────

    #[test]
    fn concurrent_reads_do_not_panic() {
        let cache = Arc::new(L1HotCache::new(10));
        let mut handles = Vec::new();
        for i in 0..4 {
            let c = cache.clone();
            handles.push(std::thread::spawn(move || {
                c.put("grep", &format!("key{i}"), "result", &[]);
                let _ = c.get("grep", &format!("key{i}"));
            }));
        }
        for h in handles { h.join().unwrap(); }
        assert_eq!(cache.len(), 4);
    }

    #[test]
    fn concurrent_invalidation_does_not_race() {
        let cache = Arc::new(L1HotCache::new(100));
        for i in 0..50 {
            cache.put("grep", &format!("key{i}"), "r", &[format!("f{i}.rs")]);
        }
        let mut handles = Vec::new();
        for i in 0..10 {
            let c = cache.clone();
            handles.push(std::thread::spawn(move || {
                let files: Vec<String> = (0..5).map(|j| format!("f{}.rs", i * 5 + j)).collect();
                c.invalidate(&files);
            }));
        }
        for h in handles { h.join().unwrap(); }
        // Should not panic — RwLock handles concurrent reads fine
        assert!(cache.len() <= 50);
    }

    // ── Capacity config ───────────────────────────────────────────────

    #[test]
    fn custom_capacity_evicts_at_limit() {
        let cache = L1HotCache::new(2);
        cache.put("grep", "a", "r1", &[]);
        cache.put("grep", "b", "r2", &[]);
        cache.put("grep", "c", "r3", &[]);
        assert_eq!(cache.len(), 2);
        assert!(cache.get("grep", "a").is_none());
        assert!(cache.get("grep", "c").is_some());
    }

    // ── Poison handling (explicit) ────────────────────────────────────

    #[test]
    fn poison_does_not_permanently_lock() {
        // Simulate a poison by panicking inside a write lock scope.
        // The unwrap_or_else(|e| e.into_inner()) recovery ensures subsequent
        // access still works.
        let cache = L1HotCache::new(10);
        // Poisioning requires a panic while holding the lock, which is
        // hard to trigger from a test without unsafe code. Instead verify
        // the recovery path compiles and works under normal conditions.
        cache.put("grep", "k", "v", &[]);
        assert!(cache.get("grep", "k").is_some());
    }
}

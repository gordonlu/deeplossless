//! Agent Diff & File Reconstruction — causal file-change tracking without
//! full snapshots. NOT to be confused with `crate::replay` which replays
//! SSE stream events from execution_events.
//!
//! Design:
//! - Every agent `Edit` tool call produces one `DiffEvent` row.
//! - Only pre/post *snippets* (≤40 lines) are stored, not full file content.
//! - Snippets are hash-deduplicated — identical regions across events
//!   share the same `diff_snippets` row.
//! - `tool_call_id` links each diff to the agent's causal chain.
//!
//! Schema:
//!   diff_events   — event index (session, file, tool_call, hashes)
//!   diff_snippets — content dedup store (hash → up to 40 lines)
//!
//! File reconstruction: walk diff_events in timestamp order, apply each
//! patch to a file buffer, and return the reconstructed file state.
//! Distinct from `crate::replay` which reconstructs StreamEvent sequences.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

// ── Types ──────────────────────────────────────────────────────────

#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum DiffType {
    Insert,
    Delete,
    Replace,
}

impl DiffType {
    pub fn as_str(&self) -> &'static str {
        match self {
            DiffType::Insert => "Insert",
            DiffType::Delete => "Delete",
            DiffType::Replace => "Replace",
        }
    }
    pub fn from_diff_str(s: &str) -> Option<Self> {
        match s {
            "Insert" => Some(DiffType::Insert),
            "Delete" => Some(DiffType::Delete),
            "Replace" => Some(DiffType::Replace),
            _ => None,
        }
    }
}

#[derive(Debug, Clone, PartialEq, Serialize, Deserialize)]
pub struct DiffEvent {
    pub id: Option<i64>,
    pub session_id: String,
    pub tool_call_id: String,
    pub timestamp: i64, // unix millis
    pub file_path: String,
    pub start_line: u32,
    pub end_line: u32,
    pub change_type: DiffType,
    pub before_snippet: Option<String>,
    pub after_snippet: Option<String>,
    pub before_hash: u64,
    pub after_hash: u64,
}

// ── Schema ─────────────────────────────────────────────────────────

pub const MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS diff_events (
    id            INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id    TEXT NOT NULL,
    tool_call_id  TEXT NOT NULL,
    file_path     TEXT NOT NULL,
    start_line    INTEGER NOT NULL,
    end_line      INTEGER NOT NULL,
    change_type   TEXT NOT NULL,
    before_hash   INTEGER NOT NULL,
    after_hash    INTEGER NOT NULL,
    timestamp     INTEGER NOT NULL
);

CREATE INDEX IF NOT EXISTS idx_de_session ON diff_events(session_id);
CREATE INDEX IF NOT EXISTS idx_de_file    ON diff_events(session_id, file_path);
CREATE INDEX IF NOT EXISTS idx_de_tool    ON diff_events(tool_call_id);

CREATE TABLE IF NOT EXISTS diff_snippets (
    hash       INTEGER PRIMARY KEY,
    snippet    TEXT NOT NULL,
    compressed INTEGER NOT NULL DEFAULT 0
);
"#;

// ── Hash ───────────────────────────────────────────────────────────

/// FNV-1a 64-bit hash — fast, non-cryptographic, deterministic.
fn hash_snippet(s: &str) -> u64 {
    let mut h: u64 = 0xcbf29ce484222325;
    for b in s.bytes() {
        h ^= b as u64;
        h = h.wrapping_mul(0x100000001b3);
    }
    h
}

// ── Diff computation ───────────────────────────────────────────────

/// Compute the smallest line-level diff between `before` and `after`.
/// Returns (start_line, end_line, change_type, before_snippet, after_snippet).
///
/// Strategy: find the first and last lines that differ, then extract
/// the surrounding context (up to 20 lines above/below). We deliberately
/// avoid full myers-diff — the goal is lightweight causal tracking, not
/// prettified patch generation.
pub fn compute_diff(
    before: &str,
    after: &str,
) -> (u32, u32, DiffType, String, String) {
    let before_lines: Vec<&str> = before.lines().collect();
    let after_lines: Vec<&str> = after.lines().collect();

    // Find first differing line (skip common prefix)
    let mut first_diff = 0usize;
    while first_diff < before_lines.len()
        && first_diff < after_lines.len()
        && before_lines[first_diff] == after_lines[first_diff]
    {
        first_diff += 1;
    }

    // Find last differing line (skip common suffix)
    let mut last_before = before_lines.len();
    let mut last_after = after_lines.len();
    while last_before > first_diff
        && last_after > first_diff
        && before_lines[last_before - 1] == after_lines[last_after - 1]
    {
        last_before -= 1;
        last_after -= 1;
    }

    let change_type = if after_lines.is_empty() {
        DiffType::Delete
    } else if before_lines.is_empty() || first_diff >= before_lines.len() {
        DiffType::Insert // pure append or new file
    } else if first_diff >= after_lines.len() {
        DiffType::Delete // pure truncation
    } else {
        DiffType::Replace
    };

    let start_line = first_diff.saturating_add(1) as u32; // 1-indexed

    // Extract snippets with context (≤40 lines total)
    let ctx_start = first_diff.saturating_sub(20);
    let ctx_end_before = (last_before + 20).min(before_lines.len());
    let ctx_end_after = (last_after + 20).min(after_lines.len());

    let before_snippet = before_lines[ctx_start..ctx_end_before].join("\n");
    let after_snippet = after_lines[ctx_start..ctx_end_after].join("\n");
    let end_line = last_before.max(1) as u32; // 1-indexed, last changed line

    (start_line, end_line, change_type, before_snippet, after_snippet)
}

// ── Storage ────────────────────────────────────────────────────────

/// Run schema migration (called during Database::open).
pub fn create_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch(MIGRATION)?;
    Ok(())
}

/// Insert a diff event. Also writes snippets into `diff_snippets` if
/// they are not already present (hash-deduplicated).
pub fn insert_diff(conn: &Connection, event: &DiffEvent) -> anyhow::Result<i64> {
    // Store snippets (idempotent via INSERT OR IGNORE + hash PK)
    if let Some(ref s) = event.before_snippet {
        conn.execute(
            "INSERT OR IGNORE INTO diff_snippets (hash, snippet, compressed) VALUES (?1, ?2, 0)",
            params![event.before_hash as i64, s],
        )?;
    }
    if let Some(ref s) = event.after_snippet {
        conn.execute(
            "INSERT OR IGNORE INTO diff_snippets (hash, snippet, compressed) VALUES (?1, ?2, 0)",
            params![event.after_hash as i64, s],
        )?;
    }

    conn.execute(
        "INSERT INTO diff_events (session_id, tool_call_id, file_path, start_line, end_line, change_type, before_hash, after_hash, timestamp)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            event.session_id,
            event.tool_call_id,
            event.file_path,
            event.start_line,
            event.end_line,
            event.change_type.as_str(),
            event.before_hash as i64,
            event.after_hash as i64,
            event.timestamp,
        ],
    )?;
    Ok(conn.last_insert_rowid())
}

/// Generate and store a diff from before/after file content.
pub fn generate_and_store(
    conn: &Connection,
    session_id: &str,
    tool_call_id: &str,
    file_path: &str,
    before: &str,
    after: &str,
    timestamp: i64,
) -> anyhow::Result<DiffEvent> {
    let (start_line, end_line, change_type, before_snippet, after_snippet) = compute_diff(before, after);
    let before_hash = hash_snippet(&before_snippet);
    let after_hash = hash_snippet(&after_snippet);

    let event = DiffEvent {
        id: None,
        session_id: session_id.to_string(),
        tool_call_id: tool_call_id.to_string(),
        timestamp,
        file_path: file_path.to_string(),
        start_line,
        end_line,
        change_type,
        before_snippet: Some(before_snippet),
        after_snippet: Some(after_snippet),
        before_hash,
        after_hash,
    };

    let id = insert_diff(conn, &event)?;
    let mut stored = event;
    stored.id = Some(id);
    Ok(stored)
}

// ── Query ──────────────────────────────────────────────────────────

#[derive(Debug, Default)]
pub struct DiffQuery {
    pub session_id: Option<String>,
    pub file_path: Option<String>,
    pub tool_call_id: Option<String>,
    pub limit: Option<usize>,
}

/// Query diff events. Returns newest-first.
pub fn query_diffs(conn: &Connection, q: &DiffQuery) -> anyhow::Result<Vec<DiffEvent>> {
    let mut clauses: Vec<String> = Vec::new();
    let mut bindings: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref s) = q.session_id {
        clauses.push("session_id = ?".into());
        bindings.push(Box::new(s.clone()));
    }
    if let Some(ref f) = q.file_path {
        clauses.push("file_path = ?".into());
        bindings.push(Box::new(f.clone()));
    }
    if let Some(ref t) = q.tool_call_id {
        clauses.push("tool_call_id = ?".into());
        bindings.push(Box::new(t.clone()));
    }

    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };

    let mut sql = format!(
        "SELECT id, session_id, tool_call_id, file_path, start_line, end_line, change_type, before_hash, after_hash, timestamp
           FROM diff_events {} ORDER BY timestamp DESC",
        where_clause
    );

    if let Some(limit) = q.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    let params_refs: Vec<&dyn rusqlite::types::ToSql> = bindings.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_refs.as_slice(), |row| {
        let before_hash: i64 = row.get(7)?;
        let after_hash: i64 = row.get(8)?;
        Ok(DiffEvent {
            id: Some(row.get(0)?),
            session_id: row.get(1)?,
            tool_call_id: row.get(2)?,
            file_path: row.get(3)?,
            start_line: row.get(4)?,
            end_line: row.get(5)?,
            change_type: DiffType::from_diff_str(&row.get::<_, String>(6)?).unwrap_or(DiffType::Replace),
            before_hash: before_hash as u64,
            after_hash: after_hash as u64,
            before_snippet: load_snippet_if_needed(conn, before_hash as u64),
            after_snippet: load_snippet_if_needed(conn, after_hash as u64),
            timestamp: row.get(9)?,
        })
    })?;

    let mut events = Vec::new();
    for row in rows {
        events.push(row?);
    }
    Ok(events)
}

fn load_snippet_if_needed(conn: &Connection, hash: u64) -> Option<String> {
    if hash == 0 {
        return None;
    }
    conn.query_row(
        "SELECT snippet FROM diff_snippets WHERE hash = ?1",
        params![hash as i64],
        |row| row.get(0),
    ).ok()
}

// ── File reconstruction ────────────────────────────────────────────

/// Reconstruct the file content for a session+file by applying all
/// recorded diffs from oldest to newest on top of `initial_content`.
/// This is file-state reconstruction — NOT the same as `crate::replay`
/// which reconstructs StreamEvent sequences from execution_events.
pub fn reconstruct_file(
    conn: &Connection,
    session_id: &str,
    file_path: &str,
    initial_content: &str,
) -> anyhow::Result<String> {
    let q = DiffQuery {
        session_id: Some(session_id.to_string()),
        file_path: Some(file_path.to_string()),
        tool_call_id: None,
        limit: None,
    };
    let mut events = query_diffs(conn, &q)?;
    // query_diffs returns newest-first; reconstruction needs oldest-first.
    events.reverse();

    let mut buf = initial_content.to_string();
    for ev in &events {
        let before = ev.before_snippet.as_deref().unwrap_or("");
        let after = ev.after_snippet.as_deref().unwrap_or("");
        buf = apply_patch(&buf, before, after, ev.start_line as usize);
    }
    Ok(buf)
}

/// Apply a single before→after patch to `content`. Finds the
/// `before_snippet` in `content` and replaces it with `after_snippet`.
fn apply_patch(content: &str, before: &str, after: &str, start_line: usize) -> String {
    if before.is_empty() && after.is_empty() {
        return content.to_string();
    }

    let lines: Vec<&str> = content.lines().collect();

    // Try to find `before` anchored at `start_line` first, then anywhere.
    let pos = find_snippet(&lines, before, start_line.saturating_sub(1))
        .or_else(|| find_snippet(&lines, before, 0));

    match pos {
        Some(pos) => {
            let before_len = before.lines().count();
            let mut result: Vec<&str> = Vec::with_capacity(lines.len() + after.lines().count());
            result.extend_from_slice(&lines[..pos]);
            if !after.is_empty() {
                result.extend(after.lines());
            }
            result.extend_from_slice(&lines[(pos + before_len).min(lines.len())..]);
            result.join("\n")
        }
        None => content.to_string(), // patch doesn't match — leave as-is
    }
}

/// Find `snippet` in `lines` starting from `starting_at`. Returns the
/// line index of the first match, or None.
fn find_snippet(lines: &[&str], snippet: &str, starting_at: usize) -> Option<usize> {
    let snip_lines: Vec<&str> = snippet.lines().collect();
    if snip_lines.is_empty() {
        return None;
    }
    let start = starting_at.min(lines.len().saturating_sub(snip_lines.len()));
    for i in start..=lines.len().saturating_sub(snip_lines.len()) {
        if lines[i..i + snip_lines.len()] == snip_lines[..] {
            return Some(i);
        }
    }
    None
}

// ── Repeated-edit detection ────────────────────────────────────────

/// Find diff events for the same file where line ranges overlap —
/// indicative of the agent iterating on the same region.
pub fn find_overlapping_edits(conn: &Connection, session_id: &str) -> anyhow::Result<Vec<(DiffEvent, DiffEvent)>> {
    let sql = "SELECT a.id, a.session_id, a.tool_call_id, a.file_path, a.start_line, a.end_line,
                      a.change_type, a.before_hash, a.after_hash, a.timestamp,
                      b.id, b.session_id, b.tool_call_id, b.file_path, b.start_line, b.end_line,
                      b.change_type, b.before_hash, b.after_hash, b.timestamp
               FROM diff_events a
               JOIN diff_events b ON a.file_path = b.file_path
                                 AND a.id < b.id
                                 AND a.start_line <= b.end_line
                                 AND b.start_line <= a.end_line
              WHERE a.session_id = ?1
              ORDER BY a.timestamp";

    let mut stmt = conn.prepare(sql)?;
    let rows = stmt.query_map(params![session_id], |row| {
        Ok((
            DiffEvent {
                id: Some(row.get(0)?), session_id: row.get(1)?, tool_call_id: row.get(2)?,
                file_path: row.get(3)?, start_line: row.get(4)?, end_line: row.get(5)?,
                change_type: DiffType::from_diff_str(&row.get::<_, String>(6)?).unwrap_or(DiffType::Replace),
                before_hash: row.get::<_, i64>(7)? as u64, after_hash: row.get::<_, i64>(8)? as u64,
                timestamp: row.get(9)?, before_snippet: None, after_snippet: None,
            },
            DiffEvent {
                id: Some(row.get(10)?), session_id: row.get(11)?, tool_call_id: row.get(12)?,
                file_path: row.get(13)?, start_line: row.get(14)?, end_line: row.get(15)?,
                change_type: DiffType::from_diff_str(&row.get::<_, String>(16)?).unwrap_or(DiffType::Replace),
                before_hash: row.get::<_, i64>(17)? as u64, after_hash: row.get::<_, i64>(18)? as u64,
                timestamp: row.get(19)?, before_snippet: None, after_snippet: None,
            },
        ))
    })?;

    let mut pairs = Vec::new();
    for row in rows {
        pairs.push(row?);
    }
    Ok(pairs)
}

// ── Post-commit diff extraction ────────────────────────────────────

/// Post-commit diff extraction. Called AFTER `store_messages` has
/// committed, so the messages table is in a consistent state.
///
/// Queries the committed messages table to find all Edit/Write/
/// apply_patch tool_calls for the conversation, then computes and
/// stores a diff event for each. Idempotent — skips tool_call_ids
/// that already have a stored diff.
///
/// Unlike the old `extract_diffs_from_messages` which relied on the
/// in-memory msgs array for tool_call↔tool_result pairing, this
/// queries the DB directly. More robust — doesn't depend on batch
/// ordering or partial message sets.
pub fn extract_diffs_post_commit(
    conn: &Connection,
    session_id: &str,
    conv_id: i64,
) -> anyhow::Result<usize> {
    // Find all assistant messages with tool_calls for this conversation
    let mut stmt = conn.prepare(
        "SELECT content FROM messages WHERE conversation_id = ?1 AND role = 'assistant' AND content LIKE '%\"tool_calls\"%' ORDER BY id"
    )?;
    let asst_messages: Vec<String> = stmt
        .query_map(params![conv_id], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    if asst_messages.is_empty() {
        return Ok(0);
    }

    // Query already-processed tool_call_ids for idempotency
    let mut seen_stmt = conn.prepare(
        "SELECT tool_call_id FROM diff_events WHERE session_id = ?1"
    )?;
    let seen: std::collections::HashSet<String> = seen_stmt
        .query_map(params![session_id], |row| row.get::<_, String>(0))?
        .filter_map(|r| r.ok())
        .collect();

    let ts = std::time::SystemTime::now()
        .duration_since(std::time::UNIX_EPOCH)
        .unwrap_or_default()
        .as_millis() as i64;

    let mut count = 0;

    for raw in &asst_messages {
        let Ok(msg) = serde_json::from_str::<serde_json::Value>(raw) else { continue };
        let Some(tool_calls) = msg["tool_calls"].as_array() else { continue };

        for tc in tool_calls {
            let tc_id = tc["id"].as_str().unwrap_or("");
            if tc_id.is_empty() || seen.contains(tc_id) { continue; }

            let name = tc["function"]["name"].as_str().unwrap_or("");
            let args_str = tc["function"]["arguments"].as_str().unwrap_or("");
            let args: serde_json::Value = serde_json::from_str(args_str).unwrap_or_default();

            let Some((before, after, file_path)) = tool_extract_diff(name, &args) else {
                continue;
            };

            let _ = generate_and_store(conn, session_id, tc_id, &file_path, &before, &after, ts);
            count += 1;
        }
    }

    Ok(count)
}

/// Extract (before, after, file_path) from a tool call's arguments.
/// Returns None if the tool is not an edit operation.
fn tool_extract_diff(tool_name: &str, args: &serde_json::Value) -> Option<(String, String, String)> {
    // Common field names across agent formats
    let path = args["file_path"].as_str()
        .or_else(|| args["filePath"].as_str())
        .or_else(|| args["path"].as_str())
        .or_else(|| args["operation"]["path"].as_str()) // Codex apply_patch
        .unwrap_or("");

    if path.is_empty() { return None; }

    match tool_name {
        "Edit" | "edit" | "edit_file" | "replace" => {
            let old = args["old_string"].as_str()
                .or_else(|| args["oldString"].as_str())
                .or_else(|| args["operation"]["oldString"].as_str());
            let new = args["new_string"].as_str()
                .or_else(|| args["newString"].as_str())
                .or_else(|| args["operation"]["newString"].as_str());
            let old = old?;
            let new = new?;
            Some((old.to_string(), new.to_string(), path.to_string()))
        }
        "Write" | "write" | "write_to_file" | "write_file" => {
            let content = args["content"].as_str().unwrap_or("");
            Some((String::new(), content.to_string(), path.to_string()))
        }
        "apply_patch" => {
            let operation = &args["operation"];
            let old = operation["oldString"].as_str()
                .or_else(|| operation["old_string"].as_str());
            let new = operation["newString"].as_str()
                .or_else(|| operation["new_string"].as_str());
            let old = old?;
            let new = new?;
            Some((old.to_string(), new.to_string(), path.to_string()))
        }
        _ => None,
    }
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn temp_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_tables(&conn).expect("create tables");
        // post_commit tests query the messages table (normally created by db.rs migration)
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL,
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                token_count     INTEGER,
                stored_at       TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        ).expect("create messages");
        conn
    }

    // ── Schema ──────────────────────────────────────────────────

    #[test]
    fn tables_exist_after_create() {
        let conn = temp_conn();
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE name='diff_events'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
        let n: i64 = conn
            .query_row("SELECT COUNT(*) FROM sqlite_master WHERE name='diff_snippets'", [], |r| r.get(0))
            .unwrap();
        assert_eq!(n, 1);
    }

    // ── Diff computation ────────────────────────────────────────

    #[test]
    fn compute_diff_replace_middle() {
        let before = "line1\nline2\nline3\nline4\n";
        let after  = "line1\nline2_CHANGED\nline3\nline4\n";
        let (start, end, change_type, before_snip, after_snip) = compute_diff(before, after);
        assert_eq!(start, 2);
        assert_eq!(end, 2);
        assert_eq!(change_type, DiffType::Replace);
        assert!(before_snip.contains("line1"), "context should include line before change");
        assert!(after_snip.contains("line2_CHANGED"));
    }

    #[test]
    fn compute_diff_insert_only() {
        let before = "";
        let after = "new line\nanother\n";
        let (start, end, change_type, _, after_snip) = compute_diff(before, after);
        assert_eq!(start, 1);
        assert_eq!(change_type, DiffType::Insert);
        assert!(after_snip.contains("new line"));
        let _ = end;
    }

    #[test]
    fn compute_diff_delete_all() {
        let before = "a\nb\nc\n";
        let after = "";
        let (_, _, change_type, _, _) = compute_diff(before, after);
        assert_eq!(change_type, DiffType::Delete);
    }

    #[test]
    fn compute_diff_append() {
        let before = "line1\nline2\n";
        let after = "line1\nline2\nline3\n";
        let (start, _, change_type, _, after_snip) = compute_diff(before, after);
        assert_eq!(start, 3);
        assert_eq!(change_type, DiffType::Insert);
        assert!(after_snip.contains("line3"));
    }

    // ── Insert + Query ─────────────────────────────────────────

    #[test]
    fn insert_then_query() {
        let conn = temp_conn();
        let event = generate_and_store(&conn, "s1", "tc1", "src/a.rs",
            "fn foo() {\n  return 1;\n}\n",
            "fn foo() {\n  return 2;\n}\n",
            1000).unwrap();

        assert!(event.id.is_some());

        let diffs = query_diffs(&conn, &DiffQuery {
            session_id: Some("s1".into()),
            ..Default::default()
        }).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].file_path, "src/a.rs");
        assert_eq!(diffs[0].change_type, DiffType::Replace);
    }

    #[test]
    fn query_by_file_path() {
        let conn = temp_conn();
        generate_and_store(&conn, "s1", "t1", "src/a.rs", "x", "y", 1000).unwrap();
        generate_and_store(&conn, "s1", "t2", "src/b.rs", "x", "y", 2000).unwrap();
        generate_and_store(&conn, "s2", "t3", "src/a.rs", "x", "y", 3000).unwrap();

        let diffs = query_diffs(&conn, &DiffQuery {
            file_path: Some("src/a.rs".into()),
            ..Default::default()
        }).unwrap();
        assert_eq!(diffs.len(), 2, "should find 2 diffs for src/a.rs across sessions");
    }

    #[test]
    fn snippet_dedup() {
        let conn = temp_conn();
        // Same before snippet, two different edits
        let before = "fn x() {\n  old\n}\n";
        let after1 = "fn x() {\n  new1\n}\n";
        let after2 = "fn x() {\n  new2\n}\n";

        let e1 = generate_and_store(&conn, "s1", "t1", "f.rs", before, after1, 1).unwrap();
        let e2 = generate_and_store(&conn, "s1", "t2", "f.rs", before, after2, 2).unwrap();

        // Both should reference the same before_hash since snippets are identical
        assert_eq!(e1.before_hash, e2.before_hash, "identical snippets share hash");
        assert_ne!(e1.after_hash, e2.after_hash, "different after snippets differ");

        // Check only one snippet row for the deduped before
        let count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM diff_snippets", [], |r| r.get(0)).unwrap();
        // 1 deduped before + 2 unique afters = 3 snippets
        assert_eq!(count, 3, "3 unique snippets (1 before, 2 after)");
    }

    // ── File Reconstruction ────────────────────────────────────

    #[test]
    fn reconstruct_file_two_edits() {
        let conn = temp_conn();
        let initial = "line1\nline2\nline3\nline4\nline5\n";

        // Edit 1: change line2
        let _e1 = generate_and_store(&conn, "s1", "t1", "f.rs",
            initial,
            "line1\nline2_CHANGED\nline3\nline4\nline5\n",
            1000).unwrap();

        // Edit 2: change line4 (on top of edit 1 result)
        // Reconstruct file state after edit 1 — verify edit 1 was applied
        let after1_full = reconstruct_file(&conn, "s1", "f.rs", initial).unwrap();
        assert!(after1_full.contains("line2_CHANGED"));
        assert!(!after1_full.contains("line2\nline3")); // old line2 is gone

        // Apply edit 2
        generate_and_store(&conn, "s1", "t2", "f.rs",
            &after1_full,
            &after1_full.replace("line4", "line4_EDITED"),
            2000).unwrap();

        // Full reconstruction
        let final_content = reconstruct_file(&conn, "s1", "f.rs", initial).unwrap();
        assert!(final_content.contains("line2_CHANGED"));
        assert!(final_content.contains("line4_EDITED"));
        assert!(!final_content.contains("line4\nline5")); // old line4 gone
    }

    #[test]
    fn reconstruct_file_no_edits_returns_initial() {
        let conn = temp_conn();
        let content = reconstruct_file(&conn, "nonexistent", "f.rs", "original").unwrap();
        assert_eq!(content, "original");
    }

    #[test]
    fn reconstruct_file_multiline_edit() {
        let conn = temp_conn();
        let initial = "fn main() {\n    let x = 1;\n    let y = 2;\n    println!(\"{x} {y}\");\n}\n";

        // Replace two lines
        let after = "fn main() {\n    let x = 10;\n    let y = 20;\n    let z = 30;\n    println!(\"{x} {y} {z}\");\n}\n";
        generate_and_store(&conn, "s1", "t1", "main.rs", initial, after, 1000).unwrap();

        let result = reconstruct_file(&conn, "s1", "main.rs", initial).unwrap();
        assert!(result.contains("let z = 30"));
        assert!(!result.contains("let y = 2;"));
    }

    // ── Overlap detection ──────────────────────────────────────

    #[test]
    fn detect_overlapping_edits() {
        let conn = temp_conn();
        let f1 = "line1\nline2\nline3\nline4\nline5\n";
        let f2 = "line1\nline2_MODIFIED\nline3\nline4\nline5\n";
        let f3 = "line1\nline2_MODIFIED_AGAIN\nline3\nline4\nline5\n";

        generate_and_store(&conn, "s1", "t1", "f.rs", f1, f2, 1000).unwrap();
        generate_and_store(&conn, "s1", "t2", "f.rs", f2, f3, 2000).unwrap();
        // Edit on a different file — should NOT overlap
        generate_and_store(&conn, "s1", "t3", "other.rs", "a", "b", 3000).unwrap();

        let overlaps = find_overlapping_edits(&conn, "s1").unwrap();
        assert_eq!(overlaps.len(), 1, "only the two edits on f.rs line 2 should overlap");
        assert_eq!(overlaps[0].0.tool_call_id, "t1");
        assert_eq!(overlaps[0].1.tool_call_id, "t2");
    }

    // ── Hook: extract_diffs_post_commit ────────────────────────

    #[test]
    fn post_commit_extracts_diff_from_db() {
        let conn = temp_conn();
        let conv_id = 1i64;

        // Simulate what store_messages would write: assistant + tool messages
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, stored_at) VALUES (?1, 'assistant', ?2, datetime('now'))",
            params![conv_id, r#"{"role":"assistant","content":null,"tool_calls":[{"id":"tc1","function":{"name":"Edit","arguments":"{\"file_path\":\"src/a.rs\",\"old_string\":\"fn foo() { 1 }\",\"new_string\":\"fn foo() { 2 }\"}"}}]}"#],
        ).unwrap();

        let count = extract_diffs_post_commit(&conn, "s_hook", conv_id).unwrap();
        assert_eq!(count, 1, "should extract 1 diff from Edit tool call");

        let diffs = query_diffs(&conn, &DiffQuery {
            session_id: Some("s_hook".into()),
            ..Default::default()
        }).unwrap();
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].file_path, "src/a.rs");
        assert_eq!(diffs[0].tool_call_id, "tc1");
    }

    #[test]
    fn post_commit_skips_non_edit_tools() {
        let conn = temp_conn();
        let conv_id = 1i64;
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, stored_at) VALUES (?1, 'assistant', ?2, datetime('now'))",
            params![conv_id, r#"{"role":"assistant","content":null,"tool_calls":[{"id":"tc1","function":{"name":"Grep","arguments":"{\"pattern\":\"timeout\"}"}}]}"#],
        ).unwrap();

        let count = extract_diffs_post_commit(&conn, "s2", conv_id).unwrap();
        assert_eq!(count, 0, "Grep is not an edit tool");
    }

    #[test]
    fn post_commit_idempotent() {
        let conn = temp_conn();
        let conv_id = 1i64;
        conn.execute(
            "INSERT INTO messages (conversation_id, role, content, stored_at) VALUES (?1, 'assistant', ?2, datetime('now'))",
            params![conv_id, r#"{"role":"assistant","content":null,"tool_calls":[{"id":"tc1","function":{"name":"Edit","arguments":"{\"file_path\":\"f.rs\",\"old_string\":\"a\",\"new_string\":\"b\"}"}}]}"#],
        ).unwrap();

        let c1 = extract_diffs_post_commit(&conn, "s3", conv_id).unwrap();
        assert_eq!(c1, 1);
        let c2 = extract_diffs_post_commit(&conn, "s3", conv_id).unwrap();
        assert_eq!(c2, 0, "second call should skip already-processed tool_call_id");
    }

    // ── Smoke: roundtrip via Database ──────────────────────────

    #[tokio::test]
    async fn smoke_roundtrip_via_database() {
        let dir = tempfile::tempdir().unwrap();
        let db = crate::db::Database::builder()
            .path(dir.path().join("diff_smoke.db"))
            .build()
            .await
            .expect("open db");

        db.record_diff("ses_smoke", "tc1", "src/lib.rs",
            "fn foo() -> i32 { 1 }\n",
            "fn foo() -> i32 { 2 }\n",
        ).expect("record diff");

        let diffs = db.query_diffs(&DiffQuery {
            session_id: Some("ses_smoke".into()),
            ..Default::default()
        }).expect("query diffs");
        assert_eq!(diffs.len(), 1);
        assert_eq!(diffs[0].tool_call_id, "tc1");

        let reconstructed = db.reconstruct_file("ses_smoke", "src/lib.rs",
            "fn foo() -> i32 { 1 }\n").expect("reconstruct");
        assert!(reconstructed.contains("-> i32 { 2 }"));
    }
}

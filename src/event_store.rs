//! Proxy Event Store — structured event index for all three proxy handlers.
//!
//! Complements the existing capture pipeline (execution_events, messages,
//! --record) with a searchable, cross-handler event table. One row per
//! observable event (request, message, tool_call, tool_result, error, ...).
//!
//! Schema
//! ──────
//!   proxy_events         — one row per event (event_type, tool, path, ...)
//!   proxy_events_fts     — FTS5 on `content` for text search
//!
//! Design decisions
//! ────────────────
//! - Do NOT replace execution_events or messages. This is a search layer.
//! - Cover ALL three handlers (chat_completions, responses, anthropic).
//! - Session identity reuses the existing session_id / prompt_cache_key.

use rusqlite::{Connection, params};
use serde::{Deserialize, Serialize};

// ── Taxonomy ────────────────────────────────────────────────────────

/// Every event the proxy can observe.
#[derive(Debug, Clone, PartialEq, Eq, Serialize, Deserialize)]
pub enum EventType {
    RequestStart,
    RequestEnd,
    UserMessage,
    AssistantMessage,
    Reasoning,
    ToolCall,
    ToolResult,
    Error,
}

impl EventType {
    pub fn as_str(&self) -> &'static str {
        match self {
            EventType::RequestStart => "request_start",
            EventType::RequestEnd => "request_end",
            EventType::UserMessage => "user_message",
            EventType::AssistantMessage => "assistant_message",
            EventType::Reasoning => "reasoning",
            EventType::ToolCall => "tool_call",
            EventType::ToolResult => "tool_result",
            EventType::Error => "error",
        }
    }

pub fn from_event_str(s: &str) -> Option<Self> {

        match s {
            "request_start" => Some(EventType::RequestStart),
            "request_end" => Some(EventType::RequestEnd),
            "user_message" => Some(EventType::UserMessage),
            "assistant_message" => Some(EventType::AssistantMessage),
            "reasoning" => Some(EventType::Reasoning),
            "tool_call" => Some(EventType::ToolCall),
            "tool_result" => Some(EventType::ToolResult),
            "error" => Some(EventType::Error),
            _ => None,
        }
    }
}

// ── Event record ────────────────────────────────────────────────────

#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct ProxyEvent {
    pub id: Option<i64>,
    pub event_type: EventType,
    pub session_id: String,
    pub timestamp: String, // ISO-8601
    pub tool_name: Option<String>,
    pub path: Option<String>,
    pub status: Option<String>,
    pub content: String,
    pub metadata: serde_json::Value,
}

// ── Query filter ────────────────────────────────────────────────────

/// Structured query filter. All fields are optional — unset means
/// "match anything". `content_match` triggers an FTS5 lookup via a
/// subquery; all other filters are SQL `= ?` or `LIKE ?` on indexed
/// columns.
#[derive(Debug, Default)]
pub struct EventFilter {
    pub event_type: Option<EventType>,
    pub tool_name: Option<String>,
    pub session_id: Option<String>,
    pub status: Option<String>,
    pub path_pattern: Option<String>,
    pub content_match: Option<String>,
    pub limit: Option<usize>,
}

// ── Schema ──────────────────────────────────────────────────────────

pub const MIGRATION: &str = r#"
CREATE TABLE IF NOT EXISTS proxy_events (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    event_type  TEXT NOT NULL,
    session_id  TEXT NOT NULL DEFAULT '',
    timestamp   TEXT NOT NULL DEFAULT (datetime('now')),
    tool_name   TEXT,
    path        TEXT,
    status      TEXT,
    content     TEXT NOT NULL DEFAULT '',
    content_len INTEGER NOT NULL DEFAULT 0,
    metadata    TEXT NOT NULL DEFAULT '{}'
);

CREATE INDEX IF NOT EXISTS idx_pe_type     ON proxy_events(event_type);
CREATE INDEX IF NOT EXISTS idx_pe_session  ON proxy_events(session_id);
CREATE INDEX IF NOT EXISTS idx_pe_tool     ON proxy_events(tool_name);
CREATE INDEX IF NOT EXISTS idx_pe_status   ON proxy_events(status);
CREATE INDEX IF NOT EXISTS idx_pe_path     ON proxy_events(path);
CREATE INDEX IF NOT EXISTS idx_pe_ts       ON proxy_events(timestamp);
"#;

pub const FTS_MIGRATION: &str = r#"
CREATE VIRTUAL TABLE IF NOT EXISTS proxy_events_fts
    USING fts5(content, event_type UNINDEXED, session_id UNINDEXED, tokenize='unicode61');
"#;

// ── Operations ──────────────────────────────────────────────────────

/// Run the schema migration (called once during `Database::open`).
pub fn create_tables(conn: &Connection) -> anyhow::Result<()> {
    conn.execute_batch("DROP TABLE IF EXISTS proxy_events_fts;").ok();
    conn.execute_batch(MIGRATION)?;
    conn.execute_batch(FTS_MIGRATION)?;
    Ok(())
}

/// Insert one event. Also writes a short mirror row into the FTS5 table.
pub fn insert_event(conn: &Connection, event: &ProxyEvent) -> anyhow::Result<i64> {
    conn.execute(
        "INSERT INTO proxy_events (event_type, session_id, timestamp, tool_name, path, status, content, content_len, metadata)
         VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9)",
        params![
            event.event_type.as_str(),
            event.session_id,
            event.timestamp,
            event.tool_name,
            event.path,
            event.status,
            event.content,
            event.content.len() as i64,
            serde_json::to_string(&event.metadata).unwrap_or_default(),
        ],
    )?;
    let id = conn.last_insert_rowid();

    // Mirror into FTS5 for content search
    if !event.content.is_empty() {
        conn.execute(
            "INSERT INTO proxy_events_fts (rowid, content, event_type, session_id) VALUES (?1, ?2, ?3, ?4)",
            params![id, event.content, event.event_type.as_str(), event.session_id],
        )?;
    }

    Ok(id)
}

/// Query events with structured filters. Returns newest-first.
pub fn query_events(conn: &Connection, filter: &EventFilter) -> anyhow::Result<Vec<ProxyEvent>> {
    let mut clauses: Vec<String> = Vec::new();
    let mut bindings: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    // FTS subquery — must be the first clause because it uses a
    // separate virtual table and joins via rowid.
    if let Some(ref c) = filter.content_match {
        clauses.push("id IN (SELECT rowid FROM proxy_events_fts WHERE proxy_events_fts MATCH ?)".into());
        bindings.push(Box::new(c.clone()));
    }

    if let Some(ref t) = filter.event_type {
        clauses.push("event_type = ?".into());
        bindings.push(Box::new(t.as_str().to_string()));
    }
    if let Some(ref t) = filter.tool_name {
        clauses.push("tool_name = ?".into());
        bindings.push(Box::new(t.clone()));
    }
    if let Some(ref s) = filter.session_id {
        clauses.push("session_id = ?".into());
        bindings.push(Box::new(s.clone()));
    }
    if let Some(ref s) = filter.status {
        clauses.push("status = ?".into());
        bindings.push(Box::new(s.clone()));
    }
    if let Some(ref p) = filter.path_pattern {
        clauses.push("path LIKE ?".into());
        bindings.push(Box::new(p.clone()));
    }

    let where_clause = if clauses.is_empty() {
        String::new()
    } else {
        format!("WHERE {}", clauses.join(" AND "))
    };

    let mut sql = format!(
        "SELECT id, event_type, session_id, timestamp, tool_name, path, status, content, metadata FROM proxy_events {} ORDER BY timestamp DESC",
        where_clause
    );

    if let Some(limit) = filter.limit {
        sql.push_str(&format!(" LIMIT {limit}"));
    }

    let params_refs: Vec<&dyn rusqlite::types::ToSql> = bindings.iter().map(|b| b.as_ref()).collect();
    let mut stmt = conn.prepare(&sql)?;
    let rows = stmt.query_map(params_refs.as_slice(), |row| {
        Ok(ProxyEvent {
            id: Some(row.get(0)?),
            event_type: EventType::from_event_str(&row.get::<_, String>(1)?).unwrap_or(EventType::Error),
            session_id: row.get(2)?,
            timestamp: row.get(3)?,
            tool_name: row.get(4)?,
            path: row.get(5)?,
            status: row.get(6)?,
            content: row.get(7)?,
            metadata: serde_json::from_str(&row.get::<_, String>(8)?).unwrap_or_default(),
        })
    })?;

    let mut events = Vec::new();
    for row in rows {
        events.push(row?);
    }
    Ok(events)
}

/// Extract events from a Chat Completions `messages` array and insert
/// them into the store. Called once per request by each proxy handler
/// after the session key is known.
pub fn extract_and_insert(
    conn: &Connection,
    session_id: &str,
    messages: &[serde_json::Value],
) -> anyhow::Result<usize> {
    let now = chrono::Utc::now().to_rfc3339();
    let mut count = 0;

    for msg in messages {
        let role = msg["role"].as_str().unwrap_or("");
        match role {
            "user" => {
                let content = msg["content"].as_str().unwrap_or("");
                if content.is_empty() { continue; }
                insert_event(conn, &ProxyEvent {
                    id: None,
                    event_type: EventType::UserMessage,
                    session_id: session_id.to_string(),
                    timestamp: now.clone(),
                    tool_name: None,
                    path: None,
                    status: None,
                    content: content.to_string(),
                    metadata: serde_json::json!({"role": role}),
                })?;
                count += 1;
            }

            "assistant" => {
                // Emit assistant text content (if any)
                let content = msg["content"].as_str().unwrap_or("");
                if !content.is_empty() {
                    insert_event(conn, &ProxyEvent {
                        id: None,
                        event_type: EventType::AssistantMessage,
                        session_id: session_id.to_string(),
                        timestamp: now.clone(),
                        tool_name: None,
                        path: None,
                        status: None,
                        content: content.to_string(),
                        metadata: serde_json::json!({"role": role}),
                    })?;
                    count += 1;
                }

                // Emit tool calls
                if let Some(tool_calls) = msg["tool_calls"].as_array() {
                    for tc in tool_calls {
                        let name = tc["function"]["name"].as_str().unwrap_or("");
                        let args = tc["function"]["arguments"].as_str().unwrap_or("");
                        let tc_id = tc["id"].as_str().unwrap_or("");
                        insert_event(conn, &ProxyEvent {
                            id: None,
                            event_type: EventType::ToolCall,
                            session_id: session_id.to_string(),
                            timestamp: now.clone(),
                            tool_name: Some(name.to_string()),
                            path: self::tool_arg_path(name, args),
                            status: None,
                            content: format!("{name}({args})"),
                            metadata: serde_json::json!({"tool_call_id": tc_id, "tool_name": name, "args": args}),
                        })?;
                        count += 1;
                    }
                }
            }

            "tool" => {
                let content = msg["content"].as_str().unwrap_or("");
                let tc_id = msg["tool_call_id"].as_str().unwrap_or("");
                insert_event(conn, &ProxyEvent {
                    id: None,
                    event_type: EventType::ToolResult,
                    session_id: session_id.to_string(),
                    timestamp: now.clone(),
                    tool_name: None, // could extract from context
                    path: None,
                    status: if content.contains("error") || content.contains("Error") { Some("error".into()) } else { Some("success".into()) },
                    content: content.to_string(),
                    metadata: serde_json::json!({"tool_call_id": tc_id}),
                })?;
                count += 1;
            }

            _ => {}
        }
    }

    // Also emit a RequestStart for session tracking
    insert_event(conn, &ProxyEvent {
        id: None,
        event_type: EventType::RequestStart,
        session_id: session_id.to_string(),
        timestamp: now,
        tool_name: None,
        path: None,
        status: None,
        content: String::new(),
        metadata: serde_json::json!({"msg_count": messages.len()}),
    })?;
    count += 1;

    Ok(count)
}

/// Extract a file path from a tool call's arguments for searchable
/// path column. Handles common field names across tool call types.
fn tool_arg_path(_tool_name: &str, args_str: &str) -> Option<String> {
    let v: serde_json::Value = serde_json::from_str(args_str).ok()?;
    if let Some(p) = v["file_path"].as_str() { return Some(p.to_string()); }
    if let Some(p) = v["filePath"].as_str() { return Some(p.to_string()); }
    if let Some(p) = v["path"].as_str() { return Some(p.to_string()); }
    if let Some(p) = v["operation"]["path"].as_str() { return Some(p.to_string()); } // Codex apply_patch
    None
}

// ── Tests ──────────────────────────────────────────────────────────

#[cfg(test)]
mod tests {
    use super::*;
    use rusqlite::Connection;

    fn temp_conn() -> Connection {
        let conn = Connection::open_in_memory().expect("in-memory db");
        create_tables(&conn).expect("create tables");
        conn
    }

    fn ev(event_type: EventType, content: &str) -> ProxyEvent {
        ProxyEvent {
            id: None,
            event_type,
            session_id: "ses_01".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            tool_name: None,
            path: None,
            status: None,
            content: content.into(),
            metadata: serde_json::json!({}),
        }
    }

    fn ev_tool(tool: &str, content: &str) -> ProxyEvent {
        ProxyEvent {
            id: None,
            event_type: EventType::ToolCall,
            session_id: "ses_01".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            tool_name: Some(tool.into()),
            path: None,
            status: None,
            content: content.into(),
            metadata: serde_json::json!({}),
        }
    }

    fn ev_path(event_type: EventType, path: &str, tool: &str) -> ProxyEvent {
        ProxyEvent {
            id: None,
            event_type,
            session_id: "ses_01".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            tool_name: Some(tool.into()),
            path: Some(path.into()),
            status: None,
            content: String::new(),
            metadata: serde_json::json!({}),
        }
    }

    fn ev_status(event_type: EventType, status: &str) -> ProxyEvent {
        ProxyEvent {
            id: None,
            event_type,
            session_id: "ses_01".into(),
            timestamp: "2026-01-01T00:00:00Z".into(),
            tool_name: None,
            path: None,
            status: Some(status.into()),
            content: String::new(),
            metadata: serde_json::json!({}),
        }
    }

    fn count(conn: &Connection) -> usize {
        query_events(conn, &EventFilter::default())
            .unwrap()
            .len()
    }

    // ── Schema ──────────────────────────────────────────────────

    #[test]
    fn tables_exist_after_create() {
        let conn = temp_conn();
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'proxy_events'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "proxy_events table should exist");
        let n: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM sqlite_master WHERE name = 'proxy_events_fts'",
                [],
                |r| r.get(0),
            )
            .unwrap();
        assert_eq!(n, 1, "proxy_events_fts table should exist");
    }

    #[test]
    fn indices_exist() {
        let conn = temp_conn();
        let names: Vec<String> = conn
            .prepare("SELECT name FROM sqlite_master WHERE type = 'index'")
            .unwrap()
            .query_map([], |r| r.get(0))
            .unwrap()
            .filter_map(|r| r.ok())
            .collect();
        assert!(names.iter().any(|n| n == "idx_pe_type"), "type index: {names:?}");
        assert!(names.iter().any(|n| n == "idx_pe_session"), "session index: {names:?}");
        assert!(names.iter().any(|n| n == "idx_pe_tool"), "tool index: {names:?}");
        assert!(names.iter().any(|n| n == "idx_pe_status"), "status index: {names:?}");
        assert!(names.iter().any(|n| n == "idx_pe_path"), "path index: {names:?}");
    }

    // ── EventType ───────────────────────────────────────────────

    #[test]
    fn event_type_roundtrip() {
        for t in [
            EventType::RequestStart,
            EventType::RequestEnd,
            EventType::UserMessage,
            EventType::AssistantMessage,
            EventType::Reasoning,
            EventType::ToolCall,
            EventType::ToolResult,
            EventType::Error,
        ] {
            let s = t.as_str();
            let back = EventType::from_event_str(s).unwrap();
            assert_eq!(back, t, "roundtrip failed for {s}");
        }
    }

    #[test]
    fn event_type_unknown_returns_none() {
        assert!(EventType::from_event_str("nonexistent").is_none());
    }

    // ── Insert + Query ──────────────────────────────────────────

    #[test]
    fn insert_then_query_single() {
        let conn = temp_conn();
        let e = ev(EventType::UserMessage, "hello world");
        let id = insert_event(&conn, &e).unwrap();
        assert!(id > 0);

        let results = query_events(&conn, &EventFilter::default()).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].content, "hello world");
        assert_eq!(results[0].event_type, EventType::UserMessage);
    }

    #[test]
    fn filter_by_event_type() {
        let conn = temp_conn();
        insert_event(&conn, &ev(EventType::UserMessage, "")).unwrap();
        insert_event(&conn, &ev(EventType::ToolCall, "")).unwrap();
        insert_event(&conn, &ev(EventType::ToolCall, "")).unwrap();

        let f = EventFilter {
            event_type: Some(EventType::ToolCall),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn filter_by_tool_name() {
        let conn = temp_conn();
        insert_event(&conn, &ev_tool("Read", "read a.rs")).unwrap();
        insert_event(&conn, &ev_tool("Edit", "edit b.rs")).unwrap();
        insert_event(&conn, &ev_tool("Read", "read c.rs")).unwrap();

        let f = EventFilter {
            tool_name: Some("Read".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn filter_by_status() {
        let conn = temp_conn();
        insert_event(&conn, &ev_status(EventType::ToolResult, "success")).unwrap();
        insert_event(&conn, &ev_status(EventType::ToolResult, "error")).unwrap();
        insert_event(&conn, &ev_status(EventType::ToolResult, "success")).unwrap();

        let f = EventFilter {
            status: Some("error".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 1);
    }

    #[test]
    fn filter_by_path_pattern() {
        let conn = temp_conn();
        insert_event(&conn, &ev_path(EventType::ToolCall, "/src/config.rs", "Read")).unwrap();
        insert_event(&conn, &ev_path(EventType::ToolCall, "/src/auth.rs", "Read")).unwrap();
        insert_event(&conn, &ev_path(EventType::ToolCall, "/tests/main.rs", "Edit")).unwrap();

        let f = EventFilter {
            path_pattern: Some("%config%".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 1);
        assert!(results[0].path.as_deref().unwrap().contains("config"));
    }

    #[test]
    fn filter_by_session() {
        let conn = temp_conn();
        let mut e1 = ev(EventType::UserMessage, ""); e1.session_id = "aaa".into();
        let mut e2 = ev(EventType::UserMessage, ""); e2.session_id = "bbb".into();
        let mut e3 = ev(EventType::UserMessage, ""); e3.session_id = "aaa".into();
        insert_event(&conn, &e1).unwrap();
        insert_event(&conn, &e2).unwrap();
        insert_event(&conn, &e3).unwrap();

        let f = EventFilter {
            session_id: Some("aaa".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 2);
    }

    #[test]
    fn filter_limit() {
        let conn = temp_conn();
        for i in 0..10 {
            insert_event(&conn, &ev(EventType::UserMessage, &format!("msg{i}"))).unwrap();
        }
        let f = EventFilter {
            limit: Some(3),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 3);
    }

    // ── FTS ─────────────────────────────────────────────────────

    #[test]
    fn fts_search_on_content() {
        let conn = temp_conn();
        insert_event(&conn, &ev(EventType::UserMessage, "find the timeout bug")).unwrap();
        insert_event(&conn, &ev(EventType::UserMessage, "fix config.rs")).unwrap();
        insert_event(&conn, &ev(EventType::UserMessage, "another timeout issue")).unwrap();

        let f = EventFilter {
            content_match: Some("timeout".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 2, "FTS should find 2 events with 'timeout'");
    }

    #[test]
    fn fts_combined_with_structured_filter() {
        let conn = temp_conn();
        insert_event(&conn, &ev_tool("Read", "read config.rs")).unwrap();
        insert_event(&conn, &ev_tool("Edit", "edit config.rs")).unwrap();
        insert_event(&conn, &ev_tool("Read", "read auth.rs")).unwrap();

        let f = EventFilter {
            tool_name: Some("Read".into()),
            content_match: Some("config".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 1);
        assert_eq!(results[0].tool_name.as_deref(), Some("Read"));
        assert!(results[0].content.contains("config"));
    }

    #[test]
    fn fts_no_match_returns_empty() {
        let conn = temp_conn();
        insert_event(&conn, &ev(EventType::UserMessage, "hello")).unwrap();

        let f = EventFilter {
            content_match: Some("nonexistent".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert!(results.is_empty());
    }

    // ── Combinatorial ──────────────────────────────────────────

    #[test]
    fn filter_multiple_fields() {
        let conn = temp_conn();
        // Match
        insert_event(&conn, &ProxyEvent {
            id: None, event_type: EventType::ToolCall,
            session_id: "ses_X".into(), timestamp: "t".into(),
            tool_name: Some("Read".into()),
            path: Some("/a.rs".into()),
            status: Some("error".into()),
            content: "find timeout".into(),
            metadata: serde_json::json!({}),
        }).unwrap();
        // Mismatch (wrong tool)
        insert_event(&conn, &ProxyEvent {
            id: None, event_type: EventType::ToolCall,
            session_id: "ses_X".into(), timestamp: "t".into(),
            tool_name: Some("Edit".into()),
            path: Some("/a.rs".into()),
            status: Some("error".into()),
            content: "find timeout".into(),
            metadata: serde_json::json!({}),
        }).unwrap();
        // Mismatch (wrong session)
        insert_event(&conn, &ProxyEvent {
            id: None, event_type: EventType::ToolCall,
            session_id: "ses_Y".into(), timestamp: "t".into(),
            tool_name: Some("Read".into()),
            path: Some("/a.rs".into()),
            status: Some("error".into()),
            content: "find timeout".into(),
            metadata: serde_json::json!({}),
        }).unwrap();

        let f = EventFilter {
            tool_name: Some("Read".into()),
            session_id: Some("ses_X".into()),
            status: Some("error".into()),
            content_match: Some("timeout".into()),
            ..Default::default()
        };
        let results = query_events(&conn, &f).unwrap();
        assert_eq!(results.len(), 1, "only one event matches all filters");
    }

    #[test]
    fn empty_db_returns_no_results() {
        let conn = temp_conn();
        let results = query_events(&conn, &EventFilter::default()).unwrap();
        assert!(results.is_empty());
        assert_eq!(count(&conn), 0);
    }

    #[test]
    fn insert_many_then_query_all() {
        let conn = temp_conn();
        for i in 0..50 {
            insert_event(&conn, &ev(EventType::ToolCall, &format!("call {i}"))).unwrap();
        }
        assert_eq!(count(&conn), 50);
    }

    // ── extract_and_insert ──────────────────────────────────────

    #[test]
    fn extract_from_messages_inserts_all_event_types() {
        let conn = temp_conn();
        let messages = serde_json::json!([
            {"role": "user", "content": "fix the bug"},
            {"role": "assistant", "content": "I'll search first"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "t1", "function": {"name": "Read", "arguments": r#"{"file_path":"/src/a.rs"}"#}}
            ]},
            {"role": "tool", "tool_call_id": "t1", "content": "fn hello() {}"}
        ]);
        let arr = messages.as_array().unwrap();
        let count = extract_and_insert(&conn, "ses_test", arr).unwrap();
        assert!(count >= 4, "should insert user, assistant, tool_call, tool_result + request_start: got {count}");

        // Check that each event type exists
        let types: Vec<String> = query_events(&conn, &EventFilter::default())
            .unwrap()
            .iter()
            .map(|e| e.event_type.as_str().to_string())
            .collect();
        assert!(types.contains(&"request_start".to_string()), "types: {types:?}");
        assert!(types.contains(&"user_message".to_string()), "types: {types:?}");
        assert!(types.contains(&"assistant_message".to_string()), "types: {types:?}");
        assert!(types.contains(&"tool_call".to_string()), "types: {types:?}");
        assert!(types.contains(&"tool_result".to_string()), "types: {types:?}");
    }

    #[test]
    fn extract_tool_call_parses_path_from_args() {
        let conn = temp_conn();
        let messages = serde_json::json!([
            {"role": "user", "content": "fix"},
            {"role": "assistant", "content": null, "tool_calls": [
                {"id": "t1", "function": {"name": "Read", "arguments": r#"{"file_path":"/src/config.rs"}"#}}
            ]},
        ]);
        let arr = messages.as_array().unwrap();
        extract_and_insert(&conn, "s1", arr).unwrap();

        let f = EventFilter {
            event_type: Some(EventType::ToolCall),
            ..Default::default()
        };
        let events = query_events(&conn, &f).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].path.as_deref(), Some("/src/config.rs"));
        assert_eq!(events[0].tool_name.as_deref(), Some("Read"));
    }

    #[test]
    fn extract_tool_result_status_is_error() {
        let conn = temp_conn();
        let messages = serde_json::json!([
            {"role": "user", "content": "fix"},
            {"role": "tool", "tool_call_id": "t1", "content": "Error: file not found"}
        ]);
        let arr = messages.as_array().unwrap();
        extract_and_insert(&conn, "s1", arr).unwrap();

        let f = EventFilter {
            event_type: Some(EventType::ToolResult),
            ..Default::default()
        };
        let events = query_events(&conn, &f).unwrap();
        assert_eq!(events.len(), 1);
        assert_eq!(events[0].status.as_deref(), Some("error"),
            "tool result with 'Error' in content should have status=error");
    }

    #[test]
    fn extract_skips_system_and_empty_messages() {
        let conn = temp_conn();
        let messages = serde_json::json!([
            {"role": "system", "content": "you are helpful"},
            {"role": "user", "content": ""},
            {"role": "user", "content": "real message"},
        ]);
        let arr = messages.as_array().unwrap();
        extract_and_insert(&conn, "s1", arr).unwrap();

        // Should have: request_start + the real user_message
        let f = EventFilter {
            event_type: Some(EventType::UserMessage),
            ..Default::default()
        };
        let events = query_events(&conn, &f).unwrap();
        assert_eq!(events.len(), 1, "only non-empty user messages counted");
    }

    // ── Smoketest: full roundtrip via Database ─────────────────

    #[tokio::test]
    async fn smoke_roundtrip_via_database() {
        let dir = tempfile::tempdir().unwrap();
        let db_path = dir.path().join("smoke.db");

        // Use the crate-level Database, which runs the full migration.
        let db = crate::db::Database::builder()
            .path(db_path)
            .build()
            .await
            .expect("open db");

        // Insert via the Database convenience method.
        db.insert_event_simple(
            EventType::ToolCall,
            "ses_smoke",
            "search for timeout",
            serde_json::json!({"tool": "Grep"}),
        )
        .expect("insert");

        db.insert_event_simple(
            EventType::ToolResult,
            "ses_smoke",
            "Error: connection refused",
            serde_json::json!({"tool_call_id": "t1"}),
        )
        .expect("insert");

        // Verify FTS can find by content.
        let fts = db
            .query_proxy_events(&EventFilter {
                content_match: Some("timeout".into()),
                ..Default::default()
            })
            .expect("fts query");

        assert_eq!(fts.len(), 1);
        assert_eq!(fts[0].session_id, "ses_smoke");
    }
}

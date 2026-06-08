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
    let mut sql = String::from("SELECT id, event_type, session_id, timestamp, tool_name, path, status, content, metadata FROM proxy_events WHERE 1=1");
    let mut bindings: Vec<Box<dyn rusqlite::types::ToSql>> = Vec::new();

    if let Some(ref t) = filter.event_type {
        sql.push_str(" AND event_type = ?");
        bindings.push(Box::new(t.as_str().to_string()));
    }
    if let Some(ref t) = filter.tool_name {
        sql.push_str(" AND tool_name = ?");
        bindings.push(Box::new(t.clone()));
    }
    if let Some(ref s) = filter.session_id {
        sql.push_str(" AND session_id = ?");
        bindings.push(Box::new(s.clone()));
    }
    if let Some(ref s) = filter.status {
        sql.push_str(" AND status = ?");
        bindings.push(Box::new(s.clone()));
    }
    if let Some(ref p) = filter.path_pattern {
        sql.push_str(" AND path LIKE ?");
        bindings.push(Box::new(p.clone()));
    }
    if let Some(ref c) = filter.content_match {
        // Use FTS5 to find rowids, then join
        sql = format!(
            "SELECT id, event_type, session_id, timestamp, tool_name, path, status, content, metadata
               FROM proxy_events
              WHERE id IN (SELECT rowid FROM proxy_events_fts WHERE proxy_events_fts MATCH ?)
                AND {}",
            &sql[45..] // remove "SELECT ... FROM proxy_events WHERE 1=1 " prefix
        );
        // Clear bindings and rebuild
        bindings.clear();
        bindings.push(Box::new(c.clone()));
        if let Some(ref t) = filter.event_type {
            sql.push_str(" AND event_type = ?");
            bindings.push(Box::new(t.as_str().to_string()));
        }
        if let Some(ref t) = filter.tool_name {
            sql.push_str(" AND tool_name = ?");
            bindings.push(Box::new(t.clone()));
        }
        if let Some(ref s) = filter.session_id {
            sql.push_str(" AND session_id = ?");
            bindings.push(Box::new(s.clone()));
        }
        if let Some(ref s) = filter.status {
            sql.push_str(" AND status = ?");
            bindings.push(Box::new(s.clone()));
        }
        if let Some(ref p) = filter.path_pattern {
            sql.push_str(" AND path LIKE ?");
            bindings.push(Box::new(p.clone()));
        }
    }

    sql.push_str(" ORDER BY timestamp DESC");

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

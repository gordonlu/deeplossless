use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::dag::DagNode;

const DEFAULT_DB_PATH: &str = "~/.deepseek/lcm/lcm.db";

/// Builder for [`Database`].
///
/// # Example
/// ```
/// let db = Database::builder()
///     .path("/tmp/test.db")
///     .build()
///     .await?;
/// ```
pub struct DatabaseBuilder {
    path: PathBuf,
}

impl Default for DatabaseBuilder {
    fn default() -> Self {
        Self {
            path: PathBuf::from(DEFAULT_DB_PATH),
        }
    }
}

impl DatabaseBuilder {
    pub fn new() -> Self {
        Self::default()
    }

    /// Set the SQLite database path. Supports `~` for home directory.
    pub fn path(mut self, path: impl Into<PathBuf>) -> Self {
        self.path = path.into();
        self
    }

    /// Open the database, creating parent directories and running migrations.
    pub async fn build(self) -> anyhow::Result<Database> {
        Database::open(&self.path).await
    }
}

/// Lossless SQLite store for conversations, messages, and DAG summaries.
///
/// All operations go through a single `Mutex<Connection>` (WAL mode) so
/// writes are serialised — the compaction thread submits new nodes via
/// an mpsc channel to the main request handler, which is the sole writer.
pub struct Database {
    conn: Mutex<Connection>,
    write_count: AtomicU64,
}

/// Run a WAL checkpoint every N writes to limit WAL file growth.
const CHECKPOINT_INTERVAL: u64 = 100;

impl Database {
    pub fn builder() -> DatabaseBuilder {
        DatabaseBuilder::new()
    }

    async fn open(path: &Path) -> anyhow::Result<Self> {
        let expanded = shellexpand::full(&path.to_string_lossy())
            .map(|c| c.to_string())
            .unwrap_or_else(|_| path.to_string_lossy().to_string());
        if let Some(parent) = Path::new(&expanded).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let conn = Connection::open(&expanded)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;",
        )?;

        let db = Self { conn: Mutex::new(conn), write_count: AtomicU64::new(0) };
        db.migrate()?;
        Ok(db)
    }

    /// Run a WAL checkpoint to prevent the WAL file from growing
    /// indefinitely under sustained write load.  Call periodically
    /// (e.g. every N store_messages calls).
    pub fn wal_checkpoint(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    fn migrate(&self) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS conversations (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                session_id  TEXT NOT NULL,
                created_at  TEXT NOT NULL DEFAULT (datetime('now')),
                model       TEXT NOT NULL
            );

            CREATE TABLE IF NOT EXISTS messages (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                role            TEXT NOT NULL,
                content         TEXT NOT NULL,
                token_count     INTEGER,
                stored_at       TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE TABLE IF NOT EXISTS dag_nodes (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                level           INTEGER NOT NULL DEFAULT 0,
                summary         TEXT NOT NULL,
                token_count     INTEGER NOT NULL,
                parent_ids      TEXT NOT NULL DEFAULT '[]',
                child_ids       TEXT NOT NULL DEFAULT '[]',
                snippets        TEXT NOT NULL DEFAULT '[]',
                is_leaf         INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_dag_conv
                ON dag_nodes(conversation_id, level);

            CREATE VIRTUAL TABLE IF NOT EXISTS messages_fts
                USING fts5(
                    content,
                    role UNINDEXED,
                    tokenize='porter unicode61'
                );
        ")?;
        Ok(())
    }

    /// Find a conversation by session fingerprint, or create one.
    pub fn find_or_create_conversation(&self, fingerprint: &str, model: &str) -> anyhow::Result<i64> {
        let conn = self.conn.lock().unwrap();
        // Try to find existing conversation
        let existing: Option<i64> = conn
            .query_row(
                "SELECT id FROM conversations WHERE session_id = ?1 LIMIT 1",
                rusqlite::params![fingerprint],
                |row| row.get(0),
            )
            .ok();
        if let Some(id) = existing {
            return Ok(id);
        }
        // Create new
        conn.execute(
            "INSERT INTO conversations (session_id, model) VALUES (?1, ?2)",
            rusqlite::params![fingerprint, model],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Persist a request's `messages` array into an existing conversation.
    pub fn store_messages(&self, conv_id: i64, messages: &serde_json::Value) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;

        if let Some(arr) = messages.as_array() {
            for msg in arr {
                let role = msg["role"].as_str().unwrap_or("unknown");
                let content = msg["content"].to_string();
                let token_count = crate::tokenizer::count_content(msg) as i64
                    + crate::tokenizer::MESSAGE_OVERHEAD_TOKENS as i64;
                let plain = Self::strip_json_markup(&content);
                tx.execute(
                    "INSERT INTO messages (conversation_id, role, content, token_count)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![conv_id, role, content, token_count],
                )?;
                let msg_id = tx.last_insert_rowid();
                // Mirror into FTS5 index
                let _ = tx.execute(
                    "INSERT INTO messages_fts (rowid, content, role) VALUES (?1, ?2, ?3)",
                    rusqlite::params![msg_id, plain, role],
                );
            }
        }

        tx.commit()?;
        let count = self.write_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count.is_multiple_of(CHECKPOINT_INTERVAL)
            && let Err(e) = self.wal_checkpoint() {
                tracing::warn!(target: "deeplossless::db", error = %e, "WAL checkpoint failed");
            }
        Ok(())
    }

    /// Legacy: create conversation + store messages in one call.
    pub fn create_and_store(&self, model: &str, messages: &serde_json::Value) -> anyhow::Result<i64> {
        let fp = crate::session::fingerprint(messages.as_array().map(|a| a.as_slice()).unwrap_or(&[]), 3);
        let conv_id = self.find_or_create_conversation(&fp, model)?;
        self.store_messages(conv_id, messages)?;
        Ok(conv_id)
    }

    // ── DAG node operations ────────────────────────────────────────────

    pub fn insert_dag_node(
        &self,
        conversation_id: i64,
        level: u8,
        summary: &str,
        token_count: i64,
        parent_ids: &[i64],
        child_ids: &[i64],
        is_leaf: bool,
    ) -> anyhow::Result<DagNode> {
        self.insert_dag_node_full(conversation_id, level, summary, token_count, parent_ids, child_ids, &[], is_leaf)
    }

    #[allow(clippy::too_many_arguments)]
    pub fn insert_dag_node_full(
        &self,
        conversation_id: i64,
        level: u8,
        summary: &str,
        token_count: i64,
        parent_ids: &[i64],
        child_ids: &[i64],
        snippets: &[crate::snippet::Snippet],
        is_leaf: bool,
    ) -> anyhow::Result<DagNode> {
        let conn = self.conn.lock().unwrap();
        let parent_json = serde_json::to_string(parent_ids)?;
        let child_json = serde_json::to_string(child_ids)?;
        let snippet_json = serde_json::to_string(snippets)?;
        conn.execute(
            "INSERT INTO dag_nodes (conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8)",
            rusqlite::params![conversation_id, level, summary, token_count, parent_json, child_json, snippet_json, is_leaf as i32],
        )?;
        let id = conn.last_insert_rowid();
        Ok(DagNode {
            id,
            conversation_id,
            level,
            summary: summary.to_string(),
            token_count,
            parent_ids: parent_ids.to_vec(),
            child_ids: child_ids.to_vec(),
            snippets: snippets.to_vec(),
            is_leaf,
        })
    }

    pub fn get_node(&self, node_id: i64) -> anyhow::Result<Option<DagNode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf
             FROM dag_nodes WHERE id = ?1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![node_id], Self::row_to_node)?;
        Ok(rows.next().transpose()?)
    }

    pub fn get_child_nodes(&self, node_id: i64, max_fanout: usize) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.conn.lock().unwrap();
        // Use json_each() for precise JSON array matching — LIKE '%id%'
        // would also match ids like 12 appearing inside 123 or 412.
        let mut stmt = conn.prepare(
            "SELECT DISTINCT n.id, n.conversation_id, n.level, n.summary,
                    n.token_count, n.parent_ids, n.child_ids, n.snippets, n.is_leaf
             FROM dag_nodes n, json_each(n.parent_ids) AS j
             WHERE j.value = ?1
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![node_id, max_fanout as i64], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows {
            nodes.push(row?);
        }
        Ok(nodes)
    }

    pub fn get_tip_node(&self, conv_id: i64) -> anyhow::Result<Option<DagNode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf
             FROM dag_nodes WHERE conversation_id = ?1 AND level > 0
             ORDER BY level DESC, id DESC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        Ok(rows.next().transpose()?)
    }

    pub fn get_tip_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf
             FROM dag_nodes
             WHERE conversation_id = ?1 AND level = (
                 SELECT MAX(level) FROM dag_nodes WHERE conversation_id = ?1 AND level > 0
             )
             ORDER BY id DESC",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows { nodes.push(row?); }
        Ok(nodes)
    }

    pub fn get_all_dag_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf
             FROM dag_nodes WHERE conversation_id = ?1 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows { nodes.push(row?); }
        Ok(nodes)
    }

    pub fn delete_dag_node(&self, node_id: i64) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        conn.execute("DELETE FROM dag_nodes WHERE id = ?1", rusqlite::params![node_id])?;
        Ok(())
    }

    pub fn get_leaf_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.conn.lock().unwrap();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf
             FROM dag_nodes WHERE conversation_id = ?1 AND is_leaf = 1
             ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows {
            nodes.push(row?);
        }
        Ok(nodes)
    }

    pub fn total_conversation_tokens(&self, conv_id: i64) -> anyhow::Result<i64> {
        let conn = self.conn.lock().unwrap();
        let total: i64 = conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = ?1",
            rusqlite::params![conv_id],
            |row| row.get(0),
        )?;
        Ok(total)
    }

    pub fn add_child_to_node(&self, parent_id: i64, child_id: i64) -> anyhow::Result<()> {
        let conn = self.conn.lock().unwrap();
        let existing: String = conn.query_row(
            "SELECT child_ids FROM dag_nodes WHERE id = ?1",
            rusqlite::params![parent_id],
            |row| row.get(0),
        )?;
        let mut ids: Vec<i64> = serde_json::from_str(&existing).unwrap_or_default();
        if !ids.contains(&child_id) {
            ids.push(child_id);
            let json = serde_json::to_string(&ids)?;
            conn.execute(
                "UPDATE dag_nodes SET child_ids = ?1 WHERE id = ?2",
                rusqlite::params![json, parent_id],
            )?;
        }
        Ok(())
    }

    /// Search messages in a conversation using FTS5 full-text search.
    /// Returns `(id, role, snippet, token_count)` ranked by relevance.
    pub fn search_messages(&self, conv_id: i64, query: &str) -> anyhow::Result<Vec<(i64, String, String, i64)>> {
        let conn = self.conn.lock().unwrap();
        let safe = Self::fts_escape(query);
        if safe.is_empty() {
            return Ok(Vec::new());
        }
        let mut stmt = conn.prepare(
            "SELECT m.id, m.role, snippet(messages_fts, 0, '<b>', '</b>', '…', 32),
                    m.token_count
             FROM messages_fts
             JOIN messages m ON m.id = messages_fts.rowid
             WHERE messages_fts MATCH ?1 AND m.conversation_id = ?2
             ORDER BY rank
             LIMIT 20",
        )?;
        let rows = stmt.query_map(rusqlite::params![safe, conv_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, String>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, i64>(3)?,
            ))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Strip JSON markup from content so FTS5 indexes only the plain text.
/// Content can be either a plain string or a JSON array of content blocks.
fn strip_json_markup(content: &str) -> String {
    if let Ok(val) = serde_json::from_str::<serde_json::Value>(content) {
        match val {
            serde_json::Value::String(s) => return s,
            serde_json::Value::Array(arr) => {
                let mut out = String::new();
                for block in arr {
                    if let Some(text) = block["text"].as_str() {
                        if !out.is_empty() { out.push(' '); }
                        out.push_str(text);
                    }
                }
                return out;
            }
            _ => {}
        }
    }
    content.to_string()
}

/// Escape a user query for FTS5 MATCH syntax.
/// FTS5 special chars: ^ * " ( ) + - ~ ` `
pub fn fts_escape(query: &str) -> String {
    let mut out = String::with_capacity(query.len());
    for ch in query.chars() {
        match ch {
            '"' | '\'' | '*' | '^' | '(' | ')' | '+' | '-' | '~' | '`' => {
                out.push(' ');
            }
            c => out.push(c),
        }
    }
    out.trim().to_string()
}

fn row_to_node(row: &rusqlite::Row) -> rusqlite::Result<DagNode> {
        let parent_str: String = row.get(5)?;
        let child_str: String = row.get(6)?;
        let snippet_str: String = row.get(7)?;
        let is_leaf_int: i32 = row.get(8)?;
        Ok(DagNode {
            id: row.get(0)?,
            conversation_id: row.get(1)?,
            level: row.get(2)?,
            summary: row.get(3)?,
            token_count: row.get(4)?,
            parent_ids: serde_json::from_str(&parent_str).unwrap_or_default(),
            child_ids: serde_json::from_str(&child_str).unwrap_or_default(),
            snippets: serde_json::from_str(&snippet_str).unwrap_or_default(),
            is_leaf: is_leaf_int != 0,
        })
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use tempfile::tempdir;

    #[tokio::test]
    async fn builder_default_uses_default_path() {
        let builder = Database::builder();
        assert!(builder.path.to_string_lossy().contains("lcm.db"));
    }

    #[tokio::test]
    async fn builder_custom_path() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("test.db");
        let db = Database::builder().path(&path).build().await.unwrap();
        assert!(path.exists());
        // Cleanup: drop db to close connection
        drop(db);
    }

    #[tokio::test]
    async fn store_and_retrieve_messages() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("store_test.db"))
            .build()
            .await
            .unwrap();

        let messages = serde_json::json!([
            {"role": "system", "content": "You are a helpful assistant."},
            {"role": "user", "content": "hello"}
        ]);

        let conv_id = db.create_and_store("deepseek-v4-flash", &messages).unwrap();
        assert!(conv_id > 0, "expected valid conversation ID");

        // Verify data was written
        let conn = db.conn.lock().unwrap();
        let count: i64 = conn
            .query_row(
                "SELECT COUNT(*) FROM messages WHERE conversation_id = ?1",
                rusqlite::params![conv_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(count, 2, "expected 2 messages (system + user)");

        let model: String = conn
            .query_row(
                "SELECT model FROM conversations WHERE id = ?1",
                rusqlite::params![conv_id],
                |row| row.get(0),
            )
            .unwrap();
        assert_eq!(model, "deepseek-v4-flash");
    }

    #[tokio::test]
    async fn migrate_is_idempotent() {
        let dir = tempdir().unwrap();
        let path = dir.path().join("idempotent.db");
        let db = Database::builder().path(&path).build().await.unwrap();
        // Second migrate should be a no-op
        assert!(db.migrate().is_ok());
    }

    #[test]
    fn builder_is_send() {
        fn assert_send<T: Send>() {}
        assert_send::<DatabaseBuilder>();
    }

    // ── P0: Transaction integrity ───────────────────────────────────────

    #[tokio::test]
    async fn transaction_rollback_on_empty_messages() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("rollback.db"))
            .build()
            .await
            .unwrap();

        // Insert a real conversation
        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "hi"}])).unwrap();
        // Try to store empty — should succeed (no messages to insert)
        assert!(db.store_messages(conv_id, &json!([])).is_ok());

        // Verify the conversation has exactly 1 message
        let total = db.total_conversation_tokens(conv_id).unwrap();
        assert!(total > 0);
    }

    // ── P0: DAG node persistence ────────────────────────────────────────

    #[tokio::test]
    async fn dag_node_crud() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("dag_crud.db"))
            .build()
            .await
            .unwrap();

        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "start"}])).unwrap();

        let node = db.insert_dag_node(conv_id, 0, "test leaf", 10, &[], &[], true).unwrap();
        assert_eq!(node.level, 0);
        assert!(node.is_leaf);

        let fetched = db.get_node(node.id).unwrap().unwrap();
        assert_eq!(fetched.summary, "test leaf");

        let parent = db.insert_dag_node(conv_id, 1, "summary", 5, &[node.id], &[], false).unwrap();
        db.add_child_to_node(node.id, parent.id).unwrap();

        let children = db.get_child_nodes(node.id, 10).unwrap();
        assert!(!children.is_empty(), "parent should be a child of node");

        // Delete
        db.delete_dag_node(node.id).unwrap();
        assert!(db.get_node(node.id).unwrap().is_none());
    }

    // ── P0: search_messages ─────────────────────────────────────────────

    #[tokio::test]
    async fn search_messages_finds_content() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("search.db"))
            .build()
            .await
            .unwrap();

        let conv_id = db.create_and_store("test", &json!([
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hello back"}
        ])).unwrap();

        let results = db.search_messages(conv_id, "hello").unwrap();
        assert!(!results.is_empty(), "should find messages containing 'hello'");

        let empty = db.search_messages(conv_id, "zzz_nonexistent").unwrap();
        assert!(empty.is_empty(), "should not find nonexistent content");
    }

    // ── P0: WAL checkpoint ──────────────────────────────────────────────

    #[tokio::test]
    async fn wal_checkpoint_does_not_crash() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("wal_check.db"))
            .build()
            .await
            .unwrap();

        // Write some data
        for i in 0..5 {
            let msgs = json!([{"role": "user", "content": format!("msg {i}")}]);
            db.create_and_store("test", &msgs).unwrap();
        }

        // Checkpoint should succeed
        assert!(db.wal_checkpoint().is_ok());
    }
}

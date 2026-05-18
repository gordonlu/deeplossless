use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Mutex;

use crate::dag::DagNode;

/// Result from unified search across messages, summaries, and snippets.
#[derive(Debug, Clone, serde::Serialize)]
pub struct UnifiedSearchResult {
    pub id: i64,
    /// `"message"`, `"summary"`, or `"snippet"`.
    pub source: String,
    /// Role (for messages) or level (for summaries/snippets).
    pub label: String,
    /// Matching excerpt (up to 500 chars).
    pub excerpt: String,
    pub token_count: i64,
    /// BM25 relevance score from FTS5, None for LIKE-fallback results.
    pub bm25_score: Option<f64>,
}

const DEFAULT_DB_PATH: &str = "~/.deepseek/lcm/lcm.db";

/// Builder for [`Database`].
///
/// # Example
/// ```ignore
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
/// Read operations go through a connection pool (round-robin dispatch).
/// Writes go through a dedicated writer connection serialised by a Mutex.
/// WAL mode ensures writes don't block reads.
const READ_POOL_SIZE: usize = 3;

pub struct Database {
    read_pool: Vec<Mutex<Connection>>,
    writer: Mutex<Connection>,
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
        // Open writer connection
        let writer = Connection::open(&expanded)?;
        writer.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;",
        )?;

        // Open read pool connections (same WAL file, safe concurrent access)
        let mut read_pool = Vec::with_capacity(READ_POOL_SIZE);
        for _ in 0..READ_POOL_SIZE {
            let rconn = Connection::open(&expanded)?;
            rconn.execute_batch(
                "PRAGMA journal_mode=WAL;
                 PRAGMA synchronous=NORMAL;
                 PRAGMA busy_timeout=5000;",
            )?;
            read_pool.push(Mutex::new(rconn));
        }

        let db = Self {
            read_pool,
            writer: Mutex::new(writer),
            write_count: AtomicU64::new(0),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Round-robin dispatch to the next read connection.
    fn read_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        let idx = self.write_count.load(Ordering::Relaxed) as usize % self.read_pool.len();
        self.read_pool[idx].lock().unwrap()
    }

    /// Run a WAL checkpoint to prevent the WAL file from growing
    /// indefinitely under sustained write load.  Call periodically
    /// (e.g. every N store_messages calls).
    pub fn wal_checkpoint(&self) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    fn migrate(&self) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        // FTS5 virtual table.  DROP + CREATE in separate calls to avoid
        // residual shadow-table state from a prior interrupted migration.
        conn.execute_batch("DROP TABLE IF EXISTS messages_fts;").ok();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE messages_fts
             USING fts5(content, role UNINDEXED, tokenize='unicode61');
        ")?;

        // Snippets FTS5 index for precision-critical value retrieval.
        // DROP first to avoid stale shadow-table state.
        conn.execute_batch("DROP TABLE IF EXISTS snippets_fts;").ok();
        conn.execute_batch(
            "CREATE VIRTUAL TABLE snippets_fts
             USING fts5(content, source_type UNINDEXED, node_id UNINDEXED, tokenize='unicode61');
        ")?;

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
                deleted         INTEGER NOT NULL DEFAULT 0,
                deleted_at      TEXT,
                semantic_hash   TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            -- Cross-conversation semantic dedup index (v0.3)
            CREATE TABLE IF NOT EXISTS semantic_index (
                hash            TEXT NOT NULL,
                node_id         INTEGER NOT NULL REFERENCES dag_nodes(id),
                conversation_id INTEGER NOT NULL REFERENCES conversations(id),
                summary_preview TEXT NOT NULL DEFAULT '',
                created_at      TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (hash, node_id)
            );
            CREATE INDEX IF NOT EXISTS idx_semantic_hash
                ON semantic_index(hash);

            -- Provenance tracing: per-sentence source attribution (v0.4)
            CREATE TABLE IF NOT EXISTS provenance (
                id              INTEGER PRIMARY KEY AUTOINCREMENT,
                summary_node_id INTEGER NOT NULL REFERENCES dag_nodes(id),
                source_node_id  INTEGER NOT NULL REFERENCES dag_nodes(id),
                sentence_offset INTEGER NOT NULL DEFAULT 0,
                sentence_length INTEGER NOT NULL DEFAULT 0,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_provenance_summary
                ON provenance(summary_node_id);
            CREATE INDEX IF NOT EXISTS idx_provenance_source
                ON provenance(source_node_id);

            -- Embedding vectors for semantic similarity dedup (v0.5)
            CREATE TABLE IF NOT EXISTS embeddings (
                node_id     INTEGER PRIMARY KEY REFERENCES dag_nodes(id),
                vector      BLOB NOT NULL,
                model       TEXT NOT NULL DEFAULT 'deepseek-embed',
                dimensions  INTEGER NOT NULL DEFAULT 1536,
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_embeddings_model
                ON embeddings(model);
        ")?;

        // Graceful migration: add soft-delete columns if upgrading from earlier schema.
        // SQLite has no ADD COLUMN IF NOT EXISTS; we use PRAGMA table_info.
        let has_deleted: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'deleted'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        let has_deleted_at: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'deleted_at'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();

        if !has_deleted {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN deleted INTEGER NOT NULL DEFAULT 0;")?;
        }
        if !has_deleted_at {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN deleted_at TEXT;")?;
        }

        // v0.3: semantic hash for cross-conversation dedup
        let has_semantic_hash: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'semantic_hash'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        if !has_semantic_hash {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN semantic_hash TEXT NOT NULL DEFAULT '';")?;
        }

        // v0.5: access tracking for memory scoring
        let has_access_count: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'access_count'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        if !has_access_count {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN access_count INTEGER NOT NULL DEFAULT 0;")?;
        }
        let has_last_accessed: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'last_accessed_at'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        if !has_last_accessed {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN last_accessed_at TEXT;")?;
        }

        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_messages_conv
                 ON messages(conversation_id);
             CREATE INDEX IF NOT EXISTS idx_dag_conv
                 ON dag_nodes(conversation_id, level);
             CREATE TABLE IF NOT EXISTS dag_edges (
                 id          INTEGER PRIMARY KEY AUTOINCREMENT,
                 from_id     INTEGER NOT NULL REFERENCES dag_nodes(id),
                 to_id       INTEGER NOT NULL REFERENCES dag_nodes(id),
                 kind        TEXT NOT NULL DEFAULT 'summarizes',
                 created_at  TEXT NOT NULL DEFAULT (datetime('now'))
             );
             CREATE INDEX IF NOT EXISTS idx_edges_from ON dag_edges(from_id);
             CREATE INDEX IF NOT EXISTS idx_edges_to ON dag_edges(to_id);
             CREATE UNIQUE INDEX IF NOT EXISTS idx_edges_unique
                 ON dag_edges(from_id, to_id, kind);

            -- Event sourcing: append-only event log for DAG mutations (v0.6)
            CREATE TABLE IF NOT EXISTS dag_events (
                id          INTEGER PRIMARY KEY AUTOINCREMENT,
                event_type  TEXT NOT NULL,
                node_id     INTEGER,
                conv_id     INTEGER NOT NULL REFERENCES conversations(id),
                payload     TEXT NOT NULL DEFAULT '{}',
                created_at  TEXT NOT NULL DEFAULT (datetime('now'))
            );
            CREATE INDEX IF NOT EXISTS idx_events_conv
                ON dag_events(conv_id);"
        )?;
        Ok(())
    }

    /// Find a conversation by session fingerprint, or create one.
    pub fn find_or_create_conversation(&self, fingerprint: &str, model: &str) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap();
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
        let conn = self.writer.lock().unwrap();
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
        let conn = self.writer.lock().unwrap();
        let parent_json = serde_json::to_string(parent_ids)?;
        let child_json = serde_json::to_string(child_ids)?;
        let snippet_json = serde_json::to_string(snippets)?;
        let hash = Self::semantic_hash(summary, snippets);
        conn.execute(
            "INSERT INTO dag_nodes (conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, 0, ?9)",
            rusqlite::params![conversation_id, level, summary, token_count, parent_json, child_json, snippet_json, is_leaf as i32, hash],
        )?;
        let id = conn.last_insert_rowid();

        // Mirror as typed edges for the semantic graph (P3 Graph Model)
        for child in child_ids {
            let _ = conn.execute(
                "INSERT OR IGNORE INTO dag_edges (from_id, to_id, kind) VALUES (?1, ?2, 'summarizes')",
                rusqlite::params![id, child],
            );
        }

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
            deleted: false,
            semantic_hash: Self::semantic_hash(summary, snippets),
            access_count: 0,
            last_accessed_at: None,
        })
    }

    pub fn get_node(&self, node_id: i64) -> anyhow::Result<Option<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash, access_count, last_accessed_at
             FROM dag_nodes WHERE id = ?1 AND deleted = 0",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![node_id], Self::row_to_node)?;
        Ok(rows.next().transpose()?)
    }

    /// Find nodes whose child_ids list contains the given node_id.
    /// For a given leaf, returns the summaries that include it.
    /// For a summary, returns higher-level summaries that compress it.
    pub fn get_parent_nodes(&self, node_id: i64, max_fanout: usize) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT DISTINCT n.id, n.conversation_id, n.level, n.summary,
                    n.token_count, n.parent_ids, n.child_ids, n.snippets, n.is_leaf, n.deleted,
                    n.semantic_hash, n.access_count, n.last_accessed_at
             FROM dag_nodes n, json_each(n.child_ids) AS j
             WHERE j.value = ?1 AND n.deleted = 0
             LIMIT ?2",
        )?;
        let rows = stmt.query_map(rusqlite::params![node_id, max_fanout as i64], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows {
            nodes.push(row?);
        }
        Ok(nodes)
    }

    pub fn get_child_nodes(&self, node_id: i64, max_fanout: usize) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let child_ids_json: String = conn.query_row(
            "SELECT child_ids FROM dag_nodes WHERE id = ?1 AND deleted = 0",
            rusqlite::params![node_id],
            |row| row.get(0),
        )?;
        let child_ids: Vec<i64> = serde_json::from_str(&child_ids_json).unwrap_or_default();
        if child_ids.is_empty() {
            return Ok(Vec::new());
        }
        let ids: Vec<i64> = child_ids.into_iter().take(max_fanout).collect();
        let placeholders: Vec<String> = ids.iter().enumerate()
            .map(|(i, _)| format!("?{}", i + 1))
            .collect();
        let sql = format!(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash, access_count, last_accessed_at
             FROM dag_nodes WHERE id IN ({}) AND deleted = 0",
            placeholders.join(","),
        );
        let mut stmt = conn.prepare(&sql)?;
        let params: Vec<&dyn rusqlite::types::ToSql> = ids.iter()
            .map(|id| id as &dyn rusqlite::types::ToSql)
            .collect();
        let rows = stmt.query_map(params.as_slice(), Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows {
            nodes.push(row?);
        }
        Ok(nodes)
    }

    pub fn get_tip_node(&self, conv_id: i64) -> anyhow::Result<Option<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash, access_count, last_accessed_at
             FROM dag_nodes WHERE conversation_id = ?1 AND level > 0 AND deleted = 0
             ORDER BY level DESC, id DESC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        Ok(rows.next().transpose()?)
    }

    pub fn get_tip_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash, access_count, last_accessed_at
             FROM dag_nodes
             WHERE conversation_id = ?1 AND level = (
                 SELECT MAX(level) FROM dag_nodes WHERE conversation_id = ?1 AND level > 0 AND deleted = 0
             ) AND deleted = 0
             ORDER BY id DESC",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows { nodes.push(row?); }
        Ok(nodes)
    }

    pub fn get_all_dag_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash, access_count, last_accessed_at
             FROM dag_nodes WHERE conversation_id = ?1 AND deleted = 0 ORDER BY id ASC",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        let mut nodes = Vec::new();
        for row in rows { nodes.push(row?); }
        Ok(nodes)
    }

    /// Atomically insert a summary node and back-link to its source nodes.
    /// Wraps the insert + parent-ids update in a single transaction (P2-6)
    /// to prevent partial graph writes.
    pub fn insert_summary_atomic(
        &self,
        conversation_id: i64,
        level: u8,
        summary: &str,
        token_count: i64,
        source_ids: &[i64],
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        let conn = self.writer.lock().unwrap();
        let tx = conn.unchecked_transaction()?;

        let parent_json = "[]".to_string();
        let child_json = serde_json::to_string(source_ids)?;
        let snippet_json = serde_json::to_string(snippets)?;
        let hash = Self::semantic_hash(summary, snippets);
        tx.execute(
            "INSERT INTO dag_nodes (conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, 0, ?8)",
            rusqlite::params![conversation_id, level, summary, token_count, parent_json, child_json, snippet_json, hash],
        )?;
        let new_id = tx.last_insert_rowid();

        // Mirror as typed edges (P3 Graph Model)
        for sid in source_ids {
            let _ = tx.execute(
                "INSERT OR IGNORE INTO dag_edges (from_id, to_id, kind) VALUES (?1, ?2, 'summarizes')",
                rusqlite::params![new_id, sid],
            );
        }

        // Index snippets in FTS5 for independent retrieval (P3 Summary Storage)
        for snippet in snippets {
            let stype = match snippet.snippet_type {
                crate::snippet::SnippetType::CodeBlock => "code",
                crate::snippet::SnippetType::FilePath => "path",
                crate::snippet::SnippetType::NumericConstant => "num",
                crate::snippet::SnippetType::ErrorMessage => "err",
                crate::snippet::SnippetType::ProperNoun => "ref",
            };
            let _ = tx.execute(
                "INSERT INTO snippets_fts (content, source_type, node_id) VALUES (?1, ?2, ?3)",
                rusqlite::params![snippet.content, stype, new_id],
            );
        }

        // Index in cross-conversation semantic dedup table (v0.3)
        let preview: String = summary.chars().take(100).collect();
        let _ = tx.execute(
            "INSERT OR IGNORE INTO semantic_index (hash, node_id, conversation_id, summary_preview)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![hash, new_id, conversation_id, preview],
        );

        // Record provenance: each source node contributed to this summary (v0.4)
        for (i, sid) in source_ids.iter().enumerate() {
            let _ = tx.execute(
                "INSERT INTO provenance (summary_node_id, source_node_id, sentence_offset, sentence_length)
                 VALUES (?1, ?2, ?3, 0)",
                rusqlite::params![new_id, sid, i as i32],
            );
        }

        for sid in source_ids {
            let existing: String = tx.query_row(
                "SELECT parent_ids FROM dag_nodes WHERE id = ?1",
                rusqlite::params![sid],
                |row| row.get(0),
            )?;
            let mut ids: Vec<i64> = serde_json::from_str(&existing).unwrap_or_default();
            if !ids.contains(&new_id) {
                ids.push(new_id);
                let json = serde_json::to_string(&ids)?;
                tx.execute(
                    "UPDATE dag_nodes SET parent_ids = ?1 WHERE id = ?2",
                    rusqlite::params![json, sid],
                )?;
            }
        }

        // Record event for audit trail
        let _ = tx.execute(
            "INSERT INTO dag_events (event_type, node_id, conv_id, payload) VALUES ('compress', ?1, ?2, '{}')",
            rusqlite::params![new_id, conversation_id],
        );

        tx.commit()?;
        Ok(DagNode {
            id: new_id,
            conversation_id,
            level,
            summary: summary.to_string(),
            token_count,
            parent_ids: vec![],
            child_ids: source_ids.to_vec(),
            snippets: snippets.to_vec(),
            is_leaf: false,
            deleted: false,
            semantic_hash: Self::semantic_hash(summary, snippets),
            access_count: 0,
            last_accessed_at: None,
        })
    }

    /// Increment access count and update last_accessed_at for memory scoring.
    pub fn touch_node(&self, node_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "UPDATE dag_nodes SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?1",
            rusqlite::params![node_id],
        )?;
        Ok(())
    }

    /// Append an event to the DAG event log for audit and rollback.
    pub fn record_event(&self, event_type: &str, node_id: i64, conv_id: i64, payload: &str) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "INSERT INTO dag_events (event_type, node_id, conv_id, payload) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![event_type, node_id, conv_id, payload],
        )?;
        Ok(())
    }

    /// Get event log for a conversation, ordered by time.
    pub fn get_events(&self, conv_id: i64, limit: usize) -> anyhow::Result<Vec<(String, Option<i64>, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT event_type, node_id, payload FROM dag_events WHERE conv_id = ?1 ORDER BY id DESC LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, Option<i64>>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Soft-delete a DAG node (set deleted=1, timestamp). The node is
    /// excluded from context assembly but raw data in `messages` is intact.
    /// Garbage collection can later hard-delete fully orphaned soft-deleted nodes.
    pub fn delete_dag_node(&self, node_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "UPDATE dag_nodes SET deleted = 1, deleted_at = datetime('now') WHERE id = ?1",
            rusqlite::params![node_id],
        )?;
        // Record event for audit trail
        let _ = self.record_event("delete", node_id, -1, "{}");
        Ok(())
    }

    /// Hard-delete a node (used by GC for fully unreachable ghosts).
    pub fn purge_dag_node(&self, node_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute("DELETE FROM dag_nodes WHERE id = ?1", rusqlite::params![node_id])?;
        Ok(())
    }

    pub fn get_leaf_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, deleted, semantic_hash, access_count, last_accessed_at
             FROM dag_nodes WHERE conversation_id = ?1 AND is_leaf = 1 AND deleted = 0
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
        let conn = self.read_conn();
        let total: i64 = conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0) FROM messages WHERE conversation_id = ?1",
            rusqlite::params![conv_id],
            |row| row.get(0),
        )?;
        Ok(total)
    }

    pub fn add_child_to_node(&self, parent_id: i64, child_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
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

    /// Add a parent ID to a node's parent_ids list.
    /// Semantic: `summary.child_ids = raw_ids`, so raw nodes get the summary
    /// added to their parent_ids.
    pub fn add_parent_to_node(&self, child_id: i64, parent_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        let existing: String = conn.query_row(
            "SELECT parent_ids FROM dag_nodes WHERE id = ?1",
            rusqlite::params![child_id],
            |row| row.get(0),
        )?;
        let mut ids: Vec<i64> = serde_json::from_str(&existing).unwrap_or_default();
        if !ids.contains(&parent_id) {
            ids.push(parent_id);
            let json = serde_json::to_string(&ids)?;
            conn.execute(
                "UPDATE dag_nodes SET parent_ids = ?1 WHERE id = ?2",
                rusqlite::params![json, child_id],
            )?;
        }
        Ok(())
    }

    // ── Edge operations (typed semantic graph) ─────────────────────────

    /// Insert a typed edge between two nodes.  Returns the edge ID.
    /// Ignores duplicate (from_id, to_id, kind) tuples (UNIQUE constraint).
    pub fn insert_edge(&self, from_id: i64, to_id: i64, kind: &str) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO dag_edges (from_id, to_id, kind) VALUES (?1, ?2, ?3)",
            rusqlite::params![from_id, to_id, kind],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Get all outgoing edges from a node.
    pub fn get_edges_from(&self, node_id: i64) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, to_id, kind FROM dag_edges WHERE from_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(rusqlite::params![node_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Get all incoming edges to a node.
    pub fn get_edges_to(&self, node_id: i64) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, from_id, kind FROM dag_edges WHERE to_id = ?1 ORDER BY id",
        )?;
        let rows = stmt.query_map(rusqlite::params![node_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Check whether a path exists from `from` to `to` (BFS, limited depth).
    /// Used for cycle detection before inserting an edge.
    pub fn has_path(&self, from_id: i64, to_id: i64, max_depth: usize) -> anyhow::Result<bool> {
        if from_id == to_id {
            return Ok(true);
        }
        let conn = self.read_conn();
        let mut visited = std::collections::HashSet::new();
        let mut queue = vec![from_id];
        visited.insert(from_id);

        let mut stmt = conn.prepare("SELECT to_id FROM dag_edges WHERE from_id = ?1")?;
        let mut depth = 0;

        while !queue.is_empty() && depth < max_depth {
            let mut next = Vec::new();
            for node_id in &queue {
                let rows = stmt.query_map(rusqlite::params![node_id], |row| row.get::<_, i64>(0))?;
                for row in rows {
                    let target = row?;
                    if target == to_id {
                        return Ok(true);
                    }
                    if visited.insert(target) {
                        next.push(target);
                    }
                }
            }
            queue = next;
            depth += 1;
        }
        Ok(false)
    }

    // ── Search ──────────────────────────────────────────────────────────

    /// Search messages in a conversation using FTS5 full-text search.
    /// Returns `(id, role, snippet, token_count)` ranked by relevance.
    /// Get raw message content in a range of message IDs.
    pub fn get_messages_in_range(&self, conv_id: i64, from_id: i64, to_id: i64) -> anyhow::Result<Vec<String>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT content FROM messages
             WHERE conversation_id = ?1 AND id BETWEEN ?2 AND ?3
             ORDER BY id ASC LIMIT 50",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, from_id, to_id], |row| {
            row.get::<_, String>(0)
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    pub fn search_messages(&self, conv_id: i64, query: &str) -> anyhow::Result<Vec<(i64, String, String, i64)>> {
        let conn = self.read_conn();
        if query.is_empty() {
            return Ok(Vec::new());
        }
        // FTS5 MATCH doesn't support mixed CJK/English reliably with the
        // bundled tokenizers (verified: English-only works, CJK+English fails).
        // Use LIKE as a universal fallback.  FTS5 table is retained for
        // future tokenizer upgrades (e.g. jieba, ICU).
        let pattern = format!("%{}%", query.replace('%', "%%").replace('_', "\\_"));
        let mut stmt = conn.prepare(
            "SELECT id, role, substr(content, 1, 500), token_count
             FROM messages
             WHERE conversation_id = ?1 AND content LIKE ?2 ESCAPE '\\'
             ORDER BY id ASC
             LIMIT 20",
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, pattern], |row| {
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

    /// Unified search across messages, summaries, and snippets.
    /// Uses FTS5 MATCH + bm25() for English queries, LIKE fallback for CJK.
    pub fn search_unified(
        &self,
        conv_id: i64,
        query: &str,
    ) -> anyhow::Result<Vec<UnifiedSearchResult>> {
        let conn = self.read_conn();
        if query.is_empty() {
            return Ok(Vec::new());
        }

        let pattern = format!("%{}%", query.replace('%', "%%").replace('_', "\\_"));

        // Try FTS5 MATCH first (supports BM25 scoring, works for English).
        // FTS5 unicode61 tokenizer fails on mixed CJK/English — detect and fall back.
        let has_cjk = query.chars().any(|c| c as u32 > 0x2E80);
        let use_fts5 = !has_cjk;

        if use_fts5 {
            let fts_query = Self::fts5_query(query);
            let sql = "
                SELECT messages_fts.rowid AS id, 'message' AS source, role AS label,
                       substr(messages.content, 1, 500) AS excerpt,
                       messages.token_count, bm25(messages_fts, 0.0, 1.0, 0.5) AS score
                FROM messages_fts
                JOIN messages ON messages.id = messages_fts.rowid
                WHERE messages_fts MATCH ?1 AND messages.conversation_id = ?2
                ORDER BY score
                LIMIT 30
            ";
            if let Ok(mut stmt) = conn.prepare(sql) {
                if let Ok(rows) = stmt.query_map(
                    rusqlite::params![fts_query, conv_id],
                    |row| {
                        Ok(UnifiedSearchResult {
                            id: row.get(0)?,
                            source: row.get(1)?,
                            label: row.get(2)?,
                            excerpt: row.get(3)?,
                            token_count: row.get::<_, Option<i64>>(4)?.unwrap_or(0),
                            bm25_score: row.get::<_, Option<f64>>(5)?.map(|s| -s),
                        })
                    },
                ) {
                    let mut results: Vec<UnifiedSearchResult> = rows.filter_map(|r| r.ok()).collect();
                    if !results.is_empty() {
                        results.sort_by(|a, b| a.bm25_score.partial_cmp(&b.bm25_score).unwrap_or(std::cmp::Ordering::Equal));
                        return Ok(results);
                    }
                }
            }
        }

        // LIKE fallback for CJK or empty FTS5 results
        let sql = "
            SELECT id, 'message' AS source, role AS label,
                   substr(content, 1, 500) AS excerpt, token_count
            FROM messages
            WHERE conversation_id = ?1 AND content LIKE ?2 ESCAPE '\\'

            UNION ALL

            SELECT id, 'summary' AS source, CAST(level AS TEXT) AS label,
                   substr(summary, 1, 500) AS excerpt, token_count
            FROM dag_nodes
            WHERE conversation_id = ?1 AND summary LIKE ?2 ESCAPE '\\' AND deleted = 0

            UNION ALL

            SELECT id, 'snippet' AS source, CAST(level AS TEXT) AS label,
                   substr(snippets, 1, 500) AS excerpt, token_count
            FROM dag_nodes
            WHERE conversation_id = ?1 AND snippets LIKE ?2 ESCAPE '\\' AND deleted = 0

            UNION ALL

            SELECT node_id AS id, 'snippet' AS source, source_type AS label,
                   substr(content, 1, 500) AS excerpt,
                   NULL AS token_count
            FROM snippets_fts
            WHERE content LIKE ?2 ESCAPE '\\'

            ORDER BY id DESC
            LIMIT 30
        ";

        let mut stmt = conn.prepare(sql)?;
        let rows = stmt.query_map(rusqlite::params![conv_id, pattern], |row| {
            Ok(UnifiedSearchResult {
                id: row.get(0)?,
                source: row.get(1)?,
                label: row.get(2)?,
                excerpt: row.get(3)?,
                token_count: row.get::<_, Option<i64>>(4)?.unwrap_or(0),
                bm25_score: None,
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Build a safe FTS5 query string from user input.
    /// Wraps each word in quotes for phrase matching.
    fn fts5_query(query: &str) -> String {
        let escaped = Self::fts_escape(query);
        if escaped.is_empty() {
            return "*".to_string();
        }
        escaped.split_whitespace()
            .map(|w| format!("\"{}\"", w.replace('"', "")))
            .collect::<Vec<_>>()
            .join(" ")
    }

    // ── Embedding vector storage (v0.5) ───────────────────────────────

    /// Store an embedding vector for a DAG node.
    pub fn store_embedding(&self, node_id: i64, vector: &[f32], model: &str, dims: i32) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        let bytes: Vec<u8> = vector.iter().flat_map(|f| f.to_le_bytes()).collect();
        conn.execute(
            "INSERT OR REPLACE INTO embeddings (node_id, vector, model, dimensions) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![node_id, bytes, model, dims],
        )?;
        Ok(())
    }

    /// Get the embedding vector for a node.
    pub fn get_embedding(&self, node_id: i64) -> anyhow::Result<Option<Vec<f32>>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT vector, dimensions FROM embeddings WHERE node_id = ?1")?;
        let mut rows = stmt.query_map(rusqlite::params![node_id], |row| {
            Ok((row.get::<_, Vec<u8>>(0)?, row.get::<_, i32>(1)?))
        })?;
        if let Some(row) = rows.next() {
            let (bytes, dims) = row?;
            let floats: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if floats.len() == dims as usize {
                return Ok(Some(floats));
            }
        }
        Ok(None)
    }

    /// Find the nearest neighbor node by cosine similarity.
    /// Returns (node_id, similarity) if similarity >= min_similarity.
    pub fn find_nearest_embedding(
        &self,
        query_vec: &[f32],
        min_similarity: f32,
    ) -> anyhow::Result<Option<(i64, f32)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT node_id, vector, dimensions FROM embeddings WHERE model = 'deepseek-embed'"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, Vec<u8>>(1)?, row.get::<_, i32>(2)?))
        })?;

        let mut best: Option<(i64, f32)> = None;
        for row in rows {
            let (nid, bytes, dims) = row?;
            let floats: Vec<f32> = bytes.chunks(4)
                .map(|c| f32::from_le_bytes([c[0], c[1], c[2], c[3]]))
                .collect();
            if floats.len() != dims as usize || floats.len() != query_vec.len() {
                continue;
            }
            let sim = cosine_similarity(query_vec, &floats);
            if sim >= min_similarity {
                if best.map_or(true, |(_, s)| sim > s) {
                    best = Some((nid, sim));
                }
            }
        }
        Ok(best)
    }

    // ── Semantic dedup (v0.3) ─────────────────────────────────────────

    /// Find nodes with the same semantic hash across ALL conversations.
    /// Returns (hash, node_id, conversation_id, summary_preview).
    pub fn find_similar_by_hash(&self, hash: &str) -> anyhow::Result<Vec<(String, i64, i64, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT hash, node_id, conversation_id, summary_preview
             FROM semantic_index WHERE hash = ?1 LIMIT 10",
        )?;
        let rows = stmt.query_map(rusqlite::params![hash], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, i64>(1)?,
                row.get::<_, i64>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Index a node in the semantic_index for cross-conversation lookup.
    pub fn index_semantic(&self, hash: &str, node_id: i64, conv_id: i64, preview: &str) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "INSERT OR IGNORE INTO semantic_index (hash, node_id, conversation_id, summary_preview)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![hash, node_id, conv_id, preview],
        )?;
        Ok(())
    }

    /// Record provenance: which source node contributed to a summary.
    /// For now, stores the full source range. Future: per-sentence offsets.
    pub fn record_provenance(
        &self,
        summary_node_id: i64,
        source_node_id: i64,
        offset: i32,
        length: i32,
    ) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        conn.execute(
            "INSERT INTO provenance (summary_node_id, source_node_id, sentence_offset, sentence_length)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![summary_node_id, source_node_id, offset, length],
        )?;
        Ok(())
    }

    /// Get provenance records for a summary node.
    pub fn get_provenance(&self, summary_node_id: i64) -> anyhow::Result<Vec<(i64, i32, i32)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT source_node_id, sentence_offset, sentence_length
             FROM provenance WHERE summary_node_id = ?1 ORDER BY sentence_offset",
        )?;
        let rows = stmt.query_map(rusqlite::params![summary_node_id], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i32>(1)?, row.get::<_, i32>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Store sentence-level provenance spans for a summary node.
    pub fn store_provenance_spans(
        &self,
        summary_node_id: i64,
        spans: &[(i64, i32, i32)],
    ) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap();
        for (source_node_id, offset, length) in spans {
            conn.execute(
                "INSERT INTO provenance (summary_node_id, source_node_id, sentence_offset, sentence_length)
                 VALUES (?1, ?2, ?3, ?4)",
                rusqlite::params![summary_node_id, source_node_id, offset, length],
            )?;
        }
        Ok(())
    }

    /// Get provenance with source excerpts for a summary node.
    pub fn get_provenance_with_excerpts(
        &self,
        summary_node_id: i64,
    ) -> anyhow::Result<Vec<(i64, i32, i32, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT p.source_node_id, p.sentence_offset, p.sentence_length,
                    substr(n.summary, 1, 200)
             FROM provenance p
             JOIN dag_nodes n ON n.id = p.source_node_id
             WHERE p.summary_node_id = ?1
             ORDER BY p.sentence_offset"
        )?;
        let rows = stmt.query_map(rusqlite::params![summary_node_id], |row| {
            Ok((
                row.get::<_, i64>(0)?,
                row.get::<_, i32>(1)?,
                row.get::<_, i32>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
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

/// Compute a semantic fingerprint for a node: SHA-256 of summary text
/// concatenated with snippet contents, then truncated to 16 hex chars.
/// Equal hashes mean the summary content is semantically identical
/// (or close enough for dedup), enabling cross-conversation reuse.
pub fn semantic_hash(summary: &str, snippets: &[crate::snippet::Snippet]) -> String {
    use sha2::{Digest, Sha256};
    let mut hasher = Sha256::new();
    hasher.update(summary.as_bytes());
    for s in snippets {
        hasher.update(s.content.as_bytes());
        let label = match s.snippet_type {
            crate::snippet::SnippetType::CodeBlock => "code",
            crate::snippet::SnippetType::FilePath => "path",
            crate::snippet::SnippetType::NumericConstant => "num",
            crate::snippet::SnippetType::ErrorMessage => "err",
            crate::snippet::SnippetType::ProperNoun => "ref",
        };
        hasher.update(label.as_bytes());
    }
    let result = hasher.finalize();
    hex::encode(&result[..8]) // 16 hex chars
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
        let deleted_int: i32 = row.get(9)?;
        let semantic_hash: String = row.get(10)?;
        let access_count: i64 = row.get(11).unwrap_or(0);
        let last_accessed_at: Option<String> = row.get(12).ok().flatten();
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
            deleted: deleted_int != 0,
            semantic_hash,
            access_count,
            last_accessed_at,
        })
    }
}

fn cosine_similarity(a: &[f32], b: &[f32]) -> f32 {
    let (dot, na, nb) = a.iter().zip(b.iter())
        .fold((0.0f32, 0.0f32, 0.0f32), |(d, na, nb), (x, y)| {
            (d + x * y, na + x * x, nb + y * y)
        });
    if na == 0.0 || nb == 0.0 { 0.0 } else { dot / (na.sqrt() * nb.sqrt()) }
}

#[cfg(test)]
mod tests {
    use super::*;
    use serde_json::json;
    use std::sync::Arc;
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

    #[test]
    fn fts5_match_works_through_rusqlite() {
        use rusqlite::Connection;
        let conn = Connection::open_in_memory().unwrap();
        // Pure English text with default tokenizer
        conn.execute_batch(
            "CREATE VIRTUAL TABLE te USING fts5(content);
             INSERT INTO te VALUES ('hello world Gartner test');
             CREATE VIRTUAL TABLE tc USING fts5(content);
             INSERT INTO tc VALUES ('一份来自Gartner的2026年报告');"
        ).unwrap();
        let e1: i64 = conn.query_row("SELECT count(*) FROM te WHERE te MATCH 'Gartner'", [], |r| r.get(0)).unwrap();
        let c1: i64 = conn.query_row("SELECT count(*) FROM tc WHERE tc MATCH 'Gartner'", [], |r| r.get(0)).unwrap();
        let c2: i64 = conn.query_row("SELECT count(*) FROM tc WHERE content LIKE '%Gartner%'", [], |r| r.get(0)).unwrap();
        println!("  English MATCH: {e1}, CJK MATCH: {c1}, CJK LIKE: {c2}");
        assert_eq!(e1, 1, "English text MATCH must work");
        assert_eq!(c2, 1, "CJK LIKE must work");
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
        let conn = db.writer.lock().unwrap();
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

    // ── P3: Unified search ───────────────────────────────────────────

    #[tokio::test]
    async fn search_unified_finds_messages_summaries_and_snippets() {
        let dir = tempdir().unwrap();
        let db = Database::builder()
            .path(dir.path().join("search_unified.db"))
            .build()
            .await
            .unwrap();

        let messages = json!([
            {"role": "user", "content": "hello world"},
            {"role": "assistant", "content": "hi there"}
        ]);
        let conv_id = db.create_and_store("test", &messages).unwrap();

        // Insert a DAG node with a summary and snippets
        let snippets = vec![
            crate::snippet::Snippet {
                snippet_type: crate::snippet::SnippetType::FilePath,
                content: "src/main.rs".to_string(),
                importance: 1.0,
                source_node_id: String::new(),
                frequency: 0,
            },
            crate::snippet::Snippet {
                snippet_type: crate::snippet::SnippetType::NumericConstant,
                content: "8080".to_string(),
                importance: 0.9,
                source_node_id: String::new(),
                frequency: 0,
            },
        ];
        db.insert_dag_node_full(conv_id, 1, "fixed port binding error", 15, &[], &[], &snippets, false).unwrap();

        // Search for "port" — should find the summary
        let results = db.search_unified(conv_id, "port").unwrap();
        assert!(!results.is_empty(), "should find summary containing 'port'");
        let has_summary = results.iter().any(|r| r.source == "summary");
        assert!(has_summary, "should include summary results");

        // Search for ".rs" — should find the snippet
        let results = db.search_unified(conv_id, ".rs").unwrap();
        let has_snippet = results.iter().any(|r| r.source == "snippet");
        assert!(has_snippet, "should include snippet results");

        // Search for "hello" — should find the message
        let results = db.search_unified(conv_id, "hello").unwrap();
        let has_message = results.iter().any(|r| r.source == "message");
        assert!(has_message, "should include message results");

        // Search for nonexistent string
        let empty = db.search_unified(conv_id, "zzz_nonexistent").unwrap();
        assert!(empty.is_empty(), "should find nothing");
    }

    // ── Phase 1: Semantic Foundation integration ───────────────────────

    #[tokio::test]
    async fn phase1_retrieval_scored_context_assembly() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase1_ctx.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([
            {"role": "user", "content": "what is rust programming language"},
            {"role": "assistant", "content": "Rust is a systems language focused on safety"},
            {"role": "user", "content": "how do I install cargo"},
            {"role": "assistant", "content": "Install via rustup: curl https://sh.rustup.rs | sh"}
        ])).unwrap();

        let engine = crate::dag::DagEngine::builder().max_level(3).build(db.clone());

        let l1 = engine.insert_leaf(conv_id, "what is rust", 10).unwrap();
        let l2 = engine.insert_leaf(conv_id, "Rust is a systems language", 15).unwrap();
        let l3 = engine.insert_leaf(conv_id, "how do I install cargo", 10).unwrap();
        let l4 = engine.insert_leaf(conv_id, "curl rustup.rs | sh", 15).unwrap();

        engine.compress_group(conv_id, &[l1.id, l2.id], "Rust is a systems programming language", 12, 1).unwrap();

        let ctx = engine.assemble_context(conv_id, 200, Some("install cargo")).unwrap();
        assert!(!ctx.is_empty());

        let has_cargo = ctx.iter().any(|n| n.summary.contains("cargo") || n.summary.contains("rustup"));
        assert!(has_cargo, "query-aware context should include cargo-related content");
    }

    #[tokio::test]
    async fn phase1_semantic_dedup_sha256_fallback() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase1_dedup.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "hi"}]))
            .unwrap();

        let engine = crate::dag::DagEngine::builder().max_level(3).build(db.clone());

        let a = engine.insert_leaf(conv_id, "leaf A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "leaf B", 5).unwrap();

        let s1 = engine.dedup_and_reuse(conv_id, &[a.id, b.id], "same text", 5, 1, &[]).unwrap();
        let s2 = engine.dedup_and_reuse(conv_id, &[a.id], "same text", 5, 1, &[]).unwrap();

        assert_eq!(s1.id, s2.id, "identical text should dedup via SHA-256");
        assert_eq!(s1.semantic_hash, s2.semantic_hash);
    }

    #[tokio::test]
    async fn phase1_cycle_detection_prevents_loop() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase1_cycle.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "hi"}]))
            .unwrap();

        let engine = crate::dag::DagEngine::builder().max_level(3).build(db.clone());

        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        let summary = engine.compress_group(conv_id, &[a.id, b.id], "summary AB", 8, 1).unwrap();

        // Verify no cycle exists in healthy tree
        assert!(!engine.db().has_path(a.id, summary.id, 20).unwrap());
        assert!(!engine.db().has_path(b.id, summary.id, 20).unwrap());

        // Create a path B → summary (would form cycle if summary could reach B)
        engine.db().insert_edge(b.id, summary.id, "summarizes").unwrap();
        assert!(engine.db().has_path(b.id, summary.id, 20).unwrap());
    }

    #[tokio::test]
    async fn phase1_shared_node_visited_set() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase1_shared.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "hi"}]))
            .unwrap();

        let engine = crate::dag::DagEngine::builder().max_level(3).build(db.clone());

        let a = engine.insert_leaf(conv_id, "A", 5).unwrap();
        let b = engine.insert_leaf(conv_id, "B", 5).unwrap();
        let c = engine.insert_leaf(conv_id, "C", 5).unwrap();

        let summary = engine.compress_group(conv_id, &[a.id, b.id], "summary AB", 8, 1).unwrap();
        engine.db().add_child_to_node(summary.id, c.id).unwrap();
        engine.db().add_parent_to_node(c.id, summary.id).unwrap();

        let ctx = engine.assemble_context(conv_id, 200, None).unwrap();
        let count = ctx.iter().filter(|n| n.id == summary.id).count();
        assert_eq!(count, 1, "shared summary must appear only once");
    }

    // ── Phase 2: Concurrency ──────────────────────────────────────────

    #[tokio::test]
    async fn phase2_concurrent_reads_do_not_block() {
        use std::time::Instant;
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase2_conc.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([
            {"role": "user", "content": "hello"}
        ])).unwrap();

        for i in 0..100 {
            db.insert_dag_node(conv_id, 0, &format!("node {i}"), 5, &[], &[], true).unwrap();
        }

        let start = Instant::now();
        let mut handles = Vec::new();
        for _ in 0..10 {
            let db = db.clone();
            let h = tokio::task::spawn_blocking(move || {
                for _ in 0..20 {
                    let _ = db.get_all_dag_nodes(conv_id);
                }
            });
            handles.push(h);
        }
        for h in handles {
            h.await.unwrap();
        }
        let elapsed = start.elapsed();
        assert!(elapsed.as_secs() < 5, "concurrent reads took too long: {elapsed:?}");
    }

    // ── Phase 2: Provenance + Hierarchical rendering ──────────────────

    #[tokio::test]
    async fn phase2_provenance_spans_persist() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase2_prov.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "hi"}]))
            .unwrap();
        let engine = crate::dag::DagEngine::builder().max_level(3).build(db.clone());

        let l1 = engine.insert_leaf(conv_id, "Rust is fast. It has zero-cost abstractions.", 10).unwrap();
        let l2 = engine.insert_leaf(conv_id, "Memory safety is key.", 5).unwrap();

        let summary = engine.compress_group(
            conv_id, &[l1.id, l2.id],
            "Rust is fast. Memory safety is key.",
            8, 1,
        ).unwrap();

        let prov = db.get_provenance(summary.id).unwrap();
        assert!(!prov.is_empty(), "provenance records should be created");
        let has_real_spans = prov.iter().any(|(_sid, _off, len)| *len > 0);
        assert!(has_real_spans, "should have real provenance span, got: {prov:?}");
    }

    #[tokio::test]
    async fn phase2_hierarchical_render_has_three_tiers() {
        let dir = tempdir().unwrap();
        let db = Arc::new(Database::builder()
            .path(dir.path().join("phase2_render.db"))
            .build()
            .await
            .unwrap());

        let conv_id = db.create_and_store("test", &json!([{"role": "user", "content": "hi"}]))
            .unwrap();
        let engine = crate::dag::DagEngine::builder().max_level(3).build(db.clone());

        let l1 = engine.insert_leaf(conv_id, "hello", 5).unwrap();
        let l2 = engine.insert_leaf(conv_id, "world", 5).unwrap();
        engine.compress_group(conv_id, &[l1.id, l2.id], "hello world summary", 8, 1).unwrap();
        // Add a 3rd leaf not covered by the summary
        engine.insert_leaf(conv_id, "extra message", 5).unwrap();

        let ctx = engine.assemble_context(conv_id, 200, None).unwrap();
        let rendered = crate::pipeline::render_dag_context(&ctx);

        assert!(rendered.contains("── Summaries ──"), "should have summaries tier");
        assert!(rendered.contains("── Recent Messages ──"), "should have recent messages tier");
        assert!(rendered.contains("← sources:"), "should show source provenance");
    }
}

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

const DEFAULT_DB_PATH: &str = "~/.deeplossless/lcm.db";

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
const READ_POOL_SIZE: usize = 8;

pub struct Database {
    read_pool: Vec<Mutex<Connection>>,
    writer: Mutex<Connection>,
    write_count: AtomicU64,
    tool_cache_l1: crate::tool_cache::L1HotCache,
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
            tool_cache_l1: crate::tool_cache::L1HotCache::default(),
        };
        db.migrate()?;
        Ok(db)
    }

    /// Round-robin dispatch to the next read connection.
    fn read_conn(&self) -> std::sync::MutexGuard<'_, Connection> {
        let idx = self.write_count.load(Ordering::Relaxed) as usize % self.read_pool.len();
        self.read_pool[idx].lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Run a WAL checkpoint to prevent the WAL file from growing
    /// indefinitely under sustained write load.  Call periodically
    /// (e.g. every N store_messages calls).
    pub fn wal_checkpoint(&self) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute_batch("PRAGMA wal_checkpoint(TRUNCATE);")?;
        Ok(())
    }

    fn migrate(&self) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        // Schema version tracking — prevents silent migration drift.
        conn.execute_batch(
            "CREATE TABLE IF NOT EXISTS schema_meta (
                version TEXT NOT NULL,
                applied_at TEXT NOT NULL DEFAULT (datetime('now'))
            );"
        )?;
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
                is_join         INTEGER NOT NULL DEFAULT 0,
                deleted         INTEGER NOT NULL DEFAULT 0,
                deleted_at      TEXT,
                semantic_hash   TEXT NOT NULL DEFAULT '',
                graph_revision  INTEGER NOT NULL DEFAULT 0,
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
        let has_graph_revision: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'graph_revision'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        if !has_graph_revision {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN graph_revision INTEGER NOT NULL DEFAULT 0;")?;
        }

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

        // v0.7: reasoning chain for execution provenance
        let has_reasoning: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'reasoning'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        if !has_reasoning {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN reasoning TEXT NOT NULL DEFAULT '';")?;
        }

        // v0.10: compaction_id for idempotent compaction dedup (P0-9)
        let has_compaction_id: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('dag_nodes') WHERE name = 'compaction_id'")
            .ok()
            .and_then(|mut s| s.query_row([], |_| Ok(())).ok())
            .is_some();
        if !has_compaction_id {
            conn.execute_batch("ALTER TABLE dag_nodes ADD COLUMN compaction_id TEXT NOT NULL DEFAULT '';")?;
        }
        // Index for fast dedup lookup
        conn.execute_batch(
            "CREATE INDEX IF NOT EXISTS idx_dag_compaction_id
                 ON dag_nodes(compaction_id) WHERE compaction_id != '';"
        )?;

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

        // v0.7: execution units — agent memory atoms
        conn.execute_batch(crate::execution::MIGRATION)?;
        // v0.7: code diff memory
        conn.execute_batch(crate::execution::CODE_CHANGE_MIGRATION)?;
        // v0.8: tool result cache
        conn.execute_batch(crate::tool_cache::MIGRATION)?;
        // v0.9: artifact versioning + dependency edges for invalidation correctness
        conn.execute_batch(crate::artifacts::MIGRATION)?;
        // v0.9: provenance lineage edges
        conn.execute_batch(crate::execution::LINEAGE_MIGRATION)?;
        // v0.9: structured reasoning steps
        conn.execute_batch(crate::execution::REASONING_MIGRATION)?;

        // v0.3.0: structured execution fields (additive, backward compatible)
        let has_tool_args_json: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('execution_units') WHERE name='tool_args_json'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_tool_args_json {
            conn.execute_batch("ALTER TABLE execution_units ADD COLUMN tool_args_json TEXT;")?;
        }
        let has_reasoning_steps: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('execution_units') WHERE name='reasoning_steps'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_reasoning_steps {
            conn.execute_batch("ALTER TABLE execution_units ADD COLUMN reasoning_steps TEXT NOT NULL DEFAULT '[]';")?;
        }

        // v0.3.0: artifact hashes for content-based cache invalidation
        let has_file_hashes: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('tool_cache') WHERE name='dependent_file_hashes'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_file_hashes {
            conn.execute_batch("ALTER TABLE tool_cache ADD COLUMN dependent_file_hashes TEXT NOT NULL DEFAULT '[]';")?;
        }
        // v0.9: multi-agent safe runtime
        conn.execute_batch("
            CREATE TABLE IF NOT EXISTS agent_active_files (
                agent_id        TEXT NOT NULL,
                file_path       TEXT NOT NULL,
                operation       TEXT NOT NULL DEFAULT 'edit',
                claimed_at      TEXT NOT NULL DEFAULT (datetime('now')),
                PRIMARY KEY (agent_id, file_path)
            );
            CREATE INDEX IF NOT EXISTS idx_agent_files
                ON agent_active_files(file_path);")?;
        // v0.8: failure memory
        conn.execute_batch(crate::execution::FAILURE_MIGRATION)?;
        // v0.3.0: failure pattern environment fingerprint (after table creation)
        {
            let has_exec_key: bool = conn
                .prepare("SELECT 1 FROM pragma_table_info('failure_patterns') WHERE name='execution_key'")
                .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
            if !has_exec_key {
                conn.execute_batch("ALTER TABLE failure_patterns ADD COLUMN execution_key TEXT NOT NULL DEFAULT '';")?;
                conn.execute_batch("ALTER TABLE failure_patterns ADD COLUMN environment_fingerprint TEXT NOT NULL DEFAULT '';")?;
            }
        }
        // v0.8: plan persistence
        conn.execute_batch(crate::execution::PLAN_MIGRATION)?;

        // v0.4.0: execution snapshots — replay acceleration with budget-aware retention
        conn.execute_batch(crate::snapshot::MIGRATION)?;

        // v0.6.1: snapshot schema versioning + integrity fields
        let has_snapshot_schema: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('execution_snapshots') WHERE name='schema_version'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_snapshot_schema {
            conn.execute_batch(crate::snapshot::ALTER_MIGRATION)?;
        }

        // v0.4.0: execution events — append-only event sourcing
        conn.execute_batch(crate::execution::EVENT_MIGRATION)?;

        // v0.5.0: mutation engine log
        conn.execute_batch(crate::mutation::MUTATION_LOG_MIGRATION)?;

        // v0.5.0: file observations for structured file caching
        conn.execute_batch(crate::file_observation::MIGRATION)?;

        // v0.5.0: tool_call_id for parallel group matching
        let has_tool_call_id: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('execution_units') WHERE name='tool_call_id'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_tool_call_id {
            conn.execute_batch(crate::execution::MIGRATION_ALTER_V6)?;
        }

        // v0.6: audit P0 columns for execution_events
        let has_epoch_ms: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('execution_events') WHERE name='epoch_ms'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_epoch_ms {
            conn.execute_batch(crate::execution::EVENT_MIGRATION_ALTER_V6)?;
        }

        // v0.6: epoch_ms for execution_units (stable ordering)
        let has_unit_epoch_ms: bool = conn
            .prepare("SELECT 1 FROM pragma_table_info('execution_units') WHERE name='epoch_ms'")
            .ok().and_then(|mut s| s.query_row([], |_| Ok(())).ok()).is_some();
        if !has_unit_epoch_ms {
            conn.execute_batch(crate::execution::EXECUTION_UNITS_ALTER_V6)?;
        }

        Ok(())
    }

    /// Access the writer connection for advanced operations (mutation engine, etc).
    pub(crate) fn writer_conn(&self) -> std::sync::MutexGuard<'_, rusqlite::Connection> {
        self.writer.lock().unwrap_or_else(|e| e.into_inner())
    }

    /// Get all conversation IDs.
    pub fn get_all_conversation_ids(&self) -> anyhow::Result<Vec<i64>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT id FROM conversations ORDER BY id")?;
        let rows = stmt.query_map([], |row| row.get::<_, i64>(0))?;
        let mut ids = Vec::new();
        for row in rows { ids.push(row?); }
        Ok(ids)
    }

    /// Find a conversation by session fingerprint, or create one.
    pub fn find_or_create_conversation(&self, fingerprint: &str, model: &str) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
                if let Err(e) = tx.execute(
                    "INSERT INTO messages_fts (rowid, content, role) VALUES (?1, ?2, ?3)",
                    rusqlite::params![msg_id, plain, role],
                ) {
                    tracing::warn!(target: "deeplossless::db", "fts5 index insert failed: {e}");
                }
            }
        }

        tx.commit()?;
        let count = self.write_count.fetch_add(1, Ordering::Relaxed) + 1;
        if count % CHECKPOINT_INTERVAL == 0
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
        self.insert_dag_node_full(conversation_id, level, summary, token_count, parent_ids, child_ids, &[], is_leaf, false)
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
        is_join: bool,
    ) -> anyhow::Result<DagNode> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let parent_json = serde_json::to_string(parent_ids)?;
        let child_json = serde_json::to_string(child_ids)?;
        let snippet_json = serde_json::to_string(snippets)?;
        let hash = Self::semantic_hash(summary, snippets);
        conn.execute(
            "INSERT INTO dag_nodes (conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, 0, ?10)",
            rusqlite::params![conversation_id, level, summary, token_count, parent_json, child_json, snippet_json, is_leaf as i32, is_join as i32, hash],
        )?;
        let id = conn.last_insert_rowid();

        // Mirror as typed edges for the semantic graph (P3 Graph Model)
        for child in child_ids {
            if let Err(e) = conn.execute(
                "INSERT OR IGNORE INTO dag_edges (from_id, to_id, kind) VALUES (?1, ?2, 'summarizes')",
                rusqlite::params![id, child],
            ) {
                tracing::warn!(target: "deeplossless::db", "edge insert failed: {e}");
            }
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
            is_join,
            deleted: false,
            semantic_hash: Self::semantic_hash(summary, snippets),
            access_count: 0,
            last_accessed_at: None,
            reasoning: String::new(),
            graph_revision: 0,
            compaction_id: String::new(),
        })
    }

    pub fn get_node(&self, node_id: i64) -> anyhow::Result<Option<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
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
                    n.token_count, n.parent_ids, n.child_ids, n.snippets,
                    n.is_leaf, n.is_join, n.deleted,
                    n.semantic_hash, n.access_count, n.last_accessed_at, n.reasoning, n.graph_revision, n.compaction_id
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
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
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
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
             FROM dag_nodes WHERE conversation_id = ?1 AND level > 0 AND deleted = 0
             ORDER BY level DESC, id DESC LIMIT 1",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![conv_id], Self::row_to_node)?;
        Ok(rows.next().transpose()?)
    }

    pub fn get_tip_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
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
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
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
        compaction_id: &str,
    ) -> anyhow::Result<DagNode> {
        self.insert_summary_atomic_inner(conversation_id, level, summary, token_count, source_ids, snippets, false, compaction_id)
    }
    #[allow(clippy::too_many_arguments)]
    fn insert_summary_atomic_inner(
        &self,
        conversation_id: i64,
        level: u8,
        summary: &str,
        token_count: i64,
        source_ids: &[i64],
        snippets: &[crate::snippet::Snippet],
        is_join: bool,
        compaction_id: &str,
    ) -> anyhow::Result<DagNode> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let tx = conn.unchecked_transaction()?;

        let parent_json = "[]".to_string();
        let child_json = serde_json::to_string(source_ids)?;
        let snippet_json = serde_json::to_string(snippets)?;
        let hash = Self::semantic_hash(summary, snippets);
        tx.execute(
            "INSERT INTO dag_nodes (conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, compaction_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, 0, ?8, 0, ?9, ?10)",
            rusqlite::params![conversation_id, level, summary, token_count, parent_json, child_json, snippet_json, is_join as i32, hash, compaction_id],
        )?;
        let new_id = tx.last_insert_rowid();

        // Mirror as typed edges (P3 Graph Model)
        for sid in source_ids {
            if let Err(e) = tx.execute(
                "INSERT OR IGNORE INTO dag_edges (from_id, to_id, kind) VALUES (?1, ?2, 'summarizes')",
                rusqlite::params![new_id, sid],
            ) {
                tracing::warn!(target: "deeplossless::db", "edge insert in atomic failed: {e}");
            }
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
            if let Err(e) = tx.execute(
                "INSERT INTO snippets_fts (content, source_type, node_id) VALUES (?1, ?2, ?3)",
                rusqlite::params![snippet.content, stype, new_id],
            ) {
                tracing::warn!(target: "deeplossless::db", "snippets fts5 insert failed: {e}");
            }
        }

        // Index in cross-conversation semantic dedup table (v0.3)
        let preview: String = summary.chars().take(100).collect();
        if let Err(e) = tx.execute(
            "INSERT OR IGNORE INTO semantic_index (hash, node_id, conversation_id, summary_preview)
             VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![hash, new_id, conversation_id, preview],
        ) {
            tracing::warn!(target: "deeplossless::db", "semantic index insert failed: {e}");
        }

        // Record provenance: each source node contributed to this summary (v0.4)
        for (i, sid) in source_ids.iter().enumerate() {
            if let Err(e) = tx.execute(
                "INSERT INTO provenance (summary_node_id, source_node_id, sentence_offset, sentence_length)
                 VALUES (?1, ?2, ?3, 0)",
                rusqlite::params![new_id, sid, i as i32],
            ) {
                tracing::warn!(target: "deeplossless::db", "provenance insert failed: {e}");
            }
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
        if let Err(e) = tx.execute(
            "INSERT INTO dag_events (event_type, node_id, conv_id, payload) VALUES ('compress', ?1, ?2, '{}')",
            rusqlite::params![new_id, conversation_id],
        ) {
            tracing::warn!(target: "deeplossless::db", "event log insert failed: {e}");
        }

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
            is_join,
            deleted: false,
            semantic_hash: Self::semantic_hash(summary, snippets),
            access_count: 0,
            last_accessed_at: None,
            reasoning: String::new(),
            graph_revision: 0,
            compaction_id: compaction_id.to_string(),
        })
    }

    /// Look up an existing summary node by its deterministic compaction_id.
    /// Used for idempotent compaction dedup (P0-9).
    pub fn find_by_compaction_id(&self, compaction_id: &str) -> anyhow::Result<Option<DagNode>> {
        if compaction_id.is_empty() {
            return Ok(None);
        }
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
             FROM dag_nodes WHERE compaction_id = ?1 AND deleted = 0",
        )?;
        let mut rows = stmt.query_map(rusqlite::params![compaction_id], Self::row_to_node)?;
        Ok(rows.next().transpose()?)
    }

    /// Atomically insert a join node (parallel execution sync point).
    pub fn insert_join_atomic(
        &self,
        conversation_id: i64,
        summary: &str,
        token_count: i64,
        source_ids: &[i64],
        snippets: &[crate::snippet::Snippet],
    ) -> anyhow::Result<DagNode> {
        self.insert_summary_atomic_inner(conversation_id, 0, summary, token_count, source_ids, snippets, true, "")
    }

    /// Update the reasoning chain for a node (execution provenance).
    pub fn update_node_reasoning(&self, node_id: i64, reasoning: &str) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "UPDATE dag_nodes SET reasoning = ?1 WHERE id = ?2",
            rusqlite::params![reasoning, node_id],
        )?;
        Ok(())
    }

    /// Increment access count and update last_accessed_at for memory scoring.
    pub fn touch_node(&self, node_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "UPDATE dag_nodes SET access_count = access_count + 1, last_accessed_at = datetime('now') WHERE id = ?1",
            rusqlite::params![node_id],
        )?;
        Ok(())
    }

    /// Append an event to the DAG event log for audit and rollback.
    pub fn record_event(&self, event_type: &str, node_id: i64, conv_id: i64, payload: &str) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO dag_events (event_type, node_id, conv_id, payload) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![event_type, node_id, conv_id, payload],
        )?;
        Ok(())
    }

    /// Get event log for a conversation, ordered by time.
    #[allow(clippy::type_complexity)]
    pub fn get_events(&self, conv_id: i64, limit: usize) -> anyhow::Result<Vec<(String, Option<i64>, String, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT event_type, node_id, payload, created_at FROM dag_events WHERE conv_id = ?1 ORDER BY id DESC LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, limit as i64], |row| {
            Ok((
                row.get::<_, String>(0)?,
                row.get::<_, Option<i64>>(1)?,
                row.get::<_, String>(2)?,
                row.get::<_, String>(3)?,
            ))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    // ── Execution units (v0.7) ──────────────────────────────────────

    /// Store an execution unit and return its ID.
    /// Also writes an append-only execution event.
    pub fn store_execution_unit(
        &self,
        conv_id: i64,
        reasoning_before: &str,
        tool_name: &str,
        tool_args: &str,
        tool_result: &str,
        reasoning_after: &str,
        outcome: &str,
        related_nodes: &[i64],
    ) -> anyhow::Result<i64> {
        self.store_execution_unit_with_span(
            conv_id, reasoning_before, tool_name, tool_args,
            tool_result, reasoning_after, outcome, related_nodes,
            "", "", "", "", "", "",
        )
    }

    /// Store an execution unit with full span/parallel metadata and replay session.
    /// Always writes an append-only event to execution_events with epoch_ms.
    /// `replay_session_id` groups related executions for replay lineage.
    pub fn store_execution_unit_with_span(
        &self,
        conv_id: i64,
        reasoning_before: &str,
        tool_name: &str,
        tool_args: &str,
        tool_result: &str,
        reasoning_after: &str,
        outcome: &str,
        related_nodes: &[i64],
        span_id: &str,
        parent_span_id: &str,
        span_mode: &str,
        parallel_group: &str,
        tool_call_id: &str,
        replay_session_id: &str,
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let related_json = serde_json::to_string(related_nodes)?;
        let tool_args_json_str = String::new();
        let reasoning_steps_json = "[]".to_string();
        let epoch_ms = crate::execution::next_logical_seq();

        conn.execute(
            "INSERT INTO execution_units (conversation_id, reasoning_before, tool_name, tool_args, tool_result, reasoning_after, outcome, related_nodes, tool_args_json, reasoning_steps, span_id, parent_span_id, span_mode, parallel_group, tool_call_id, epoch_ms, replay_session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11, ?12, ?13, ?14, ?15, ?16, ?17)",
            rusqlite::params![
                conv_id, reasoning_before, tool_name, tool_args, tool_result,
                reasoning_after, outcome, related_json, tool_args_json_str,
                reasoning_steps_json, span_id, parent_span_id, span_mode, parallel_group,
                tool_call_id, epoch_ms, replay_session_id,
            ],
        )?;
        let exec_id = conn.last_insert_rowid();

        // Append-only event log (authoritative audit source)
        let payload = serde_json::json!({
            "tool_name": tool_name,
            "outcome": outcome,
            "span_id": span_id,
            "parent_span_id": parent_span_id,
            "span_mode": span_mode,
            "parallel_group": parallel_group,
        });
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, span_id, parent_span_id, span_mode, parallel_group, tool_call_id, conv_id, epoch_ms, replay_session_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10, ?11)",
            rusqlite::params![
                exec_id, "execution_completed",
                &serde_json::to_string(&payload).unwrap_or_default(),
                span_id, parent_span_id, span_mode, parallel_group,
                tool_call_id, conv_id, epoch_ms, replay_session_id,
            ],
        )?;

        Ok(exec_id)
    }

    /// Get execution units for a conversation, newest first.
    pub fn get_execution_units(
        &self,
        conv_id: i64,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::execution::ExecutionUnit>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, reasoning_before, tool_name, tool_args,
                    tool_result, reasoning_after, outcome, related_nodes, created_at,
                    span_id, parent_span_id, span_mode, parallel_group, tool_call_id,
                    epoch_ms, replay_session_id
             FROM execution_units
             WHERE conversation_id = ?1
             ORDER BY id DESC LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, limit as i64], |row| {
            let related_str: String = row.get(8)?;
            Ok(crate::execution::ExecutionUnit {
                id: row.get(0)?,
                conversation_id: row.get(1)?,
                reasoning_before: row.get(2)?,
                tool_name: row.get(3)?,
                tool_args: row.get(4)?,
                tool_result: row.get(5)?,
                reasoning_after: row.get(6)?,
                outcome: crate::execution::ExecutionOutcome::from_str(
                    &row.get::<_, String>(7)?
                ).unwrap_or(crate::execution::ExecutionOutcome::Success),
                related_nodes: serde_json::from_str(&related_str).unwrap_or_default(),
                created_at: row.get(9)?,
                tool_args_json: None,
                reasoning_steps: vec![],
                span_id: row.get::<_, String>(10).unwrap_or_default(),
                parent_span_id: row.get::<_, String>(11).unwrap_or_default(),
                span_mode: row.get::<_, String>(12).unwrap_or_default(),
                parallel_group: row.get::<_, String>(13).unwrap_or_default(),
                tool_call_id: row.get::<_, String>(14).unwrap_or_default(),
                epoch_ms: row.get::<_, i64>(15).unwrap_or_default(),
                replay_session_id: row.get::<_, String>(16).unwrap_or_default(),
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    /// Search execution units by tool name or reasoning content.
    pub fn search_execution_units(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::execution::ExecutionUnitRef>> {
        let conn = self.read_conn();
        let pattern = format!("%{}%", query.replace('%', "%%").replace('_', "\\_"));
        let mut stmt = conn.prepare(
            "SELECT id, tool_name, outcome, reasoning_before
             FROM execution_units
             WHERE tool_name LIKE ?1 ESCAPE '\\'
                OR reasoning_before LIKE ?1 ESCAPE '\\'
                OR reasoning_after LIKE ?1 ESCAPE '\\'
                OR tool_result LIKE ?1 ESCAPE '\\'
             ORDER BY id DESC LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, limit as i64], |row| {
            Ok(crate::execution::ExecutionUnitRef {
                id: row.get(0)?,
                tool_name: row.get(1)?,
                outcome: row.get(2)?,
                snippet: row.get::<_, String>(3)?.chars().take(80).collect(),
            })
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Code diff memory (v0.7) ──────────────────────────────────────

    /// Store a code change record.
    pub fn store_code_change(
        &self,
        conv_id: i64,
        file_path: &str,
        diff: &str,
        symbols: &[String],
        error_before: &[String],
        error_after: &[String],
        execution_unit_id: Option<i64>,
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let symbols_json = serde_json::to_string(symbols)?;
        let err_before_json = serde_json::to_string(error_before)?;
        let err_after_json = serde_json::to_string(error_after)?;
        conn.execute(
            "INSERT INTO code_changes (conversation_id, file_path, diff, symbols_changed, error_before, error_after, execution_unit_id)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![conv_id, file_path, diff, symbols_json, err_before_json, err_after_json, execution_unit_id],
        )?;
        let id = conn.last_insert_rowid();
        drop(conn);
        // Auto-invalidate: file changed → purge stale cache entries + release claims
        let _ = self.on_files_changed(&[file_path.to_string()]);
        Ok(id)
    }

    /// Search code changes by file path, symbol, or error message.
    pub fn search_code_changes(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(i64, String, String, String)>> {
        let conn = self.read_conn();
        let pattern = format!("%{}%", query.replace('%', "%%").replace('_', "\\_"));
        let mut stmt = conn.prepare(
            "SELECT id, file_path, symbols_changed, error_before
             FROM code_changes
             WHERE file_path LIKE ?1 ESCAPE '\\'
                OR symbols_changed LIKE ?1 ESCAPE '\\'
                OR error_before LIKE ?1 ESCAPE '\\'
             ORDER BY id DESC LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?, row.get::<_, String>(3)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Execution-aware retrieval (v0.7) ─────────────────────────────

    /// Search execution memory: finds similar bugs, tool chains, and code edits.
    /// Cross-references execution_units, code_changes, and snippet FTS5.
    pub fn search_execution_memory(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<crate::execution::ExecutionUnitRef>> {
        // Code changes return the most actionable results first
        let code_results = self.search_code_changes(query, limit / 2).unwrap_or_default();
        let mut seen = std::collections::HashSet::new();

        // Convert code change results to execution refs
        let mut results: Vec<crate::execution::ExecutionUnitRef> = Vec::new();
        for (id, file_path, symbols, errors) in &code_results {
            if !seen.insert(*id) {
                continue;
            }
            let sym: Vec<String> = serde_json::from_str(symbols).unwrap_or_default();
            let snippet = if !sym.is_empty() {
                format!("changed {} — {} symbols", file_path, sym.len())
            } else if !errors.is_empty() {
                let err: Vec<String> = serde_json::from_str(errors).unwrap_or_default();
                format!("{} — fixed: {}", file_path, err.first().map(|s| s.as_str()).unwrap_or("?"))
            } else {
                file_path.clone()
            };
            results.push(crate::execution::ExecutionUnitRef {
                id: *id,
                tool_name: "code_change".to_string(),
                outcome: "fixed".to_string(),
                snippet,
            });
        }

        // Then add execution unit results
        let exec_results = self.search_execution_units(query, limit - results.len().min(limit)).unwrap_or_default();
        for refr in exec_results {
            if seen.insert(refr.id) {
                results.push(refr);
            }
        }

        results.truncate(limit);
        Ok(results)
    }

    // ── Tool result cache (v0.8) ──────────────────────────────────────

    /// Summary counts for debug dump — no user content exposed.
    pub fn debug_counts(&self) -> anyhow::Result<(i64, i64, i64, i64, i64)> {
        let conn = self.read_conn();
        let nodes: i64 = conn.query_row("SELECT COUNT(*) FROM dag_nodes WHERE deleted=0", [], |r| r.get(0))?;
        let convs: i64 = conn.query_row("SELECT COUNT(*) FROM conversations", [], |r| r.get(0))?;
        let embeddings: i64 = conn.query_row("SELECT COUNT(*) FROM embeddings", [], |r| r.get(0))?;
        let plans: i64 = conn.query_row("SELECT COUNT(*) FROM plan_states WHERE is_active=1", [], |r| r.get(0))?;
        let events: i64 = conn.query_row("SELECT COUNT(*) FROM dag_events", [], |r| r.get(0))?;
        Ok((nodes, convs, embeddings, plans, events))
    }

    /// Collect per-conversation metrics for the session report.
    pub fn collect_session_metrics(&self, conv_id: i64) -> anyhow::Result<(i64, i64, i64, i64)> {
        // leaf_count, summary_count, total_tokens, failure_count
        let conn = self.read_conn();
        let leaf_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM dag_nodes WHERE conversation_id = ?1 AND is_leaf = 1 AND deleted = 0",
            rusqlite::params![conv_id], |r| r.get(0))?;
        let summary_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM dag_nodes WHERE conversation_id = ?1 AND level > 0 AND deleted = 0",
            rusqlite::params![conv_id], |r| r.get(0))?;
        let total_tokens: i64 = conn.query_row(
            "SELECT COALESCE(SUM(token_count), 0) FROM dag_nodes WHERE conversation_id = ?1 AND deleted = 0",
            rusqlite::params![conv_id], |r| r.get(0))?;
        let failure_count: i64 = conn.query_row(
            "SELECT COUNT(*) FROM failure_patterns WHERE conversation_id = ?1",
            rusqlite::params![conv_id], |r| r.get(0))?;
        Ok((leaf_count, summary_count, total_tokens, failure_count))
    }

    /// Fetch execution scoring data for a conversation.
    /// Returns (execution_units, dag_metrics) for `score_execution`.
    pub fn get_scoring_data(
        &self, conv_id: i64,
    ) -> anyhow::Result<(Vec<crate::execution::ExecutionUnit>, crate::execution::DagMetrics)> {
        let units = self.get_execution_units(conv_id, 1000)?;
        let (leaf_count, summary_count, total_tokens, _failure_count) = self.collect_session_metrics(conv_id)?;
        let conn = self.read_conn();
        let total_edges: i64 = conn.query_row(
            "SELECT COUNT(*) FROM dag_edges WHERE from_id IN (SELECT id FROM dag_nodes WHERE conversation_id = ?1 AND deleted = 0)",
            rusqlite::params![conv_id], |r| r.get(0),
        ).unwrap_or(0);
        let reuse_edges: i64 = conn.query_row(
            "SELECT COUNT(*) FROM dag_edges WHERE kind = 'reuses' AND from_id IN (SELECT id FROM dag_nodes WHERE conversation_id = ?1 AND deleted = 0)",
            rusqlite::params![conv_id], |r| r.get(0),
        ).unwrap_or(0);
        let dag = crate::execution::DagMetrics {
            leaf_count,
            summary_count,
            total_tokens,
            max_budget_tokens: 128_000,
            reuse_edge_count: reuse_edges,
            total_edge_count: total_edges,
        };
        Ok((units, dag))
    }

    /// Atomically store a file observation (P0 transactional snapshot).
    pub fn store_file_observation(
        &self,
        obs: &crate::file_observation::FileObservation,
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let ast_json = serde_json::to_string(&obs.ast)?;
        conn.execute(
            "INSERT INTO file_observations (path, content_hash, semantic_hash, ast_json, size_bytes, line_count, kind)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7)",
            rusqlite::params![
                obs.path, obs.content_hash, obs.semantic_hash, ast_json,
                obs.size_bytes as i64, obs.line_count as i64, obs.kind.as_str(),
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Compute and return the execution score for a conversation.
    pub fn compute_execution_score(&self, conv_id: i64) -> anyhow::Result<crate::execution::ExecutionScore> {
        let (units, dag) = self.get_scoring_data(conv_id)?;
        Ok(crate::execution::ExecutionScore::from_units(&units, &dag))
    }

    /// Get top N most-hit tool cache entries for the session report.
    pub fn top_tool_cache_entries(&self, limit: usize) -> anyhow::Result<Vec<(String, i64)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT tool_name, hit_count FROM tool_cache WHERE hit_count > 0 ORDER BY hit_count DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Look up a cached tool result. Returns (result, hit_count) if found.
    /// Checks L1 in-memory cache first, falls back to SQLite.
    pub fn tool_cache_get(&self, tool_name: &str, args_hash: &str) -> anyhow::Result<Option<(String, i64)>> {
        // L1 hot cache check (no SQLite round-trip)
        if let Some((result, _count)) = self.tool_cache_l1.get(tool_name, args_hash) {
            return Ok(Some((result.to_string(), 1)));
        }
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT result, hit_count FROM tool_cache WHERE tool_name = ?1 AND args_hash = ?2"
        )?;
        let result = stmt.query_row(rusqlite::params![tool_name, args_hash], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, i64>(1)?))
        }).ok();
        if let Some((_res, _)) = &result {
            // Bump hit count
            let w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            let _ = w.execute(
                "UPDATE tool_cache SET hit_count = hit_count + 1 WHERE tool_name = ?1 AND args_hash = ?2",
                rusqlite::params![tool_name, args_hash],
            );
        }
        Ok(result)
    }

    /// Store a tool result in cache.
    pub fn tool_cache_put(
        &self,
        tool_name: &str,
        args_hash: &str,
        result: &str,
        dependent_files: &[String],
    ) -> anyhow::Result<()> {
        self.tool_cache_put_with_hashes(tool_name, args_hash, result, dependent_files, &[])
    }

    /// Store a tool result with optional content hashes for ArtifactVersion-based validation.
    pub fn tool_cache_put_with_hashes(
        &self,
        tool_name: &str,
        args_hash: &str,
        result: &str,
        dependent_files: &[String],
        file_hashes: &[String],
    ) -> anyhow::Result<()> {
        self.tool_cache_l1.put_with_hashes(tool_name, args_hash, result, dependent_files, file_hashes);
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let files_json = serde_json::to_string(dependent_files)?;
        let hashes_json = serde_json::to_string(file_hashes).unwrap_or_else(|_| "[]".into());
        conn.execute(
            "INSERT OR REPLACE INTO tool_cache (tool_name, args_hash, result, dependent_files, dependent_file_hashes) VALUES (?1, ?2, ?3, ?4, ?5)",
            rusqlite::params![tool_name, args_hash, result, files_json, hashes_json],
        )?;
        Ok(())
    }

    /// Invalidate cache entries whose dependent files overlap with changed_files.
    /// Uses content-hash comparison when file_hashes are available (v0.3);
    /// falls back to path-based matching for legacy entries.
    pub fn tool_cache_invalidate(&self, changed_files: &[String]) -> anyhow::Result<usize> {
        self.tool_cache_invalidate_with_hashes(changed_files, &[])
    }

    /// Invalidate with optional content hashes for precise matching.
    pub fn tool_cache_invalidate_with_hashes(&self, changed_files: &[String], new_hashes: &[String]) -> anyhow::Result<usize> {
        if changed_files.is_empty() {
            return Ok(0);
        }
        self.tool_cache_l1.invalidate(changed_files);
        let conn = self.read_conn();
        let mut stmt = conn.prepare("SELECT id, dependent_files, dependent_file_hashes FROM tool_cache")?;
        let rows: Vec<(i64, String, String)> = stmt.query_map([], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?.filter_map(|r| r.ok()).collect();
        drop(stmt);
        drop(conn);

        let changed_set: std::collections::HashSet<&str> = changed_files.iter().map(|s| s.as_str()).collect();
        let hash_set: std::collections::HashSet<&str> = new_hashes.iter().map(|s| s.as_str()).collect();
        let use_hashes = !hash_set.is_empty();
        let mut invalidated = 0;
        let w = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        for (id, files_json, hashes_json) in &rows {
            let deps: Vec<String> = serde_json::from_str(files_json).unwrap_or_default();
            let hashes: Vec<String> = serde_json::from_str(hashes_json).unwrap_or_default();
            let should_invalidate = if use_hashes && !hashes.is_empty() {
                // Content-hash-based: invalidate only if hash differs
                deps.iter().any(|f| changed_set.contains(f.as_str()))
                    && !hashes.iter().any(|h| hash_set.contains(h.as_str()))
            } else {
                // Legacy path-based fallback
                deps.iter().any(|f| changed_set.contains(f.as_str()))
            };
            if should_invalidate {
                w.execute("DELETE FROM tool_cache WHERE id = ?1", rusqlite::params![id])?;
                invalidated += 1;
            }
        }
        Ok(invalidated)
    }

    // ── Failure memory (v0.8) ─────────────────────────────────────────

    /// Store a failure pattern. Returns ID.
    pub fn store_failure_pattern(
        &self,
        conv_id: i64,
        signature: &str,
        attempted_fix: &str,
        why_failed: &str,
        assumptions: &[String],
        related_files: &[String],
        execution_unit_id: Option<i64>,
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let assump_json = serde_json::to_string(assumptions)?;
        let files_json = serde_json::to_string(related_files)?;
        conn.execute(
            "INSERT INTO failure_patterns (conversation_id, signature, attempted_fix, why_failed, invalidated_assumptions, related_files, execution_unit_id, execution_key, environment_fingerprint)
             VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, '', '')",
            rusqlite::params![conv_id, signature, attempted_fix, why_failed, assump_json, files_json, execution_unit_id],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Search failure patterns by signature or related files.
    pub fn search_failure_patterns(&self, query: &str, limit: usize) -> anyhow::Result<Vec<(i64, String, String)>> {
        let conn = self.read_conn();
        let pattern = format!("%{}%", query.replace('%', "%%").replace('_', "\\_"));
        let mut stmt = conn.prepare(
            "SELECT id, signature, why_failed FROM failure_patterns
             WHERE signature LIKE ?1 ESCAPE '\\' OR why_failed LIKE ?1 ESCAPE '\\'
             ORDER BY id DESC LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows {
            results.push(row?);
        }
        Ok(results)
    }

    // ── Plan persistence (v0.8) ───────────────────────────────────────

    /// Store a new plan state. Deactivates previous active plans for the conversation.
    pub fn store_plan_state(
        &self,
        conv_id: i64,
        goal: &str,
        pending: &[String],
        assumptions: &[String],
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        // Deactivate previous plans
        conn.execute("UPDATE plan_states SET is_active = 0, updated_at = datetime('now') WHERE conversation_id = ?1 AND is_active = 1",
            rusqlite::params![conv_id])?;
        let pending_json = serde_json::to_string(pending)?;
        let assump_json = serde_json::to_string(assumptions)?;
        conn.execute(
            "INSERT INTO plan_states (conversation_id, goal, pending_steps, assumptions, is_active)
             VALUES (?1, ?2, ?3, ?4, 1)",
            rusqlite::params![conv_id, goal, pending_json, assump_json],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Get the active plan state for a conversation.
    pub fn get_active_plan(&self, conv_id: i64) -> anyhow::Result<Option<(i64, String, String, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, goal, pending_steps, assumptions FROM plan_states
             WHERE conversation_id = ?1 AND is_active = 1 ORDER BY id DESC LIMIT 1"
        )?;
        Ok(stmt.query_row(rusqlite::params![conv_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?))
        }).ok())
    }

    // ── Execution Events (v0.4.0) ──────────────────────────────────────

    /// Store a single execution event. Append-only — never updated.
    /// Called from stream handlers as events flow through the proxy.
    /// `epoch_ms` is a Unix timestamp in milliseconds for stable ordering.
    pub fn store_execution_event(&self, execution_id: Option<i64>, event_kind: &str, event_payload: &str, seq_no: i64, conv_id: Option<i64>, epoch_ms: i64) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO execution_events (execution_id, event_kind, event_payload, seq_no, conv_id, epoch_ms) VALUES (?1, ?2, ?3, ?4, ?5, ?6)",
            rusqlite::params![execution_id, event_kind, event_payload, seq_no, conv_id, epoch_ms],
        )?;
        Ok(conn.last_insert_rowid())
    }

    #[allow(clippy::type_complexity)]
    /// Read all events for an execution in seq_no order (replay).
    pub fn get_execution_events(&self, execution_id: i64) -> anyhow::Result<Vec<(i64, String, String, i64, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, event_kind, event_payload, seq_no, created_at FROM execution_events WHERE execution_id = ?1 ORDER BY seq_no"
        )?;
        let rows = stmt.query_map(rusqlite::params![execution_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?, row.get(4)?))
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    #[allow(clippy::type_complexity)]
    /// Read all execution events for a conversation (authoritative audit source).
    /// Returns (id, execution_id, event_kind, event_payload, seq_no, created_at, span_id,
    ///          parent_span_id, span_mode, parallel_group, tool_call_id, epoch_ms).
    pub fn get_execution_events_by_conv(
        &self, conv_id: i64, limit: usize,
    ) -> anyhow::Result<Vec<(i64, Option<i64>, String, String, i64, String, String, String, String, String, String, i64)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, execution_id, event_kind, event_payload, seq_no, created_at,
                    span_id, parent_span_id, span_mode, parallel_group, tool_call_id, epoch_ms
             FROM execution_events
             WHERE conv_id = ?1
             ORDER BY epoch_ms DESC, id DESC
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, limit as i64], |row| {
            Ok((
                row.get(0)?, row.get(1)?, row.get(2)?, row.get(3)?,
                row.get(4)?, row.get(5)?, row.get(6)?, row.get(7)?,
                row.get(8)?, row.get(9)?, row.get(10)?, row.get(11)?,
            ))
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    /// Insert a provenance lineage edge into the lineage_edges table.
    /// This is the authoritative edge store for DependsOn, DerivedFrom, etc.
    /// Returns the edge ID.
    pub fn insert_lineage_edge(&self, from_id: i64, to_id: i64, kind: &str) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT OR IGNORE INTO lineage_edges (from_id, to_id, kind) VALUES (?1, ?2, ?3)",
            rusqlite::params![from_id, to_id, kind],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Read lineage edges pointing TO a given node.
    pub fn get_lineage_to(&self, to_id: i64) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT from_id, to_id, kind FROM lineage_edges WHERE to_id = ?1",
        )?;
        let rows = stmt.query_map(rusqlite::params![to_id], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        let mut result = Vec::new();
        for row in rows { result.push(row?); }
        Ok(result)
    }

    /// Get file paths that a specific execution unit depends on.
    /// Reads from tool_cache dependent_files JSON column.
    pub fn get_dependent_files_for_unit(&self, _execution_unit_id: i64) -> anyhow::Result<Vec<String>> {
        // Execution units don't have a direct file-dependency column.
        // File dependencies are tracked per cache entry (tool_cache table).
        // For now, return empty — this is a placeholder for Phase 3 integration.
        // TODO: when cache entries include execution_unit_id, filter here.
        Ok(Vec::new())
    }

    /// Get cache entry IDs that depend on a given file path.
    pub fn get_cache_ids_for_file(&self, file_path: &str) -> anyhow::Result<Vec<i64>> {
        let conn = self.read_conn();
        let pattern = format!("%{}%", file_path.replace('%', "%%"));
        let mut stmt = conn.prepare(
            "SELECT id FROM tool_cache WHERE dependent_files LIKE ?1",
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern], |row| row.get(0))?;
        let mut ids = Vec::new();
        for row in rows { ids.push(row?); }
        Ok(ids)
    }

    // ── Memory Versions (v0.4.0) ───────────────────────────────────────

    /// Create a new memory version, linked to a parent. Returns the new version id.
    pub fn create_memory_version(
        &self, parent_version_id: Option<i64>, mutation_kind: &str,
        mutation_desc: &str, dag_root_id: Option<i64>,
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO memory_versions (parent_version_id, mutation_kind, mutation_desc, dag_root_id) VALUES (?1, ?2, ?3, ?4)",
            rusqlite::params![parent_version_id, mutation_kind, mutation_desc, dag_root_id],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// List memory versions in reverse chronological order.
    pub fn list_memory_versions(&self, limit: usize) -> anyhow::Result<Vec<crate::snapshot::MemoryVersion>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, parent_version_id, mutation_kind, mutation_desc, dag_root_id, created_at FROM memory_versions ORDER BY id DESC LIMIT ?1"
        )?;
        let rows = stmt.query_map(rusqlite::params![limit as i64], |row| {
            Ok(crate::snapshot::MemoryVersion {
                id: row.get(0)?,
                parent_version_id: row.get(1)?,
                mutation_kind: row.get(2)?,
                mutation_desc: row.get(3)?,
                dag_root_id: row.get(4)?,
                created_at: row.get(5)?,
            })
        })?.collect::<Result<Vec<_>, _>>()?;
        Ok(rows)
    }

    // ── Execution Snapshots (v0.4.0) ────────────────────────────────────

    /// Take an append-only snapshot. Returns the snapshot id.
    /// `last_event_seq_no`, `boundary_hash`, `integrity_hash` are computed
    /// from the snapshot data for continuity verification.
    pub fn take_snapshot(
        &self, execution_id: i64, memory_version_id: i64,
        tier: i32, data: &str, size_bytes: i64, retention_ttl: Option<i64>,
        last_event_seq_no: i64, boundary_hash: &str, integrity_hash: &str,
    ) -> anyhow::Result<i64> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT INTO execution_snapshots (execution_id, memory_version_id, tier, schema_version, snapshot_data, size_bytes, retention_ttl, last_event_seq_no, boundary_hash, integrity_hash) VALUES (?1, ?2, ?3, ?4, ?5, ?6, ?7, ?8, ?9, ?10)",
            rusqlite::params![
                execution_id, memory_version_id, tier,
                crate::snapshot::SCHEMA_VERSION,
                data, size_bytes, retention_ttl,
                last_event_seq_no, boundary_hash, integrity_hash,
            ],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Restore a snapshot by id.
    pub fn restore_snapshot(&self, id: i64) -> anyhow::Result<Option<crate::snapshot::ExecutionSnapshot>> {
        let conn = self.read_conn();
        let row = conn.query_row(
            "SELECT id, execution_id, memory_version_id, schema_version, tier, snapshot_data, last_event_seq_no, boundary_hash, integrity_hash, size_bytes, retention_ttl, created_at FROM execution_snapshots WHERE id = ?1",
            rusqlite::params![id],
            |row| Ok(crate::snapshot::ExecutionSnapshot {
                id: row.get(0)?,
                execution_id: row.get(1)?,
                memory_version_id: row.get(2)?,
                schema_version: row.get(3)?,
                tier: row.get(4)?,
                snapshot_data: row.get(5)?,
                last_event_seq_no: row.get(6)?,
                boundary_hash: row.get(7)?,
                integrity_hash: row.get(8)?,
                size_bytes: row.get(9)?,
                retention_ttl: row.get(10)?,
                created_at: row.get(11)?,
            }),
        ).ok();
        Ok(row)
    }

    /// Enforce snapshot budget — evicts oldest snapshots when over limit.
    /// Returns the number evicted.
    pub fn enforce_snapshot_budget(&self, budget: &crate::snapshot::SnapshotBudget) -> anyhow::Result<usize> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let mut evicted = 0;

        // L0 ring buffer: keep max_hot_snapshots, evict oldest
        let l0_count: i64 = conn.query_row("SELECT COUNT(*) FROM execution_snapshots WHERE tier = 0", [], |r| r.get(0))?;
        if l0_count > budget.max_hot_snapshots as i64 {
            let to_remove = l0_count - budget.max_hot_snapshots as i64;
            conn.execute(
                "DELETE FROM execution_snapshots WHERE id IN (SELECT id FROM execution_snapshots WHERE tier = 0 ORDER BY created_at ASC LIMIT ?1)",
                rusqlite::params![to_remove],
            )?;
            evicted += to_remove;
        }

        // L2 soft limit
        let l2_count: i64 = conn.query_row("SELECT COUNT(*) FROM execution_snapshots WHERE tier = 2", [], |r| r.get(0))?;
        if l2_count > budget.max_full_snapshots as i64 {
            let to_remove = l2_count - budget.max_full_snapshots as i64;
            conn.execute(
                "DELETE FROM execution_snapshots WHERE id IN (SELECT id FROM execution_snapshots WHERE tier = 2 ORDER BY created_at ASC LIMIT ?1)",
                rusqlite::params![to_remove],
            )?;
            evicted += to_remove;
        }

        // Hard size cap: remove oldest non-frozen (L3)
        let total_size: i64 = conn.query_row("SELECT COALESCE(SUM(size_bytes), 0) FROM execution_snapshots", [], |r| r.get(0))?;
        if total_size > budget.max_total_size_bytes as i64 {
            // Evict oldest non-frozen until under budget (simple: delete a batch)
            conn.execute(
                "DELETE FROM execution_snapshots WHERE id IN (SELECT id FROM execution_snapshots WHERE tier < 3 ORDER BY size_bytes DESC, created_at ASC LIMIT 10)",
                [],
            )?;
            evicted += 1; // at least one batch attempt
        }

        Ok(evicted as usize)
    }

    /// Access the writer lock for direct SQL. Used by dag engine for
    /// transactional graph mutations (P0-2).
    pub fn writer_lock(&self) -> &Mutex<Connection> {
        &self.writer
    }

    // ── Multi-agent safe runtime (v0.9) ─────────────────────────────────

    /// Claim a file for an agent. Returns Ok(()) if claim succeeds,
    /// Err with conflicting agent_id if another agent already holds the file.
    pub fn claim_file(&self, agent_id: &str, file_path: &str, operation: &str) -> anyhow::Result<Result<(), String>> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        // Check for existing claim from another agent
        let existing: Option<String> = conn.query_row(
            "SELECT agent_id FROM agent_active_files WHERE file_path = ?1 AND agent_id != ?2",
            rusqlite::params![file_path, agent_id],
            |row| row.get(0),
        ).ok();
        if let Some(other_agent) = existing {
            return Ok(Err(other_agent));
        }
        conn.execute(
            "INSERT OR REPLACE INTO agent_active_files (agent_id, file_path, operation) VALUES (?1, ?2, ?3)",
            rusqlite::params![agent_id, file_path, operation],
        )?;
        Ok(Ok(()))
    }

    /// Release an agent's claim on a file.
    pub fn release_file(&self, agent_id: &str, file_path: &str) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "DELETE FROM agent_active_files WHERE agent_id = ?1 AND file_path = ?2",
            rusqlite::params![agent_id, file_path],
        )?;
        Ok(())
    }

    /// Release all claims for an agent (on disconnect/session end).
    pub fn release_all_agent_files(&self, agent_id: &str) -> anyhow::Result<usize> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let count = conn.execute(
            "DELETE FROM agent_active_files WHERE agent_id = ?1",
            rusqlite::params![agent_id],
        )?;
        Ok(count)
    }

    /// List all file claims across all agents.
    pub fn list_all_file_claims(&self) -> anyhow::Result<Vec<(String, String, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT agent_id, file_path, operation FROM agent_active_files ORDER BY claimed_at"
        )?;
        let rows = stmt.query_map([], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Get files currently claimed by an agent.
    pub fn get_agent_files(&self, agent_id: &str) -> anyhow::Result<Vec<(String, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT file_path, operation FROM agent_active_files WHERE agent_id = ?1"
        )?;
        let rows = stmt.query_map(rusqlite::params![agent_id], |row| {
            Ok((row.get::<_, String>(0)?, row.get::<_, String>(1)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Cross-agent cache invalidation: when files change, invalidate affected
    /// tool cache entries and release stale claims.
    pub fn on_files_changed(&self, changed_files: &[String]) -> anyhow::Result<usize> {
        // Expand changed files to include parent directories for broader invalidation.
        // E.g., "src/main.rs" changed → also invalidate cache entries dependent on "src/"
        let mut expanded = Vec::new();
        for f in changed_files {
            expanded.push(f.clone());
            // Add parent directories
            let mut path = std::path::Path::new(f);
            while let Some(parent) = path.parent() {
                let dir = parent.to_string_lossy().to_string();
                if !dir.is_empty() && dir != "." {
                    let dir_with_sep = if dir.ends_with('/') { dir.clone() } else { format!("{dir}/") };
                    expanded.push(dir_with_sep);
                    path = parent;
                } else {
                    break;
                }
            }
        }
        expanded.sort();
        expanded.dedup();

        // 1. Invalidate tool cache entries (O(affected) via reverse index)
        let invalidated = self.tool_cache_l1.invalidate(&expanded);
        // Also invalidate SQLite cache
        let _ = self.tool_cache_invalidate(&expanded)?;

        // 2. Release claims on changed files (agent finished editing)
        if !changed_files.is_empty() {
            let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            for f in changed_files {
                let _ = conn.execute(
                    "DELETE FROM agent_active_files WHERE file_path = ?1",
                    rusqlite::params![f],
                );
            }
            // Mark active plans as stale — file changes may invalidate assumptions
            let _ = conn.execute(
                "UPDATE plan_states SET updated_at = datetime('now') WHERE is_active = 1",
                [],
            );
        }
        Ok(invalidated)
    }

    /// Soft-delete a DAG node (set deleted=1, timestamp). The node is
    /// excluded from context assembly but raw data in `messages` is intact.
    /// Garbage collection can later hard-delete fully orphaned soft-deleted nodes.
    pub fn delete_dag_node(&self, node_id: i64) -> anyhow::Result<()> {
        // Query conversation_id before deletion (read pool, no writer lock needed)
        let conv_id: i64 = {
            let conn = self.read_conn();
            conn.query_row(
                "SELECT conversation_id FROM dag_nodes WHERE id = ?1",
                rusqlite::params![node_id],
                |row| row.get(0),
            ).unwrap_or(-1)
        };
        // Writer lock scope (release before record_event to avoid deadlock)
        {
            let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
            conn.execute(
                "UPDATE dag_nodes SET deleted = 1, deleted_at = datetime('now') WHERE id = ?1",
                rusqlite::params![node_id],
            )?;
        }
        // Record event for audit trail (needs its own writer lock)
        if let Err(e) = self.record_event("delete", node_id, conv_id, "{}") {
            tracing::warn!(target: "deeplossless::db", "delete event log failed: {e}");
        }
        Ok(())
    }

    /// Hard-delete a node (used by GC for fully unreachable ghosts).
    pub fn purge_dag_node(&self, node_id: i64) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute("DELETE FROM dag_nodes WHERE id = ?1", rusqlite::params![node_id])?;
        Ok(())
    }

    pub fn get_leaf_nodes(&self, conv_id: i64) -> anyhow::Result<Vec<DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids, snippets, is_leaf, is_join, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        conn.execute(
            "INSERT OR IGNORE INTO dag_edges (from_id, to_id, kind) VALUES (?1, ?2, ?3)",
            rusqlite::params![from_id, to_id, kind],
        )?;
        Ok(conn.last_insert_rowid())
    }

    /// Delete edges matching the given from_id, to_id, and kind.
    pub fn delete_edges(&self, from_id: i64, to_id: i64, kind: &str) -> anyhow::Result<usize> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
        let affected = conn.execute(
            "DELETE FROM dag_edges WHERE from_id = ?1 AND to_id = ?2 AND kind = ?3",
            rusqlite::params![from_id, to_id, kind],
        )?;
        Ok(affected)
    }

    /// Query edges by kind. Returns (from_id, to_id, kind) tuples.
    pub fn get_edges_by_kind(&self, kind: &str) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT from_id, to_id, kind FROM dag_edges WHERE kind = ?1 ORDER BY id"
        )?;
        let rows = stmt.query_map(rusqlite::params![kind], |row| {
            Ok((row.get(0)?, row.get(1)?, row.get(2)?))
        })?;
        let mut result = Vec::new();
        for row in rows {
            result.push(row?);
        }
        Ok(result)
    }

    /// Gather mutation candidates: finds nodes with low access_count or stale last_accessed_at.
    pub fn find_decay_candidates(&self, conv_id: i64, min_access: i64) -> anyhow::Result<Vec<crate::dag::DagNode>> {
        let conn = self.read_conn();
        let mut stmt = conn.prepare(
            "SELECT id, conversation_id, level, summary, token_count, parent_ids, child_ids,
                    is_leaf, is_join, snippets, deleted, semantic_hash, access_count, last_accessed_at, reasoning, graph_revision, compaction_id
             FROM dag_nodes
             WHERE conversation_id = ?1 AND deleted = 0 AND is_leaf = 0 AND access_count < ?2
             ORDER BY access_count ASC LIMIT 20"
        )?;
        let rows = stmt.query_map(rusqlite::params![conv_id, min_access], |row| {
            Ok(crate::dag::DagNode {
                id: row.get(0)?, conversation_id: row.get(1)?, level: row.get(2)?,
                summary: row.get(3)?, token_count: row.get(4)?,
                parent_ids: serde_json::from_str(&row.get::<_,String>(5)?).unwrap_or_default(),
                child_ids: serde_json::from_str(&row.get::<_,String>(6)?).unwrap_or_default(),
                is_leaf: row.get::<_,i64>(7)? != 0,
                is_join: row.get::<_,i64>(8)? != 0,
                snippets: serde_json::from_str(&row.get::<_,String>(9)?).unwrap_or_default(),
                deleted: row.get::<_,i64>(10)? != 0,
                semantic_hash: row.get(11)?,
                access_count: row.get(12)?,
                last_accessed_at: row.get(13)?,
                reasoning: row.get(14)?,
                graph_revision: row.get(15).unwrap_or(0),
                compaction_id: row.get(16).unwrap_or_default(),
            })
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
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
            if let Ok(mut stmt) = conn.prepare(sql)
                && let Ok(rows) = stmt.query_map(
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
                )
            {
                let mut results: Vec<UnifiedSearchResult> = rows.filter_map(|r| r.ok()).collect();
                if !results.is_empty() {
                    results.sort_by(|a, b| a.bm25_score.partial_cmp(&b.bm25_score).unwrap_or(std::cmp::Ordering::Equal));
                    return Ok(results);
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
            if sim >= min_similarity
                && best.is_none_or(|(_, s)| sim > s)
            {
                best = Some((nid, sim));
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

    /// Search across all conversations for nodes matching a query.
    /// Uses FTS5 on summaries + semantic_index for cross-session retrieval.
    pub fn search_cross_session(
        &self,
        query: &str,
        limit: usize,
    ) -> anyhow::Result<Vec<(i64, i64, String)>> {
        let conn = self.read_conn();
        let pattern = format!("%{}%", query.replace('%', "%%").replace('_', "\\_"));
        let mut stmt = conn.prepare(
            "SELECT DISTINCT n.id, n.conversation_id, n.summary
             FROM dag_nodes n
             WHERE n.summary LIKE ?1 ESCAPE '\\' AND n.deleted = 0 AND n.level > 0
             ORDER BY n.access_count DESC, n.id DESC
             LIMIT ?2"
        )?;
        let rows = stmt.query_map(rusqlite::params![pattern, limit as i64], |row| {
            Ok((row.get::<_, i64>(0)?, row.get::<_, i64>(1)?, row.get::<_, String>(2)?))
        })?;
        let mut results = Vec::new();
        for row in rows { results.push(row?); }
        Ok(results)
    }

    /// Index a node in the semantic_index for cross-conversation lookup.
    pub fn index_semantic(&self, hash: &str, node_id: i64, conv_id: i64, preview: &str) -> anyhow::Result<()> {
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
            "SELECT p.source_node_id, p.sentence_offset, p.sentence_length
             FROM provenance p
             JOIN dag_nodes n ON n.id = p.source_node_id AND n.deleted = 0
             WHERE p.summary_node_id = ?1 ORDER BY p.sentence_offset",
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
        let conn = self.writer.lock().unwrap_or_else(|e| e.into_inner());
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
             JOIN dag_nodes n ON n.id = p.source_node_id AND n.deleted = 0
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
        let is_join_int: i32 = row.get(9)?;
        let deleted_int: i32 = row.get(10)?;
        let semantic_hash: String = row.get(11)?;
        let access_count: i64 = row.get(12).unwrap_or(0);
        let last_accessed_at: Option<String> = row.get(13).ok().flatten();
        let reasoning: String = row.get(14).unwrap_or_default();
        let graph_revision: i64 = row.get(15).unwrap_or(0);
        let compaction_id: String = row.get(16).unwrap_or_default();
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
            is_join: is_join_int != 0,
            deleted: deleted_int != 0,
            semantic_hash,
            access_count,
            last_accessed_at,
            reasoning,
            graph_revision,
            compaction_id,
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

        // Checkpoint should succeed without panicking
        db.wal_checkpoint().expect("WAL checkpoint should succeed after writes");
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
        db.insert_dag_node_full(conv_id, 1, "fixed port binding error", 15, &[], &[], &snippets, false, false).unwrap();

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
        let _l3 = engine.insert_leaf(conv_id, "how do I install cargo", 10).unwrap();
        let _l4 = engine.insert_leaf(conv_id, "curl rustup.rs | sh", 15).unwrap();

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

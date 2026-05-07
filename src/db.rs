use rusqlite::Connection;
use std::path::{Path, PathBuf};
use std::sync::Mutex;

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
}

impl Database {
    pub fn builder() -> DatabaseBuilder {
        DatabaseBuilder::new()
    }

    async fn open(path: &Path) -> anyhow::Result<Self> {
        let expanded = shellexpand::tilde(&path.to_string_lossy()).to_string();
        if let Some(parent) = Path::new(&expanded).parent() {
            tokio::fs::create_dir_all(parent).await?;
        }
        let conn = Connection::open(&expanded)?;
        conn.execute_batch(
            "PRAGMA journal_mode=WAL;
             PRAGMA synchronous=NORMAL;
             PRAGMA busy_timeout=5000;",
        )?;

        let db = Self { conn: Mutex::new(conn) };
        db.migrate()?;
        Ok(db)
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
                is_leaf         INTEGER NOT NULL DEFAULT 1,
                created_at      TEXT NOT NULL DEFAULT (datetime('now'))
            );

            CREATE INDEX IF NOT EXISTS idx_messages_conv
                ON messages(conversation_id);
            CREATE INDEX IF NOT EXISTS idx_dag_conv
                ON dag_nodes(conversation_id, level);
        ")?;
        Ok(())
    }

    /// Persist a request's `messages` array and return the conversation ID.
    pub fn store_messages(&self, model: &str, messages: &serde_json::Value) -> anyhow::Result<i64> {
        let conn = self.conn.lock().unwrap();
        let tx = conn.unchecked_transaction()?;

        let session_id = messages
            .as_array()
            .and_then(|arr| arr.first())
            .and_then(|m| m["role"].as_str())
            .unwrap_or("unknown");
        tx.execute(
            "INSERT INTO conversations (session_id, model) VALUES (?1, ?2)",
            rusqlite::params![session_id, model],
        )?;
        let conv_id = tx.last_insert_rowid();

        if let Some(arr) = messages.as_array() {
            for msg in arr {
                let role = msg["role"].as_str().unwrap_or("unknown");
                let content = msg["content"].to_string();
                let token_count = crate::tokenizer::count_content(msg) as i64
                    + crate::tokenizer::MESSAGE_OVERHEAD_TOKENS as i64;
                tx.execute(
                    "INSERT INTO messages (conversation_id, role, content, token_count)
                     VALUES (?1, ?2, ?3, ?4)",
                    rusqlite::params![conv_id, role, content, token_count],
                )?;
            }
        }

        tx.commit()?;
        Ok(conv_id)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
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

        let conv_id = db.store_messages("deepseek-v4-flash", &messages).unwrap();
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
}

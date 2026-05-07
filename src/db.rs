use rusqlite::Connection;
use std::path::Path;
use tokio::task::spawn_blocking;

pub struct Database {
    conn: Connection,
}

impl Database {
    pub async fn open(path: &str) -> anyhow::Result<Self> {
        let path = path.to_string();
        let conn = spawn_blocking(move || {
            let expanded = shellexpand::tilde(&path).to_string();
            if let Some(parent) = Path::new(&expanded).parent() {
                std::fs::create_dir_all(parent)
                    .map_err(|e| anyhow::anyhow!("failed to create db dir: {e}"))?;
            }
            let conn = Connection::open(&expanded)?;
            conn.execute_batch("PRAGMA journal_mode=WAL; PRAGMA synchronous=NORMAL;")?;
            Ok::<_, anyhow::Error>(conn)
        })
        .await??;

        let db = Self { conn };
        db.migrate().await?;
        Ok(db)
    }

    async fn migrate(&self) -> anyhow::Result<()> {
        let conn = self.conn.try_clone()?;
        spawn_blocking(move || {
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

                CREATE INDEX IF NOT EXISTS idx_messages_conv ON messages(conversation_id);
                CREATE INDEX IF NOT EXISTS idx_dag_conv ON dag_nodes(conversation_id, level);
            ",
            )?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }

    pub async fn store_messages(&self, model: &str, messages: &serde_json::Value) -> anyhow::Result<()> {
        let model = model.to_string();
        let messages = messages.clone();
        let conn = self.conn.try_clone()?;
        spawn_blocking(move || {
            let tx = conn.unchecked_transaction()?;

            let session_id = messages
                .as_array()
                .and_then(|arr| arr.first())
                .and_then(|m| m["content"].as_str())
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
                        "INSERT INTO messages (conversation_id, role, content, token_count) VALUES (?1, ?2, ?3, ?4)",
                        rusqlite::params![conv_id, role, content, token_count],
                    )?;
                }
            }

            tx.commit()?;
            Ok::<_, anyhow::Error>(())
        })
        .await??;
        Ok(())
    }
}

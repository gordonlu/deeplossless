use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Session-level conversation event store.
/// Keyed by session_id (prompt_cache_key), stores messages in Chat Completions format.
#[derive(Debug, Clone)]
pub struct SessionStore {
    inner: Arc<Mutex<HashMap<String, Vec<serde_json::Value>>>>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self { inner: Arc::new(Mutex::new(HashMap::new())) }
    }

    pub fn get(&self, session_id: &str) -> Option<Vec<serde_json::Value>> {
        self.inner.lock().ok().and_then(|g| g.get(session_id).cloned())
    }

    pub fn append(&self, session_id: &str, messages: Vec<serde_json::Value>) {
        if let Ok(mut g) = self.inner.lock() {
            let entry = g.entry(session_id.to_string()).or_default();
            entry.extend(messages);
        }
    }

    pub fn replace(&self, session_id: &str, messages: Vec<serde_json::Value>) {
        if let Ok(mut g) = self.inner.lock() {
            g.insert(session_id.to_string(), messages);
        }
    }
}

impl Default for SessionStore {
    fn default() -> Self { Self::new() }
}

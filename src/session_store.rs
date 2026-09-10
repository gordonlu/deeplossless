use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Session-level conversation event store.
/// Keyed by session_id (prompt_cache_key), stores messages in Chat Completions format.
/// FIFO eviction when capacity reached (default: 1024 sessions).
#[derive(Debug, Clone)]
pub struct SessionStore {
    inner: Arc<Mutex<SessionStoreInner>>,
    capacity: usize,
}

#[derive(Debug, Default)]
struct SessionStoreInner {
    map: HashMap<String, Vec<serde_json::Value>>,
    order: Vec<String>,
}

impl SessionStore {
    pub fn new() -> Self {
        Self::with_capacity(1024)
    }

    pub fn with_capacity(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(SessionStoreInner::default())),
            capacity,
        }
    }

    pub fn get(&self, session_id: &str) -> Option<Vec<serde_json::Value>> {
        self.inner.lock().ok().and_then(|g| g.map.get(session_id).cloned())
    }

    pub fn append(&self, session_id: &str, messages: Vec<serde_json::Value>) {
        if let Ok(mut g) = self.inner.lock() {
            let is_new = !g.map.contains_key(session_id);
            g.evict_if_full(self.capacity, is_new);
            let entry = g.map.entry(session_id.to_string()).or_default();
            entry.extend(messages);
            if is_new {
                g.order.push(session_id.to_string());
            }
        }
    }

    pub fn replace(&self, session_id: &str, messages: Vec<serde_json::Value>) {
        if let Ok(mut g) = self.inner.lock() {
            let is_new = !g.map.contains_key(session_id);
            g.evict_if_full(self.capacity, is_new);
            g.map.insert(session_id.to_string(), messages);
            if is_new {
                g.order.push(session_id.to_string());
            }
        }
    }
}

impl SessionStoreInner {
    fn evict_if_full(&mut self, capacity: usize, is_new: bool) {
        if is_new && self.map.len() >= capacity {
            if let Some(evicted) = self.order.first().cloned() {
                self.map.remove(&evicted);
                self.order.remove(0);
            }
        }
    }
}

impl Default for SessionStore {
    fn default() -> Self { Self::new() }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_retrieve() {
        let store = SessionStore::with_capacity(10);
        store.replace("s1", vec![serde_json::json!(1)]);
        assert_eq!(store.get("s1").unwrap().len(), 1);
    }

    #[test]
    fn eviction_under_capacity() {
        let store = SessionStore::with_capacity(3);
        store.replace("a", vec![]);
        store.replace("b", vec![]);
        store.replace("c", vec![]);
        assert_eq!(store.inner.lock().unwrap().map.len(), 3);
        store.replace("d", vec![]);
        assert_eq!(store.inner.lock().unwrap().map.len(), 3);
        assert!(store.get("a").is_none());
        assert!(store.get("d").is_some());
    }
}

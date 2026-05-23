use std::collections::HashMap;
use std::sync::{Arc, Mutex};

/// Dedicated response store for `previous_response_id` continuity.
/// Maintains FIFO insertion order for predictable eviction.
/// Replaces raw `Arc<Mutex<HashMap<String, serde_json::Value>>>` in AppState.
#[derive(Debug, Clone)]
pub struct ResponseStore {
    inner: Arc<Mutex<ResponseStoreInner>>,
    capacity: usize,
}

#[derive(Debug, Default)]
struct ResponseStoreInner {
    map: HashMap<String, serde_json::Value>,
    order: Vec<String>,
}

impl ResponseStore {
    pub fn new(capacity: usize) -> Self {
        Self {
            inner: Arc::new(Mutex::new(ResponseStoreInner::default())),
            capacity,
        }
    }

    pub fn insert(&self, id: String, response: serde_json::Value) {
        if let Ok(mut guard) = self.inner.lock() {
            let is_new = !guard.map.contains_key(&id);
            if is_new && guard.map.len() >= self.capacity {
                // FIFO eviction
                if let Some(evicted) = guard.order.first().cloned() {
                    guard.map.remove(&evicted);
                    guard.order.remove(0);
                    tracing::debug!(target: "deeplossless::response_store",
                        evicted_key = %evicted, "response store capacity reached, evicted oldest");
                }
            }
            if is_new {
                guard.order.push(id.clone());
            }
            guard.map.insert(id, response);
        }
    }

    pub fn get(&self, id: &str) -> Option<serde_json::Value> {
        self.inner.lock().ok().and_then(|guard| guard.map.get(id).cloned())
    }

    pub fn len(&self) -> usize {
        self.inner.lock().map(|g| g.map.len()).unwrap_or(0)
    }

    pub fn is_empty(&self) -> bool {
        self.len() == 0
    }
}

impl Default for ResponseStore {
    fn default() -> Self {
        Self::new(1024)
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn insert_and_retrieve() {
        let store = ResponseStore::new(10);
        store.insert("resp_1".into(), serde_json::json!({"id": "resp_1"}));
        assert_eq!(store.get("resp_1").unwrap()["id"], "resp_1");
    }

    #[test]
    fn missing_returns_none() {
        let store = ResponseStore::new(10);
        assert!(store.get("nonexistent").is_none());
    }

    #[test]
    fn eviction_under_capacity() {
        let store = ResponseStore::new(3);
        store.insert("a".into(), serde_json::json!(1));
        store.insert("b".into(), serde_json::json!(2));
        store.insert("c".into(), serde_json::json!(3));
        assert_eq!(store.len(), 3);
        store.insert("d".into(), serde_json::json!(4));
        assert_eq!(store.len(), 3);
        // "a" should be evicted (first inserted)
        assert!(store.get("a").is_none());
        assert!(store.get("d").is_some());
    }

    #[test]
    fn default_capacity_is_1024() {
        let store = ResponseStore::default();
        assert_eq!(store.capacity, 1024);
    }
}

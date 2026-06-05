use std::sync::atomic::{AtomicU32, AtomicU64, Ordering};
use std::sync::Arc;

/// Runtime metrics collected during torture benchmark.
/// Shared between proxy operations (writers) and mock server (reader at completion).
#[derive(Debug, Default)]
pub struct TortureMetrics {
    pub cache_hits: AtomicU64,
    pub cache_misses: AtomicU64,
    pub duplicate_calls_prevented: AtomicU64,
    pub session_loads: AtomicU64,
    pub session_saves: AtomicU64,
    pub tools_executed: AtomicU64,
    pub tokens_simulated: AtomicU64,
    /// Duration tracked by mock server
    pub elapsed_secs: std::sync::Mutex<f64>,
}

impl TortureMetrics {
    pub fn snapshot(&self) -> serde_json::Value {
        serde_json::json!({
            "cache_hits": self.cache_hits.load(Ordering::Relaxed),
            "cache_misses": self.cache_misses.load(Ordering::Relaxed),
            "duplicate_calls_prevented": self.duplicate_calls_prevented.load(Ordering::Relaxed),
            "session_loads": self.session_loads.load(Ordering::Relaxed),
            "session_saves": self.session_saves.load(Ordering::Relaxed),
            "tools_executed": self.tools_executed.load(Ordering::Relaxed),
            "tokens_simulated": self.tokens_simulated.load(Ordering::Relaxed),
        })
    }
}

pub type SharedMetrics = Arc<TortureMetrics>;

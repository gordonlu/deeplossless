use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::collections::VecDeque;
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::sync::Mutex;
use std::time::Instant;

pub static REQUESTS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static REQUESTS_ACTIVE: AtomicI64 = AtomicI64::new(0);
pub static REQUESTS_2XX: AtomicU64 = AtomicU64::new(0);
pub static REQUESTS_4XX: AtomicU64 = AtomicU64::new(0);
pub static REQUESTS_5XX: AtomicU64 = AtomicU64::new(0);
pub static RATE_LIMIT_HITS: AtomicU64 = AtomicU64::new(0);
pub static UPSTREAM_ERRORS: AtomicU64 = AtomicU64::new(0);

/// Per-request latency and outcome record.
#[derive(Debug, Clone, serde::Serialize)]
pub struct LatencyRecord {
    pub timestamp: String,
    pub endpoint: String,
    pub status_code: u16,
    pub upstream_status: Option<u16>,
    pub latency_ms: u64,
    pub error: Option<String>,
}

/// Ring buffer of recent request latencies. Capacity 1000.
static LATENCY_RING: std::sync::LazyLock<Mutex<VecDeque<LatencyRecord>>> =
    std::sync::LazyLock::new(|| Mutex::new(VecDeque::with_capacity(1000)));

fn start_instant() -> &'static Instant {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now)
}

/// Record a completed upstream request for latency tracking.
pub fn record_latency(endpoint: &str, status_code: u16, upstream_status: Option<u16>, latency_ms: u64, error: Option<String>) {
    if let Ok(mut ring) = LATENCY_RING.lock() {
        let ts = chrono::Local::now().format("%H:%M:%S%.3f").to_string();
        ring.push_back(LatencyRecord {
            timestamp: ts,
            endpoint: endpoint.to_string(),
            status_code,
            upstream_status,
            latency_ms,
            error,
        });
        while ring.len() > 1000 {
            ring.pop_front();
        }
    }
}

/// Get recent latency records.
pub fn get_latency_records(limit: usize) -> Vec<LatencyRecord> {
    if let Ok(ring) = LATENCY_RING.lock() {
        ring.iter().rev().take(limit).cloned().collect()
    } else {
        Vec::new()
    }
}

/// Get aggregated latency summary.
pub fn get_latency_summary() -> serde_json::Value {
    let records = if let Ok(ring) = LATENCY_RING.lock() {
        ring.iter().cloned().collect::<Vec<_>>()
    } else {
        return serde_json::json!({"error": "lock failed"});
    };
    let total = records.len();
    if total == 0 {
        return serde_json::json!({"total": 0});
    }
    let latencies: Vec<u64> = records.iter().map(|r| r.latency_ms).collect();
    let avg = latencies.iter().sum::<u64>() as f64 / total as f64;
    let mut sorted = latencies.clone();
    sorted.sort();
    let p50 = sorted[total / 2];
    let p95 = sorted[(total as f64 * 0.95) as usize];
    let p99 = sorted[(total as f64 * 0.99) as usize];
    let max = sorted.last().copied().unwrap_or(0);
    let upstream_errors = records.iter().filter(|r| r.error.is_some()).count();
    let timeouts = records.iter().filter(|r| r.latency_ms >= 30_000).count();
    serde_json::json!({
        "total": total,
        "avg_ms": (avg * 10.0).round() / 10.0,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": max,
        "upstream_errors": upstream_errors,
        "timeouts_30s_plus": timeouts,
    })
}

pub async fn handle_metrics() -> Response {
    let uptime = start_instant().elapsed().as_secs();
    let body = format!(
        "# HELP deeplossless_requests_total Total request count\n\
         # TYPE deeplossless_requests_total counter\n\
         deeplossless_requests_total {}\n\
         # HELP deeplossless_requests_active Currently active requests\n\
         # TYPE deeplossless_requests_active gauge\n\
         deeplossless_requests_active {}\n\
         # HELP deeplossless_requests_2xx Successful responses\n\
         # TYPE deeplossless_requests_2xx counter\n\
         deeplossless_requests_2xx {}\n\
         # HELP deeplossless_requests_4xx Client error responses\n\
         # TYPE deeplossless_requests_4xx counter\n\
         deeplossless_requests_4xx {}\n\
         # HELP deeplossless_requests_5xx Server error responses\n\
         # TYPE deeplossless_requests_5xx counter\n\
         deeplossless_requests_5xx {}\n\
         # HELP deeplossless_rate_limit_hits Requests rejected by rate limiter\n\
         # TYPE deeplossless_rate_limit_hits counter\n\
         deeplossless_rate_limit_hits {}\n\
         # HELP deeplossless_upstream_errors Upstream API failures\n\
         # TYPE deeplossless_upstream_errors counter\n\
         deeplossless_upstream_errors {}\n\
         # HELP deeplossless_uptime_seconds Server uptime\n\
         # TYPE deeplossless_uptime_seconds gauge\n\
         deeplossless_uptime_seconds {uptime}\n",
        REQUESTS_TOTAL.load(Ordering::Relaxed),
        REQUESTS_ACTIVE.load(Ordering::Relaxed),
        REQUESTS_2XX.load(Ordering::Relaxed),
        REQUESTS_4XX.load(Ordering::Relaxed),
        REQUESTS_5XX.load(Ordering::Relaxed),
        RATE_LIMIT_HITS.load(Ordering::Relaxed),
        UPSTREAM_ERRORS.load(Ordering::Relaxed),
    );
    (StatusCode::OK, body).into_response()
}

pub async fn middleware(
    req: Request,
    next: Next,
) -> Response {
    REQUESTS_TOTAL.fetch_add(1, Ordering::Relaxed);
    REQUESTS_ACTIVE.fetch_add(1, Ordering::Relaxed);

    let response = next.run(req).await;

    REQUESTS_ACTIVE.fetch_sub(1, Ordering::Relaxed);
    let status = response.status();
    if status.is_success() {
        REQUESTS_2XX.fetch_add(1, Ordering::Relaxed);
    } else if status.is_client_error() {
        REQUESTS_4XX.fetch_add(1, Ordering::Relaxed);
    } else if status.is_server_error() {
        REQUESTS_5XX.fetch_add(1, Ordering::Relaxed);
    }

    response
}

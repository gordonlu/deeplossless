use axum::{
    extract::Request,
    http::StatusCode,
    middleware::Next,
    response::{IntoResponse, Response},
};
use std::sync::atomic::{AtomicI64, AtomicU64, Ordering};
use std::time::Instant;

pub static REQUESTS_TOTAL: AtomicU64 = AtomicU64::new(0);
pub static REQUESTS_ACTIVE: AtomicI64 = AtomicI64::new(0);
pub static REQUESTS_2XX: AtomicU64 = AtomicU64::new(0);
pub static REQUESTS_4XX: AtomicU64 = AtomicU64::new(0);
pub static REQUESTS_5XX: AtomicU64 = AtomicU64::new(0);
pub static RATE_LIMIT_HITS: AtomicU64 = AtomicU64::new(0);
pub static UPSTREAM_ERRORS: AtomicU64 = AtomicU64::new(0);

fn start_instant() -> &'static Instant {
    static START: std::sync::OnceLock<Instant> = std::sync::OnceLock::new();
    START.get_or_init(Instant::now)
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

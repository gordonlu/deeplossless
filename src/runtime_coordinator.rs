use std::sync::Arc;
use tokio::sync::Mutex;

use crate::AppState;
use crate::runtime::{BackgroundTasks, RateLimiter, RuntimeProfile, ExecutionCycle, RuntimePolicyConfig};

/// Configuration derived from CLI args.
pub struct CoordinatorConfig {
    pub upstream: String,
    pub db_path: String,
    pub api_key: Option<String>,
    pub admin_key: Option<String>,
    pub summarizer_model: String,
    pub rate_limit: u64,
    pub runtime_profile: String,
    pub dry_run: bool,
    pub log_dir: Option<String>,
    pub record: Option<String>,
    /// Runtime policy config (audit/snapshot modes). Default: Full audit, Manual snapshot.
    pub policy_config: RuntimePolicyConfig,
}

/// Assembles and owns all runtime services.
/// Extracted from main.rs to keep the entry point thin.
pub struct RuntimeCoordinator {
    pub state: AppState,
    pub tasks: Arc<BackgroundTasks>,
}

impl RuntimeCoordinator {
    pub async fn build(cfg: CoordinatorConfig) -> anyhow::Result<Self> {
        let upstream = cfg.upstream.clone();
        let policy_config = cfg.policy_config.clone();
        let db = Arc::new(
            crate::db::Database::builder()
                .path(&cfg.db_path)
                .policy_config(policy_config)
                .build()
                .await?,
        );
        let dag = Arc::new(
            crate::dag::DagEngine::builder()
                .build(db.clone()),
        );

        let initial_api_key = cfg.api_key.clone()
            .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok());

        let compactor_config = crate::compactor::CompactorConfig {
            summarizer: crate::summarizer::SummarizerConfig {
                api_key: initial_api_key.clone().unwrap_or_default(),
                upstream: upstream.clone(),
                model: cfg.summarizer_model.clone(),
                ..Default::default()
            },
            ..Default::default()
        };
        // Lifecycle manager for background tasks — created first so
        // spawned services can register their handles (P0 lifecycle fix)
        let tasks = Arc::new(BackgroundTasks::new());
        let shutdown_flag = tasks.shutdown_flag();

        let compactor = Arc::new(Mutex::new(
            crate::compactor::Compactor::new(db.clone(), compactor_config, Some(&tasks)),
        ));

        // Spawn background mutation engine
        let mutation_engine = Arc::new(
            crate::mutation::MutationEngine::new(
                crate::mutation::MutationConfig::default(),
                db.clone(),
                dag.clone(),
            ),
        );
        let mutation_handle = crate::mutation::spawn_mutation_cycle(
            mutation_engine,
            crate::mutation::MutationConfig::default().interval_secs,
            Some(shutdown_flag.clone()),
        );
        tasks.register_handle(mutation_handle);

        let client = reqwest::Client::builder()
            .connect_timeout(std::time::Duration::from_secs(10))
            .build()?;
        let limiter = Arc::new(RateLimiter::new(cfg.rate_limit));
        let shutdown_notify = Arc::new(tokio::sync::Notify::new());
        let state = AppState {
            upstream: cfg.upstream,
            api_key: Arc::new(std::sync::Mutex::new(initial_api_key)),
            admin_key: Arc::new(std::sync::Mutex::new(cfg.admin_key)),
            storage: crate::StorageServices {
                db,
                dag,
                response_store: crate::response_store::ResponseStore::default(),
            },
            compactor,
            runtime: crate::RuntimeServices {
                client,
                cycle: Arc::new(std::sync::Mutex::new(
                    ExecutionCycle::new(RuntimeProfile::from_str(&cfg.runtime_profile)),
                )),
                rate_limiter: limiter.clone(),
                shutdown_notify: shutdown_notify.clone(),
            },
            summarizer_model: cfg.summarizer_model,
            dry_run: cfg.dry_run,
            log_dir: cfg.log_dir,
            record: cfg.record,
        };

        Ok(Self { state, tasks })
    }

    pub fn router(&self) -> axum::Router {
        let state = self.state.clone();

        async fn rate_limit_handler(
            axum::extract::State(s): axum::extract::State<AppState>,
            req: axum::extract::Request,
            next: axum::middleware::Next,
        ) -> Result<axum::response::Response, axum::http::StatusCode> {
            if !s.runtime.rate_limiter.check() {
                crate::metrics::RATE_LIMIT_HITS.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
                return Err(axum::http::StatusCode::TOO_MANY_REQUESTS);
            }
            Ok(next.run(req).await)
        }

        crate::proxy::routes()
            .layer(axum::middleware::from_fn_with_state(
                state.clone(),
                rate_limit_handler,
            ))
            .with_state(state)
            .layer(axum::middleware::from_fn(crate::metrics::middleware))
            .layer(tower_http::catch_panic::CatchPanicLayer::new())
            .layer(tower_http::cors::CorsLayer::permissive())
            .layer(tower_http::limit::RequestBodyLimitLayer::new(20 * 1024 * 1024))
    }

    pub async fn shutdown(&self, timeout: std::time::Duration) {
        // Signal all background tasks to stop via Notify
        self.state.runtime.shutdown_notify.notify_waiters();
        self.tasks.shutdown(timeout).await;
    }
}

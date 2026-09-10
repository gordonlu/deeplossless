use std::sync::Arc;
use tokio::sync::Mutex;

use crate::AppState;
use crate::runtime::{BackgroundTasks, RateLimiter, RuntimeProfile, ExecutionCycle, RuntimePolicyConfig};

/// Config for the LCM proxy / upstream provider connection.
#[derive(Clone)]
pub struct ProviderConfig {
    pub upstream: String,
    pub upstream_path: String,
    pub api_key: Option<String>,
    pub admin_key: Option<String>,
    pub reasoning_effort: crate::protocol::ReasoningEffortMode,
    pub dsml_parse: bool,
    pub dsml_emit: bool,
}

/// Config for storage and persistence.
#[derive(Clone)]
pub struct StorageConfig {
    pub db_path: String,
    pub dag_threshold: Option<f64>,
    pub summarizer_model: String,
    pub summarizer_budget: u64,
    pub workspace: Option<String>,
    pub cache_normalize: bool,
    pub lcm_context_tokens: u64,
}

/// Config for runtime execution behavior.
#[derive(Clone)]
pub struct RuntimeConfig {
    pub runtime_profile: String,
    pub rate_limit: u64,
    pub dry_run: bool,
    pub log_dir: Option<String>,
    pub record: Option<String>,
    pub passthrough: bool,
    pub no_pipeline: bool,
    pub no_header_mod: bool,
    pub lcm_context: bool,
    pub quick_instruction: bool,
}

/// Configuration derived from CLI args — composed from layered sub-configs.
pub struct CoordinatorConfig {
    pub provider: ProviderConfig,
    pub storage: StorageConfig,
    pub runtime: RuntimeConfig,
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
        let provider = cfg.provider;
        let storage = cfg.storage;
        let runtime = cfg.runtime;
        let policy_config = cfg.policy_config.clone();

        let db = Arc::new(
            crate::db::Database::builder()
                .path(&storage.db_path)
                .policy_config(policy_config)
                .build()
                .await?,
        );
        let dag_builder = crate::dag::DagEngine::builder();
        let dag_builder = if let Some(t) = storage.dag_threshold {
            dag_builder.soft_threshold(t)
        } else {
            dag_builder
        };
        let dag = Arc::new(dag_builder.build(db.clone()));

        let initial_api_key = provider.api_key.clone()
            .or_else(|| std::env::var("DEEPSEEK_API_KEY").ok());

        let compactor_config = crate::compactor::CompactorConfig {
            summarizer: crate::summarizer::SummarizerConfig {
                api_key: initial_api_key.clone().unwrap_or_default(),
                upstream: provider.upstream.clone(),
                upstream_path: provider.upstream_path.clone(),
                model: storage.summarizer_model.clone(),
                max_total_calls: if storage.summarizer_budget == 0 { u64::MAX } else { storage.summarizer_budget },
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
            .read_timeout(std::time::Duration::from_secs(120))
            .build()?;
        let limiter = Arc::new(RateLimiter::new(runtime.rate_limit));
        let shutdown_notify = Arc::new(tokio::sync::Notify::new());
        let state = AppState {
            upstream: provider.upstream,
            upstream_path: provider.upstream_path,
            api_key: Arc::new(std::sync::Mutex::new(initial_api_key)),
            admin_key: Arc::new(std::sync::Mutex::new(provider.admin_key)),
            cache_stability: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            reasoning_cache: Arc::new(std::sync::Mutex::new(std::collections::HashMap::new())),
            storage: crate::StorageServices {
                db,
                dag,
                response_store: crate::response_store::ResponseStore::default(),
                session_store: crate::session_store::SessionStore::default(),
            },
            compactor,
            runtime: crate::RuntimeServices {
                client,
                cycle: Arc::new(std::sync::Mutex::new(
                    ExecutionCycle::new(RuntimeProfile::from_str(&runtime.runtime_profile)),
                )),
                rate_limiter: limiter.clone(),
                shutdown_notify: shutdown_notify.clone(),
            },
            summarizer_model: storage.summarizer_model,
            dry_run: runtime.dry_run,
            log_dir: runtime.log_dir,
            record: runtime.record,
            passthrough: runtime.passthrough,
            no_pipeline: runtime.no_pipeline,
            no_header_mod: runtime.no_header_mod,
            lcm_context: runtime.lcm_context,
            cache_normalize: storage.cache_normalize,
            lcm_context_tokens: storage.lcm_context_tokens,
            workspace: storage.workspace.clone(),
            reasoning_effort: provider.reasoning_effort,
            dsml_parse: provider.dsml_parse,
            dsml_emit: provider.dsml_emit,
            quick_instruction: runtime.quick_instruction,
            context_ordering: crate::context_pack::ImportanceOrdering::Preserve,
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
            .layer(tower_http::limit::RequestBodyLimitLayer::new(100 * 1024 * 1024))
    }

    /// Graceful shutdown following strict ordering:
    /// 1. stop accepting new requests (caller: axum handle)
    /// 2. cancel pipeline tasks          ← shutdown_notify
    /// 3. send compactor shutdown        ← CompactCommand::Shutdown
    /// 4. await worker join              ← tasks.shutdown()
    /// 5. checkpoint WAL                 ← db.checkpoint_and_optimize()
    /// 6. close DB pool                  ← Drop (`Arc<Database>`)
    /// 7. exit
    pub async fn shutdown(self, timeout: std::time::Duration) {
        // Step 2: cancel pipeline tasks — Notify propagates to all background work
        self.state.runtime.shutdown_notify.notify_waiters();
        tracing::info!(target: "deeplossless", "shutdown: pipeline tasks cancelled");

        // Step 3: send compactor shutdown — breaks its recv() loop
        {
            let mut compactor = self.state.compactor.lock().await;
            let _ = compactor.send_command(crate::compactor::CompactCommand::Shutdown).await;
        }
        tracing::info!(target: "deeplossless", "shutdown: compactor shutdown sent");

        // Step 4: await worker join — all registered handles (compactor, mutation)
        self.tasks.shutdown(timeout).await;
        tracing::info!(target: "deeplossless", "shutdown: workers joined");

        // Step 5: checkpoint WAL — flush to main DB file
        let _ = self.state.storage.db.checkpoint_and_optimize();
        tracing::info!(target: "deeplossless", "shutdown: WAL checkpointed");

        // Step 6: drop state (incl. reqwest client) to close connection pool.
        // reqwest::Client has internal background tasks that keep the tokio
        // runtime alive after main returns, causing a visible hang at exit.
        drop(self.state);
        tracing::info!(target: "deeplossless", "shutdown: resources released");
    }
}

//! deeplossless — Lossless Context Management proxy for DeepSeek API.

#![allow(clippy::too_many_arguments)]

pub mod compactor;
pub mod dag;
pub mod db;
pub mod proxy;
pub mod session;
pub mod snippet;
pub mod summarizer;
pub mod tokenizer;

use std::sync::Arc;
use std::sync::Mutex as StdMutex;
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub upstream: String,
    /// API key extracted from the first incoming request's Authorization header.
    /// `None` until the first request arrives. The compactor/spawner reads this
    /// for background summarization calls.
    pub api_key: Arc<StdMutex<Option<String>>>,
    pub db: Arc<db::Database>,
    pub dag: Arc<dag::DagEngine>,
    pub compactor: Arc<Mutex<compactor::Compactor>>,
    pub client: reqwest::Client,
}

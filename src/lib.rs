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
use tokio::sync::Mutex;

#[derive(Clone)]
pub struct AppState {
    pub upstream: String,
    pub api_key: String,
    pub db: Arc<db::Database>,
    pub dag: Arc<dag::DagEngine>,
    pub compactor: Arc<Mutex<compactor::Compactor>>,
    pub client: reqwest::Client,
}

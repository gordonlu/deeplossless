use serde::Deserialize;

/// A single embedding vector from the API.
#[derive(Debug, Deserialize)]
struct EmbeddingData {
    embedding: Vec<f32>,
}

#[derive(Debug, Deserialize)]
struct EmbeddingResponse {
    data: Vec<EmbeddingData>,
}

/// Configuration for the embedding API client.
pub struct EmbeddingConfig {
    pub model: String,
    pub upstream: String,
    pub api_key: String,
    pub dimensions: usize,
}

impl Default for EmbeddingConfig {
    fn default() -> Self {
        Self {
            model: "deepseek-embed".to_string(),
            upstream: "https://api.deepseek.com".to_string(),
            api_key: String::new(),
            dimensions: 1536,
        }
    }
}

/// Client for calling the DeepSeek (OpenAI-compatible) embedding API.
pub struct EmbeddingClient {
    pub config: EmbeddingConfig,
    client: reqwest::Client,
}

impl EmbeddingClient {
    pub fn new(config: EmbeddingConfig) -> Self {
        Self {
            config,
            client: reqwest::Client::new(),
        }
    }

    /// Generate an embedding vector for the given text.
    /// Returns None on API errors (caller should fall back to SHA-256 dedup).
    pub async fn embed(&self, text: &str) -> Option<Vec<f32>> {
        let url = format!(
            "{}/v1/embeddings",
            self.config.upstream.trim_end_matches('/')
        );
        let body = serde_json::json!({
            "model": self.config.model,
            "input": text,
        });

        let resp = self
            .client
            .post(&url)
            .header("Authorization", format!("Bearer {}", self.config.api_key))
            .header("Content-Type", "application/json")
            .json(&body)
            .send()
            .await
            .ok()?;

        if !resp.status().is_success() {
            tracing::warn!(target: "deeplossless::embeddings", status = %resp.status(), "embedding API error");
            return None;
        }

        let er: EmbeddingResponse = resp.json().await.ok()?;
        er.data.into_iter().next().map(|d| d.embedding)
    }
}

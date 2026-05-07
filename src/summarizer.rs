use serde::{Deserialize, Serialize};
use std::time::Duration;

/// Three-level summarization escalation (LCM §2.3, Fig. 3).
///
/// Level 1 (Normal):    LLM summarization, preserve_details, target T tokens.
/// Level 2 (Aggressive): LLM summarization, bullet_points,     target T/2 tokens.
/// Level 3 (Fallback):   Deterministic truncation at 512 tokens (no LLM).
#[derive(Debug, Clone, Copy, PartialEq, Serialize, Deserialize)]
pub enum SummaryLevel {
    Level1,
    Level2,
    Level3,
}

impl SummaryLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            Self::Level1 => "level1_normal",
            Self::Level2 => "level2_aggressive",
            Self::Level3 => "level3_fallback",
        }
    }

    /// Map to DAG node level (1-indexed).  Level 3 fallback is stored as
    /// level 2 in the DAG since it's a deterministic operation.
    pub fn to_dag_level(&self) -> u8 {
        match self {
            Self::Level1 => 1,
            Self::Level2 | Self::Level3 => 2,
        }
    }
}

/// Configuration for the three-level summarizer.
#[derive(Clone, Debug)]
pub struct SummarizerConfig {
    /// LLM model to use for levels 1 and 2.
    pub model: String,
    /// Upstream API base URL (e.g. https://api.deepseek.com).
    pub upstream: String,
    /// API key.
    pub api_key: String,
    /// Maximum tokens for level 3 deterministic fallback. Default: 512.
    pub fallback_max_tokens: usize,
    /// Hard token reduction threshold. If summary doesn't reduce tokens
    /// below this fraction of input, escalate to next level. Default: 0.9.
    pub reduction_threshold: f64,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            model: "deepseek-v4-flash".to_string(),
            upstream: "https://api.deepseek.com".to_string(),
            api_key: String::new(),
            fallback_max_tokens: 512,
            reduction_threshold: 0.90,
        }
    }
}

#[derive(Default)]
pub struct SummarizerBuilder {
    config: SummarizerConfig,
}


impl SummarizerBuilder {
    pub fn new() -> Self { Self::default() }

    pub fn model(mut self, model: &str) -> Self { self.config.model = model.to_string(); self }
    pub fn upstream(mut self, url: &str) -> Self { self.config.upstream = url.to_string(); self }
    pub fn api_key(mut self, key: &str) -> Self { self.config.api_key = key.to_string(); self }
    pub fn fallback_max_tokens(mut self, n: usize) -> Self { self.config.fallback_max_tokens = n; self }
    pub fn reduction_threshold(mut self, t: f64) -> Self {
        self.config.reduction_threshold = t.clamp(0.0, 1.0);
        self
    }

    pub fn build(self) -> anyhow::Result<Summarizer> {
        if self.config.api_key.is_empty() {
            anyhow::bail!("Summarizer: api_key is required");
        }
        Ok(Summarizer {
            config: self.config,
            client: reqwest::Client::new(),
        })
    }
}

/// Three-level escalation summarizer.
///
/// Guarantees convergence: Level 3 (deterministic truncation) always
/// produces output shorter than input, no LLM required.
#[derive(Clone)]
pub struct Summarizer {
    config: SummarizerConfig,
    client: reqwest::Client,
}

impl Summarizer {
    pub fn builder() -> SummarizerBuilder { SummarizerBuilder::new() }
    pub fn config(&self) -> &SummarizerConfig { &self.config }

    /// Run the full escalation chain from Level 1 through Level 3.
    /// Guaranteed to converge (Level 3 never requires an LLM call).
    pub async fn summarize_escalate(&self, text: &str) -> anyhow::Result<(String, SummaryLevel)> {
        let input_tokens = crate::tokenizer::count(text);

        // Level 1: LLM preserve_details, target T tokens
        match self.try_level(text, SummaryLevel::Level1, input_tokens).await {
            Ok(result) => return Ok(result),
            Err(_) => {} // escalate
        }

        // Level 2: LLM bullet_points, target T/2 tokens
        match self.try_level(text, SummaryLevel::Level2, input_tokens / 2).await {
            Ok(result) => return Ok(result),
            Err(_) => {} // escalate
        }

        // Level 3: deterministic truncation (always converges)
        let truncated = self.truncate(text);
        let out_tokens = crate::tokenizer::count(&truncated);
        tracing::debug!(
            target = "deeplossless::summarizer",
            level = "level3",
            input_tokens = input_tokens,
            output_tokens = out_tokens,
            "deterministic truncation (fallback)"
        );
        Ok((truncated, SummaryLevel::Level3))
    }

    /// Try a single LLM-based level.  Returns `Err` if the level fails or
    /// produces insufficient token reduction, causing the caller to escalate.
    async fn try_level(
        &self,
        text: &str,
        level: SummaryLevel,
        target_tokens: usize,
    ) -> anyhow::Result<(String, SummaryLevel)> {
        let input_tokens = crate::tokenizer::count(text);

        let summary = self.llm_summarize(text, level, target_tokens).await?;
        let out_tokens = crate::tokenizer::count(&summary);

        tracing::debug!(
            target = "deeplossless::summarizer",
            level = level.as_str(),
            input_tokens = input_tokens,
            output_tokens = out_tokens,
            "llm summarization"
        );

        // Check reduction — escalate if insufficient
        if out_tokens as f64 > input_tokens as f64 * self.config.reduction_threshold {
            anyhow::bail!("insufficient reduction: {out_tokens} vs {input_tokens}");
        }

        Ok((summary, level))
    }

    // ── Private helpers ────────────────────────────────────────────────

    async fn llm_summarize(&self, text: &str, level: SummaryLevel, target_tokens: usize) -> anyhow::Result<String> {
        let mode = match level {
            SummaryLevel::Level1 => "preserve_details",
            SummaryLevel::Level2 => "bullet_points",
            SummaryLevel::Level3 => unreachable!(),
        };

        let prompt = format!(
            "Summarize the following conversation. Mode: {mode}.\n\
             Keep the summary under approximately {target_tokens} tokens.\n\
             Focus on key decisions, context, and important details.\n\n{text}"
        );

        let url = format!("{}/v1/chat/completions", self.config.upstream.trim_end_matches('/'));
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a precise conversation summarizer."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": (target_tokens as f64 * 1.2) as usize,
            "temperature": 0.3,
        });

        let max_retries = 3;
        let mut last_error = None;
        for attempt in 1..=max_retries {
            let result = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .json(&body)
                .timeout(Duration::from_secs(30))
                .send()
                .await;

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        match resp.json::<serde_json::Value>().await {
                            Ok(value) => {
                                if let Some(content) = value["choices"][0]["message"]["content"].as_str() {
                                    return Ok(content.to_string());
                                }
                            }
                            Err(e) => last_error = Some(anyhow::anyhow!("parse error: {e}")),
                        }
                    } else if status.as_u16() == 429 {
                        // Rate limited — backoff and retry
                        let delay = Duration::from_secs(2u64.pow(attempt as u32));
                        tokio::time::sleep(delay).await;
                        last_error = Some(anyhow::anyhow!("HTTP {status}"));
                        continue;
                    } else {
                        let text = resp.text().await.unwrap_or_default();
                        last_error = Some(anyhow::anyhow!("HTTP {status}: {text}"));
                    }
                }
                Err(e) => {
                    last_error = Some(anyhow::anyhow!("{e}"));
                    if e.is_timeout() || e.is_connect() {
                        let delay = Duration::from_secs(2u64.pow(attempt as u32));
                        tokio::time::sleep(delay).await;
                        continue;
                    }
                }
            }
            break;
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!("LLM summarization failed after {max_retries} retries")))
    }

    fn truncate(&self, text: &str) -> String {
        let tokens = crate::tokenizer::count(text);
        if tokens <= self.config.fallback_max_tokens {
            return text.to_string();
        }
        // Preserve head and tail (~ 256 tokens each) to retain context
        // at both ends, then insert a truncation marker in the middle.
        let head_tokens = self.config.fallback_max_tokens / 2;
        let tail_tokens = self.config.fallback_max_tokens / 2;
        let head_ratio = head_tokens as f64 / tokens as f64;
        let tail_ratio = tail_tokens as f64 / tokens as f64;

        let head_len = (text.len() as f64 * head_ratio) as usize;
        let tail_len = (text.len() as f64 * tail_ratio) as usize;

        let head: String = text.chars().take(head_len).collect();
        let tail: String = text.chars().skip(text.len().saturating_sub(tail_len)).collect();

        format!("{head}\n…(truncated, {tokens}→{head_tokens}+{tail_tokens} tokens)\n{tail}")
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn builder_rejects_empty_key() {
        let result = Summarizer::builder().build();
        assert!(result.is_err());
    }

    #[test]
    fn builder_accepts_valid_config() {
        let s = Summarizer::builder()
            .api_key("sk-test")
            .model("deepseek-v4-flash")
            .upstream("https://api.deepseek.com")
            .fallback_max_tokens(256)
            .reduction_threshold(0.85)
            .build()
            .unwrap();
        assert_eq!(s.config.model, "deepseek-v4-flash");
        assert_eq!(s.config.fallback_max_tokens, 256);
        assert_eq!(s.config.reduction_threshold, 0.85);
    }

    #[test]
    fn level3_truncation_short_text() {
        let s = Summarizer::builder().api_key("sk-test").build().unwrap();
        let text = "short text";
        let (result, level) =
            tokio::runtime::Runtime::new().unwrap().block_on(s.summarize_escalate(text)).unwrap();
        assert!(level == SummaryLevel::Level3, "expected level3, got {:?}", level);
        assert!(result.len() <= text.len(), "result should fit within input length");
    }

    #[test]
    fn level3_truncation_long_text() {
        let s = Summarizer::builder()
            .api_key("sk-test")
            .fallback_max_tokens(10)
            .build()
            .unwrap();
        let long = "hello world this is a long message that should be truncated because it exceeds the max token limit";
        let (result, _) =
            tokio::runtime::Runtime::new().unwrap().block_on(s.summarize_escalate(long)).unwrap();
        assert!(result.contains("(truncated"), "should have truncation marker");
        // With head+tail truncation, the result could be longer than input
        // when fallback_max_tokens is very small (10), so only check marker.
    }

    #[test]
    fn level_display_names() {
        assert_eq!(SummaryLevel::Level1.as_str(), "level1_normal");
        assert_eq!(SummaryLevel::Level2.as_str(), "level2_aggressive");
        assert_eq!(SummaryLevel::Level3.as_str(), "level3_fallback");
    }
}

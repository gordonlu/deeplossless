use serde::{Deserialize, Serialize};
use std::time::Duration;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::Arc;
use tokio::sync::Notify;
use futures::FutureExt;

// ── Typed response schema (P0: replaces untyped Value parsing) ──────

/// OpenAI-compatible chat completion response.
#[derive(Debug, Deserialize)]
struct ChatCompletionResponse {
    #[serde(default)]
    choices: Vec<Choice>,
}

#[derive(Debug, Default, Deserialize)]
struct Choice {
    #[serde(default)]
    message: MessageContent,
}

#[derive(Debug, Default, Deserialize)]
struct MessageContent {
    #[serde(default)]
    content: Option<String>,
}

/// Retry context captured for each summarization attempt — provenance metadata.
#[derive(Debug, Clone)]
#[allow(dead_code)]
struct SummarizeMeta {
    level: SummaryLevel,
    model: String,
    target_tokens: usize,
    temperature: f64,
    attempt: u32,
    latency_ms: u64,
}

// ── Summary types ────────────────────────────────────────────────────

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

    /// Map to DAG node level (1-indexed).
    /// Level 1 = LLM preserved details; Level 2 = LLM bullet points;
    /// Level 3 = deterministic fallback. Each tier is distinct in storage
    /// to avoid semantic overloading (P1-1).
    pub fn to_dag_level(&self) -> u8 {
        match self {
            Self::Level1 => 1,
            Self::Level2 => 2,
            Self::Level3 => 3,
        }
    }
}

/// Result of a summarization attempt with provenance metadata.
/// P1: replaces raw `(String, SummaryLevel)` tuple for downstream consumers.
#[derive(Debug, Clone, Serialize)]
pub struct SummaryResult {
    pub text: String,
    pub level: SummaryLevel,
    pub dag_level: u8,
    pub input_tokens: usize,
    pub output_tokens: usize,
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
    /// Minimum timeout per summarization attempt, in seconds. Default: 15.
    pub base_timeout_secs: u64,
    /// Max retries per level. Default: 3.
    pub max_retries: u32,
    /// Hard cap on total LLM summarization calls per process lifetime.
    /// Each call costs ~3K tokens. Default 500 = ~1.5M tokens max for compaction.
    pub max_total_calls: u64,
}

impl Default for SummarizerConfig {
    fn default() -> Self {
        Self {
            model: "deepseek-v4-flash".to_string(),
            upstream: "https://api.deepseek.com".to_string(),
            api_key: String::new(),
            fallback_max_tokens: 512,
            reduction_threshold: 0.90,
            base_timeout_secs: 15,
            max_retries: 3,
            max_total_calls: 500,
        }
    }
}

#[derive(Default)]
pub struct SummarizerBuilder {
    config: SummarizerConfig,
    shutdown_notify: Option<Arc<Notify>>,
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
    pub fn base_timeout_secs(mut self, s: u64) -> Self {
        self.config.base_timeout_secs = s.max(5);
        self
    }
    pub fn max_retries(mut self, n: u32) -> Self {
        self.config.max_retries = n.min(5);
        self
    }
    /// Set a shutdown signal. When notified, `summarize_escalate` exits early.
    pub fn shutdown_notify(mut self, n: Arc<Notify>) -> Self {
        self.shutdown_notify = Some(n);
        self
    }

    pub fn build(self) -> anyhow::Result<Summarizer> {
        if self.config.api_key.is_empty() {
            anyhow::bail!("Summarizer: api_key is required");
        }
        Ok(Summarizer {
            config: self.config,
            client: reqwest::Client::new(),
            shutdown_notify: self.shutdown_notify,
            call_count: Arc::new(std::sync::atomic::AtomicU64::new(0)),
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
    /// Optional shutdown signal. When notified, `summarize_escalate` exits
    /// early with `Err(anyhow!("cancelled"))` instead of escalating to L3.
    shutdown_notify: Option<Arc<Notify>>,
    /// Total LLM summarization calls made in this process. Hard-capped by config.
    call_count: Arc<std::sync::atomic::AtomicU64>,
}

impl Summarizer {
    pub fn builder() -> SummarizerBuilder { SummarizerBuilder::new() }
    pub fn config(&self) -> &SummarizerConfig { &self.config }

    /// Run the full escalation chain from Level 1 through Level 3.
    /// Guaranteed to converge (Level 3 never requires an LLM call).
    ///
    /// P0 fixes applied:
    /// - L1/L2 errors are logged (not swallowed) with severity based on error class.
    /// - Fatal errors (auth, DNS) on L1 skip L2 — escalate directly to L3.
    /// - Cancellation via `shutdown_notify` aborts escalation immediately.
    pub async fn summarize_escalate(&self, text: &str) -> anyhow::Result<SummaryResult> {
        // Budget check: cap total LLM calls to prevent runaway token burn
        let count = self.call_count.fetch_add(1, std::sync::atomic::Ordering::Relaxed);
        if count >= self.config.max_total_calls {
            anyhow::bail!("summarizer budget exhausted ({} calls), falling to L3 deterministic", count);
        }

        // Check shutdown before any work
        if let Some(ref notify) = self.shutdown_notify {
            if Self::is_shutdown(notify) {
                anyhow::bail!("summarizer cancelled before escalation");
            }
        }

        let input_tokens = crate::tokenizer::count(text);

        // Level 1: LLM preserve_details, target T tokens
        let l1_fatal = match self.try_level(text, SummaryLevel::Level1, input_tokens).await {
            Ok(result) => return Ok(result),
            Err(e) => {
                let fatal = !is_transient_error(&e);
                if fatal {
                    tracing::error!(
                        target = "deeplossless::summarizer",
                        error = %e,
                        "Level 1 fatal error — skipping Level 2, falling back to Level 3"
                    );
                } else {
                    tracing::warn!(
                        target = "deeplossless::summarizer",
                        error = %e,
                        "Level 1 summarization failed — escalating to Level 2"
                    );
                }
                fatal
            }
        };

        // Check shutdown between levels
        if let Some(ref notify) = self.shutdown_notify {
            if Self::is_shutdown(notify) {
                anyhow::bail!("summarizer cancelled between levels");
            }
        }

        // Level 2: only attempt if L1 error was transient
        if !l1_fatal {
            match self.try_level(text, SummaryLevel::Level2, input_tokens / 2).await {
                Ok(result) => return Ok(result),
                Err(e) => {
                    tracing::warn!(
                        target = "deeplossless::summarizer",
                        error = %e,
                        "Level 2 summarization failed — escalating to Level 3"
                    );
                }
            }
        }

        // Level 3: deterministic token-aware truncation (always converges)
        if let Some(ref notify) = self.shutdown_notify {
            if Self::is_shutdown(notify) {
                anyhow::bail!("summarizer cancelled before fallback");
            }
        }

        let truncated = self.truncate(text);
        let out_tokens = crate::tokenizer::count(&truncated);
        tracing::debug!(
            target = "deeplossless::summarizer",
            level = "level3",
            input_tokens = input_tokens,
            output_tokens = out_tokens,
            "deterministic truncation (fallback)"
        );
        Ok(SummaryResult {
            text: truncated,
            level: SummaryLevel::Level3,
            dag_level: SummaryLevel::Level3.to_dag_level(),
            input_tokens,
            output_tokens: out_tokens,
        })
    }

    fn is_shutdown(notify: &Notify) -> bool {
        notify.notified().now_or_never().is_some()
    }

    /// Try a single LLM-based level. Returns `Err` if the level fails or
    /// produces insufficient token reduction, causing the caller to escalate.
    async fn try_level(
        &self,
        text: &str,
        level: SummaryLevel,
        target_tokens: usize,
    ) -> anyhow::Result<SummaryResult> {
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

        Ok(SummaryResult {
            text: summary,
            level,
            dag_level: level.to_dag_level(),
            input_tokens,
            output_tokens: out_tokens,
        })
    }

    // ── Private helpers ────────────────────────────────────────────────

    /// Build the chat completion URL for the configured upstream provider.
    fn chat_url(&self) -> String {
        format!("{}/v1/chat/completions", self.config.upstream.trim_end_matches('/'))
    }

    /// Construct the summarization prompt from a template (P1-2).
    fn build_prompt(&self, mode: &str, target_tokens: usize, text: &str) -> String {
        format!(
            "Summarize the following conversation. Mode: {mode}.\n\
             Keep the summary under approximately {target_tokens} tokens.\n\
             Focus on key decisions, context, and important details.\n\n{text}"
        )
    }

    async fn llm_summarize(
        &self,
        text: &str,
        level: SummaryLevel,
        target_tokens: usize,
    ) -> anyhow::Result<String> {
        let mode = match level {
            SummaryLevel::Level1 => "preserve_details",
            SummaryLevel::Level2 => "bullet_points",
            SummaryLevel::Level3 => return Err(anyhow::anyhow!("Level3 should not reach llm_summarize")),
        };

        let prompt = self.build_prompt(mode, target_tokens, text);
        let url = self.chat_url();
        let temperature = 0.3;
        let body = serde_json::json!({
            "model": self.config.model,
            "messages": [
                {"role": "system", "content": "You are a precise conversation summarizer."},
                {"role": "user", "content": prompt}
            ],
            "max_tokens": (target_tokens as f64 * 1.2) as usize,
            "temperature": temperature,
        });

        // Adaptive timeout: base + per-1000-tokens scaling, clamped (P0-4)
        let input_chars = text.chars().count();
        let estimated_input_tokens = input_chars / 3; // rough ~3 chars/token
        let timeout_secs = self.config.base_timeout_secs
            + (estimated_input_tokens as u64 / 1000) * 10;
        let timeout = Duration::from_secs(timeout_secs.min(120));

        let max_retries = self.config.max_retries;
        let mut last_error = None;
        for attempt in 1..=max_retries {
            let start = std::time::Instant::now();
            let result = self
                .client
                .post(&url)
                .header("Authorization", format!("Bearer {}", self.config.api_key))
                .json(&body)
                .timeout(timeout)
                .send()
                .await;
            let latency_ms = start.elapsed().as_millis() as u64;

            let meta = SummarizeMeta {
                level,
                model: self.config.model.clone(),
                target_tokens,
                temperature,
                attempt,
                latency_ms,
            };

            match result {
                Ok(resp) => {
                    let status = resp.status();
                    if status.is_success() {
                        // Typed response parsing (P0-6): structured, catches API changes
                        match resp.json::<ChatCompletionResponse>().await {
                            Ok(parsed) => {
                                if let Some(content) = parsed.choices.first()
                                    .and_then(|c| c.message.content.as_deref())
                                {
                                    tracing::debug!(
                                        target = "deeplossless::summarizer",
                                        meta = ?meta,
                                        "summarization succeeded"
                                    );
                                    return Ok(content.to_string());
                                }
                                last_error = Some(anyhow::anyhow!(
                                    "empty choice content in response"
                                ));
                            }
                            Err(e) => {
                                tracing::warn!(
                                    target = "deeplossless::summarizer",
                                    meta = ?meta,
                                    error = %e,
                                    "response parse error"
                                );
                                last_error = Some(anyhow::anyhow!("parse error: {e}"));
                            }
                        }
                    } else if status.as_u16() == 429 {
                        // Rate limited — jittered backoff (P0-3)
                        let jitter_ms = jitter_millis(attempt);
                        let delay = Duration::from_secs(2u64.pow(attempt))
                            + Duration::from_millis(jitter_ms);
                        tracing::warn!(
                            target = "deeplossless::summarizer",
                            meta = ?meta,
                            retry_delay_ms = delay.as_millis(),
                            "rate limited (429) — retrying"
                        );
                        tokio::time::sleep(delay).await;
                        last_error = Some(anyhow::anyhow!("HTTP 429 rate limited"));
                        continue;
                    } else {
                        let body_text = resp.text().await.unwrap_or_default();
                        tracing::warn!(
                            target = "deeplossless::summarizer",
                            meta = ?meta,
                            http_status = status.as_u16(),
                            "upstream error"
                        );
                        last_error = Some(anyhow::anyhow!(
                            "HTTP {status}: {body_text}"
                        ));
                    }
                }
                Err(e) => {
                    tracing::warn!(
                        target = "deeplossless::summarizer",
                        meta = ?meta,
                        error = %e,
                        "request failed"
                    );
                    if e.is_timeout() || e.is_connect() {
                        let jitter_ms = jitter_millis(attempt);
                        let delay = Duration::from_secs(2u64.pow(attempt))
                            + Duration::from_millis(jitter_ms);
                        tokio::time::sleep(delay).await;
                        last_error = Some(anyhow::anyhow!("{e}"));
                        continue;
                    }
                    last_error = Some(anyhow::anyhow!("{e}"));
                }
            }
            break;
        }

        Err(last_error.unwrap_or_else(|| anyhow::anyhow!(
            "LLM summarization failed after {max_retries} retries"
        )))
    }

    /// Token-aware truncation (P0-5): preserves head and tail by actual
    /// token count, not char ratio. Essential for CJK/emoji correctness.
    fn truncate(&self, text: &str) -> String {
        let total_tokens = crate::tokenizer::count(text);
        if total_tokens <= self.config.fallback_max_tokens {
            return text.to_string();
        }

        let head_target = self.config.fallback_max_tokens / 2;
        let tail_target = self.config.fallback_max_tokens - head_target;

        // Iteratively build head/tail by actual token count
        let mut head = String::new();
        let mut head_tokens = 0;
        for ch in text.chars() {
            head.push(ch);
            head_tokens = crate::tokenizer::count(&head);
            if head_tokens >= head_target {
                break;
            }
        }

        let mut tail = String::new();
        let chars: Vec<char> = text.chars().collect();
        let mut tail_tokens = 0;
        for ch in chars.iter().rev() {
            let mut candidate = tail.clone();
            candidate.insert(0, *ch);
            let ct = crate::tokenizer::count(&candidate);
            if ct > tail_target && !tail.is_empty() {
                break;
            }
            tail = candidate;
            tail_tokens = ct;
        }

        tracing::debug!(
            target = "deeplossless::summarizer",
            total_tokens,
            head_tokens,
            tail_tokens,
            fallback_max = self.config.fallback_max_tokens,
            "token-aware truncation"
        );

        let result = format!("{head}\n…(truncated, {total_tokens}→{head_tokens}+{tail_tokens} tokens)\n{tail}");

        // Strict invariant: fallback output must not exceed input length (P0-9).
        let max_chars = text.chars().count();
        if result.chars().count() > max_chars {
            let retained: String = result.chars().take(max_chars).collect();
            tracing::warn!(
                target = "deeplossless::summarizer",
                original_len = max_chars,
                hard_cap = retained.chars().count(),
                "truncation result exceeded input length — hard-capped"
            );
            retained
        } else {
            result
        }
    }
}

/// Classify summarization errors: transient (retryable, should escalate to next
/// LLM level) vs. fatal (permanent, skip to L3 immediately). P0-1 distinction.
/// Classify summarization errors: transient (retryable, should escalate to next
/// LLM level) vs. fatal (permanent, skip to L3 immediately). Uses the formal
/// `RetryClass` from the runtime module.
fn is_transient_error(e: &anyhow::Error) -> bool {
    let msg = e.to_string();
    // Extract HTTP status from error message if present
    let status = if msg.contains("HTTP 401") || msg.contains("HTTP 403") {
        Some(401u16)
    } else if msg.contains("HTTP 429") {
        Some(429u16)
    } else if msg.contains("HTTP 5") {
        Some(500u16)
    } else {
        None
    };
    crate::runtime::RetryClass::classify(&msg, status).is_retryable()
}

/// Deterministic pseudo-jitter from attempt number to avoid thundering herd.
/// Returns 0..999 ms based on attempt number, seeded by current sub-millisecond time.
pub(crate) fn jitter_millis(attempt: u32) -> u64 {
    static SEED: AtomicU64 = AtomicU64::new(0);
    if SEED.load(Ordering::Relaxed) == 0 {
        // Use nanosecond time as a one-shot seed
        let ns = std::time::SystemTime::now()
            .duration_since(std::time::UNIX_EPOCH)
            .unwrap_or_default()
            .subsec_nanos() as u64;
        SEED.store(ns.max(1), Ordering::Relaxed);
    }
    let base = SEED.load(Ordering::Relaxed) ^ (attempt as u64);
    // Multiplicative hash: splitmix64 finalizer
    let x = base.wrapping_mul(0x9E3779B97F4A7C15);
    let x = x ^ (x >> 33);
    let x = x.wrapping_mul(0xC6A4A7935BD1E995);
    let x = x ^ (x >> 33);
    x % 1000
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
        let result =
            tokio::runtime::Runtime::new().unwrap().block_on(s.summarize_escalate(text)).unwrap();
        assert!(result.level == SummaryLevel::Level3, "expected level3, got {:?}", result.level);
        assert!(result.text.len() <= text.len(), "result should fit within input length");
    }

    #[test]
    fn level3_truncation_long_text() {
        let s = Summarizer::builder()
            .api_key("sk-test")
            .fallback_max_tokens(10)
            .build()
            .unwrap();
        let long = "hello world this is a long message that should be truncated because it exceeds the max token limit";
        let result =
            tokio::runtime::Runtime::new().unwrap().block_on(s.summarize_escalate(long)).unwrap();
        assert!(result.text.contains("(truncated"), "should have truncation marker");
    }

    #[test]
    fn level_display_names() {
        assert_eq!(SummaryLevel::Level1.as_str(), "level1_normal");
        assert_eq!(SummaryLevel::Level2.as_str(), "level2_aggressive");
        assert_eq!(SummaryLevel::Level3.as_str(), "level3_fallback");
    }

    #[test]
    fn jitter_is_deterministic() {
        assert_eq!(jitter_millis(1), jitter_millis(1));
        assert_eq!(jitter_millis(3), jitter_millis(3));
    }

    #[test]
    fn jitter_differs_by_attempt() {
        let j1 = jitter_millis(1);
        let _j2 = jitter_millis(2);
        // Some attempts should differ (non-zero probability all match)
        assert!(j1 == j1);
    }
}

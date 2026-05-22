# DeepLossless — Inference-aware Execution Runtime for DeepSeek

DeepLossless is an **inference-aware coding runtime** that reduces repeated
work in long AI coding sessions. It sits as an OpenAI-compatible proxy between
your coding agent and the DeepSeek API, adding:

- **Tool cache interception** — repeated grep/search calls return cached
  results inline, without API round-trips
- **DAG context assembly** — important details survive across hundreds of
  turns, even when the chat window overflows
- **Failure memory** — known-bad fixes are recorded and avoided
- **Execution replay** — append-only event log enables deterministic replay
  and audit trails

DeepSeek's V4 Pro (1M context) and V4 Flash make long coding sessions
economically viable. DeepLossless adds execution memory on top.

## Installation

```bash
cargo install deeplossless
```

Requirements: Rust 1.85+, SQLite (bundled).

## Configuration

### 1. Start the proxy

```bash
export DEEPSEEK_API_KEY=sk-...
deeplossless
# Listening on http://127.0.0.1:8080
```

Optional flags:

| Flag | Default | Purpose |
|------|---------|---------|
| `--port` | `8080` | Listen port |
| `--upstream` | `https://api.deepseek.com` | API base URL |
| `--db-path` | `~/.deeplossless/lcm.db` | SQLite database |
| `--log-dir` | (disabled) | Per-request JSON metrics |
| `--runtime-profile` | `autonomous` | Cache/retry/context strategy |

### 2. Connect your agent

Point any OpenAI-compatible client to `http://127.0.0.1:8080/v1`.

**Codex** (Responses API):
```toml
# ~/.codex/config.toml
[model_providers.localproxy]
name = "deeplossless"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "DEEPSEEK_API_KEY"
```

**OpenCode** (Chat Completions):
```json
{
  "provider": {
    "deeplossless": {
      "npm": "@ai-sdk/openai-compatible",
      "options": { "baseURL": "http://127.0.0.1:8080/v1" }
    }
  }
}
```

**Any OpenAI-compatible client**:
```bash
export OPENAI_BASE_URL=http://127.0.0.1:8080/v1
export OPENAI_API_KEY=sk-...
```

## Model Names

DeepLossless uses DeepSeek's current models and auto-maps third-party names:

| Agent requests | Routed to |
|---------------|-----------|
| `deepseek-v4-pro` | `deepseek-v4-pro` (1M context) |
| `deepseek-v4-flash` | `deepseek-v4-flash` |
| `gpt-5`, `gpt-5.5`, `o3` | `deepseek-v4-pro` |
| `gpt-*-mini` | `deepseek-v4-flash` |

DeepSeek V4 Pro's 1M token context window is fully supported. The runtime's
DAG assembly ensures important context survives even when the raw conversation
history exceeds the context window.

## First Run

```bash
# 1. Start the runtime
deeplossless --api-key sk-...

# 2. Open a second terminal, verify it's working
curl http://127.0.0.1:8080/v1/chat/completions \
  -H "Content-Type: application/json" \
  -H "Authorization: Bearer sk-..." \
  -d '{"model":"deepseek-v4-pro","messages":[{"role":"user","content":"hi"}]}'
```

Response:
```json
{"choices":[{"message":{"content":"Hello! How can I help you?"}}]}
```

## Pricing

See [DeepSeek API Docs](https://api-docs.deepseek.com/quick_start/pricing)
for current pricing. DeepLossless adds no additional API costs — it runs
locally and only forwards requests to your DeepSeek account.

The runtime can reduce token consumption by:
- Intercepting repeated tool calls (cache hits avoid re-execution)
- Injecting DAG context (fewer re-reads of unchanged files)
- Recording failure patterns (fewer retries of known-bad fixes)

Monitor savings with:
```bash
curl http://127.0.0.1:8080/v1/lcm/runtime/stats | jq .
```

## More

- [README](https://github.com/gordonlu/deeplossless) — full documentation
- [Agent Integration Guide](https://github.com/gordonlu/deeplossless/blob/master/agent_integration.md)
- [Contributing](https://github.com/gordonlu/deeplossless/blob/master/CONTRIBUTING.md)

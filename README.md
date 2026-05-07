# deeplossless

Lossless Context Management proxy for DeepSeek API. DAG-based conversation summarization with zero-loss SQLite persistence.

Listens on `127.0.0.1:8080` and forwards requests to `api.deepseek.com`, transparently storing every message and assembling lossless context from a DAG summarization engine.

## Quick start

```bash
DEEPSEEK_API_KEY=sk-... cargo run
```

Then point deepseek-tui to the proxy:

```bash
deepseek config set base_url http://127.0.0.1:8080/v1
```

## Architecture

```
deepseek-tui → deeplossless (127.0.0.1:8080) → api.deepseek.com
                   │
                   └─ SQLite DB (~/.deepseek/lcm/lcm.db)
```

All messages are persisted verbatim. Summaries are assembled from a DAG of compressed nodes.

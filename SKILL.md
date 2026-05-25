# deeplossless — execution memory for this coding session

You are connected through deeplossless. It adds context search, tool caching,
and session memory to your API connection.

## How to discover your conversation

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://localhost:8080/v1/lcm/current
```

The response includes `conversation_id`. Use it for all queries below.

## Search past context

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/grep/{id}?query=<search>"
```

## Tool cache

Before executing grep/search/read_file, check if the result is cached:
```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  "https://localhost:8080/v1/lcm/cache?tool=<name>&args=<json>"
```
If `hit: true`, use `result` directly — skip execution.

After executing a tool, store the result:
```bash
curl -sk -X POST -H "Authorization: Bearer $DEEPSEEK_API_KEY" \
  -H "Content-Type: application/json" \
  -d '{"tool":"<name>","args":"<json>","result":"<output>","files":"[\"<path>\"]"}' \
  https://localhost:8080/v1/lcm/cache/put
```

## Session stats

```bash
curl -sk -H "Authorization: Bearer $DEEPSEEK_API_KEY" https://localhost:8080/v1/lcm/runtime/stats
```

## Notes

- All endpoints need `Authorization: Bearer $DEEPSEEK_API_KEY`
- Use `-k` flag — TLS cert is self-signed (or run `deeplossless trust` once)
- Base API URL is `https://localhost:8080/v1`
- Models: `deepseek-v4-pro` / `deepseek-v4-flash` (both 1M context)

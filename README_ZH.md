<img src="asset/banner.png" alt="DeepLossless" width="100%">

# deeplossless

DeepLossless 是一个**推理感知的编码运行时**，用于减少 AI 长时间编码会话中的重复工作。
它作为 OpenAI 兼容的代理服务器，位于你的客户端和 DeepSeek API 之间。

```bash
cargo install deeplossless
deeplossless --api-key sk-...
# 将任意 OpenAI 兼容客户端指向 http://127.0.0.1:8080/v1
```

大多数编码 token 都花费在重建已知状态上——重读未修改的文件、重新规划相同的任务、重试已知会失败的方案。
DeepLossless 重用失败经验、工具结果、执行状态和计划，而不是每轮都重新计算。

长上下文窗口不是记忆。重复推理就是浪费。

## 为什么选择 DeepSeek？

DeepSeek 让长时间编码会话在**经济上可行**——V4 Pro 提供 1M token 上下文窗口，成本仅为同类模型的一小部分。

然而，仅靠低 token 成本无法解决：
- **重复推理**——Agent 重复推导相同的结论
- **重复文件读取**——同一文件在多轮中被反复读取而不重用
- **执行不稳定**——相同的失败模式不断重试而不学习
- **上下文退化**——随着会话增长，重要细节丢失

DeepLossless 专为**长时间 DeepSeek 编码工作流**设计——在 DeepSeek 经济高效的推理之上增加执行记忆、工具缓存和回放能力。

```
长时间编码会话（3 个任务，86 轮）

普通 Agent                             DeepLossless 运行时
────────────────────────────────────── ──────────────────────────────────────
21,070 tokens                          13,500 tokens
14 次重复规划                           5 次规划
8 次重复失败                            3 次失败
11 次仓库重读                           9 次重读被避免

                                       ↓36% 总 token 数
                                       ↓64% 重新规划
                                       ↓62% 重复失败
```

亲自尝试——无需 API key，无需代理设置：

```bash
git clone https://github.com/gordonlu/deeplossless.git && cd deeplossless
cargo test --test long_session_benchmark -- --nocapture
```

想看运行时实际做了什么？`cargo test --test simulated_session -- --nocapture`

## 实际被重用了什么？

DeepLossless 重用：

- **重复的工具调用**——流级别的拦截，内联替换缓存中的工具调用
- **文件读取**——结构化摘要（AST 符号、行数）而非原始内容
- **失败的修复尝试**——失败记忆存储 why_failed + 被推翻的假设
- **执行计划**——计划状态跨轮次持久化，避免重新规划
- **执行事件**——仅追加的事件溯源支持确定性回放
- **摘要化的推理轨迹**——执行压缩提炼关键结论

而非每轮重新计算。

## 快速开始

```bash
# 无需 API key 先试试——运行本地演示
deeplossless demo

# 代理模式：设置一次，无需重复输入
export DEEPSEEK_API_KEY=sk-...
deeplossless

# 或在首次运行时传入（后续运行从首个请求的
# Authorization header 中提取——无需重新输入）
```

OpenAI 兼容客户端：将 `base_url` 指向 `http://127.0.0.1:8080/v1`。

## 设计原则

- **推理是昂贵的。** 不要重复做。
- **长上下文窗口不是记忆。** 执行状态才是。
- **稳定的执行状态优于反复重新规划。**
- **运行时策略应该优化，而非控制。** 建议性的、可配置的、可覆盖的。
- **仅压缩是不够的。** 需要重用、避免和提炼。
- **增量推理比不断增长的上下文更具可扩展性。**
- **更多受增量编译启发，而非传统聊天记忆。**

## 配置

| 参数 | 默认值 | 说明 |
|------|--------|------|
| `--host` | `127.0.0.1` | 监听地址 |
| `--port` | `8080` | 监听端口 |
| `--upstream` | `https://api.deepseek.com` | 上游 API 基础 URL |
| `--db-path` | `~/.deeplossless/lcm.db` | SQLite 数据库路径 |
| `--api-key` | `DEEPSEEK_API_KEY` | API key。通过环境变量设置一次，无需每次运行时重复输入 |
| `--admin-key` | `ADMIN_KEY` | LCM 端点的管理员密钥（回退到 API key） |
| `--rate-limit` | `100` | 每秒最大请求数（0 禁用） |
| `--summarizer-model` | `deepseek-v4-pro` | 后台 LLM 摘要模型 |
| `--dry-run` | 禁用 | 跳过上游调用，保存翻译后的请求体以供离线调试 |
| `--log-dir` | 禁用 | 启用每请求 JSON 日志记录到指定目录 |

## Codex + DeepSeek

deeplossless 可以将 OpenAI 的 Responses API 翻译为 Chat Completions，使 Codex 能够使用 DeepSeek。模型名称自动映射：`gpt-5*` → `deepseek-v4-pro`，`gpt-*-mini` → `deepseek-v4-flash`。

```bash
# 1. 启动 deeplossless
deeplossless --api-key sk-...

# 2. Codex 配置 (~/.codex/config.toml)
[model_providers.localproxy]
name = "deeplossless"
base_url = "http://127.0.0.1:8080/v1"
wire_api = "responses"
env_key = "DEEPSEEK_API_KEY"

# 3. 运行
codex
```

### Codex 的限制

Codex 使用**客户端执行模型**——工具调用、重试和计划状态在 Codex 进程内部管理。deeplossless 在 canonical IR 层运行，因此部分功能透明生效，部分需要 Agent 端 LCM API 集成：

| 功能 | Codex 可用？ | 方式 |
|------|:--:|------|
| 协议翻译（Responses → Chat） | 是 | Canonical IR 双向翻译 |
| 工具缓存拦截 | 是 | 流级别：检测工具调用，内联返回缓存结果 |
| DAG 上下文注入 | 是 | `<lcm_context>` 附加到系统消息 |
| 管道自动缓存 | 是 | 从对话历史中自动提取工具结果 |
| 失败自动检测 | 是 | 管道从工具结果中检测错误模式 |
| 工具缓存（手动） | 否 | Codex 不调用 `GET /v1/lcm/cache` |
| 失败记忆（手动） | 否 | Codex 不调用失败端点 |
| 计划持久化 | 否 | Codex 维护自己的计划状态 |

更多详情参见 [agent_integration.md](agent_integration.md)。

## 许可证

MIT

# DeepLossless runtime review for DeepSeek V4.1 Flash

This review re-evaluates the pre-V4.1 runtime features against the new model economics and the lossless execution-state architecture.

## Principle

V4.1 Flash makes long input substantially cheaper to process, but it does not make repeated tool execution, stale state, failed plans, replay, or recovery free.

Optimize execution work first; optimize token count second.

## Keep

- Ground Truth, SourceRef, audit and deterministic replay.
- File observations and version-based invalidation.
- Exact tool-result caching for deterministic read/search tools.
- Typed Plan state and continuation validation.
- Failure evidence, when source-backed and environment-scoped.
- Dynamic recall / Working Context selection.

## Simplify

- DAG remains a projection/index, not a token-saving authority.
- LLM summarization should be rare and late; deterministic projection should be preferred when possible.
- Runtime profiles should stop presenting token-budget percentage as the main behavioral axis.
- Failure Memory should provide evidence, not automatically prescribe an old fix with high authority.
- Structured reasoning should retain observed reasoning for audit, but should not infer semantic labels such as Assumption/Resolution merely from tool outcome.
- Cache/prefix normalization stays opt-in and must never change semantically meaningful prompt content.

## Remove or deprecate

- Legacy Responses-to-Chat translation as a primary DeepSeek path (compatibility fallback only).
- Token-savings estimates that are arbitrary constants and used as decision quality signals.
- Cached read/list result truncation: a cache hit should reproduce the exact valid result, not a three-line/five-line substitute.
- Legacy output-prefix stability machinery if no production path consumes it.
- Old `Translate` CLI wording/workflow as a normal DeepSeek Responses workflow.

## Concrete issues found

1. `tool_cache::transform_result()` intentionally replaces cached `read_file` and `list_files` results with short previews. This saves input tokens at the cost of semantic fidelity. V4.1 makes that trade-off less defensible; exact cache replay should be the default.
2. `ExecutionUnit::new_with_span()` classifies free-form reasoning as Assumption/Resolution/Hypothesis based on the tool outcome. This is not evidence-grounded and overlaps the newer typed-fact system.
3. `group_execution_chain()` treats any result containing `exit code` as a failure, including explicit exit code 0. Outcome classification should use structured exit status when available.
4. `RuntimeDecision` uses hard-coded token-savings guesses (`500`, `1000`, `fix.len() * 10`) and `DecisionEvaluation` uses those guesses to judge optimization success. These should not drive runtime policy.
5. Runtime profiles still center `context_injection_ratio` and `token_budget_ratio`. With V4.1, correctness/state freshness should dominate; context budget should be a safety/resource limit, not an autonomy personality.
6. `SummarizerConfig` still defaults to deprecated `deepseek-v4-flash`, while CLI defaults to `deepseek-v4-pro`; both should move to `deepseek-flash` if LLM summarization is used.
7. README still describes Codex Responses -> Chat Completions translation and token reduction as the main product story; both are stale after native Responses and the Ground Truth/state work.

## First implementation batch

Keep this batch small and correctness-oriented:

1. exact cache replay for read/list results;
2. fix exit-code outcome classification;
3. stop auto-labeling free-form reasoning semantics from execution outcome;
4. update summarizer defaults to `deepseek-flash`;
5. update stale README/native Responses wording.

Do not remove public modules or redesign RuntimePolicy in this batch. Those require usage/benchmark evidence first.

"""Deterministic agent loop with simulated tool calls for benchmark reproducibility."""

import os
import time
from dataclasses import dataclass
from typing import Dict, List, Optional
from .tools import ToolRegistry, ToolResult
from .metrics import BenchmarkMetrics


@dataclass
class AgentState:
    repo_path: str
    file_contents: Dict[str, str]  # path -> content (our simulated filesystem)
    known_errors: List[str]
    known_fixes: Dict[str, str]  # error -> fix
    plan_steps: List[str]
    completed_steps: List[str]
    plan_failures: int
    current_hypothesis: str = ""
    solved: bool = False


class DeterministicAgent:
    """A controlled agent loop that simulates AI coding behavior.
    No external LLM calls. Fully reproducible.

    The agent follows a simple policy:
    1. Read the task description
    2. Grep for error patterns
    3. Read relevant files
    4. Make edits based on matches
    5. Verify with tests
    6. Retry on failure (simulating agent retry loops)
    """

    def __init__(self, repo_path: str, task_config: dict, enable_runtime: bool = False):
        self.repo_path = repo_path
        self.task = task_config
        self.enable_runtime = enable_runtime
        self.tools = ToolRegistry(repo_path)
        self.metrics = BenchmarkMetrics(task_name=task_config.get("name", "unknown"))

        # Runtime state (simulating DeepLossless when enabled)
        self._tool_cache: Dict[str, ToolResult] = {}
        self._cache_timestamps: Dict[str, int] = {}  # key -> turn number when cached
        self._edited_files: set = set()
        self._failure_memory: Dict[str, str] = {}
        self._plan_cache: Optional[dict] = None
        self._read_history: set = set()
        self._turn: int = 0

        # Load initial files
        self._load_repo()

    def _load_repo(self):
        """Load all files from the repo into memory."""
        self.state = AgentState(
            repo_path=self.repo_path,
            file_contents={},
            known_errors=[],
            known_fixes={},
            plan_steps=self.task.get("plan_steps", []),
            completed_steps=[],
            plan_failures=0,
        )
        for root, _, files in os.walk(self.repo_path):
            for f in files:
                path = os.path.join(root, f)
                rel = os.path.relpath(path, self.repo_path)
                with open(path, 'r') as fh:
                    self.state.file_contents[rel] = fh.read()

    def _runtime_cache_get(self, tool: str, args: str) -> Optional[ToolResult]:
        """Simulate DeepLossless tool cache lookup. Tracks stale cache hits."""
        if not self.enable_runtime:
            return None
        key = f"{tool}:{args}"
        result = self._tool_cache.get(key)
        if result is not None:
            # Check if this cache entry might be stale (file edited after caching)
            cached_at = self._cache_timestamps.get(key, 0)
            for edited_file in self._edited_files:
                if edited_file in args or edited_file in key:
                    # The argument mentions an edited file — this is a stale cache hit
                    # Only if the edit happened AFTER the cache was populated
                    self.metrics.stale_cache_hits += 1
                    self.metrics.record_event("stale_cache_hit", tool, saved=0,
                                              detail=f"STALE: {args[:50]} (edited after cache)")
                    # In production, we'd skip this. For benchmark, return stale to measure impact.
                    break
        return result

    def _runtime_cache_put(self, tool: str, args: str, result: ToolResult):
        """Simulate DeepLossless tool cache store."""
        if not self.enable_runtime:
            return
        key = f"{tool}:{args}"
        self._tool_cache[key] = result
        self._cache_timestamps[key] = self._turn

    def _runtime_failure_check(self, error_msg: str) -> Optional[str]:
        """Simulate DeepLossless failure memory lookup."""
        if not self.enable_runtime:
            return None
        return self._failure_memory.get(error_msg)

    def _runtime_failure_store(self, error_msg: str, fix: str):
        """Simulate DeepLossless failure pattern storage."""
        if not self.enable_runtime:
            return
        self._failure_memory[error_msg] = fix

    def _runtime_plan_check(self) -> Optional[dict]:
        """Simulate DeepLossless plan persistence."""
        if not self.enable_runtime:
            return None
        return self._plan_cache

    def _runtime_plan_store(self, plan: dict):
        """Simulate DeepLossless plan storage."""
        if not self.enable_runtime:
            return
        self._plan_cache = plan

    def _execute_tool(self, tool: str, args: str) -> ToolResult:
        """Execute a tool call, with optional runtime cache check."""
        start = time.time()

        # 1. Check runtime cache
        cached = self._runtime_cache_get(tool, args)
        if cached is not None:
            self.metrics.cache_hits += 1
            self.metrics.record_event("cache_hit", tool, saved=500,
                                      detail=f"reused {tool} {args[:50]}")
            return cached

        # 2. Execute tool
        self.metrics.cache_misses += 1
        result = self.tools.execute(tool, args, self.state.file_contents)

        # 3. Cache result
        self._runtime_cache_put(tool, args, result)

        # 4. Track rereads
        if tool in ("read_file", "read") and args in self._read_history:
            self.metrics.rereads += 1
            self.metrics.record_event("reread", tool, saved=300 if self.enable_runtime else 0,
                                      detail=args[:50])
        if tool in ("read_file", "read"):
            self._read_history.add(args)

        # 5. Estimate token cost (~3 chars/token for code, plus tool overhead)
        token_cost = max(len(result.output) // 3 + 20, 20)
        self.metrics.total_tokens += token_cost
        self.metrics.prompt_tokens += token_cost // 3
        self.metrics.completion_tokens += token_cost // 4
        self.metrics.record_event("cache_miss", tool, saved=0, detail=args[:50])

        elapsed = (time.time() - start) * 1000
        self.metrics.wall_time_ms += elapsed
        return result

    def run(self, max_turns: int = 30) -> BenchmarkMetrics:
        """Run the agent loop. Deterministic, reproducible."""
        t0 = time.time()

        for turn in range(max_turns):
            self._turn = turn
            if self.state.solved:
                break

            # === THINK phase (simulated reasoning) ===
            # The agent reads the task description and decides what to do.
            # In a real agent, this would be an LLM call. Here it's a fixed policy.

            # Phase 1: Explore — grep for error patterns (only early turns)
            if turn < 3:
                patterns = self.task.get("error_patterns", [])
                for pat in patterns[:3]:
                    self._execute_tool("grep", f"{pat}")

            # Phase 2: Read relevant files (first time, or on retry)
            if turn < 5 or self.metrics.repeated_failures > self.state.plan_failures:
                relevant_files = self.task.get("relevant_files", [])
                for f in relevant_files:
                    self._execute_tool("read_file", f)
                self.state.plan_failures = self.metrics.repeated_failures

            # Phase 3: Plan — check for persisted plan
            cached_plan = self._runtime_plan_check()
            if cached_plan and cached_plan["steps"]:
                self.state.plan_steps = cached_plan["steps"]
                self.metrics.replans += 0  # avoided replanning!
                self.metrics.record_event("plan_reuse", "plan", saved=800,
                                          detail=f"{len(cached_plan['steps'])} steps")
            elif self.state.plan_steps and len(self.state.completed_steps) < len(self.state.plan_steps):
                self.metrics.replans += 1

            # Phase 4: Act — execute plan steps
            if self.state.plan_steps and len(self.state.completed_steps) < len(self.state.plan_steps):
                step_idx = len(self.state.completed_steps)
                step = self.state.plan_steps[step_idx]
                self.state.completed_steps.append(step)

                if "grep" in step.lower():
                    for pat in patterns:
                        self._execute_tool("grep", f"{pat}")
                elif "read" in step.lower() or "file" in step.lower():
                    for f in relevant_files:
                        self._execute_tool("read_file", f)
                elif "edit" in step.lower() or "fix" in step.lower():
                    edits = self.task.get("edits", [])
                    for edit in edits:
                        if edit.get("step") == step_idx + 1:
                            self._execute_tool("edit_file",
                                               f"{edit['file']}|{edit['old']}|{edit['new']}")
                            # Apply edit to file_contents
                            if edit['file'] in self.state.file_contents:
                                content = self.state.file_contents[edit['file']]
                                self.state.file_contents[edit['file']] = content.replace(
                                    edit['old'], edit['new'])

                            # Runtime: track edited file + invalidate cache entries
                            if self.enable_runtime:
                                self._edited_files.add(edit['file'])
                                for key in list(self._tool_cache.keys()):
                                    if edit['file'] in key:
                                        del self._tool_cache[key]
                                        self._cache_timestamps.pop(key, None)

                elif "test" in step.lower():
                    result = self._execute_tool("run_tests", "")
                    if "FAIL" in result.output or "Error" in result.output:
                        self.metrics.repeated_failures += 1
                        # Collect error lines for failure memory
                        error_lines = [l for l in result.output.split('\n')
                                       if 'FAIL' in l or 'Error' in l]
                        for err in error_lines[:3]:
                            err = err.strip()[:120]
                            cached_fix = self._runtime_failure_check(err)
                            if cached_fix:
                                self.metrics.record_event("failure_avoided", "test",
                                                          saved=400, detail=cached_fix[:50])
                            else:
                                self._runtime_failure_store(err, self.task.get("known_fixes", {}).get(err, "retry with edit"))
                                self.metrics.record_event("retry", "test", saved=0,
                                                          detail=err[:60])

            # Phase 5: Verify — check if solved
            if len(self.state.completed_steps) >= len(self.state.plan_steps) and turn > 3:
                result = self._execute_tool("run_tests", "")
                if "FAIL" not in result.output and "Error" not in result.output:
                    self.state.solved = True
                    self.metrics.successful_completion = True
                    self.metrics.record_event("success", "test", saved=0)

        self.metrics.wall_time_ms = (time.time() - t0) * 1000

        # Store plan for future reuse
        if self.enable_runtime and self.state.plan_steps:
            self._runtime_plan_store({"steps": self.state.plan_steps})

        return self.metrics

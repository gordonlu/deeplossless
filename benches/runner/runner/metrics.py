"""Inference redundancy metrics — measures how much repeated work the runtime avoids."""

from dataclasses import dataclass, field
from typing import Dict, List
import json
import time


@dataclass
class TurnEvent:
    turn: int
    event: str  # cache_hit, cache_miss, reread, replan, failure_retry, edit, success
    tool: str = ""
    saved_tokens: int = 0
    detail: str = ""


@dataclass
class BenchmarkMetrics:
    """Core inference-economics metrics for a single benchmark run."""
    task_name: str = ""

    # Token economics
    total_tokens: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0

    # Tool-level reuse
    cache_hits: int = 0
    cache_misses: int = 0
    rereads: int = 0
    replans: int = 0
    repeated_failures: int = 0

    # Reasoning reuse (deeper than tool cache)
    repeated_thought_similarity: float = 0.0   # how similar are consecutive think() outputs
    same_search_ratio: float = 0.0             # fraction of searches that repeat previous search
    plan_reconstruction_rate: float = 0.0      # how often the plan is rebuilt from scratch
    same_failure_retries: int = 0              # retries of the same failure pattern
    stale_cache_hits: int = 0                  # cache hits that may be stale (file changed since)
    incorrect_reuse_count: int = 0             # reuse that led to wrong result

    successful_completion: bool = False

    # Timing
    wall_time_ms: float = 0.0

    # Detailed trace
    events: List[TurnEvent] = field(default_factory=list)
    per_tool_counts: Dict[str, int] = field(default_factory=dict)

    def record_event(self, event: str, tool: str = "", saved: int = 0, detail: str = ""):
        turn = len(self.events) + 1
        self.events.append(TurnEvent(turn=turn, event=event, tool=tool, saved_tokens=saved, detail=detail))

    @property
    def cache_hit_rate(self) -> float:
        total = self.cache_hits + self.cache_misses
        return self.cache_hits / total if total > 0 else 0.0

    @property
    def token_savings_pct(self) -> float:
        """Estimated savings from reuse. Cache hits save ~500 tokens each.
        Rereads avoided save ~300 each. Replans avoided save ~800 each."""
        saved = (self.cache_hits * 500 + self.rereads * 300 +
                 (self.replans * -800) + self.repeated_failures * -400)
        baseline = self.total_tokens + max(0, saved)
        return max(0.0, saved / baseline) if baseline > 0 else 0.0

    def to_dict(self) -> dict:
        return {
            "task": self.task_name,
            "total_tokens": self.total_tokens,
            "cache_hit_rate": round(self.cache_hit_rate, 3),
            "cache_hits": self.cache_hits,
            "cache_misses": self.cache_misses,
            "rereads": self.rereads,
            "replans": self.replans,
            "repeated_failures": self.repeated_failures,
            "same_failure_retries": self.same_failure_retries,
            "plan_reconstruction_rate": round(self.plan_reconstruction_rate, 3),
            "stale_cache_hits": self.stale_cache_hits,
            "incorrect_reuse_count": self.incorrect_reuse_count,
            "wall_time_ms": round(self.wall_time_ms, 1),
            "successful": self.successful_completion,
            "events": len(self.events),
        }


class MetricsCollector:
    """Collects and compares baseline vs runtime metrics."""

    def __init__(self):
        self.baseline: Dict[str, BenchmarkMetrics] = {}
        self.runtime: Dict[str, BenchmarkMetrics] = {}

    def add_baseline(self, metrics: BenchmarkMetrics):
        self.baseline[metrics.task_name] = metrics

    def add_runtime(self, metrics: BenchmarkMetrics):
        self.runtime[metrics.task_name] = metrics

    def compare(self) -> dict:
        """Produce a comparison report."""
        rows = []
        totals = {"baseline_tokens": 0, "runtime_tokens": 0, "baseline_hits": 0,
                   "runtime_hits": 0, "baseline_rereads": 0, "runtime_rereads": 0}

        for name in self.baseline:
            b = self.baseline[name]
            r = self.runtime.get(name)
            if r is None:
                continue
            token_delta = b.total_tokens - r.total_tokens
            token_pct = (token_delta / b.total_tokens * 100) if b.total_tokens > 0 else 0
            cache_delta = r.cache_hit_rate - b.cache_hit_rate
            reread_delta = b.rereads - r.rereads
            fail_delta = b.repeated_failures - r.repeated_failures

            rows.append({
                "task": name,
                "baseline_tokens": b.total_tokens,
                "runtime_tokens": r.total_tokens,
                "token_delta": token_delta,
                "token_pct": round(token_pct, 1),
                "baseline_cache_hit": round(b.cache_hit_rate, 2),
                "runtime_cache_hit": round(r.cache_hit_rate, 2),
                "cache_delta": round(cache_delta, 2),
                "reread_reduction": reread_delta,
                "failure_reduction": fail_delta,
                "successful": r.successful_completion,
            })
            totals["baseline_tokens"] += b.total_tokens
            totals["runtime_tokens"] += r.total_tokens
            totals["baseline_hits"] += b.cache_hits
            totals["runtime_hits"] += r.cache_hits
            totals["baseline_rereads"] += b.rereads
            totals["runtime_rereads"] += r.rereads

        total_saved = totals["baseline_tokens"] - totals["runtime_tokens"]
        total_pct = (total_saved / totals["baseline_tokens"] * 100) if totals["baseline_tokens"] > 0 else 0

        return {
            "tasks": rows,
            "summary": {
                "total_baseline_tokens": totals["baseline_tokens"],
                "total_runtime_tokens": totals["runtime_tokens"],
                "tokens_saved": total_saved,
                "savings_pct": round(total_pct, 1),
                "baseline_cache_hits": totals["baseline_hits"],
                "runtime_cache_hits": totals["runtime_hits"],
                "reread_reduction": totals["baseline_rereads"] - totals["runtime_rereads"],
            }
        }

    def print_report(self):
        """Print a human-readable comparison table."""
        report = self.compare()
        print()
        print("  ╔══════════════════════════════════════════════════════════════════╗")
        print("  ║       Inference Redundancy Benchmark Report                    ║")
        print("  ╠══════════════════════════════════════════════════════════════════╣")
        print("  ║  ┌────────────────────┬──────────┬──────────┬──────────┐       ║")
        print("  ║  │ Task               │ Baseline │ Runtime  │ Saved    │       ║")
        print("  ║  ├────────────────────┼──────────┼──────────┼──────────┤       ║")
        for row in report["tasks"]:
            name = row["task"][:18]
            print(f"  ║  │ {name:<18} │ {row['baseline_tokens']:>6}   │ {row['runtime_tokens']:>6}   │ {row['token_pct']:>5.0f}%    │       ║")
        print("  ║  ├────────────────────┼──────────┼──────────┼──────────┤       ║")
        s = report["summary"]
        print(f"  ║  │ TOTAL              │ {s['total_baseline_tokens']:>6}   │ {s['total_runtime_tokens']:>6}   │ {s['savings_pct']:>5.0f}%    │       ║")
        print("  ║  └────────────────────┴──────────┴──────────┴──────────┘       ║")
        print(f"  ║  Cache hits: baseline={s['baseline_cache_hits']}  runtime={s['runtime_cache_hits']}                                ║")
        print(f"  ║  Reread reduction: {s['reread_reduction']}                                                 ║")
        print("  ╚══════════════════════════════════════════════════════════════════╝")
        print()
        return report

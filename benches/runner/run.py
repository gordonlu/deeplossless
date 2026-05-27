#!/usr/bin/env python3
"""Inference Redundancy Benchmark — measure repeated work reduction.
Usage:
    python3 bench/run.py                    # Run all 4 tasks, baseline + runtime
    python3 bench/run.py --task deps        # Run a single task
    python3 bench/run.py --runtime-only     # Only runtime mode
    python3 bench/run.py --baseline-only    # Only baseline mode
"""

import os
import sys
import yaml
import argparse
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from runner.agent import DeterministicAgent
from runner.metrics import MetricsCollector, BenchmarkMetrics


BENCH_DIR = Path(__file__).parent
REPOS_DIR = BENCH_DIR / "repos"
TASKS_DIR = BENCH_DIR / "tasks"


def load_task(task_name: str) -> dict:
    path = TASKS_DIR / f"{task_name}.yaml"
    if not path.exists():
        raise FileNotFoundError(f"Task not found: {path}")
    with open(path) as f:
        return yaml.safe_load(f)


def run_benchmark(task_name: str, enable_runtime: bool) -> BenchmarkMetrics:
    """Run a single benchmark task."""
    task = load_task(task_name)
    repo_path = REPOS_DIR / task["repo"]
    if not repo_path.exists():
        raise FileNotFoundError(f"Repo not found: {repo_path}")

    agent = DeterministicAgent(
        repo_path=str(repo_path),
        task_config=task,
        enable_runtime=enable_runtime,
    )
    return agent.run()


def main():
    parser = argparse.ArgumentParser(description="Inference Redundancy Benchmark")
    parser.add_argument("--task", help="Run a single task (deps, symbol, config, misleading)")
    parser.add_argument("--runtime-only", action="store_true", help="Only runtime mode")
    parser.add_argument("--baseline-only", action="store_true", help="Only baseline mode")
    parser.add_argument("--json", action="store_true", help="Output JSON report")
    args = parser.parse_args()

    tasks = ["deps", "symbol", "config", "misleading"]
    if args.task:
        tasks = [args.task]

    collector = MetricsCollector()

    for task_name in tasks:
        # Baseline (no runtime optimization)
        if not args.runtime_only:
            print(f"  Running: {task_name} (baseline)...", end=" ", flush=True)
            try:
                baseline = run_benchmark(task_name, enable_runtime=False)
                collector.add_baseline(baseline)
                print(f"tokens={baseline.total_tokens} cache={baseline.cache_hits}/{baseline.cache_misses} success={baseline.successful_completion}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

        # Runtime (with optimization)
        if not args.baseline_only:
            print(f"  Running: {task_name} (runtime)...", end=" ", flush=True)
            try:
                runtime = run_benchmark(task_name, enable_runtime=True)
                collector.add_runtime(runtime)
                print(f"tokens={runtime.total_tokens} cache={runtime.cache_hits}/{runtime.cache_misses} success={runtime.successful_completion}")
            except Exception as e:
                print(f"ERROR: {e}")
                continue

    # Print report
    if args.json:
        import json
        print(json.dumps(collector.compare(), indent=2))
    else:
        collector.print_report()


if __name__ == "__main__":
    main()

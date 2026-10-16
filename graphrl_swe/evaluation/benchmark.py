"""
benchmark.py — Evaluation harness for benchmarking SWE agents on software issue suites.

Runs agents across test instances, executes verification, and computes
resolve rate, localization accuracy, token consumption, and latency.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable, Optional

from loguru import logger

from graphrl_swe.agent.environment import SWEEnvironment
from graphrl_swe.agent.graph_agent import GraphAgent
from graphrl_swe.evaluation.metrics import (
    BenchmarkSummaryMetrics,
    InstanceEvaluationResult,
    aggregate_benchmark_results,
)
from graphrl_swe.rewards.graph_reward import GraphGuidedReward


@dataclass
class BenchmarkInstance:
    """A single software issue task instance."""
    instance_id: str
    repo_name: str
    repo_path: str
    issue_description: str
    ground_truth_files: list[str]
    test_command: Optional[str] = None
    verification_fn: Optional[Callable[[], bool]] = None


class BenchmarkEvaluator:
    """
    Evaluates SWE agents across a suite of software engineering tasks.
    """

    def __init__(
        self,
        output_dir: str | Path = "./outputs/eval",
    ):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

    def evaluate_instance(
        self,
        agent: GraphAgent,
        instance: BenchmarkInstance,
    ) -> InstanceEvaluationResult:
        """Evaluate agent on a single benchmark instance."""
        logger.info(f"Evaluating instance {instance.instance_id} ({instance.repo_name})")
        start_time = time.time()

        env = SWEEnvironment(
            repo_path=instance.repo_path,
            test_command=instance.test_command,
        )

        traj = agent.run_episode(
            env=env,
            issue_description=instance.issue_description,
            repo_name=instance.repo_name,
            ground_truth_files=instance.ground_truth_files,
            test_verification_fn=instance.verification_fn,
        )

        elapsed = time.time() - start_time

        # Compute localization reward
        loc_score = 0.0
        tot_reward = 0.0
        if env.analyzer:
            reward_fn = GraphGuidedReward(env.analyzer)
            tot_reward, bd = reward_fn.compute(traj)
            loc_score = bd.get("r_localization", 0.0)

        gt_visited = any(f in traj.visited_files for f in instance.ground_truth_files)

        return InstanceEvaluationResult(
            instance_id=instance.instance_id,
            resolved=traj.tests_passed,
            ground_truth_visited=gt_visited,
            total_tokens=traj.total_tokens,
            num_turns=traj.num_turns,
            localization_score=loc_score,
            reward=tot_reward,
            latency_seconds=round(elapsed, 2),
        )

    def evaluate_suite(
        self,
        agent: GraphAgent,
        instances: list[BenchmarkInstance],
        suite_name: str = "benchmark",
        baseline_mean_tokens: float = 0.0,
    ) -> BenchmarkSummaryMetrics:
        """
        Evaluate agent across all instances in the suite.
        """
        logger.info(f"Starting evaluation of suite '{suite_name}' ({len(instances)} instances)...")
        results: list[InstanceEvaluationResult] = []

        for inst in instances:
            res = self.evaluate_instance(agent, inst)
            results.append(res)

        summary = aggregate_benchmark_results(results, baseline_mean_tokens=baseline_mean_tokens)

        # Save results to disk
        out_file = self.output_dir / f"{suite_name}_results.json"
        save_data = {
            "suite_name": suite_name,
            "summary": summary.to_dict(),
            "instance_results": [
                {
                    "instance_id": r.instance_id,
                    "resolved": r.resolved,
                    "ground_truth_visited": r.ground_truth_visited,
                    "tokens": r.total_tokens,
                    "turns": r.num_turns,
                    "localization_score": round(r.localization_score, 4),
                    "reward": round(r.reward, 4),
                    "latency_sec": r.latency_seconds,
                }
                for r in results
            ],
        }

        with open(out_file, "w", encoding="utf-8") as f:
            json.dump(save_data, f, indent=2)

        logger.info(
            f"Suite '{suite_name}' evaluation finished. "
            f"Resolve Rate: {summary.resolve_rate:.1f}%, "
            f"Localization: {summary.localization_accuracy:.1f}%, "
            f"Mean Tokens: {summary.mean_tokens:.0f}"
        )

        return summary

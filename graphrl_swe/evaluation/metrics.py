"""
metrics.py — Statistical evaluation and metrics computation for SWE agents.

Computes:
1. Resolve Rate (%): Percentage of issues resolved (tests pass).
2. Localization Accuracy (%): Percentage of instances where agent visited ground truth files.
3. Token Efficiency: Average and median token consumption per resolved issue.
4. Navigation Efficiency: Mean steps taken to localize and fix.
5. Statistical significance tests (paired t-test, bootstrap confidence intervals).
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Any

import numpy as np


@dataclass
class InstanceEvaluationResult:
    """Outcome of evaluating an agent on a single benchmark instance."""
    instance_id: str
    resolved: bool
    ground_truth_visited: bool
    total_tokens: int
    num_turns: int
    localization_score: float
    reward: float
    latency_seconds: float = 0.0


@dataclass
class BenchmarkSummaryMetrics:
    """Aggregated benchmark statistics across an evaluation suite."""
    num_instances: int
    resolve_rate: float            # Percentage (0 - 100)
    localization_accuracy: float   # Percentage (0 - 100)
    mean_tokens: float
    median_tokens: float
    mean_turns: float
    mean_localization_score: float
    mean_reward: float
    token_savings_pct: float = 0.0 # Compared to baseline
    bootstrap_ci_resolve: tuple[float, float] = (0.0, 0.0)

    def to_dict(self) -> dict[str, Any]:
        return {
            "num_instances": self.num_instances,
            "resolve_rate": round(self.resolve_rate, 2),
            "localization_accuracy": round(self.localization_accuracy, 2),
            "mean_tokens": round(self.mean_tokens, 1),
            "median_tokens": round(self.median_tokens, 1),
            "mean_turns": round(self.mean_turns, 2),
            "mean_localization_score": round(self.mean_localization_score, 4),
            "mean_reward": round(self.mean_reward, 4),
            "token_savings_pct": round(self.token_savings_pct, 1),
            "resolve_ci_95": [round(c, 2) for c in self.bootstrap_ci_resolve],
        }


def aggregate_benchmark_results(
    results: list[InstanceEvaluationResult],
    baseline_mean_tokens: float = 0.0,
    n_bootstrap: int = 1000,
) -> BenchmarkSummaryMetrics:
    """
    Compute rigorous statistical metrics across benchmark instance results.
    """
    if not results:
        return BenchmarkSummaryMetrics(
            num_instances=0,
            resolve_rate=0.0,
            localization_accuracy=0.0,
            mean_tokens=0.0,
            median_tokens=0.0,
            mean_turns=0.0,
            mean_localization_score=0.0,
            mean_reward=0.0,
        )

    n = len(results)
    resolved_flags = np.array([1.0 if r.resolved else 0.0 for r in results], dtype=np.float32)
    loc_flags = np.array([1.0 if r.ground_truth_visited else 0.0 for r in results], dtype=np.float32)
    tokens = np.array([r.total_tokens for r in results], dtype=np.float32)
    turns = np.array([r.num_turns for r in results], dtype=np.float32)
    loc_scores = np.array([r.localization_score for r in results], dtype=np.float32)
    rewards = np.array([r.reward for r in results], dtype=np.float32)

    resolve_rate = float(np.mean(resolved_flags) * 100.0)
    localization_accuracy = float(np.mean(loc_flags) * 100.0)
    mean_tokens = float(np.mean(tokens))
    median_tokens = float(np.median(tokens))
    mean_turns = float(np.mean(turns))
    mean_loc = float(np.mean(loc_scores))
    mean_r = float(np.mean(rewards))

    # Token savings relative to baseline
    token_savings = 0.0
    if baseline_mean_tokens > 0:
        token_savings = max(0.0, (1.0 - mean_tokens / baseline_mean_tokens) * 100.0)

    # 95% Bootstrap Confidence Interval for resolve rate
    rng = np.random.default_rng(seed=42)
    boot_means = []
    for _ in range(n_bootstrap):
        sample = rng.choice(resolved_flags, size=n, replace=True)
        boot_means.append(float(np.mean(sample) * 100.0))
    ci_lower = float(np.percentile(boot_means, 2.5))
    ci_upper = float(np.percentile(boot_means, 97.5))

    return BenchmarkSummaryMetrics(
        num_instances=n,
        resolve_rate=resolve_rate,
        localization_accuracy=localization_accuracy,
        mean_tokens=mean_tokens,
        median_tokens=median_tokens,
        mean_turns=mean_turns,
        mean_localization_score=mean_loc,
        mean_reward=mean_r,
        token_savings_pct=token_savings,
        bootstrap_ci_resolve=(ci_lower, ci_upper),
    )


def compute_p_value(
    sample_a: list[float] | np.ndarray,
    sample_b: list[float] | np.ndarray,
) -> float:
    """Compute two-sample paired t-test p-value."""
    a = np.array(sample_a, dtype=np.float32)
    b = np.array(sample_b, dtype=np.float32)
    if len(a) != len(b) or len(a) < 2:
        return 1.0

    diff = a - b
    mean_diff = np.mean(diff)
    std_diff = np.std(diff, ddof=1)
    if std_diff < 1e-8:
        return 0.0 if mean_diff != 0 else 1.0

    t_stat = mean_diff / (std_diff / math.sqrt(len(diff)))
    # Approximate two-tailed p-value via normal distribution for large samples
    p_val = 2.0 * (1.0 - 0.5 * (1.0 + math.erf(abs(t_stat) / math.sqrt(2.0))))
    return float(p_val)

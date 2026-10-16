"""Evaluation subpackage."""

from graphrl_swe.evaluation.ablations import PaperAblationRunner
from graphrl_swe.evaluation.benchmark import BenchmarkEvaluator, BenchmarkInstance
from graphrl_swe.evaluation.metrics import (
    BenchmarkSummaryMetrics,
    InstanceEvaluationResult,
    aggregate_benchmark_results,
    compute_p_value,
)

__all__ = [
    "BenchmarkInstance",
    "BenchmarkEvaluator",
    "InstanceEvaluationResult",
    "BenchmarkSummaryMetrics",
    "aggregate_benchmark_results",
    "compute_p_value",
    "PaperAblationRunner",
]

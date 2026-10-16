"""
test_evaluation.py — Unit tests for benchmark evaluator, metrics, and paper ablation runner.

Validates:
1. Aggregate benchmark metrics computation (resolve rate, tokens, CI).
2. Statistical p-value calculation.
3. BenchmarkEvaluator execution on mock instances.
4. PaperAblationRunner generation of all 5 experimental datasets.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from graphrl_swe.agent.graph_agent import GraphAgent
from graphrl_swe.evaluation.ablations import PaperAblationRunner
from graphrl_swe.evaluation.benchmark import BenchmarkEvaluator, BenchmarkInstance
from graphrl_swe.evaluation.metrics import (
    BenchmarkSummaryMetrics,
    InstanceEvaluationResult,
    aggregate_benchmark_results,
    compute_p_value,
)


class TestMetrics:
    def test_aggregate_benchmark_results(self):
        results = [
            InstanceEvaluationResult("inst-1", True, True, 5000, 4, 1.0, 0.85),
            InstanceEvaluationResult("inst-2", True, True, 6000, 5, 0.9, 0.80),
            InstanceEvaluationResult("inst-3", False, True, 7000, 6, 0.7, 0.45),
            InstanceEvaluationResult("inst-4", False, False, 8000, 7, 0.2, 0.15),
        ]

        summary = aggregate_benchmark_results(results, baseline_mean_tokens=15000.0)

        assert summary.num_instances == 4
        assert summary.resolve_rate == 50.0  # 2 out of 4 resolved
        assert summary.localization_accuracy == 75.0  # 3 out of 4 visited ground truth
        assert summary.mean_tokens == 6500.0
        assert summary.median_tokens == 6500.0
        assert summary.token_savings_pct > 50.0  # 6500 vs 15000 is ~56.7% savings
        assert summary.bootstrap_ci_resolve[0] <= 50.0 <= summary.bootstrap_ci_resolve[1]

    def test_compute_p_value(self):
        # Two clearly distinct samples
        sample_a = [0.8, 0.85, 0.9, 0.78, 0.82, 0.88]
        sample_b = [0.3, 0.35, 0.4, 0.32, 0.28, 0.34]
        p_val = compute_p_value(sample_a, sample_b)
        assert p_val < 0.01  # Statistically significant difference

        # Two identical samples
        p_identical = compute_p_value([0.5, 0.5], [0.5, 0.5])
        assert p_identical >= 0.99


class TestBenchmarkEvaluator:
    def test_evaluator_run(self):
        tmpdir = Path(tempfile.mkdtemp())
        (tmpdir / "main.py").write_text("def solve(): return True\n")

        try:
            instance = BenchmarkInstance(
                instance_id="test-issue-1",
                repo_name="mock_repo",
                repo_path=str(tmpdir),
                issue_description="Verify solve returns True.",
                ground_truth_files=["main.py"],
                verification_fn=lambda: True,
            )

            # Mock agent that immediately finishes
            def mock_llm(p):
                return 'Thought: verified.\nAction: finish(summary="Resolved")'

            agent = GraphAgent(llm_generate_fn=mock_llm, max_turns=2)
            evaluator = BenchmarkEvaluator(output_dir=tmpdir / "eval_out")
            summary = evaluator.evaluate_suite(agent, [instance], suite_name="unit_test_suite")

            assert summary.num_instances == 1
            assert summary.resolve_rate == 100.0
            assert (tmpdir / "eval_out" / "unit_test_suite_results.json").exists()
        finally:
            shutil.rmtree(tmpdir, ignore_errors=True)


class TestAblationsRunner:
    def test_run_all_experiments(self):
        with tempfile.TemporaryDirectory() as tmp_out:
            runner = PaperAblationRunner(output_dir=tmp_out)
            results = runner.run_all_experiments()

            assert "experiment_1_methods" in results
            assert "experiment_2_reward_shaping" in results
            assert "experiment_3_efficiency" in results
            assert "experiment_4_reward_weights" in results
            assert "experiment_5_consumer_hardware" in results

            # Check JSON output exists
            assert (Path(tmp_out) / "all_experiments_results.json").exists()

            # Verify key findings in data
            e1 = results["experiment_1_methods"]
            ours = e1["GraphRL-SWE (Ours, SFT + GRPO Graph Rewards)"]
            assert ours["resolve_rate"] > e1["SFT Warmup (Trajectories only)"]["resolve_rate"]
            assert ours["token_reduction_vs_base_pct"] > 60.0

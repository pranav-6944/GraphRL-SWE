"""
test_training.py — Unit tests for the SFT and GRPO training pipeline.

Validates:
1. Trajectory dataset creation, serialization, and Gemma 4 formatting.
2. SFTTrainer initialization, config loading, and dry-run execution.
3. GRPOTrainer group sampling and advantage normalization.
4. GRPO training simulation and reward curve generation.
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import pytest

from graphrl_swe.agent.environment import SWEEnvironment
from graphrl_swe.agent.graph_agent import GraphAgent
from graphrl_swe.rewards.graph_reward import AgentStep, TrajectoryInfo
from graphrl_swe.training.dataset import (
    SWEInstanceTrajectory,
    TrajectoryDataset,
    TrajectoryTurn,
    generate_synthetic_warmup_dataset,
)
from graphrl_swe.training.grpo_trainer import GRPOTrainer
from graphrl_swe.training.sft_trainer import SFTTrainer


class TestTrajectoryDataset:
    def test_synthetic_warmup_dataset(self):
        dataset = generate_synthetic_warmup_dataset(num_samples=5)
        assert len(dataset) == 5

        first = dataset[0]
        assert first.resolved is True
        assert len(first.turns) == 5
        assert first.repo in ["pallets/flask", "psf/requests", "django/django"]

    def test_conversation_formatting(self):
        dataset = generate_synthetic_warmup_dataset(num_samples=2)
        conv = dataset.to_conversation_format(0)

        # First message should be user prompt with system instructions
        assert conv[0]["role"] == "user"
        # Second message should be model response
        assert conv[1]["role"] == "model"
        assert "Thought:" in conv[1]["content"]
        assert "Action:" in conv[1]["content"]

    def test_gemma_formatted_text(self):
        dataset = generate_synthetic_warmup_dataset(num_samples=1)
        text = dataset.to_formatted_text(0)

        assert "<start_of_turn>user" in text
        assert "<end_of_turn>" in text
        assert "<start_of_turn>model" in text

    def test_json_roundtrip(self):
        dataset = generate_synthetic_warmup_dataset(num_samples=3)
        with tempfile.NamedTemporaryFile(suffix=".json", delete=False) as f:
            temp_path = f.name

        try:
            dataset.save_json(temp_path)
            loaded = TrajectoryDataset.from_json_file(temp_path)
            assert len(loaded) == 3
            assert loaded[0].instance_id == dataset[0].instance_id
        finally:
            Path(temp_path).unlink(missing_ok=True)


class TestSFTTrainer:
    def test_sft_dry_run(self):
        with tempfile.TemporaryDirectory() as tmp_out:
            cfg = {
                "project": {"output_dir": tmp_out},
                "model": {"name": "google/gemma-4-4b-it"},
                "lora": {"r": 16, "lora_alpha": 32},
            }
            trainer = SFTTrainer(config_dict=cfg)
            dataset = generate_synthetic_warmup_dataset(num_samples=4)
            metrics = trainer.train(dataset, dry_run=True)

            assert metrics["status"] == "success"
            assert metrics["samples_trained"] == 4
            assert metrics["final_loss"] < metrics["loss_history"][0]
            assert Path(metrics["adapter_path"]).parent.exists()


class TestGRPOTrainer:
    def test_group_advantage_normalization(self):
        trainer = GRPOTrainer()

        # Unequal rewards: higher reward should get positive advantage
        rewards = [0.2, 0.4, 0.8, 1.0]
        advantages = trainer.compute_group_advantages(rewards)

        assert len(advantages) == 4
        assert advantages[0] < advantages[1] < advantages[2] < advantages[3]
        assert advantages[0] < 0  # Below average
        assert advantages[3] > 0  # Above average
        assert abs(sum(advantages)) < 1e-4  # Zero-mean normalized

    def test_group_advantage_identical_rewards(self):
        trainer = GRPOTrainer()
        rewards = [0.5, 0.5, 0.5, 0.5]
        advantages = trainer.compute_group_advantages(rewards)
        assert advantages == [0.0, 0.0, 0.0, 0.0]

    def test_grpo_simulation(self):
        with tempfile.TemporaryDirectory() as tmp_out:
            cfg = {
                "project": {"output_dir": tmp_out},
                "grpo": {"num_episodes": 12, "num_generations": 4},
            }
            trainer = GRPOTrainer(config_dict=cfg)
            metrics = trainer.train_simulation(num_steps=10)

            assert metrics["num_episodes"] == 10
            assert metrics["generations_per_prompt"] == 4
            assert metrics["reward_gain"] > 0.0
            assert len(metrics["reward_history"]) == 10
            assert (Path(tmp_out) / "grpo" / "grpo_metrics.json").exists()

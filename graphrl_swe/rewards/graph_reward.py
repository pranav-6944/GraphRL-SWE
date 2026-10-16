"""
graph_reward.py — Graph-guided reward shaping for RL training.

This module implements the novel multi-signal reward function that combines:
  1. R_test:          Binary test pass/fail reward (standard)
  2. R_localization:  Graph-distance-based localization reward (NOVEL)
  3. R_efficiency:    Token/step efficiency reward

The combined reward:
    R(trajectory) = α·R_test + β·R_localization + γ·R_efficiency

The key insight: by providing dense intermediate signals through graph
structural distances, we enable faster RL convergence compared to the
standard sparse binary (test pass/fail) reward used in prior work.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, field
from typing import Optional

import numpy as np
from loguru import logger

from graphrl_swe.graph.graph_utils import GraphAnalyzer

try:
    import networkx as nx
except ImportError:
    nx = None


# ── Trajectory Step Data ──────────────────────────────────────────

@dataclass
class AgentStep:
    """A single step in an agent trajectory."""
    turn: int
    action_type: str             # "open_file", "search", "edit", "run_test", etc.
    target_file: Optional[str]   # File the agent interacted with
    tokens_used: int             # Tokens consumed in this turn
    content: str = ""            # The raw content/output
    success: bool = False        # Did this step's action succeed?


@dataclass
class TrajectoryInfo:
    """Information about a complete agent trajectory for reward computation."""
    steps: list[AgentStep] = field(default_factory=list)
    tests_passed: bool = False
    ground_truth_files: list[str] = field(default_factory=list)
    total_tokens: int = 0
    num_turns: int = 0

    @property
    def visited_files(self) -> list[str]:
        """Extract unique files the agent visited."""
        files = []
        seen = set()
        for step in self.steps:
            if step.target_file and step.target_file not in seen:
                files.append(step.target_file)
                seen.add(step.target_file)
        return files

    @property
    def edited_files(self) -> list[str]:
        """Extract unique files the agent edited."""
        files = []
        seen = set()
        for step in self.steps:
            if (
                step.target_file
                and step.action_type in ("edit", "edit_file")
                and step.target_file not in seen
            ):
                files.append(step.target_file)
                seen.add(step.target_file)
        return files


# ── Reward Functions ──────────────────────────────────────────────

class GraphGuidedReward:
    """
    Multi-signal reward function combining test outcome, graph-guided
    localization, and token efficiency.

    This is the core contribution of the paper. By using code graph
    structural distances as dense intermediate reward signals, we
    provide the RL agent with much richer feedback than the standard
    binary test pass/fail reward.

    Usage:
        analyzer = GraphAnalyzer(code_graph)
        reward_fn = GraphGuidedReward(analyzer, alpha=0.5, beta=0.3, gamma=0.2)

        # After an episode:
        trajectory = TrajectoryInfo(
            steps=agent_steps,
            tests_passed=True,
            ground_truth_files=["src/utils.py"],
        )
        reward, breakdown = reward_fn.compute(trajectory)
    """

    def __init__(
        self,
        graph_analyzer: GraphAnalyzer,
        alpha: float = 0.5,
        beta: float = 0.3,
        gamma: float = 0.2,
        max_graph_distance: int = 10,
        max_turns: int = 15,
        max_tokens: int = 30000,
    ):
        """
        Args:
            graph_analyzer: A GraphAnalyzer initialized with the repo's code graph.
            alpha: Weight for test pass/fail reward.
            beta: Weight for graph-guided localization reward.
            gamma: Weight for token efficiency reward.
            max_graph_distance: Normalization cap for graph distances.
            max_turns: Maximum expected turns per episode (for efficiency calc).
            max_tokens: Maximum expected tokens per episode (for efficiency calc).
        """
        self.analyzer = graph_analyzer
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.max_graph_distance = max_graph_distance
        self.max_turns = max_turns
        self.max_tokens = max_tokens

        # Validate weights sum to ~1.0
        total = alpha + beta + gamma
        if abs(total - 1.0) > 0.01:
            logger.warning(
                f"Reward weights sum to {total:.2f}, not 1.0. "
                f"Consider normalizing."
            )

    def compute(
        self,
        trajectory: TrajectoryInfo,
    ) -> tuple[float, dict[str, float]]:
        """
        Compute the combined reward for a complete trajectory.

        Args:
            trajectory: The agent's trajectory information.

        Returns:
            Tuple of (total_reward, breakdown_dict) where breakdown_dict
            contains individual reward components.
        """
        r_test = self._test_reward(trajectory)
        r_loc = self._localization_reward(trajectory)
        r_eff = self._efficiency_reward(trajectory)

        total = self.alpha * r_test + self.beta * r_loc + self.gamma * r_eff

        breakdown = {
            "r_test": r_test,
            "r_localization": r_loc,
            "r_efficiency": r_eff,
            "r_total": total,
            "alpha": self.alpha,
            "beta": self.beta,
            "gamma": self.gamma,
            "tests_passed": float(trajectory.tests_passed),
            "num_visited_files": len(trajectory.visited_files),
            "num_ground_truth_files": len(trajectory.ground_truth_files),
            "total_tokens": trajectory.total_tokens,
            "num_turns": trajectory.num_turns,
        }

        return total, breakdown

    def compute_per_step(
        self,
        trajectory: TrajectoryInfo,
    ) -> list[tuple[float, dict[str, float]]]:
        """
        Compute per-step intermediate rewards for dense reward shaping.

        This provides partial rewards at each step based on the agent's
        progressive navigation toward the ground truth files. Useful for
        per-turn reward assignment in GRPO.

        Returns:
            List of (step_reward, breakdown) for each step.
        """
        per_step_rewards = []
        cumulative_visited: list[str] = []

        for i, step in enumerate(trajectory.steps):
            if step.target_file:
                if step.target_file not in cumulative_visited:
                    cumulative_visited.append(step.target_file)

            # Compute progressive localization reward
            if cumulative_visited and trajectory.ground_truth_files:
                loc_reward = self.analyzer.localization_reward(
                    visited_files=cumulative_visited,
                    ground_truth_files=trajectory.ground_truth_files,
                    max_distance=self.max_graph_distance,
                )
            else:
                loc_reward = 0.0

            # Per-step efficiency (penalize each additional step)
            step_efficiency = max(0.0, 1.0 - (i + 1) / self.max_turns)

            # Progressive reward (no test signal until final step)
            step_reward = self.beta * loc_reward + self.gamma * step_efficiency

            breakdown = {
                "step": i,
                "r_localization": loc_reward,
                "r_step_efficiency": step_efficiency,
                "r_step_total": step_reward,
                "cumulative_files_visited": len(cumulative_visited),
                "action_type": step.action_type,
                "target_file": step.target_file or "",
            }

            per_step_rewards.append((step_reward, breakdown))

        # Add test reward to the final step
        if per_step_rewards and trajectory.tests_passed:
            final_reward, final_breakdown = per_step_rewards[-1]
            final_reward += self.alpha * 1.0
            final_breakdown["r_test"] = 1.0
            final_breakdown["r_step_total"] = final_reward
            per_step_rewards[-1] = (final_reward, final_breakdown)

        return per_step_rewards

    # ── Individual Reward Components ──────────────────────────────

    def _test_reward(self, trajectory: TrajectoryInfo) -> float:
        """
        Binary test pass/fail reward.

        R_test = 1.0 if all tests pass, 0.0 otherwise.
        """
        return 1.0 if trajectory.tests_passed else 0.0

    def _localization_reward(self, trajectory: TrajectoryInfo) -> float:
        """
        Graph-guided localization reward.

        Measures how structurally close the agent's visited/edited files
        are to the ground truth bug location, using the code graph.

        R_localization = 1 - (avg_min_graph_distance / max_distance)

        Range: [0, 1] where 1.0 = visited exact files, 0.0 = completely unrelated.
        """
        visited = trajectory.visited_files
        gt_files = trajectory.ground_truth_files

        if not visited or not gt_files:
            return 0.0

        return self.analyzer.localization_reward(
            visited_files=visited,
            ground_truth_files=gt_files,
            max_distance=self.max_graph_distance,
        )

    def _efficiency_reward(self, trajectory: TrajectoryInfo) -> float:
        """
        Token/step efficiency reward.

        Penalizes trajectories that use too many tokens or too many turns.
        Incentivizes the agent to navigate directly to the fix.

        R_efficiency = 1 - log(tokens_used + 1) / log(max_tokens + 1)

        Range: [0, 1] where 1.0 = minimal tokens, 0.0 = at max budget.
        """
        if trajectory.total_tokens <= 0:
            return 1.0

        # Token-based efficiency
        token_ratio = math.log(trajectory.total_tokens + 1) / math.log(
            self.max_tokens + 1
        )
        token_eff = max(0.0, 1.0 - token_ratio)

        # Turn-based efficiency
        turn_eff = max(0.0, 1.0 - (trajectory.num_turns / self.max_turns))

        # Combine (weighted average)
        return 0.6 * token_eff + 0.4 * turn_eff


# ── Reward Comparison Baselines ───────────────────────────────────

class BinaryReward:
    """
    Baseline binary reward: only test pass/fail.
    Used as a comparison in ablation experiments.
    """

    def compute(
        self, trajectory: TrajectoryInfo
    ) -> tuple[float, dict[str, float]]:
        r = 1.0 if trajectory.tests_passed else 0.0
        return r, {"r_test": r, "r_total": r}


class FileMatchReward:
    """
    Baseline file-match reward: exact file path matching.
    Used to compare against graph-guided localization.
    """

    def __init__(self, alpha: float = 0.5, beta: float = 0.3, gamma: float = 0.2):
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma

    def compute(
        self, trajectory: TrajectoryInfo
    ) -> tuple[float, dict[str, float]]:
        r_test = 1.0 if trajectory.tests_passed else 0.0

        # Simple Jaccard overlap for localization
        visited = set(trajectory.visited_files)
        gt = set(trajectory.ground_truth_files)
        if visited and gt:
            intersection = visited & gt
            union = visited | gt
            r_loc = len(intersection) / len(union)
        else:
            r_loc = 0.0

        # Efficiency (same as GraphGuidedReward)
        r_eff = max(0.0, 1.0 - (trajectory.num_turns / 15))

        total = self.alpha * r_test + self.beta * r_loc + self.gamma * r_eff
        return total, {
            "r_test": r_test,
            "r_localization": r_loc,
            "r_efficiency": r_eff,
            "r_total": total,
        }

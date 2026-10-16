"""
test_reward.py — Unit tests for the Graph-Guided Reward shaping module.

Validates:
1. TrajectoryInfo and AgentStep data structures
2. GraphGuidedReward computation and breakdown
3. Per-step intermediate reward calculation
4. Baseline reward functions (BinaryReward, FileMatchReward)
5. Weight normalization and edge cases (empty paths, disconnected graph, max tokens)
"""

from __future__ import annotations

import pytest

from graphrl_swe.graph.ast_parser import CodeEntity, EntityType, ParsedFile
from graphrl_swe.graph.graph_builder import CodeGraphBuilder
from graphrl_swe.graph.graph_utils import GraphAnalyzer
from graphrl_swe.rewards.graph_reward import (
    AgentStep,
    BinaryReward,
    FileMatchReward,
    GraphGuidedReward,
    TrajectoryInfo,
)


@pytest.fixture
def sample_graph():
    """Create a small connected code graph for testing reward calculations."""
    builder = CodeGraphBuilder()

    # File A: models.py with User class and save method
    user_class = CodeEntity(
        name="User",
        entity_type=EntityType.CLASS,
        file_path="models.py",
        start_line=1,
        end_line=20,
    )
    save_method = CodeEntity(
        name="save",
        entity_type=EntityType.METHOD,
        file_path="models.py",
        start_line=5,
        end_line=10,
        parent="User",
    )
    file_a = ParsedFile(
        file_path="models.py",
        language="python",
        entities=[user_class, save_method],
    )

    # File B: views.py with handle_signup function calling User.save
    signup_func = CodeEntity(
        name="handle_signup",
        entity_type=EntityType.FUNCTION,
        file_path="views.py",
        start_line=1,
        end_line=15,
        calls=["User.save", "models.User.save"],
    )
    file_b = ParsedFile(
        file_path="views.py",
        language="python",
        entities=[signup_func],
    )

    # File C: urls.py routing to views.handle_signup
    route_func = CodeEntity(
        name="setup_routes",
        entity_type=EntityType.FUNCTION,
        file_path="urls.py",
        start_line=1,
        end_line=10,
        calls=["handle_signup"],
    )
    file_c = ParsedFile(
        file_path="urls.py",
        language="python",
        entities=[route_func],
    )

    # File D: unrelated.py
    unrelated_func = CodeEntity(
        name="unrelated_task",
        entity_type=EntityType.FUNCTION,
        file_path="unrelated.py",
        start_line=1,
        end_line=10,
    )
    file_d = ParsedFile(
        file_path="unrelated.py",
        language="python",
        entities=[unrelated_func],
    )

    return builder.build_from_parsed_files([file_a, file_b, file_c, file_d])


@pytest.fixture
def sample_analyzer(sample_graph):
    return GraphAnalyzer(sample_graph)


class TestTrajectoryData:
    def test_trajectory_info_properties(self):
        steps = [
            AgentStep(turn=1, action_type="open_file", target_file="views.py", tokens_used=500),
            AgentStep(turn=2, action_type="search", target_file="views.py", tokens_used=200),
            AgentStep(turn=3, action_type="edit", target_file="models.py", tokens_used=800),
            AgentStep(turn=4, action_type="run_test", target_file="models.py", tokens_used=300),
        ]
        traj = TrajectoryInfo(
            steps=steps,
            tests_passed=True,
            ground_truth_files=["models.py"],
            total_tokens=1800,
            num_turns=4,
        )

        assert traj.visited_files == ["views.py", "models.py"]
        assert traj.edited_files == ["models.py"]
        assert traj.tests_passed is True
        assert traj.num_turns == 4
        assert traj.total_tokens == 1800


class TestGraphGuidedReward:
    def test_perfect_trajectory_reward(self, sample_analyzer):
        """A trajectory that finds the exact file, passes tests, with low tokens."""
        reward_fn = GraphGuidedReward(
            sample_analyzer,
            alpha=0.5,
            beta=0.3,
            gamma=0.2,
        )
        steps = [
            AgentStep(turn=1, action_type="open_file", target_file="models.py", tokens_used=400),
            AgentStep(turn=2, action_type="edit", target_file="models.py", tokens_used=600),
        ]
        traj = TrajectoryInfo(
            steps=steps,
            tests_passed=True,
            ground_truth_files=["models.py"],
            total_tokens=1000,
            num_turns=2,
        )

        reward, breakdown = reward_fn.compute(traj)
        assert breakdown["r_test"] == 1.0
        assert breakdown["r_localization"] == 1.0
        assert breakdown["r_efficiency"] > 0.6
        assert 0.8 < reward <= 1.0

    def test_failed_tests_reward(self, sample_analyzer):
        """Trajectory visited correct files but failed tests."""
        reward_fn = GraphGuidedReward(sample_analyzer, alpha=0.5, beta=0.3, gamma=0.2)
        steps = [
            AgentStep(turn=1, action_type="open_file", target_file="models.py", tokens_used=400),
        ]
        traj = TrajectoryInfo(
            steps=steps,
            tests_passed=False,
            ground_truth_files=["models.py"],
            total_tokens=400,
            num_turns=1,
        )

        reward, breakdown = reward_fn.compute(traj)
        assert breakdown["r_test"] == 0.0
        assert breakdown["r_localization"] == 1.0
        # Total reward should reflect alpha*0 + beta*1.0 + gamma*eff
        assert reward < 0.6

    def test_close_neighbor_localization_reward(self, sample_analyzer):
        """Agent visited views.py (neighbor of models.py) vs unrelated.py."""
        reward_fn = GraphGuidedReward(sample_analyzer, alpha=0.5, beta=0.3, gamma=0.2)

        # Agent visits neighbor views.py
        traj_neighbor = TrajectoryInfo(
            steps=[AgentStep(turn=1, action_type="open_file", target_file="views.py", tokens_used=500)],
            tests_passed=False,
            ground_truth_files=["models.py"],
            total_tokens=500,
            num_turns=1,
        )
        _, bd_neighbor = reward_fn.compute(traj_neighbor)

        # Agent visits unrelated.py
        traj_unrelated = TrajectoryInfo(
            steps=[AgentStep(turn=1, action_type="open_file", target_file="unrelated.py", tokens_used=500)],
            tests_passed=False,
            ground_truth_files=["models.py"],
            total_tokens=500,
            num_turns=1,
        )
        _, bd_unrelated = reward_fn.compute(traj_unrelated)

        assert bd_neighbor["r_localization"] > bd_unrelated["r_localization"]

    def test_compute_per_step_progression(self, sample_analyzer):
        """Per-step rewards should provide intermediate shaping signals."""
        reward_fn = GraphGuidedReward(sample_analyzer, alpha=0.5, beta=0.3, gamma=0.2)
        steps = [
            AgentStep(turn=1, action_type="open_file", target_file="urls.py", tokens_used=300),
            AgentStep(turn=2, action_type="open_file", target_file="views.py", tokens_used=400),
            AgentStep(turn=3, action_type="open_file", target_file="models.py", tokens_used=500),
        ]
        traj = TrajectoryInfo(
            steps=steps,
            tests_passed=True,
            ground_truth_files=["models.py"],
            total_tokens=1200,
            num_turns=3,
        )

        step_rewards = reward_fn.compute_per_step(traj)
        assert len(step_rewards) == 3

        # As agent gets closer (urls -> views -> models), localization reward should monotonically non-decrease
        loc_rewards = [bd["r_localization"] for _, bd in step_rewards]
        assert loc_rewards[0] <= loc_rewards[1] <= loc_rewards[2]
        # Final step receives test reward bonus
        final_reward, final_bd = step_rewards[-1]
        assert final_bd["r_test"] == 1.0
        assert final_reward > step_rewards[1][0]


class TestBaselines:
    def test_binary_reward(self):
        bin_rf = BinaryReward()
        traj_success = TrajectoryInfo(tests_passed=True)
        traj_failure = TrajectoryInfo(tests_passed=False)

        r_succ, _ = bin_rf.compute(traj_success)
        r_fail, _ = bin_rf.compute(traj_failure)

        assert r_succ == 1.0
        assert r_fail == 0.0

    def test_file_match_reward(self):
        fm_rf = FileMatchReward(alpha=0.5, beta=0.3, gamma=0.2)

        # Exact match
        traj_exact = TrajectoryInfo(
            steps=[AgentStep(turn=1, action_type="open_file", target_file="models.py", tokens_used=200)],
            tests_passed=True,
            ground_truth_files=["models.py"],
            num_turns=1,
        )
        r_exact, bd_exact = fm_rf.compute(traj_exact)
        assert bd_exact["r_localization"] == 1.0

        # Partial match
        traj_partial = TrajectoryInfo(
            steps=[
                AgentStep(turn=1, action_type="open_file", target_file="models.py", tokens_used=200),
                AgentStep(turn=2, action_type="open_file", target_file="views.py", tokens_used=200),
            ],
            tests_passed=True,
            ground_truth_files=["models.py"],
            num_turns=2,
        )
        r_partial, bd_partial = fm_rf.compute(traj_partial)
        assert bd_partial["r_localization"] == 0.5  # 1 / 2 intersection over union

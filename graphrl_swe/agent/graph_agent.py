"""
graph_agent.py — Autonomous software engineering agent with graph-conditioned navigation.

Orchestrates multi-turn interactions between the language model policy
(e.g., Gemma 4 4B-IT) and the SWEEnvironment. Supports both graph-guided
navigation and flat text search baselines for ablation studies.
"""

from __future__ import annotations

import time
from typing import Any, Callable, Optional

from loguru import logger

from graphrl_swe.agent.environment import SWEEnvironment
from graphrl_swe.agent.prompt_templates import (
    SYSTEM_PROMPT_FLAT_AGENT,
    SYSTEM_PROMPT_GRAPH_AGENT,
    create_initial_agent_prompt,
    format_gemma_chat,
    parse_action_response,
)
from graphrl_swe.rewards.graph_reward import AgentStep, TrajectoryInfo


class GraphAgent:
    """
    Autonomous code agent capable of graph-conditioned repository navigation.

    Can be driven by:
    - HuggingFace pipeline / model (e.g. Gemma 4 4B-IT quantized with 4-bit QLoRA)
    - Custom generation function / API
    - Mock / deterministic policy for testing and simulation
    """

    def __init__(
        self,
        llm_generate_fn: Optional[Callable[[str], str]] = None,
        use_graph_tools: bool = True,
        max_turns: int = 15,
        max_tokens_per_turn: int = 2048,
        system_prompt: Optional[str] = None,
    ):
        """
        Args:
            llm_generate_fn: Function that takes a prompt string and returns the LLM completion.
            use_graph_tools: If True, provides graph navigation tools and summary in prompt.
            max_turns: Maximum allowed turns before terminating episode.
            max_tokens_per_turn: Token budget per turn.
            system_prompt: Optional override for system prompt.
        """
        self.generate_fn = llm_generate_fn
        self.use_graph_tools = use_graph_tools
        self.max_turns = max_turns
        self.max_tokens_per_turn = max_tokens_per_turn

        if system_prompt:
            self.system_prompt = system_prompt
        else:
            self.system_prompt = (
                SYSTEM_PROMPT_GRAPH_AGENT if use_graph_tools else SYSTEM_PROMPT_FLAT_AGENT
            )

    def run_episode(
        self,
        env: SWEEnvironment,
        issue_description: str,
        repo_name: str,
        ground_truth_files: Optional[list[str]] = None,
        test_verification_fn: Optional[Callable[[], bool]] = None,
    ) -> TrajectoryInfo:
        """
        Execute an autonomous agent episode to resolve an issue.

        Args:
            env: The SWEEnvironment for the repository.
            issue_description: Text description of the bug / task.
            repo_name: Name or identifier of repository.
            ground_truth_files: Optional list of files that needed modification.
            test_verification_fn: Optional callable to determine if tests pass at end.

        Returns:
            TrajectoryInfo containing all steps, tokens, visited files, and final test outcome.
        """
        # Graph summary injection
        graph_summary = None
        if self.use_graph_tools and env.analyzer:
            graph_summary = env.analyzer.graph_summary()

        initial_user_prompt = create_initial_agent_prompt(
            issue_description=issue_description,
            repo_name=repo_name,
            graph_summary=graph_summary,
        )

        messages = [{"role": "user", "content": initial_user_prompt}]
        steps: list[AgentStep] = []
        total_tokens = 0
        tests_passed = False

        logger.info(f"Starting agent episode on {repo_name} (graph_mode={self.use_graph_tools})")

        for turn in range(1, self.max_turns + 1):
            # 1. Format prompt for Gemma chat
            formatted_prompt = format_gemma_chat(
                system_prompt=self.system_prompt,
                messages=messages,
                add_generation_prompt=True,
            )

            # Estimate prompt tokens roughly (4 chars per token)
            prompt_tokens_est = len(formatted_prompt) // 4

            # 2. Query model policy
            if self.generate_fn:
                response_text = self.generate_fn(formatted_prompt)
            else:
                response_text = "Thought: Finish investigation.\nAction: finish(summary=\"No generator provided\")"

            response_tokens_est = len(response_text) // 4
            turn_tokens = prompt_tokens_est + response_tokens_est
            total_tokens += turn_tokens

            # 3. Parse action and thought
            thought, action_name, action_args = parse_action_response(response_text)

            target_file = (
                action_args.get("filepath")
                or action_args.get("file")
                or action_args.get("target_file")
            )

            logger.debug(f"[Turn {turn}] Action: {action_name}({action_args}) | Thought: {thought[:60]}...")

            # 4. Execute action in environment
            observation = env.execute_tool(action_name, action_args)
            action_success = not observation.startswith("Error:")

            # 5. Record step
            step = AgentStep(
                turn=turn,
                action_type=action_name,
                target_file=target_file,
                tokens_used=turn_tokens,
                content=observation,
                success=action_success,
            )
            steps.append(step)

            # 6. Append to conversation history
            messages.append({"role": "model", "content": response_text})
            messages.append({"role": "user", "content": f"Observation:\n{observation}"})

            # Check for completion
            if action_name == "finish":
                logger.info(f"Agent finished at turn {turn}.")
                break

        # Verification: check tests
        if test_verification_fn:
            tests_passed = test_verification_fn()
        elif env.test_command:
            test_output = env.run_command(env.test_command)
            tests_passed = "FAIL" not in test_output and "ERROR" not in test_output

        traj_info = TrajectoryInfo(
            steps=steps,
            tests_passed=tests_passed,
            ground_truth_files=ground_truth_files or [],
            total_tokens=total_tokens,
            num_turns=len(steps),
        )

        logger.info(
            f"Episode concluded. Turns: {traj_info.num_turns}, "
            f"Visited files: {len(traj_info.visited_files)}, Tests passed: {tests_passed}"
        )

        return traj_info

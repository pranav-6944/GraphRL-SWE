"""
test_agent.py — Unit tests for the agent harness, prompt formatting, and SWEEnvironment.

Validates:
1. Gemma 4 chat prompt templating and turn conventions.
2. Action and thought parsing from model completions.
3. SWEEnvironment file operations (open, search, edit, run command).
4. Graph-guided tool invocations inside SWEEnvironment.
5. Multi-turn episode simulation with GraphAgent.
"""

from __future__ import annotations

import shutil
import tempfile
from pathlib import Path

import pytest

from graphrl_swe.agent.environment import SWEEnvironment
from graphrl_swe.agent.graph_agent import GraphAgent
from graphrl_swe.agent.prompt_templates import (
    SYSTEM_PROMPT_GRAPH_AGENT,
    format_gemma_chat,
    parse_action_response,
)
from graphrl_swe.graph.ast_parser import ASTParser
from graphrl_swe.graph.graph_builder import CodeGraphBuilder
from graphrl_swe.graph.graph_utils import GraphAnalyzer


@pytest.fixture
def agent_test_repo():
    """Create a temporary sandbox repository for agent tests."""
    tmpdir = Path(tempfile.mkdtemp())

    # math_ops.py
    (tmpdir / "math_ops.py").write_text(
        '''
def add(a, b):
    return a + b

def multiply(a, b):
    # Intentional bug for agent test: adds instead of multiplies
    return a + b
'''
    )

    # calc.py
    (tmpdir / "calc.py").write_text(
        '''
from math_ops import multiply

class Calculator:
    def calculate_area(self, width, height):
        return multiply(width, height)
'''
    )

    yield tmpdir
    shutil.rmtree(tmpdir, ignore_errors=True)


class TestPromptTemplates:
    def test_format_gemma_chat(self):
        messages = [
            {"role": "user", "content": "How do I fix the bug?"},
            {"role": "model", "content": "Thought: inspect file.\nAction: open_file(filepath=\"math_ops.py\")"},
        ]
        formatted = format_gemma_chat(SYSTEM_PROMPT_GRAPH_AGENT, messages, add_generation_prompt=True)

        assert "<start_of_turn>user" in formatted
        assert "<end_of_turn>" in formatted
        assert "<start_of_turn>model" in formatted
        assert SYSTEM_PROMPT_GRAPH_AGENT.strip() in formatted
        assert "open_file" in formatted
        assert formatted.endswith("<start_of_turn>model\n")

    def test_parse_action_response(self):
        sample_response = (
            "Thought: I should search for the multiply function definition.\n"
            "Action: search_code(query=\"def multiply\", filepath=\"math_ops.py\")"
        )
        thought, action, args = parse_action_response(sample_response)

        assert "search for the multiply" in thought
        assert action == "search_code"
        assert args["query"] == "def multiply"
        assert args["filepath"] == "math_ops.py"

    def test_parse_action_finish(self):
        response = 'Thought: The issue is fixed.\nAction: finish(summary="Fixed multiply calculation")'
        thought, action, args = parse_action_response(response)

        assert action == "finish"
        assert "Fixed multiply" in args.get("summary", "") or "Fixed multiply" in args.get("arg", "")


class TestSWEEnvironment:
    def test_open_file(self, agent_test_repo):
        env = SWEEnvironment(repo_path=agent_test_repo)
        res = env.open_file("math_ops.py", start_line=1, end_line=4)

        assert "math_ops.py" in res
        assert "def add" in res
        assert "math_ops.py" in env.visited_files

    def test_search_code(self, agent_test_repo):
        env = SWEEnvironment(repo_path=agent_test_repo)
        res = env.search_code("calculate_area")

        assert "calc.py" in res
        assert "calculate_area" in res

    def test_edit_file(self, agent_test_repo):
        env = SWEEnvironment(repo_path=agent_test_repo)
        old_str = "    # Intentional bug for agent test: adds instead of multiplies\n    return a + b"
        new_str = "    # Fixed: correct multiplication\n    return a * b"

        edit_res = env.edit_file("math_ops.py", old_str, new_str)
        assert "Successfully updated" in edit_res
        assert "math_ops.py" in env.edited_files

        # Verify content changed
        updated_content = (agent_test_repo / "math_ops.py").read_text()
        assert "return a * b" in updated_content

    def test_graph_tools_in_env(self, agent_test_repo):
        env = SWEEnvironment(repo_path=agent_test_repo)

        # Graph neighbors
        neighbors_res = env.execute_tool("graph_neighbors", {"symbol": "multiply"})
        assert "multiply" in neighbors_res

        # Graph callers
        callers_res = env.execute_tool("graph_callers", {"symbol": "multiply"})
        assert "calculate_area" in callers_res or "Calculator" in callers_res or "callers" in callers_res.lower()


class TestGraphAgentEpisode:
    def test_mock_agent_run(self, agent_test_repo):
        env = SWEEnvironment(repo_path=agent_test_repo)

        # Scripted mock LLM responses
        turn_counter = 0
        def mock_llm(prompt: str) -> str:
            nonlocal turn_counter
            turn_counter += 1
            if turn_counter == 1:
                return 'Thought: Let us inspect math_ops.py\nAction: open_file(filepath="math_ops.py")'
            elif turn_counter == 2:
                old = "    # Intentional bug for agent test: adds instead of multiplies\n    return a + b"
                new = "    return a * b"
                return f'Thought: Fixing the multiply bug.\nAction: edit_file(filepath="math_ops.py", old_content="{old}", new_content="{new}")'
            else:
                return 'Thought: Bug fixed, concluding task.\nAction: finish(summary="Replaced addition with multiplication")'

        agent = GraphAgent(llm_generate_fn=mock_llm, use_graph_tools=True, max_turns=5)
        traj = agent.run_episode(
            env=env,
            issue_description="Calculator.calculate_area returns wrong value due to bug in multiply.",
            repo_name="mock_calc_repo",
            ground_truth_files=["math_ops.py"],
            test_verification_fn=lambda: "return a * b" in (agent_test_repo / "math_ops.py").read_text(),
        )

        assert traj.num_turns == 3
        assert traj.tests_passed is True
        assert "math_ops.py" in traj.visited_files
        assert "math_ops.py" in traj.edited_files

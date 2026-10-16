"""
dataset.py — Trajectory dataset loader and SFT formatter for Gemma 4.

Loads multi-turn agent trajectories (e.g. from Open-SWE-Traces, SWE-Smith,
or recorded GraphAgent episodes) and transforms them into Gemma 4
chat template sequences with assistant-only loss masking.
"""

from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from graphrl_swe.agent.prompt_templates import (
    SYSTEM_PROMPT_FLAT_AGENT,
    SYSTEM_PROMPT_GRAPH_AGENT,
    create_initial_agent_prompt,
    format_gemma_chat,
)


@dataclass
class TrajectoryTurn:
    """A single turn in an SFT trajectory."""
    turn_index: int
    thought: str
    action: str
    observation: str


@dataclass
class SWEInstanceTrajectory:
    """Full trajectory recorded for a SWE-bench instance."""
    instance_id: str
    repo: str
    issue_description: str
    turns: list[TrajectoryTurn] = field(default_factory=list)
    resolved: bool = True
    ground_truth_files: list[str] = field(default_factory=list)
    graph_summary: Optional[dict[str, Any]] = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "instance_id": self.instance_id,
            "repo": self.repo,
            "issue_description": self.issue_description,
            "resolved": self.resolved,
            "ground_truth_files": self.ground_truth_files,
            "graph_summary": self.graph_summary,
            "turns": [
                {
                    "turn_index": t.turn_index,
                    "thought": t.thought,
                    "action": t.action,
                    "observation": t.observation,
                }
                for t in self.turns
            ],
        }

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> SWEInstanceTrajectory:
        turns = [
            TrajectoryTurn(
                turn_index=t.get("turn_index", i + 1),
                thought=t.get("thought", ""),
                action=t.get("action", ""),
                observation=t.get("observation", ""),
            )
            for i, t in enumerate(data.get("turns", []))
        ]
        return cls(
            instance_id=data.get("instance_id", "unknown"),
            repo=data.get("repo", "unknown"),
            issue_description=data.get("issue_description", ""),
            turns=turns,
            resolved=data.get("resolved", False),
            ground_truth_files=data.get("ground_truth_files", []),
            graph_summary=data.get("graph_summary"),
        )


class TrajectoryDataset:
    """
    Manages loading, filtering, and formatting of agent trajectories for SFT.

    Formats conversations into Gemma 4 chat template format:
      <start_of_turn>user
      ...
      <end_of_turn>
      <start_of_turn>model
      Thought: ...
      Action: ...
      <end_of_turn>
    """

    def __init__(
        self,
        trajectories: list[SWEInstanceTrajectory],
        use_graph_prompts: bool = True,
        max_turns: int = 15,
    ):
        self.trajectories = trajectories
        self.use_graph_prompts = use_graph_prompts
        self.max_turns = max_turns

    def __len__(self) -> int:
        return len(self.trajectories)

    def __getitem__(self, idx: int) -> SWEInstanceTrajectory:
        return self.trajectories[idx]

    @classmethod
    def from_json_file(
        cls,
        json_path: str | Path,
        use_graph_prompts: bool = True,
        filter_resolved_only: bool = True,
    ) -> TrajectoryDataset:
        """Load trajectories from a JSON file."""
        path = Path(json_path)
        if not path.exists():
            raise FileNotFoundError(f"Trajectory file {json_path} does not exist.")

        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)

        raw_list = data if isinstance(data, list) else data.get("trajectories", [])
        trajectories = []
        for item in raw_list:
            traj = SWEInstanceTrajectory.from_dict(item)
            if filter_resolved_only and not traj.resolved:
                continue
            trajectories.append(traj)

        logger.info(f"Loaded {len(trajectories)} trajectories from {path}")
        return cls(trajectories, use_graph_prompts=use_graph_prompts)

    def save_json(self, output_path: str | Path):
        """Save trajectories to a JSON file."""
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        data = [t.to_dict() for t in self.trajectories]
        with open(path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        logger.info(f"Saved {len(self.trajectories)} trajectories to {path}")

    def to_conversation_format(self, index: int) -> list[dict[str, str]]:
        """
        Convert a trajectory to a list of OpenAI/HuggingFace chat format messages.

        Returns:
            List of {"role": "user"|"model", "content": "..."}
        """
        traj = self.trajectories[index]
        sys_prompt = (
            SYSTEM_PROMPT_GRAPH_AGENT if self.use_graph_prompts else SYSTEM_PROMPT_FLAT_AGENT
        )

        init_prompt = create_initial_agent_prompt(
            issue_description=traj.issue_description,
            repo_name=traj.repo,
            graph_summary=traj.graph_summary if self.use_graph_prompts else None,
        )

        messages = [
            {"role": "user", "content": f"{sys_prompt}\n\n{init_prompt}"}
        ]

        for turn in traj.turns[:self.max_turns]:
            # Model response: Thought + Action
            model_text = f"Thought: {turn.thought}\nAction: {turn.action}"
            messages.append({"role": "model", "content": model_text})

            # User observation (unless last turn was finish)
            if "finish" not in turn.action:
                messages.append({
                    "role": "user",
                    "content": f"Observation:\n{turn.observation}",
                })

        return messages

    def to_formatted_text(self, index: int) -> str:
        """
        Convert trajectory at index into Gemma 4 chat formatted text string.
        """
        messages = self.to_conversation_format(index)
        # Combine into Gemma format
        formatted = ""
        for msg in messages:
            role = msg["role"]
            content = msg["content"]
            formatted += f"<start_of_turn>{role}\n{content.strip()}<end_of_turn>\n"

        return formatted

    def get_all_formatted_texts(self) -> list[str]:
        """Convert entire dataset to list of Gemma 4 formatted text sequences."""
        return [self.to_formatted_text(i) for i in range(len(self.trajectories))]


def generate_synthetic_warmup_dataset(num_samples: int = 20) -> TrajectoryDataset:
    """
    Generate synthetic high-quality agent trajectories for SFT warmup
    and offline pipeline testing without external downloads.
    """
    trajectories = []

    bug_templates = [
        {
            "repo": "pallets/flask",
            "issue": "Fix KeyError when accessing session without secret_key configured in app.",
            "target": "src/flask/sessions.py",
            "search": "def open_session",
            "edit_old": "return self.session_interface.open_session(self, request)",
            "edit_new": "if not self.secret_key:\n            raise RuntimeError('The session is unavailable because no secret key was set.')\n        return self.session_interface.open_session(self, request)",
        },
        {
            "repo": "psf/requests",
            "issue": "Handle ChunkedEncodingError when connection drops during streamed download.",
            "target": "requests/models.py",
            "search": "def iter_content",
            "edit_old": "yield chunk",
            "edit_new": "try:\n            yield chunk\n        except ChunkedEncodingError:\n            raise ConnectionError('Chunked transfer aborted')",
        },
        {
            "repo": "django/django",
            "issue": "QuerySet.exists() should optimize away ordering clauses for faster execution.",
            "target": "django/db/models/query.py",
            "search": "def exists(self):",
            "edit_old": "return bool(self[:1])",
            "edit_new": "clone = self.order_by()\n        return bool(clone[:1])",
        },
    ]

    for i in range(num_samples):
        tmpl = bug_templates[i % len(bug_templates)]
        instance_id = f"{tmpl['repo'].replace('/', '__')}-issue-{1000 + i}"

        turns = [
            TrajectoryTurn(
                turn_index=1,
                thought=f"I need to localize the issue in {tmpl['target']}. Let us query the code graph.",
                action=f"graph_neighbors(symbol_or_file=\"{tmpl['target']}\")",
                observation=f"File '{tmpl['target']}' contains symbols: [open_session, get_session, save_session].",
            ),
            TrajectoryTurn(
                turn_index=2,
                thought=f"Now viewing the implementation around {tmpl['search']}.",
                action=f"open_file(filepath=\"{tmpl['target']}\", start_line=1, end_line=50)",
                observation=f"10 | def open_session(self, app, request):\n11 |     {tmpl['edit_old']}",
            ),
            TrajectoryTurn(
                turn_index=3,
                thought="Identified the bug location. Applying precise fix.",
                action=f'edit_file(filepath="{tmpl['target']}", old_content="{tmpl['edit_old']}", new_content="{tmpl['edit_new']}")',
                observation=f"Successfully updated '{tmpl['target']}'.",
            ),
            TrajectoryTurn(
                turn_index=4,
                thought="Running test suite to verify fix.",
                action="run_command(cmd=\"pytest tests/\")",
                observation="================= 12 passed in 0.42s =================",
            ),
            TrajectoryTurn(
                turn_index=5,
                thought="All tests pass and the issue is resolved.",
                action=f'finish(summary="Resolved {tmpl['issue']}")',
                observation="Task completed successfully.",
            ),
        ]

        traj = SWEInstanceTrajectory(
            instance_id=instance_id,
            repo=tmpl["repo"],
            issue_description=tmpl["issue"],
            turns=turns,
            resolved=True,
            ground_truth_files=[tmpl["target"]],
            graph_summary={
                "total_nodes": 45,
                "num_modules": 4,
                "num_classes": 2,
                "num_functions": 12,
            },
        )
        trajectories.append(traj)

    return TrajectoryDataset(trajectories, use_graph_prompts=True)

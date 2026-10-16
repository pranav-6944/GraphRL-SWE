"""
prompt_templates.py — Gemma 4 chat formatted prompts for GraphRL-SWE agent.

Formats prompts following Google Gemma 4's special token conventions:
  <start_of_turn>user
  {prompt}
  <end_of_turn>
  <start_of_turn>model
  {response}
  <end_of_turn>

Provides both standard baseline prompts (flat text search) and
graph-conditioned prompts (injecting structural code graph information).
"""

from __future__ import annotations

import json
from typing import Any, Optional


SYSTEM_PROMPT_GRAPH_AGENT = """You are an expert autonomous software engineering agent running locally on consumer hardware.
Your task is to resolve an issue in a software repository by:
1. Understanding the problem description and requirements.
2. Navigating the codebase efficiently using structural code-graph tools.
3. Localizing the root cause functions and classes.
4. Editing the source code accurately.
5. Verifying the fix by running tests.

You have access to both standard file operations and structural code-graph tools:
- `open_file(filepath, start_line, end_line)`: View contents of a file within a line range.
- `edit_file(filepath, old_content, new_content)`: Make an exact string replacement.
- `search_code(query, filepath)`: Search for literal strings or symbols.
- `run_command(cmd)`: Execute shell commands (e.g., pytest, git diff).
- `graph_neighbors(symbol_or_file)`: Inspect callers, callees, parent classes, and imports.
- `graph_callers(symbol)`: Find all functions/methods calling this symbol.
- `graph_callees(symbol)`: Find all functions/methods called by this symbol.
- `graph_subgraph(symbol_or_file, radius)`: Get a structural neighborhood map.
- `finish(summary)`: Conclude when the issue is resolved and tests pass.

Always output your response in this format:
Thought: <Your reasoning about what to do next, analyzing graph connections or test results>
Action: <tool_name>(<arguments>)
"""

SYSTEM_PROMPT_FLAT_AGENT = """You are an expert autonomous software engineering agent running locally on consumer hardware.
Your task is to resolve an issue in a software repository by:
1. Understanding the problem description and requirements.
2. Navigating the codebase using search and file inspection.
3. Localizing the root cause.
4. Editing the source code accurately.
5. Verifying the fix by running tests.

Available tools:
- `open_file(filepath, start_line, end_line)`: View contents of a file within a line range.
- `edit_file(filepath, old_content, new_content)`: Make an exact string replacement.
- `search_code(query, filepath)`: Search for literal strings or symbols.
- `run_command(cmd)`: Execute shell commands (e.g., pytest, git diff).
- `finish(summary)`: Conclude when the issue is resolved and tests pass.

Always output your response in this format:
Thought: <Your reasoning about what to do next>
Action: <tool_name>(<arguments>)
"""


def format_gemma_chat(
    system_prompt: str,
    messages: list[dict[str, str]],
    add_generation_prompt: bool = True,
) -> str:
    """
    Format a conversation into Gemma 4 chat template format.

    Args:
        system_prompt: System instructions for the agent.
        messages: List of turn dicts with 'role' ('user' or 'model') and 'content'.
        add_generation_prompt: If True, appends '<start_of_turn>model\n'.
    """
    formatted = ""
    # In Gemma chat template, system instructions are prefixed in the first user turn
    first_user = True

    for msg in messages:
        role = msg["role"]
        content = msg["content"]

        if role == "user" and first_user:
            combined_content = f"{system_prompt.strip()}\n\n{content.strip()}"
            formatted += f"<start_of_turn>user\n{combined_content}<end_of_turn>\n"
            first_user = False
        else:
            formatted += f"<start_of_turn>{role}\n{content.strip()}<end_of_turn>\n"

    if first_user:
        # No messages provided yet, output system prompt in user turn
        formatted += f"<start_of_turn>user\n{system_prompt.strip()}<end_of_turn>\n"

    if add_generation_prompt:
        formatted += "<start_of_turn>model\n"

    return formatted


def create_initial_agent_prompt(
    issue_description: str,
    repo_name: str,
    graph_summary: Optional[dict[str, Any]] = None,
    hints: Optional[str] = None,
) -> str:
    """
    Construct the initial task prompt presenting the issue and repository context.

    If graph_summary is provided, injects high-level topological context:
    - Number of modules, classes, and functions
    - Central hubs / key classes identified via PageRank
    - Entry points
    """
    prompt_parts = [
        f"You are working on the repository `{repo_name}`.",
        f"An issue has been reported:\n",
        f"\"\"\"\n{issue_description.strip()}\n\"\"\"\n",
    ]

    if graph_summary:
        prompt_parts.append("### Code Graph Overview:")
        prompt_parts.append(
            f"- Total structural entities: {graph_summary.get('total_nodes', 'N/A')} "
            f"({graph_summary.get('num_modules', 0)} files/modules, "
            f"{graph_summary.get('num_classes', 0)} classes, "
            f"{graph_summary.get('num_functions', 0)} functions)"
        )
        top_hubs = graph_summary.get("top_central_entities", [])
        if top_hubs:
            hubs_str = ", ".join([f"`{h['name']}` ({h.get('type', '')})" for h in top_hubs[:8]])
            prompt_parts.append(f"- Key structural hubs (by centrality): {hubs_str}")
        prompt_parts.append(
            "Use `graph_neighbors` or `graph_subgraph` to explore relationships without reading whole files."
        )

    if hints:
        prompt_parts.append(f"\n### Additional Context/Hints:\n{hints.strip()}")

    prompt_parts.append(
        "\nPlease investigate the problem, locate the bug, make the necessary modifications, "
        "and verify with tests. Start by stating your thought and initial action."
    )

    return "\n".join(prompt_parts)


def parse_action_response(response_text: str) -> tuple[str, str, dict[str, Any]]:
    """
    Parse an agent's response containing Thought and Action.

    Returns:
        (thought, action_name, action_args_dict)
    """
    thought = ""
    action_str = ""

    lines = response_text.strip().split("\n")
    thought_lines = []
    action_line = ""

    for line in lines:
        stripped = line.strip()
        if stripped.startswith("Thought:"):
            thought_lines.append(stripped[len("Thought:"):].strip())
        elif stripped.startswith("Action:"):
            action_line = stripped[len("Action:"):].strip()
        elif action_line == "":
            thought_lines.append(stripped)

    thought = " ".join(thought_lines).strip()

    if not action_line:
        # Fallback: check if the model just output an action without prefix
        for candidate in ["open_file", "edit_file", "search_code", "run_command", "graph_neighbors", "finish"]:
            if candidate in response_text:
                action_line = response_text[response_text.index(candidate):].split("\n")[0].strip()
                break

    if not action_line:
        return thought, "noop", {}

    # Parse function call: func_name(arg1="val", arg2=123)
    if "(" in action_line and action_line.endswith(")"):
        name = action_line[:action_line.index("(")].strip()
        args_raw = action_line[action_line.index("(") + 1:-1].strip()
        parsed_args = _parse_args_string(args_raw)
        return thought, name, parsed_args

    return thought, action_line, {}


def _parse_args_string(args_raw: str) -> dict[str, Any]:
    """Helper to parse key=value arguments from string."""
    if not args_raw:
        return {}

    parsed = {}
    # Try parsing as JSON object if wrapped in braces
    if args_raw.startswith("{") and args_raw.endswith("}"):
        try:
            return json.loads(args_raw)
        except Exception:
            pass

    # Simple comma split with quotes preservation
    import re
    # Match key="value" or key=123 or positional single strings
    pattern = re.compile(r'(\w+)\s*=\s*(?:"([^"]*)"|\'([^\']*)\'|([^,]+))')
    matches = pattern.findall(args_raw)
    for m in matches:
        key = m[0]
        val = m[1] or m[2] or m[3]
        val = val.strip()
        if val.isdigit():
            val = int(val)
        elif val.lower() == "true":
            val = True
        elif val.lower() == "false":
            val = False
        parsed[key] = val

    if not parsed and args_raw:
        # Positional single string argument, e.g. finish("Fixed typo in user auth")
        cleaned = args_raw.strip('"\'')
        parsed["arg"] = cleaned

    return parsed

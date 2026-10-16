"""
environment.py — Execution environment for SWE agent interactions.

Provides sandbox operations (file reading, search, editing, test execution)
and interfaces directly with the code graph analyzer for graph-guided actions.
Supports both real repository directories and synthetic/mock test environments.
"""

from __future__ import annotations

import os
import subprocess
from pathlib import Path
from typing import Any, Optional

from loguru import logger

from graphrl_swe.graph.graph_builder import CodeGraphBuilder
from graphrl_swe.graph.graph_utils import GraphAnalyzer


class SWEEnvironment:
    """
    Execution environment for a software engineering task instance.

    Handles:
    - File reading (`open_file`)
    - Exact text search (`search_code`)
    - String replacement editing (`edit_file`)
    - Command / test execution (`run_command`)
    - Graph navigation queries (`graph_neighbors`, `graph_callers`, `graph_callees`, `graph_subgraph`, `graph_path`)
    - Modification tracking and diff generation
    """

    def __init__(
        self,
        repo_path: str | Path,
        graph_analyzer: Optional[GraphAnalyzer] = None,
        test_command: Optional[str] = None,
        timeout: int = 120,
    ):
        self.repo_path = Path(repo_path).resolve()
        self.timeout = timeout
        self.test_command = test_command
        self.edited_files: set[str] = set()
        self.visited_files: list[str] = []

        # Initialize or assign graph analyzer
        if graph_analyzer:
            self.analyzer = graph_analyzer
        else:
            self.analyzer = self._build_graph()

    def _build_graph(self) -> Optional[GraphAnalyzer]:
        """Build graph for repository if analyzer not provided."""
        try:
            builder = CodeGraphBuilder()
            graph = builder.from_repository(self.repo_path)
            return GraphAnalyzer(graph)
        except Exception as e:
            logger.warning(f"Could not build code graph for {self.repo_path}: {e}")
            return None

    def execute_tool(self, action_name: str, args: dict[str, Any]) -> str:
        """Dispatch and execute an agent action, returning the observation string."""
        action_name = action_name.lower().strip()

        try:
            if action_name == "open_file":
                filepath = args.get("filepath") or args.get("file") or args.get("arg", "")
                start = int(args.get("start_line", 1))
                end = int(args.get("end_line", -1))
                return self.open_file(filepath, start, end)

            elif action_name == "search_code":
                query = args.get("query") or args.get("pattern") or args.get("arg", "")
                filepath = args.get("filepath") or args.get("path")
                return self.search_code(query, filepath)

            elif action_name == "edit_file":
                filepath = args.get("filepath") or args.get("file", "")
                old_content = args.get("old_content") or args.get("old_str", "")
                new_content = args.get("new_content") or args.get("new_str", "")
                return self.edit_file(filepath, old_content, new_content)

            elif action_name == "run_command":
                cmd = args.get("cmd") or args.get("command") or args.get("arg", "")
                return self.run_command(cmd)

            # Graph navigation tools
            elif action_name == "graph_neighbors":
                symbol = args.get("symbol") or args.get("symbol_or_file") or args.get("arg", "")
                return self.graph_neighbors(symbol)

            elif action_name == "graph_callers":
                symbol = args.get("symbol") or args.get("arg", "")
                return self.graph_callers(symbol)

            elif action_name == "graph_callees":
                symbol = args.get("symbol") or args.get("arg", "")
                return self.graph_callees(symbol)

            elif action_name == "graph_subgraph":
                symbol = args.get("symbol") or args.get("symbol_or_file") or args.get("arg", "")
                radius = int(args.get("radius", 1))
                return self.graph_subgraph(symbol, radius)

            elif action_name == "graph_path":
                source = args.get("source") or args.get("src", "")
                target = args.get("target") or args.get("dst", "")
                return self.graph_path(source, target)

            elif action_name == "finish":
                summary = args.get("summary") or args.get("arg", "Task marked as finished.")
                return f"Task completed: {summary}"

            else:
                return f"Error: Unknown tool '{action_name}'. Available tools: open_file, search_code, edit_file, run_command, graph_neighbors, graph_callers, graph_callees, graph_subgraph, graph_path, finish."

        except Exception as e:
            return f"Execution error for action '{action_name}': {str(e)}"

    # ── Standard Tools ────────────────────────────────────────────

    def open_file(self, filepath: str, start_line: int = 1, end_line: int = -1) -> str:
        """View file content with line numbers."""
        rel_path = self._resolve_rel_path(filepath)
        full_path = self.repo_path / rel_path

        if not full_path.exists():
            return f"Error: File '{filepath}' does not exist."
        if full_path.is_dir():
            # Return directory listing
            files = [str(p.relative_to(self.repo_path)) for p in full_path.iterdir()]
            return f"Directory '{filepath}' contents:\n" + "\n".join(files[:50])

        self._record_visit(rel_path)

        try:
            lines = full_path.read_text(encoding="utf-8", errors="replace").splitlines()
            total_lines = len(lines)

            start = max(1, start_line)
            end = total_lines if end_line == -1 or end_line > total_lines else end_line

            if start > total_lines:
                return f"File '{filepath}' has {total_lines} lines (requested start at {start})."

            output_lines = []
            for i in range(start - 1, end):
                output_lines.append(f"{i + 1:4d} | {lines[i]}")

            header = f"--- {rel_path} (Lines {start}-{end} of {total_lines}) ---\n"
            return header + "\n".join(output_lines)
        except Exception as e:
            return f"Error reading file '{filepath}': {e}"

    def search_code(self, query: str, filepath: Optional[str] = None) -> str:
        """Search for a string across files or within a specific file."""
        if not query:
            return "Error: Search query cannot be empty."

        matches = []
        search_paths = [self.repo_path / filepath] if filepath else self.repo_path.rglob("*.py")

        for p in search_paths:
            if not p.is_file():
                continue
            rel = str(p.relative_to(self.repo_path))
            try:
                for idx, line in enumerate(p.read_text(encoding="utf-8", errors="replace").splitlines()):
                    if query in line:
                        matches.append(f"{rel}:{idx + 1}: {line.strip()}")
                        if len(matches) >= 30:
                            break
            except Exception:
                continue

            if len(matches) >= 30:
                break

        if not matches:
            return f"No matches found for '{query}'."

        return f"Found {len(matches)} match(es):\n" + "\n".join(matches)

    def edit_file(self, filepath: str, old_content: str, new_content: str) -> str:
        """Perform exact text replacement in file."""
        rel_path = self._resolve_rel_path(filepath)
        full_path = self.repo_path / rel_path

        if not full_path.exists():
            return f"Error: File '{filepath}' does not exist."

        self._record_visit(rel_path)
        content = full_path.read_text(encoding="utf-8", errors="replace")

        if old_content not in content:
            return (
                f"Error: Target text not found in '{filepath}'. "
                "Ensure exact match including whitespace and indentation."
            )

        occurrences = content.count(old_content)
        if occurrences > 1:
            return (
                f"Error: Found {occurrences} occurrences of target text in '{filepath}'. "
                "Please provide more surrounding context to uniquely identify the location."
            )

        updated = content.replace(old_content, new_content, 1)
        full_path.write_text(updated, encoding="utf-8")
        self.edited_files.add(rel_path)

        return f"Successfully updated '{rel_path}'."

    def run_command(self, cmd: str) -> str:
        """Execute a shell command inside the repository directory."""
        if not cmd:
            return "Error: Command cannot be empty."

        try:
            res = subprocess.run(
                cmd,
                shell=True,
                cwd=str(self.repo_path),
                capture_output=True,
                text=True,
                timeout=self.timeout,
            )
            stdout = res.stdout.strip()
            stderr = res.stderr.strip()
            output = []
            if stdout:
                output.append(stdout)
            if stderr:
                output.append(f"STDERR:\n{stderr}")
            if not output:
                output.append(f"Command exited with return code {res.returncode} (no output).")

            return "\n".join(output)[:4000]
        except subprocess.TimeoutExpired:
            return f"Error: Command timed out after {self.timeout} seconds."
        except Exception as e:
            return f"Error running command '{cmd}': {e}"

    # ── Graph Navigation Tools ────────────────────────────────────

    def graph_neighbors(self, symbol_or_file: str) -> str:
        """Get structural neighbors (callers, callees, classes, methods)."""
        if not self.analyzer:
            return "Code graph is not available for this environment."

        nodes = self.analyzer.find_nodes_by_name(symbol_or_file)
        if not nodes:
            # Check if it matches a file
            file_nodes = self.analyzer.get_file_nodes(symbol_or_file)
            if file_nodes:
                return (
                    f"File '{symbol_or_file}' contains {len(file_nodes)} structural entities:\n"
                    + "\n".join([f"- `{n}` ({self.analyzer.graph.nodes[n].get('type', '')})" for n in file_nodes[:20]])
                )
            return f"No entity or file matching '{symbol_or_file}' found in code graph."

        node_id = nodes[0]
        node_name = self.analyzer.graph.nodes[node_id].get("name", node_id)
        neighbors = self.analyzer.get_neighbors(node_id)
        if not neighbors:
            return f"Entity '{node_name}' ({node_id}) has no direct graph neighbors."

        lines = [f"Structural connections for `{node_name}` ({node_id}):"]
        for n, rel in neighbors[:25]:
            n_data = self.analyzer.graph.nodes.get(n, {})
            n_name = n_data.get("name", n)
            n_type = n_data.get("type", "")
            lines.append(f"  [{rel}] -> `{n_name}` ({n_type}) [{n}]")

        return "\n".join(lines)

    def graph_callers(self, symbol: str) -> str:
        """Find entities calling this symbol."""
        if not self.analyzer:
            return "Code graph is not available."

        callers = self.analyzer.find_callers(symbol)
        if not callers:
            return f"No callers found for '{symbol}' in code graph."

        lines = [f"Callers of `{symbol}` ({len(callers)} found):"]
        for c in callers[:20]:
            c_data = self.analyzer.graph.nodes.get(c, {})
            c_name = c_data.get("name", c)
            lines.append(f"- `{c_name}` ({c})")
        return "\n".join(lines)

    def graph_callees(self, symbol: str) -> str:
        """Find functions/methods called by this symbol."""
        if not self.analyzer:
            return "Code graph is not available."

        callees = self.analyzer.find_callees(symbol)
        if not callees:
            return f"No outgoing calls found for '{symbol}'."

        lines = [f"Callees of `{symbol}` ({len(callees)} found):"]
        for c in callees[:20]:
            c_data = self.analyzer.graph.nodes.get(c, {})
            c_name = c_data.get("name", c)
            lines.append(f"- `{c_name}` ({c})")
        return "\n".join(lines)

    def graph_subgraph(self, symbol_or_file: str, radius: int = 1) -> str:
        """Extract a sub-network surrounding an entity."""
        if not self.analyzer:
            return "Code graph is not available."

        nodes = self.analyzer.find_nodes_by_name(symbol_or_file)
        if not nodes:
            return f"Symbol '{symbol_or_file}' not found."

        center = nodes[0]
        sub = self.analyzer.extract_neighborhood(center, radius=radius)
        lines = [f"Neighborhood subgraph around `{center}` (radius={radius}, {sub.number_of_nodes()} nodes, {sub.number_of_edges()} edges):"]
        for u, v, data in list(sub.edges(data=True))[:30]:
            rel = data.get("type", "rel")
            lines.append(f"  `{u}` --({rel})--> `{v}`")

        return "\n".join(lines)

    def graph_path(self, source: str, target: str) -> str:
        """Find shortest structural path between two entities."""
        if not self.analyzer:
            return "Code graph is not available."

        src_nodes = self.analyzer.find_nodes_by_name(source)
        dst_nodes = self.analyzer.find_nodes_by_name(target)

        if not src_nodes or not dst_nodes:
            return f"Could not find entities for source='{source}' or target='{target}'."

        path = self.analyzer.shortest_path(src_nodes[0], dst_nodes[0])
        if not path:
            return f"No path found between `{source}` and `{target}`."

        return f"Shortest graph path ({len(path) - 1} hops):\n  " + " -> ".join(path)

    # ── Utilities ─────────────────────────────────────────────────

    def _resolve_rel_path(self, filepath: str) -> str:
        """Normalize relative path."""
        p = Path(filepath.strip())
        if p.is_absolute():
            try:
                return str(p.relative_to(self.repo_path)).replace("\\", "/")
            except ValueError:
                return str(p).replace("\\", "/")
        return str(p).replace("\\", "/")

    def _record_visit(self, rel_path: str):
        """Track file visit order."""
        if rel_path not in self.visited_files:
            self.visited_files.append(rel_path)

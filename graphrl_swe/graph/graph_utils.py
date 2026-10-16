"""
graph_utils.py — Graph analysis utilities for code navigation and reward shaping.

Provides methods to:
  - Compute shortest-path distances between code elements
  - Find relevant subgraphs around a target location
  - Calculate Personalized PageRank for importance-weighted navigation
  - Map file paths to graph node sets (for reward computation)
  - Extract the "blast radius" of a code change
"""

from __future__ import annotations

from collections import defaultdict
from typing import Any, Optional

import networkx as nx
import numpy as np
from loguru import logger

from graphrl_swe.graph.graph_builder import NodeType, EdgeType


class GraphAnalyzer:
    """
    Analyzes a code graph to provide structural insights for agent navigation
    and reward computation.

    Usage:
        analyzer = GraphAnalyzer(graph)

        # Get distance between two code elements
        dist = analyzer.shortest_path_distance("module.foo", "module.bar")

        # Find all nodes in a file
        nodes = analyzer.get_file_nodes("src/utils.py")

        # Get the neighborhood subgraph
        subgraph = analyzer.get_neighborhood("module.MyClass.method", radius=2)

        # Compute localization reward
        reward = analyzer.localization_reward(
            visited_files=["src/a.py", "src/b.py"],
            ground_truth_files=["src/b.py"],
        )
    """

    def __init__(self, graph: nx.DiGraph):
        """
        Args:
            graph: A code graph built by CodeGraphBuilder.
        """
        self.graph = graph

        # Build lookup indices
        self._file_to_nodes: dict[str, list[str]] = defaultdict(list)
        self._qname_to_id: dict[str, str] = {}
        self._build_indices()

    def _build_indices(self):
        """Build fast lookup indices from graph attributes."""
        # Use stored name→id mapping if available
        stored_map = self.graph.graph.get("name_to_id", {})
        if stored_map:
            self._qname_to_id = dict(stored_map)

        for node_id, data in self.graph.nodes(data=True):
            # File → nodes mapping
            filepath = data.get("filepath", "")
            if filepath:
                self._file_to_nodes[filepath].append(node_id)

            # Qualified name → id (fallback)
            qname = data.get("qualified_name", "")
            if qname and qname not in self._qname_to_id:
                self._qname_to_id[qname] = node_id

    # ── Distance Computation ──────────────────────────────────────

    def shortest_path_distance(
        self,
        source: str,
        target: str,
        max_distance: int = 50,
    ) -> int:
        """
        Compute the shortest path distance between two code elements.

        Args:
            source: Qualified name or node ID of the source.
            target: Qualified name or node ID of the target.
            max_distance: Return this value if no path exists (unreachable).

        Returns:
            Integer shortest path length, or max_distance if unreachable.
        """
        src_id = self._resolve_id(source)
        tgt_id = self._resolve_id(target)

        if src_id is None or tgt_id is None:
            return max_distance

        if src_id == tgt_id:
            return 0

        # Use undirected version for distance (edges are navigable both ways)
        undirected = self.graph.to_undirected()

        try:
            path_len = nx.shortest_path_length(undirected, src_id, tgt_id)
            return min(path_len, max_distance)
        except nx.NetworkXNoPath:
            return max_distance
        except nx.NodeNotFound:
            return max_distance

    def pairwise_distances(
        self,
        node_set_a: list[str],
        node_set_b: list[str],
        max_distance: int = 50,
    ) -> np.ndarray:
        """
        Compute pairwise shortest-path distances between two sets of nodes.

        Args:
            node_set_a: List of qualified names or node IDs.
            node_set_b: List of qualified names or node IDs.
            max_distance: Value for unreachable pairs.

        Returns:
            Distance matrix of shape (len(a), len(b)).
        """
        ids_a = [self._resolve_id(n) for n in node_set_a]
        ids_b = [self._resolve_id(n) for n in node_set_b]

        undirected = self.graph.to_undirected()
        distances = np.full((len(ids_a), len(ids_b)), max_distance, dtype=np.float32)

        for i, src in enumerate(ids_a):
            if src is None:
                continue
            try:
                lengths = nx.single_source_shortest_path_length(
                    undirected, src, cutoff=max_distance
                )
                for j, tgt in enumerate(ids_b):
                    if tgt is not None and tgt in lengths:
                        distances[i, j] = lengths[tgt]
            except nx.NodeNotFound:
                continue

        return distances

    def min_distance_to_set(
        self,
        source: str,
        target_set: list[str],
        max_distance: int = 50,
    ) -> float:
        """
        Compute the minimum distance from source to any node in target_set.

        Useful for measuring how "close" an agent's visited location is to
        the ground truth bug location.
        """
        src_id = self._resolve_id(source)
        if src_id is None:
            return float(max_distance)

        tgt_ids = {self._resolve_id(t) for t in target_set} - {None}
        if not tgt_ids:
            return float(max_distance)

        undirected = self.graph.to_undirected()
        try:
            lengths = nx.single_source_shortest_path_length(
                undirected, src_id, cutoff=max_distance
            )
            min_dist = max_distance
            for tgt_id in tgt_ids:
                if tgt_id in lengths:
                    min_dist = min(min_dist, lengths[tgt_id])
            return float(min_dist)
        except nx.NodeNotFound:
            return float(max_distance)

    # ── File-Level Operations ─────────────────────────────────────

    def get_file_nodes(self, filepath: str) -> list[str]:
        """Get all graph node IDs that belong to a given file."""
        # Try exact match first
        if filepath in self._file_to_nodes:
            return self._file_to_nodes[filepath]

        # Try suffix match (relative vs absolute paths)
        for fp, nodes in self._file_to_nodes.items():
            if fp.endswith(filepath) or filepath.endswith(fp):
                return nodes

        return []

    def file_distance(
        self,
        file_a: str,
        file_b: str,
        max_distance: int = 50,
    ) -> float:
        """
        Compute the minimum graph distance between any nodes in two files.

        This is used to measure how structurally related two files are,
        independent of their directory location.
        """
        nodes_a = self.get_file_nodes(file_a)
        nodes_b = self.get_file_nodes(file_b)

        if not nodes_a or not nodes_b:
            return float(max_distance)

        undirected = self.graph.to_undirected()
        min_dist = max_distance

        # Use BFS from all nodes in file_a
        for src_id in nodes_a:
            try:
                lengths = nx.single_source_shortest_path_length(
                    undirected, src_id, cutoff=min_dist
                )
                for tgt_id in nodes_b:
                    if tgt_id in lengths:
                        min_dist = min(min_dist, lengths[tgt_id])
            except nx.NodeNotFound:
                continue

            if min_dist == 0:
                break

        return float(min_dist)

    def files_distance_matrix(
        self,
        files: list[str],
        max_distance: int = 50,
    ) -> np.ndarray:
        """Compute pairwise file distances for a list of files."""
        n = len(files)
        dist_matrix = np.full((n, n), max_distance, dtype=np.float32)
        np.fill_diagonal(dist_matrix, 0)

        for i in range(n):
            for j in range(i + 1, n):
                d = self.file_distance(files[i], files[j], max_distance)
                dist_matrix[i, j] = d
                dist_matrix[j, i] = d

        return dist_matrix

    # ── Localization Reward ───────────────────────────────────────

    def localization_reward(
        self,
        visited_files: list[str],
        ground_truth_files: list[str],
        max_distance: int = 10,
    ) -> float:
        """
        Compute the graph-guided localization reward.

        Measures how close the agent's visited files are to the ground truth
        files where the bug actually lives, using graph structure rather than
        simple path/string matching.

        R_localization = 1 - (avg_min_distance / max_distance)

        A score of 1.0 means the agent visited the exact files.
        A score of 0.0 means the agent's visited files are completely
        structurally unrelated to the ground truth.

        Args:
            visited_files: Files the agent opened/edited during the episode.
            ground_truth_files: Files that actually need to be changed.
            max_distance: Normalization cap for distances.

        Returns:
            Float reward in [0, 1].
        """
        if not visited_files or not ground_truth_files:
            return 0.0

        # For each ground truth file, find the minimum distance to any visited file
        min_distances = []
        for gt_file in ground_truth_files:
            min_d = max_distance
            for v_file in visited_files:
                d = self.file_distance(v_file, gt_file, max_distance)
                min_d = min(min_d, d)
            min_distances.append(min_d)

        avg_min_dist = np.mean(min_distances)
        reward = 1.0 - (avg_min_dist / max_distance)
        return float(np.clip(reward, 0.0, 1.0))

    # ── Neighborhood & Subgraph Extraction ────────────────────────

    def get_neighborhood(
        self,
        center: str,
        radius: int = 2,
    ) -> nx.DiGraph:
        """
        Extract a subgraph centered on a node within a given radius.

        Useful for providing focused context to the agent about the
        structural neighborhood of a code element.

        Args:
            center: Qualified name or node ID of the center node.
            radius: BFS radius for neighborhood extraction.

        Returns:
            A subgraph (DiGraph) containing the neighborhood.
        """
        center_id = self._resolve_id(center)
        if center_id is None:
            return nx.DiGraph()

        undirected = self.graph.to_undirected()
        try:
            neighbors = nx.single_source_shortest_path_length(
                undirected, center_id, cutoff=radius
            )
            return self.graph.subgraph(neighbors.keys()).copy()
        except nx.NodeNotFound:
            return nx.DiGraph()

    def get_blast_radius(
        self,
        changed_files: list[str],
        radius: int = 2,
    ) -> set[str]:
        """
        Compute the "blast radius" — all files potentially affected by
        changes to the given files, within a graph radius.

        Args:
            changed_files: Files being modified.
            radius: How far to look for affected code.

        Returns:
            Set of file paths that are within the blast radius.
        """
        affected_files: set[str] = set(changed_files)

        for filepath in changed_files:
            file_nodes = self.get_file_nodes(filepath)
            for node_id in file_nodes:
                neighborhood = self.get_neighborhood(node_id, radius=radius)
                for _, data in neighborhood.nodes(data=True):
                    fp = data.get("filepath", "")
                    if fp:
                        affected_files.add(fp)

        return affected_files

    # ── Personalized PageRank ─────────────────────────────────────

    def importance_scores(
        self,
        seed_nodes: Optional[list[str]] = None,
        alpha: float = 0.85,
    ) -> dict[str, float]:
        """
        Compute Personalized PageRank scores, optionally seeded on specific nodes.

        When seeded on ground-truth bug locations, this gives a
        "relevance-weighted" importance map of the entire codebase relative
        to the bug.

        Args:
            seed_nodes: Qualified names or node IDs to seed PageRank on.
                        If None, uses standard PageRank.
            alpha: Damping factor (0.85 is standard).

        Returns:
            Dict mapping node IDs to importance scores.
        """
        if seed_nodes:
            personalization = {}
            for name in seed_nodes:
                nid = self._resolve_id(name)
                if nid:
                    personalization[nid] = 1.0

            if not personalization:
                logger.warning("No valid seed nodes found. Using standard PageRank.")
                return dict(nx.pagerank(self.graph, alpha=alpha))

            # Normalize personalization
            total = sum(personalization.values())
            personalization = {k: v / total for k, v in personalization.items()}

            return dict(
                nx.pagerank(self.graph, alpha=alpha, personalization=personalization)
            )
        else:
            return dict(nx.pagerank(self.graph, alpha=alpha))

    # ── Graph Summary for LLM Context ─────────────────────────────

    def generate_graph_summary(
        self,
        focus_files: Optional[list[str]] = None,
        max_nodes_in_summary: int = 50,
    ) -> str:
        """
        Generate a concise text summary of the code graph suitable for
        inclusion in an LLM's context window.

        This gives the agent a "map" of the repository structure.

        Args:
            focus_files: If provided, focus the summary on these files and
                         their neighborhoods. Otherwise, summarize the whole graph.
            max_nodes_in_summary: Cap on nodes included.

        Returns:
            A markdown-formatted string describing the code structure.
        """
        lines = ["## Repository Code Graph\n"]

        if focus_files:
            relevant_nodes = set()
            for fp in focus_files:
                nodes = self.get_file_nodes(fp)
                relevant_nodes.update(nodes)
                # Add 1-hop neighbors
                for nid in nodes:
                    for neighbor in self.graph.neighbors(nid):
                        relevant_nodes.add(neighbor)
                    for predecessor in self.graph.predecessors(nid):
                        relevant_nodes.add(predecessor)
        else:
            # Use PageRank to pick the most important nodes
            pr = nx.pagerank(self.graph)
            sorted_nodes = sorted(pr, key=pr.get, reverse=True)
            relevant_nodes = set(sorted_nodes[:max_nodes_in_summary])

        # Group by file
        file_groups: dict[str, list[tuple[str, dict]]] = defaultdict(list)
        for nid in relevant_nodes:
            if nid not in self.graph:
                continue
            data = self.graph.nodes[nid]
            fp = data.get("filepath", "unknown")
            file_groups[fp].append((nid, data))

        for filepath, nodes in sorted(file_groups.items()):
            lines.append(f"\n### `{filepath}`\n")
            for nid, data in sorted(nodes, key=lambda x: x[1].get("start_line", 0)):
                ntype = data.get("type", "?")
                name = data.get("name", "?")
                start = data.get("start_line", "?")
                end = data.get("end_line", "?")

                if ntype == NodeType.MODULE:
                    lines.append(f"- **Module**: `{name}` ({data.get('lines_of_code', '?')} lines)")
                elif ntype == NodeType.CLASS:
                    bases = data.get("base_classes", [])
                    base_str = f"({', '.join(bases)})" if bases else ""
                    lines.append(f"- **Class**: `{name}{base_str}` (L{start}–L{end})")
                elif ntype in (NodeType.FUNCTION, NodeType.METHOD):
                    params = data.get("parameters", [])
                    param_str = ", ".join(params[:5])
                    if len(params) > 5:
                        param_str += ", ..."
                    prefix = "Method" if ntype == NodeType.METHOD else "Function"
                    parent = data.get("parent_class", "")
                    if parent:
                        lines.append(f"  - **{prefix}**: `{parent}.{name}({param_str})` (L{start}–L{end})")
                    else:
                        lines.append(f"- **{prefix}**: `{name}({param_str})` (L{start}–L{end})")

        # Add edge summary
        stats = self.graph.graph.get("stats", {})
        edge_counts = stats.get("edge_type_counts", {})
        if edge_counts:
            lines.append("\n### Relationships\n")
            for etype, count in sorted(edge_counts.items()):
                lines.append(f"- `{etype}`: {count} edges")

        return "\n".join(lines)

    # ── Utility ───────────────────────────────────────────────────

    def _resolve_id(self, name_or_id: str) -> Optional[str]:
        """Resolve a qualified name or node ID to a graph node ID."""
        # Direct node ID
        if name_or_id in self.graph:
            return name_or_id
        # Qualified name lookup
        if name_or_id in self._qname_to_id:
            return self._qname_to_id[name_or_id]
        return None

    def get_node_info(self, name_or_id: str) -> Optional[dict]:
        """Get full attribute dict for a node."""
        nid = self._resolve_id(name_or_id)
        if nid and nid in self.graph:
            return dict(self.graph.nodes[nid])
        return None

    def get_all_files(self) -> list[str]:
        """Get a sorted list of all unique file paths in the graph."""
        return sorted(self._file_to_nodes.keys())

    # ── Graph Navigation & Query Utilities for Agents ─────────────

    def find_nodes_by_name(self, name: str) -> list[str]:
        """Find nodes matching a simple or qualified name."""
        matches = []
        name_lower = name.lower()
        for node_id, data in self.graph.nodes(data=True):
            node_name = data.get("name", "")
            qname = data.get("qualified_name", "")
            if (
                node_name == name
                or qname == name
                or qname.endswith(f".{name}")
                or node_id == name
                or node_name.lower() == name_lower
            ):
                matches.append(node_id)
        return matches

    def get_neighbors(self, node_id: str) -> list[tuple[str, str]]:
        """Return list of (neighbor_id, relation_type) for direct outgoing & incoming edges."""
        nid = self._resolve_id(node_id)
        if not nid or nid not in self.graph:
            return []
        neighbors = []
        for _, tgt, data in self.graph.out_edges(nid, data=True):
            neighbors.append((tgt, data.get("type", "out")))
        for src, _, data in self.graph.in_edges(nid, data=True):
            neighbors.append((src, f"in_{data.get('type', 'in')}"))
        return neighbors

    def find_callers(self, symbol_name: str) -> list[str]:
        """Find functions or methods that call the given symbol."""
        nodes = self.find_nodes_by_name(symbol_name)
        callers = set()
        for nid in nodes:
            # Check inbound call edges
            for src, _, data in self.graph.in_edges(nid, data=True):
                if data.get("type") == EdgeType.CALLS:
                    callers.add(src)
            # Also check node attribute calls
            for other_id, other_data in self.graph.nodes(data=True):
                called_list = other_data.get("calls", [])
                if any(symbol_name in c for c in called_list):
                    callers.add(other_id)
        return sorted(callers)

    def find_callees(self, symbol_name: str) -> list[str]:
        """Find functions or methods called by the given symbol."""
        nodes = self.find_nodes_by_name(symbol_name)
        callees = set()
        for nid in nodes:
            for _, tgt, data in self.graph.out_edges(nid, data=True):
                if data.get("type") == EdgeType.CALLS:
                    callees.add(tgt)
            called_list = self.graph.nodes[nid].get("calls", [])
            for c in called_list:
                callees.add(c)
        return sorted(callees)

    def shortest_path(self, source: str, target: str) -> list[str]:
        """Compute the shortest path node list between source and target."""
        src_id = self._resolve_id(source)
        tgt_id = self._resolve_id(target)
        if not src_id or not tgt_id:
            return []
        try:
            return nx.shortest_path(self.graph.to_undirected(), src_id, tgt_id)
        except (nx.NetworkXNoPath, nx.NodeNotFound):
            return []

    def extract_neighborhood(self, center: str, radius: int = 1) -> nx.DiGraph:
        """Extract neighborhood around a central entity."""
        return self.get_neighborhood(center, radius=radius)

    def graph_summary(self, top_k: int = 10) -> dict[str, Any]:
        """Return a structured dictionary summary of the graph topology."""
        total_nodes = self.graph.number_of_nodes()
        total_edges = self.graph.number_of_edges()

        type_counts = defaultdict(int)
        for _, data in self.graph.nodes(data=True):
            type_counts[data.get("type", "unknown")] += 1

        top_hubs = []
        if total_nodes > 0:
            try:
                pr = nx.pagerank(self.graph)
                sorted_nodes = sorted(pr.items(), key=lambda x: x[1], reverse=True)[:top_k]
                for nid, score in sorted_nodes:
                    d = self.graph.nodes[nid]
                    top_hubs.append({
                        "id": nid,
                        "name": d.get("name", nid),
                        "type": d.get("type", ""),
                        "filepath": d.get("filepath", ""),
                        "pagerank": round(score, 4),
                    })
            except Exception:
                pass

        return {
            "total_nodes": total_nodes,
            "total_edges": total_edges,
            "num_modules": type_counts.get(NodeType.MODULE, 0),
            "num_classes": type_counts.get(NodeType.CLASS, 0),
            "num_functions": type_counts.get(NodeType.FUNCTION, 0),
            "num_methods": type_counts.get(NodeType.METHOD, 0),
            "top_central_entities": top_hubs,
        }

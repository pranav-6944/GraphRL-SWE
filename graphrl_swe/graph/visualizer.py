"""
visualizer.py — Code graph visualization for paper figures and debugging.

Generates publication-quality visualizations of code graphs using
matplotlib and pyvis.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

from loguru import logger

try:
    import networkx as nx
    import matplotlib.pyplot as plt
    import matplotlib.patches as mpatches
    MATPLOTLIB_AVAILABLE = True
except ImportError:
    MATPLOTLIB_AVAILABLE = False

from graphrl_swe.graph.graph_builder import NodeType, EdgeType


# ── Color schemes for paper figures ───────────────────────────────

NODE_COLORS = {
    NodeType.MODULE: "#4A90D9",     # Blue
    NodeType.CLASS: "#E74C3C",      # Red
    NodeType.FUNCTION: "#2ECC71",   # Green
    NodeType.METHOD: "#F39C12",     # Orange
    NodeType.VARIABLE: "#9B59B6",   # Purple
}

NODE_SIZES = {
    NodeType.MODULE: 800,
    NodeType.CLASS: 600,
    NodeType.FUNCTION: 400,
    NodeType.METHOD: 300,
    NodeType.VARIABLE: 200,
}

EDGE_COLORS = {
    EdgeType.CONTAINS: "#BDC3C7",
    EdgeType.CALLS: "#3498DB",
    EdgeType.IMPORTS: "#1ABC9C",
    EdgeType.INHERITS: "#E74C3C",
    EdgeType.REFERENCES: "#95A5A6",
    EdgeType.HAS_METHOD: "#F39C12",
}

EDGE_STYLES = {
    EdgeType.CONTAINS: "dotted",
    EdgeType.CALLS: "solid",
    EdgeType.IMPORTS: "dashed",
    EdgeType.INHERITS: "solid",
    EdgeType.REFERENCES: "dotted",
    EdgeType.HAS_METHOD: "solid",
}


class GraphVisualizer:
    """
    Publication-quality code graph visualizer.

    Usage:
        viz = GraphVisualizer(graph)
        viz.plot_full_graph("paper/figures/full_graph.png")
        viz.plot_subgraph(subgraph, "paper/figures/subgraph.png")
        viz.plot_navigation_path(path_nodes, "paper/figures/nav_path.png")
    """

    def __init__(self, graph: nx.DiGraph):
        self.graph = graph

    def plot_full_graph(
        self,
        output_path: str | Path,
        figsize: tuple[int, int] = (16, 12),
        title: str = "Repository Code Graph",
        max_nodes: int = 200,
        dpi: int = 300,
        show_labels: bool = True,
        font_size: int = 7,
    ):
        """
        Plot the full code graph (or a subset of the most important nodes).

        Args:
            output_path: Where to save the figure.
            figsize: Figure size in inches.
            title: Plot title.
            max_nodes: Maximum nodes to display (uses PageRank to select).
            dpi: Output resolution.
            show_labels: Whether to show node name labels.
            font_size: Label font size.
        """
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available for visualization.")
            return

        # Select subset if graph is too large
        if self.graph.number_of_nodes() > max_nodes:
            pr = nx.pagerank(self.graph)
            top_nodes = sorted(pr, key=pr.get, reverse=True)[:max_nodes]
            subgraph = self.graph.subgraph(top_nodes).copy()
        else:
            subgraph = self.graph

        self._plot_graph(
            subgraph, output_path, figsize, title, dpi, show_labels, font_size
        )

    def plot_subgraph(
        self,
        subgraph: nx.DiGraph,
        output_path: str | Path,
        figsize: tuple[int, int] = (12, 8),
        title: str = "Code Subgraph",
        dpi: int = 300,
        show_labels: bool = True,
        font_size: int = 9,
    ):
        """Plot a specific subgraph (e.g., from neighborhood extraction)."""
        if not MATPLOTLIB_AVAILABLE:
            logger.error("matplotlib not available.")
            return
        self._plot_graph(
            subgraph, output_path, figsize, title, dpi, show_labels, font_size
        )

    def plot_navigation_comparison(
        self,
        graph: nx.DiGraph,
        agent_path_nodes: list[str],
        ground_truth_nodes: list[str],
        output_path: str | Path,
        figsize: tuple[int, int] = (14, 10),
        title: str = "Agent Navigation vs. Ground Truth",
        dpi: int = 300,
    ):
        """
        Plot showing the agent's navigation path vs. ground truth locations.
        Useful for paper figure showing how graph-guided navigation converges
        faster toward the bug location.

        Args:
            graph: The code graph or subgraph to plot.
            agent_path_nodes: Node IDs the agent visited (in order).
            ground_truth_nodes: Node IDs of ground truth bug locations.
            output_path: Where to save the figure.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Layout
        pos = nx.spring_layout(graph, k=2.0, iterations=50, seed=42)

        # Draw base graph
        nx.draw_networkx_edges(
            graph, pos, ax=ax, alpha=0.15, edge_color="#CCCCCC",
            arrows=True, arrowsize=8,
        )

        # Color nodes
        node_colors_list = []
        node_sizes_list = []
        for node in graph.nodes():
            if node in ground_truth_nodes:
                node_colors_list.append("#E74C3C")  # Red for ground truth
                node_sizes_list.append(600)
            elif node in agent_path_nodes:
                # Gradient based on visit order
                idx = agent_path_nodes.index(node)
                intensity = 0.3 + 0.7 * (idx / max(len(agent_path_nodes) - 1, 1))
                node_colors_list.append(plt.cm.Blues(intensity))
                node_sizes_list.append(400)
            else:
                node_colors_list.append("#EEEEEE")
                node_sizes_list.append(150)

        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_color=node_colors_list,
            node_size=node_sizes_list,
            edgecolors="#333333",
            linewidths=0.5,
        )

        # Draw agent path with arrows
        if len(agent_path_nodes) > 1:
            path_edges = [
                (agent_path_nodes[i], agent_path_nodes[i + 1])
                for i in range(len(agent_path_nodes) - 1)
                if agent_path_nodes[i] in graph and agent_path_nodes[i + 1] in graph
            ]
            if path_edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=path_edges, ax=ax,
                    edge_color="#3498DB", width=2.5, alpha=0.8,
                    arrows=True, arrowsize=15,
                    connectionstyle="arc3,rad=0.1",
                )

        # Labels
        labels = {}
        for node in graph.nodes():
            if node in agent_path_nodes or node in ground_truth_nodes:
                labels[node] = graph.nodes[node].get("name", "")
        nx.draw_networkx_labels(
            graph, pos, labels=labels, ax=ax, font_size=8, font_weight="bold"
        )

        # Legend
        legend_elements = [
            mpatches.Patch(color="#E74C3C", label="Ground Truth (Bug Location)"),
            mpatches.Patch(color="#3498DB", label="Agent Path"),
            mpatches.Patch(color="#EEEEEE", label="Other Nodes"),
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=10)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Navigation comparison plot saved to {output_path}")

    def plot_reward_landscape(
        self,
        distances: list[float],
        rewards: list[float],
        output_path: str | Path,
        figsize: tuple[int, int] = (10, 6),
        title: str = "Graph-Guided Reward vs. Distance to Bug",
        dpi: int = 300,
    ):
        """
        Plot showing how the graph-guided reward changes with distance
        from the ground truth. Useful for demonstrating the dense reward
        signal properties.
        """
        if not MATPLOTLIB_AVAILABLE:
            return

        fig, ax = plt.subplots(1, 1, figsize=figsize)

        ax.plot(distances, rewards, "o-", color="#3498DB", linewidth=2, markersize=6)
        ax.fill_between(distances, rewards, alpha=0.15, color="#3498DB")

        ax.set_xlabel("Graph Distance to Bug Location", fontsize=12)
        ax.set_ylabel("Localization Reward", fontsize=12)
        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.grid(True, alpha=0.3)
        ax.set_ylim(-0.05, 1.05)

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Reward landscape plot saved to {output_path}")

    # ── Internal plotting helper ──────────────────────────────────

    def _plot_graph(
        self,
        graph: nx.DiGraph,
        output_path: str | Path,
        figsize: tuple[int, int],
        title: str,
        dpi: int,
        show_labels: bool,
        font_size: int,
    ):
        """Internal method to render a graph to an image file."""
        fig, ax = plt.subplots(1, 1, figsize=figsize)

        # Use spring layout
        pos = nx.spring_layout(graph, k=1.5, iterations=50, seed=42)

        # Color and size nodes by type
        node_colors = []
        node_sizes = []
        for node in graph.nodes():
            ntype = graph.nodes[node].get("type", "unknown")
            node_colors.append(NODE_COLORS.get(ntype, "#CCCCCC"))
            node_sizes.append(NODE_SIZES.get(ntype, 200))

        # Draw edges by type
        for etype in EDGE_COLORS:
            edges = [
                (u, v) for u, v, d in graph.edges(data=True)
                if d.get("type") == etype
            ]
            if edges:
                nx.draw_networkx_edges(
                    graph, pos, edgelist=edges, ax=ax,
                    edge_color=EDGE_COLORS[etype],
                    style=EDGE_STYLES.get(etype, "solid"),
                    alpha=0.5, arrows=True, arrowsize=8,
                )

        # Draw nodes
        nx.draw_networkx_nodes(
            graph, pos, ax=ax,
            node_color=node_colors,
            node_size=node_sizes,
            edgecolors="#333333",
            linewidths=0.5,
        )

        # Labels
        if show_labels:
            labels = {
                n: graph.nodes[n].get("name", "")[:20]
                for n in graph.nodes()
            }
            nx.draw_networkx_labels(
                graph, pos, labels=labels, ax=ax,
                font_size=font_size, font_weight="bold",
            )

        # Legend
        legend_elements = [
            mpatches.Patch(color=color, label=ntype.capitalize())
            for ntype, color in NODE_COLORS.items()
        ]
        ax.legend(handles=legend_elements, loc="upper left", fontsize=9)

        ax.set_title(title, fontsize=14, fontweight="bold")
        ax.axis("off")

        plt.tight_layout()
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        plt.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
        plt.close()
        logger.info(f"Graph plot saved to {output_path}")

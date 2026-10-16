"""
graph_builder.py — Constructs a heterogeneous directed code graph from
parsed AST modules.

Takes the output of ASTParser (list of ParsedModule) and builds a
NetworkX DiGraph with typed nodes (function, class, module) and typed
edges (calls, imports, inherits_from, contains, references).

This graph is the core data structure used for:
  1. Graph-guided reward shaping during RL training
  2. Efficient fault localization at inference time
  3. Measuring navigation accuracy in evaluation
"""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any, Optional

import networkx as nx
from loguru import logger

from graphrl_swe.graph.ast_parser import (
    ASTParser,
    ParsedModule,
    ParsedFunction,
    ParsedClass,
    ParsedImport,
)


# ── Node & Edge Type Constants ────────────────────────────────────

class NodeType:
    MODULE = "module"
    CLASS = "class"
    FUNCTION = "function"
    METHOD = "method"
    VARIABLE = "variable"


class EdgeType:
    CONTAINS = "contains"        # module → class, module → function
    CALLS = "calls"              # function → function
    IMPORTS = "imports"          # module → module
    INHERITS = "inherits_from"   # class → class
    REFERENCES = "references"    # function → variable / class
    HAS_METHOD = "has_method"    # class → method


# ── Code Graph Builder ────────────────────────────────────────────

@dataclass
class GraphStats:
    """Statistics about the constructed code graph."""
    num_nodes: int = 0
    num_edges: int = 0
    num_modules: int = 0
    num_classes: int = 0
    num_functions: int = 0
    num_methods: int = 0
    num_variables: int = 0
    edge_type_counts: dict[str, int] = field(default_factory=dict)


class CodeGraphBuilder:
    """
    Builds a heterogeneous directed code graph from parsed repository modules.

    The graph structure:
      - Nodes have attributes: type, name, qualified_name, filepath, lines, etc.
      - Edges have attributes: type (relationship kind), weight.

    Usage:
        parser = ASTParser()
        modules = parser.parse_repository("/path/to/repo")

        builder = CodeGraphBuilder()
        graph = builder.build(modules)

        # Or directly from a repo path:
        graph = CodeGraphBuilder.from_repository("/path/to/repo")
    """

    def __init__(self, max_nodes: int = 5000):
        """
        Args:
            max_nodes: Maximum number of nodes in the graph (safety cap).
        """
        self.max_nodes = max_nodes
        self.graph: nx.DiGraph = nx.DiGraph()
        self._qualified_name_to_id: dict[str, str] = {}
        self._simple_name_to_ids: dict[str, list[str]] = {}
        self._stats = GraphStats()

    @classmethod
    def from_repository(
        cls,
        repo_path: str | Path,
        max_nodes: int = 5000,
        max_files: int = 500,
    ) -> nx.DiGraph:
        """
        Convenience method: parse a repository and build the graph in one step.

        Args:
            repo_path: Path to the repository root.
            max_nodes: Maximum graph nodes.
            max_files: Maximum files to parse.

        Returns:
            A NetworkX DiGraph representing the code structure.
        """
        parser = ASTParser()
        modules = parser.parse_repository(repo_path, max_files=max_files)
        builder = cls(max_nodes=max_nodes)
        return builder.build(modules)

    def build(self, modules: list[ParsedModule]) -> nx.DiGraph:
        """
        Build the code graph from a list of parsed modules.

        Args:
            modules: Output from ASTParser.parse_repository().

        Returns:
            A NetworkX DiGraph with typed nodes and edges.
        """
        self.graph = nx.DiGraph()
        self._qualified_name_to_id = {}
        self._simple_name_to_ids = {}
        self._stats = GraphStats()

        logger.info(f"Building code graph from {len(modules)} modules...")

        # Phase 1: Add all nodes (modules, classes, functions, variables)
        for module in modules:
            self._add_module_node(module)
            self._add_class_nodes(module)
            self._add_function_nodes(module)
            self._add_variable_nodes(module)

            if self.graph.number_of_nodes() >= self.max_nodes:
                logger.warning(
                    f"Reached max node limit ({self.max_nodes}). "
                    f"Stopping node creation."
                )
                break

        # Phase 2: Add all edges (calls, imports, inherits, contains)
        for module in modules:
            self._add_containment_edges(module)
            self._add_import_edges(module)
            self._add_inheritance_edges(module)
            self._add_call_edges(module)

        # Compute stats
        self._compute_stats()

        logger.info(
            f"Code graph built: {self._stats.num_nodes} nodes, "
            f"{self._stats.num_edges} edges"
        )
        logger.info(
            f"  Modules: {self._stats.num_modules}, "
            f"Classes: {self._stats.num_classes}, "
            f"Functions: {self._stats.num_functions}, "
            f"Methods: {self._stats.num_methods}"
        )

        # Store stats as graph attribute
        self.graph.graph["stats"] = asdict(self._stats)
        self.graph.graph["name_to_id"] = dict(self._qualified_name_to_id)

        return self.graph

    # ── Node creation ─────────────────────────────────────────────

    def _make_node_id(self, node_type: str, qualified_name: str) -> str:
        """Create a unique, deterministic node ID."""
        raw = f"{node_type}::{qualified_name}"
        # Use a short hash to keep IDs manageable
        short_hash = hashlib.md5(raw.encode()).hexdigest()[:8]
        return f"{node_type}_{short_hash}"

    def _register_name(self, node_id: str, qualified_name: str, simple_name: str):
        """Register a node's names for lookup."""
        self._qualified_name_to_id[qualified_name] = node_id
        if simple_name not in self._simple_name_to_ids:
            self._simple_name_to_ids[simple_name] = []
        self._simple_name_to_ids[simple_name].append(node_id)

    def _add_module_node(self, module: ParsedModule):
        """Add a module node to the graph."""
        node_id = self._make_node_id(NodeType.MODULE, module.module_name)
        self.graph.add_node(
            node_id,
            type=NodeType.MODULE,
            name=module.module_name,
            qualified_name=module.module_name,
            filepath=module.filepath,
            lines_of_code=module.lines_of_code,
        )
        self._register_name(node_id, module.module_name, module.module_name.split(".")[-1])

    def _add_class_nodes(self, module: ParsedModule):
        """Add class nodes for all classes in a module."""
        for cls in module.classes:
            node_id = self._make_node_id(NodeType.CLASS, cls.qualified_name)
            self.graph.add_node(
                node_id,
                type=NodeType.CLASS,
                name=cls.name,
                qualified_name=cls.qualified_name,
                filepath=cls.filepath,
                start_line=cls.start_line,
                end_line=cls.end_line,
                base_classes=cls.base_classes,
                docstring=cls.docstring or "",
            )
            self._register_name(node_id, cls.qualified_name, cls.name)

    def _add_function_nodes(self, module: ParsedModule):
        """Add function/method nodes for all functions in a module."""
        for func in module.functions:
            ntype = NodeType.METHOD if func.is_method else NodeType.FUNCTION
            node_id = self._make_node_id(ntype, func.qualified_name)
            self.graph.add_node(
                node_id,
                type=ntype,
                name=func.name,
                qualified_name=func.qualified_name,
                filepath=func.filepath,
                start_line=func.start_line,
                end_line=func.end_line,
                parent_class=func.parent_class or "",
                parameters=func.parameters,
                num_calls=len(func.calls),
                docstring=func.docstring or "",
            )
            self._register_name(node_id, func.qualified_name, func.name)

    def _add_variable_nodes(self, module: ParsedModule):
        """Add global variable nodes."""
        for var_name in module.global_variables:
            qualified = f"{module.module_name}.{var_name}"
            node_id = self._make_node_id(NodeType.VARIABLE, qualified)
            self.graph.add_node(
                node_id,
                type=NodeType.VARIABLE,
                name=var_name,
                qualified_name=qualified,
                filepath=module.filepath,
            )
            self._register_name(node_id, qualified, var_name)

    # ── Edge creation ─────────────────────────────────────────────

    def _add_containment_edges(self, module: ParsedModule):
        """
        Add 'contains' edges: module → classes, module → functions.
        Add 'has_method' edges: class → methods.
        """
        mod_id = self._qualified_name_to_id.get(module.module_name)
        if not mod_id:
            return

        # Module contains classes
        for cls in module.classes:
            cls_id = self._qualified_name_to_id.get(cls.qualified_name)
            if cls_id:
                self.graph.add_edge(
                    mod_id, cls_id, type=EdgeType.CONTAINS, weight=1.0
                )

        # Module contains top-level functions; Class has_method methods
        for func in module.functions:
            func_id = self._qualified_name_to_id.get(func.qualified_name)
            if not func_id:
                continue

            if func.is_method and func.parent_class:
                # Class → method
                cls_qualified = f"{module.module_name}.{func.parent_class}"
                cls_id = self._qualified_name_to_id.get(cls_qualified)
                if cls_id:
                    self.graph.add_edge(
                        cls_id, func_id, type=EdgeType.HAS_METHOD, weight=1.0
                    )
            else:
                # Module → function
                self.graph.add_edge(
                    mod_id, func_id, type=EdgeType.CONTAINS, weight=1.0
                )

        # Module contains variables
        for var_name in module.global_variables:
            var_qualified = f"{module.module_name}.{var_name}"
            var_id = self._qualified_name_to_id.get(var_qualified)
            if var_id:
                self.graph.add_edge(
                    mod_id, var_id, type=EdgeType.CONTAINS, weight=1.0
                )

    def _add_import_edges(self, module: ParsedModule):
        """Add 'imports' edges between modules based on import statements."""
        mod_id = self._qualified_name_to_id.get(module.module_name)
        if not mod_id:
            return

        for imp in module.imports:
            # Try to find the target module in our graph
            target_mod = imp.module
            target_id = self._qualified_name_to_id.get(target_mod)

            if target_id:
                self.graph.add_edge(
                    mod_id, target_id, type=EdgeType.IMPORTS, weight=1.0
                )

            # For 'from X import Y', also try to link to specific symbols
            if imp.is_from_import:
                for name in imp.names:
                    symbol_qualified = f"{imp.module}.{name}"
                    symbol_id = self._qualified_name_to_id.get(symbol_qualified)
                    if symbol_id:
                        self.graph.add_edge(
                            mod_id, symbol_id,
                            type=EdgeType.REFERENCES, weight=0.8,
                        )

    def _add_inheritance_edges(self, module: ParsedModule):
        """Add 'inherits_from' edges between classes."""
        for cls in module.classes:
            cls_id = self._qualified_name_to_id.get(cls.qualified_name)
            if not cls_id:
                continue

            for base_name in cls.base_classes:
                # Try qualified name first
                base_id = self._resolve_name(base_name, module.module_name)
                if base_id:
                    self.graph.add_edge(
                        cls_id, base_id,
                        type=EdgeType.INHERITS, weight=1.0,
                    )

    def _add_call_edges(self, module: ParsedModule):
        """Add 'calls' edges based on function calls within bodies."""
        for func in module.functions:
            caller_id = self._qualified_name_to_id.get(func.qualified_name)
            if not caller_id:
                continue

            for call_name in func.calls:
                callee_id = self._resolve_name(call_name, module.module_name)
                if callee_id and callee_id != caller_id:  # No self-loops
                    self.graph.add_edge(
                        caller_id, callee_id,
                        type=EdgeType.CALLS, weight=1.0,
                    )

    def _resolve_name(self, name: str, current_module: str) -> Optional[str]:
        """
        Resolve a name to a node ID using multiple strategies:
          1. Exact qualified name match
          2. Current module qualified name
          3. Simple name match (returns first match)
        """
        # Strategy 1: Exact qualified name
        if name in self._qualified_name_to_id:
            return self._qualified_name_to_id[name]

        # Strategy 2: Relative to current module
        qualified = f"{current_module}.{name}"
        if qualified in self._qualified_name_to_id:
            return self._qualified_name_to_id[qualified]

        # Strategy 3: Handle dotted names (e.g., self.method, obj.method)
        parts = name.split(".")
        if len(parts) > 1:
            # Try the last part as simple name
            simple = parts[-1]
            if simple in self._simple_name_to_ids:
                candidates = self._simple_name_to_ids[simple]
                if len(candidates) == 1:
                    return candidates[0]
                # If multiple candidates, prefer one in the same module
                for cid in candidates:
                    node_data = self.graph.nodes.get(cid, {})
                    if node_data.get("qualified_name", "").startswith(current_module):
                        return cid
                return candidates[0] if candidates else None

        # Strategy 4: Simple name lookup
        if name in self._simple_name_to_ids:
            candidates = self._simple_name_to_ids[name]
            if len(candidates) == 1:
                return candidates[0]
            # Prefer same module
            for cid in candidates:
                node_data = self.graph.nodes.get(cid, {})
                if node_data.get("qualified_name", "").startswith(current_module):
                    return cid
            return candidates[0] if candidates else None

        return None

    # ── Statistics ────────────────────────────────────────────────

    def _compute_stats(self):
        """Compute graph statistics."""
        self._stats.num_nodes = self.graph.number_of_nodes()
        self._stats.num_edges = self.graph.number_of_edges()

        edge_counts: dict[str, int] = {}
        for _, _, data in self.graph.edges(data=True):
            etype = data.get("type", "unknown")
            edge_counts[etype] = edge_counts.get(etype, 0) + 1
        self._stats.edge_type_counts = edge_counts

        for _, data in self.graph.nodes(data=True):
            ntype = data.get("type", "unknown")
            if ntype == NodeType.MODULE:
                self._stats.num_modules += 1
            elif ntype == NodeType.CLASS:
                self._stats.num_classes += 1
            elif ntype == NodeType.FUNCTION:
                self._stats.num_functions += 1
            elif ntype == NodeType.METHOD:
                self._stats.num_methods += 1
            elif ntype == NodeType.VARIABLE:
                self._stats.num_variables += 1

    def get_stats(self) -> GraphStats:
        """Return the graph statistics."""
        return self._stats

    # ── Serialization ─────────────────────────────────────────────

    def save_graph(self, path: str | Path):
        """Save the graph to a JSON file."""
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)

        data = nx.node_link_data(self.graph)
        with open(path, "w") as f:
            json.dump(data, f, indent=2, default=str)
        logger.info(f"Graph saved to {path}")

    @staticmethod
    def load_graph(path: str | Path) -> nx.DiGraph:
        """Load a graph from a JSON file."""
        with open(path, "r") as f:
            data = json.load(f)
        graph = nx.node_link_graph(data, directed=True)
        logger.info(f"Graph loaded from {path}: {graph.number_of_nodes()} nodes")
        return graph

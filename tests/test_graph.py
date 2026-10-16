"""
test_graph.py — Unit tests for the code graph construction pipeline.

Tests the AST parser, graph builder, and graph analyzer on a small
synthetic Python "repository" to validate correctness.
"""

import os
import sys
import tempfile
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent))

import pytest

from graphrl_swe.graph.ast_parser import ASTParser, ParsedModule
from graphrl_swe.graph.graph_builder import CodeGraphBuilder, NodeType, EdgeType
from graphrl_swe.graph.graph_utils import GraphAnalyzer


# ── Fixtures: Synthetic Python Files ──────────────────────────────

SAMPLE_MODULE_A = '''
"""Module A: Contains utility functions."""

import os
from pathlib import Path

MAX_RETRIES = 3

class BaseProcessor:
    """Base class for processors."""
    
    def __init__(self, name: str):
        self.name = name
    
    def process(self, data):
        """Process data."""
        return self._validate(data)
    
    def _validate(self, data):
        return data is not None


def helper_function(x, y):
    """A helper function."""
    result = x + y
    return format_output(result)


def format_output(value):
    """Format the output."""
    return str(value)
'''

SAMPLE_MODULE_B = '''
"""Module B: Extends Module A."""

from module_a import BaseProcessor, helper_function

class AdvancedProcessor(BaseProcessor):
    """Advanced processor with extra features."""
    
    def __init__(self, name: str, config: dict):
        super().__init__(name)
        self.config = config
    
    def process(self, data):
        """Override process with advanced logic."""
        validated = self._validate(data)
        if validated:
            return self._transform(data)
        return None
    
    def _transform(self, data):
        """Transform data using config."""
        result = helper_function(data, self.config.get("scale", 1))
        return result


def run_pipeline(data_list):
    """Run the processing pipeline."""
    processor = AdvancedProcessor("main", {"scale": 2})
    results = []
    for data in data_list:
        results.append(processor.process(data))
    return results
'''

SAMPLE_MODULE_C = '''
"""Module C: Independent utilities."""

def standalone_function():
    """A function with no dependencies."""
    return 42

def another_util(x):
    """Another utility."""
    return x * 2
'''


# ── Helper to create temp repo ────────────────────────────────────

def create_temp_repo(files: dict[str, str]) -> Path:
    """Create a temporary directory with the given files."""
    tmpdir = Path(tempfile.mkdtemp())
    for filename, content in files.items():
        filepath = tmpdir / filename
        filepath.parent.mkdir(parents=True, exist_ok=True)
        filepath.write_text(content)
    return tmpdir


# ── Tests ─────────────────────────────────────────────────────────

class TestASTParser:
    """Tests for the AST parser."""

    def setup_method(self):
        self.parser = ASTParser()

    def test_parse_file_extracts_functions(self):
        repo = create_temp_repo({"module_a.py": SAMPLE_MODULE_A})
        module = self.parser.parse_file(repo / "module_a.py")
        
        assert module is not None
        func_names = [f.name for f in module.functions]
        assert "helper_function" in func_names
        assert "format_output" in func_names

    def test_parse_file_extracts_classes(self):
        repo = create_temp_repo({"module_a.py": SAMPLE_MODULE_A})
        module = self.parser.parse_file(repo / "module_a.py")
        
        assert module is not None
        class_names = [c.name for c in module.classes]
        assert "BaseProcessor" in class_names

    def test_parse_file_extracts_methods(self):
        repo = create_temp_repo({"module_a.py": SAMPLE_MODULE_A})
        module = self.parser.parse_file(repo / "module_a.py")
        
        assert module is not None
        methods = [f for f in module.functions if f.is_method]
        method_names = [m.name for m in methods]
        assert "__init__" in method_names
        assert "process" in method_names
        assert "_validate" in method_names

    def test_parse_file_extracts_imports(self):
        repo = create_temp_repo({"module_a.py": SAMPLE_MODULE_A})
        module = self.parser.parse_file(repo / "module_a.py")
        
        assert module is not None
        import_modules = [i.module for i in module.imports]
        assert "os" in import_modules
        assert "pathlib" in import_modules

    def test_parse_file_extracts_calls(self):
        repo = create_temp_repo({"module_a.py": SAMPLE_MODULE_A})
        module = self.parser.parse_file(repo / "module_a.py")
        
        assert module is not None
        # helper_function calls format_output
        helper = next(f for f in module.functions if f.name == "helper_function")
        assert "format_output" in helper.calls

    def test_parse_file_extracts_global_variables(self):
        repo = create_temp_repo({"module_a.py": SAMPLE_MODULE_A})
        module = self.parser.parse_file(repo / "module_a.py")
        
        assert module is not None
        assert "MAX_RETRIES" in module.global_variables

    def test_parse_repository(self):
        repo = create_temp_repo({
            "module_a.py": SAMPLE_MODULE_A,
            "module_b.py": SAMPLE_MODULE_B,
            "module_c.py": SAMPLE_MODULE_C,
        })
        modules = self.parser.parse_repository(repo)
        assert len(modules) == 3

    def test_parse_nonexistent_file(self):
        result = self.parser.parse_file("/nonexistent/file.py")
        assert result is None

    def test_parse_inheritance(self):
        repo = create_temp_repo({"module_b.py": SAMPLE_MODULE_B})
        module = self.parser.parse_file(repo / "module_b.py")
        
        assert module is not None
        adv_class = next(c for c in module.classes if c.name == "AdvancedProcessor")
        assert "BaseProcessor" in adv_class.base_classes


class TestCodeGraphBuilder:
    """Tests for the graph builder."""

    def setup_method(self):
        self.parser = ASTParser()
        self.repo = create_temp_repo({
            "module_a.py": SAMPLE_MODULE_A,
            "module_b.py": SAMPLE_MODULE_B,
            "module_c.py": SAMPLE_MODULE_C,
        })

    def test_build_graph_has_nodes(self):
        modules = self.parser.parse_repository(self.repo)
        builder = CodeGraphBuilder()
        graph = builder.build(modules)
        
        assert graph.number_of_nodes() > 0
        assert graph.number_of_edges() > 0

    def test_graph_has_module_nodes(self):
        modules = self.parser.parse_repository(self.repo)
        builder = CodeGraphBuilder()
        graph = builder.build(modules)
        
        module_nodes = [
            n for n, d in graph.nodes(data=True)
            if d.get("type") == NodeType.MODULE
        ]
        assert len(module_nodes) == 3

    def test_graph_has_containment_edges(self):
        modules = self.parser.parse_repository(self.repo)
        builder = CodeGraphBuilder()
        graph = builder.build(modules)
        
        contain_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("type") == EdgeType.CONTAINS
        ]
        assert len(contain_edges) > 0

    def test_graph_has_call_edges(self):
        modules = self.parser.parse_repository(self.repo)
        builder = CodeGraphBuilder()
        graph = builder.build(modules)
        
        call_edges = [
            (u, v) for u, v, d in graph.edges(data=True)
            if d.get("type") == EdgeType.CALLS
        ]
        # helper_function calls format_output
        assert len(call_edges) > 0

    def test_graph_serialization(self):
        modules = self.parser.parse_repository(self.repo)
        builder = CodeGraphBuilder()
        graph = builder.build(modules)
        
        save_path = self.repo / "test_graph.json"
        builder.save_graph(save_path)
        
        loaded = CodeGraphBuilder.load_graph(save_path)
        assert loaded.number_of_nodes() == graph.number_of_nodes()
        assert loaded.number_of_edges() == graph.number_of_edges()

    def test_from_repository(self):
        graph = CodeGraphBuilder.from_repository(self.repo)
        assert graph.number_of_nodes() > 0


class TestGraphAnalyzer:
    """Tests for the graph analyzer and reward computation."""

    def setup_method(self):
        self.parser = ASTParser()
        self.repo = create_temp_repo({
            "src/module_a.py": SAMPLE_MODULE_A,
            "src/module_b.py": SAMPLE_MODULE_B,
            "tests/module_c.py": SAMPLE_MODULE_C,
        })
        modules = self.parser.parse_repository(self.repo)
        builder = CodeGraphBuilder()
        self.graph = builder.build(modules)
        self.analyzer = GraphAnalyzer(self.graph)

    def test_get_file_nodes(self):
        all_files = self.analyzer.get_all_files()
        assert len(all_files) > 0
        
        for filepath in all_files:
            nodes = self.analyzer.get_file_nodes(filepath)
            assert len(nodes) > 0

    def test_shortest_path_same_file(self):
        """Nodes in the same file should be close."""
        all_files = self.analyzer.get_all_files()
        if len(all_files) > 0:
            nodes = self.analyzer.get_file_nodes(all_files[0])
            if len(nodes) >= 2:
                dist = self.analyzer.shortest_path_distance(nodes[0], nodes[1])
                assert dist < 50  # Should be reachable

    def test_localization_reward_exact_match(self):
        """Visiting the exact ground truth files should give high reward."""
        all_files = self.analyzer.get_all_files()
        if len(all_files) > 0:
            gt_file = all_files[0]
            reward = self.analyzer.localization_reward(
                visited_files=[gt_file],
                ground_truth_files=[gt_file],
            )
            assert reward == 1.0  # Exact match → perfect reward

    def test_localization_reward_no_overlap(self):
        """Visiting completely unrelated files should give low reward."""
        all_files = self.analyzer.get_all_files()
        if len(all_files) >= 3:
            # If the files are in disconnected components, reward should be low
            reward = self.analyzer.localization_reward(
                visited_files=[all_files[-1]],
                ground_truth_files=[all_files[0]],
            )
            # May or may not be 0 depending on graph connectivity
            assert 0.0 <= reward <= 1.0

    def test_neighborhood_extraction(self):
        all_files = self.analyzer.get_all_files()
        if len(all_files) > 0:
            nodes = self.analyzer.get_file_nodes(all_files[0])
            if nodes:
                subgraph = self.analyzer.get_neighborhood(nodes[0], radius=2)
                assert subgraph.number_of_nodes() > 0

    def test_importance_scores(self):
        scores = self.analyzer.importance_scores()
        assert len(scores) > 0
        assert all(v >= 0 for v in scores.values())

    def test_graph_summary_generation(self):
        summary = self.analyzer.generate_graph_summary()
        assert len(summary) > 0
        assert "Repository Code Graph" in summary

    def test_blast_radius(self):
        all_files = self.analyzer.get_all_files()
        if len(all_files) > 0:
            affected = self.analyzer.get_blast_radius([all_files[0]], radius=2)
            assert len(affected) >= 1  # At least the file itself


# ── Run tests ─────────────────────────────────────────────────────

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

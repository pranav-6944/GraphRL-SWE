"""
ast_parser.py — Tree-sitter based AST parsing for repository code files.

Extracts structural elements (functions, classes, methods, imports, calls)
from Python source files using tree-sitter. This forms the raw material
for constructing the heterogeneous code graph.
"""

from __future__ import annotations

import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

from loguru import logger

try:
    import tree_sitter_python as tspython
    from tree_sitter import Language, Parser, Node
    TREE_SITTER_AVAILABLE = True
except ImportError:
    TREE_SITTER_AVAILABLE = False
    logger.warning(
        "tree-sitter not installed. Install with: "
        "pip install tree-sitter tree-sitter-python"
    )


# ── Data classes for parsed elements ──────────────────────────────

@dataclass
class ParsedFunction:
    """A parsed function or method definition."""
    name: str
    filepath: str
    start_line: int
    end_line: int
    start_byte: int
    end_byte: int
    # Fully qualified name: module.Class.method or module.function
    qualified_name: str = ""
    parent_class: Optional[str] = None
    decorators: list[str] = field(default_factory=list)
    parameters: list[str] = field(default_factory=list)
    calls: list[str] = field(default_factory=list)
    is_method: bool = False
    docstring: Optional[str] = None
    body_text: str = ""


@dataclass
class ParsedClass:
    """A parsed class definition."""
    name: str
    filepath: str
    start_line: int
    end_line: int
    qualified_name: str = ""
    base_classes: list[str] = field(default_factory=list)
    methods: list[str] = field(default_factory=list)
    decorators: list[str] = field(default_factory=list)
    docstring: Optional[str] = None


@dataclass
class ParsedImport:
    """A parsed import statement."""
    module: str
    names: list[str] = field(default_factory=list)
    alias: Optional[str] = None
    filepath: str = ""
    line: int = 0
    is_from_import: bool = False


@dataclass
class ParsedModule:
    """All parsed elements from a single source file."""
    filepath: str
    module_name: str
    functions: list[ParsedFunction] = field(default_factory=list)
    classes: list[ParsedClass] = field(default_factory=list)
    imports: list[ParsedImport] = field(default_factory=list)
    global_variables: list[str] = field(default_factory=list)
    lines_of_code: int = 0


# ── AST Parser ───────────────────────────────────────────────────

class ASTParser:
    """
    Tree-sitter based AST parser that extracts structural code elements
    from Python source files.

    Usage:
        parser = ASTParser()
        module = parser.parse_file("path/to/file.py")
        repo_modules = parser.parse_repository("path/to/repo")
    """

    # Python file extensions to parse
    PYTHON_EXTENSIONS = {".py"}

    # Directories to skip during repository parsing
    SKIP_DIRS = {
        "__pycache__", ".git", ".hg", ".svn", "node_modules",
        ".tox", ".eggs", "*.egg-info", "venv", ".venv", "env",
        ".env", "dist", "build", ".mypy_cache", ".pytest_cache",
        ".ruff_cache", "migrations",
    }

    # Max file size to parse (skip very large generated files)
    MAX_FILE_SIZE = 1_000_000  # 1 MB

    def __init__(self, language: str = "python"):
        """
        Initialize the AST parser.

        Args:
            language: Programming language to parse. Currently only "python"
                      is fully supported.
        """
        self.language = language
        self.parser: Optional[Parser] = None
        self._init_parser()

    def _init_parser(self):
        """Initialize the tree-sitter parser with the appropriate language."""
        if not TREE_SITTER_AVAILABLE:
            logger.error("tree-sitter is not available. Parser will not work.")
            return

        try:
            if self.language == "python":
                py_language = Language(tspython.language())
                self.parser = Parser(py_language)
                logger.info("Tree-sitter Python parser initialized successfully.")
            else:
                raise ValueError(f"Language '{self.language}' not yet supported.")
        except Exception as e:
            logger.error(f"Failed to initialize tree-sitter parser: {e}")
            self.parser = None

    def parse_file(self, filepath: str | Path) -> Optional[ParsedModule]:
        """
        Parse a single Python file and extract all structural elements.

        Args:
            filepath: Path to the Python source file.

        Returns:
            ParsedModule with all extracted elements, or None on failure.
        """
        filepath = Path(filepath)

        if not filepath.exists():
            logger.warning(f"File not found: {filepath}")
            return None

        if filepath.stat().st_size > self.MAX_FILE_SIZE:
            logger.warning(f"File too large, skipping: {filepath}")
            return None

        if self.parser is None:
            logger.error("Parser not initialized.")
            return None

        try:
            source_code = filepath.read_bytes()
            tree = self.parser.parse(source_code)
        except Exception as e:
            logger.error(f"Failed to parse {filepath}: {e}")
            return None

        source_text = source_code.decode("utf-8", errors="replace")
        lines = source_text.split("\n")

        # Derive module name from filepath
        module_name = filepath.stem

        module = ParsedModule(
            filepath=str(filepath),
            module_name=module_name,
            lines_of_code=len(lines),
        )

        # Extract all elements from the AST
        self._extract_imports(tree.root_node, source_text, module)
        self._extract_classes(tree.root_node, source_text, module)
        self._extract_functions(tree.root_node, source_text, module, parent_class=None)
        self._extract_global_variables(tree.root_node, source_text, module)

        return module

    def parse_repository(
        self,
        repo_path: str | Path,
        max_files: int = 500,
    ) -> list[ParsedModule]:
        """
        Parse all Python files in a repository.

        Args:
            repo_path: Root path of the repository.
            max_files: Maximum number of files to parse (safety cap for huge repos).

        Returns:
            List of ParsedModule objects for each parsed file.
        """
        repo_path = Path(repo_path)
        if not repo_path.is_dir():
            logger.error(f"Repository path is not a directory: {repo_path}")
            return []

        modules = []
        files_parsed = 0

        for py_file in sorted(repo_path.rglob("*.py")):
            # Skip excluded directories
            if any(skip in py_file.parts for skip in self.SKIP_DIRS):
                continue

            if files_parsed >= max_files:
                logger.warning(
                    f"Reached max file limit ({max_files}). "
                    f"Stopping repository parse."
                )
                break

            module = self.parse_file(py_file)
            if module is not None:
                # Update module name to be relative to repo root
                try:
                    rel_path = py_file.relative_to(repo_path)
                    module.module_name = str(rel_path.with_suffix("")).replace(
                        os.sep, "."
                    )
                except ValueError:
                    pass

                modules.append(module)
                files_parsed += 1

        logger.info(
            f"Parsed {files_parsed} Python files from {repo_path.name}"
        )
        return modules

    # ── Private extraction methods ────────────────────────────────

    def _extract_imports(
        self, root: "Node", source: str, module: ParsedModule
    ):
        """Extract import and from-import statements."""
        for node in self._walk_tree(root):
            if node.type == "import_statement":
                self._process_import(node, source, module, is_from=False)
            elif node.type == "import_from_statement":
                self._process_import(node, source, module, is_from=True)

    def _process_import(
        self, node: "Node", source: str, module: ParsedModule, is_from: bool
    ):
        """Process a single import node."""
        try:
            if is_from:
                # from X import Y, Z
                module_node = node.child_by_field_name("module_name")
                mod_name = (
                    self._node_text(module_node, source)
                    if module_node
                    else ""
                )
                names = []
                for child in node.children:
                    if child.type == "dotted_name" and child != module_node:
                        names.append(self._node_text(child, source))
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        if name_node:
                            names.append(self._node_text(name_node, source))

                module.imports.append(
                    ParsedImport(
                        module=mod_name,
                        names=names,
                        filepath=module.filepath,
                        line=node.start_point[0] + 1,
                        is_from_import=True,
                    )
                )
            else:
                # import X, Y
                for child in node.children:
                    if child.type == "dotted_name":
                        module.imports.append(
                            ParsedImport(
                                module=self._node_text(child, source),
                                filepath=module.filepath,
                                line=node.start_point[0] + 1,
                                is_from_import=False,
                            )
                        )
                    elif child.type == "aliased_import":
                        name_node = child.child_by_field_name("name")
                        alias_node = child.child_by_field_name("alias")
                        if name_node:
                            module.imports.append(
                                ParsedImport(
                                    module=self._node_text(name_node, source),
                                    alias=(
                                        self._node_text(alias_node, source)
                                        if alias_node
                                        else None
                                    ),
                                    filepath=module.filepath,
                                    line=node.start_point[0] + 1,
                                    is_from_import=False,
                                )
                            )
        except Exception as e:
            logger.debug(f"Failed to process import at line {node.start_point[0]}: {e}")

    def _extract_classes(
        self, root: "Node", source: str, module: ParsedModule
    ):
        """Extract class definitions and their methods."""
        for node in self._walk_tree(root):
            if node.type == "class_definition":
                self._process_class(node, source, module)

    def _process_class(
        self, node: "Node", source: str, module: ParsedModule
    ):
        """Process a single class definition node."""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            class_name = self._node_text(name_node, source)

            # Extract base classes
            base_classes = []
            arg_list = node.child_by_field_name("superclasses")
            if arg_list:
                for child in arg_list.children:
                    if child.type in ("identifier", "dotted_name", "attribute"):
                        base_classes.append(self._node_text(child, source))

            # Extract decorators
            decorators = self._extract_decorators(node, source)

            # Extract docstring
            docstring = self._extract_docstring(node, source)

            # Extract method names
            methods = []
            body = node.child_by_field_name("body")
            if body:
                for child in body.children:
                    if child.type == "function_definition":
                        method_name = child.child_by_field_name("name")
                        if method_name:
                            methods.append(self._node_text(method_name, source))

                        # Also add as a ParsedFunction with is_method=True
                        self._process_function(
                            child, source, module, parent_class=class_name
                        )

            parsed_class = ParsedClass(
                name=class_name,
                filepath=module.filepath,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                qualified_name=f"{module.module_name}.{class_name}",
                base_classes=base_classes,
                methods=methods,
                decorators=decorators,
                docstring=docstring,
            )
            module.classes.append(parsed_class)

        except Exception as e:
            logger.debug(f"Failed to process class at line {node.start_point[0]}: {e}")

    def _extract_functions(
        self,
        root: "Node",
        source: str,
        module: ParsedModule,
        parent_class: Optional[str],
    ):
        """Extract top-level function definitions (not inside classes)."""
        for node in root.children:
            if node.type == "function_definition":
                self._process_function(node, source, module, parent_class=None)
            elif node.type == "decorated_definition":
                # Decorated top-level function
                for child in node.children:
                    if child.type == "function_definition":
                        self._process_function(
                            child, source, module, parent_class=None
                        )

    def _process_function(
        self,
        node: "Node",
        source: str,
        module: ParsedModule,
        parent_class: Optional[str],
    ):
        """Process a single function/method definition node."""
        try:
            name_node = node.child_by_field_name("name")
            if not name_node:
                return

            func_name = self._node_text(name_node, source)

            # Build qualified name
            if parent_class:
                qualified = f"{module.module_name}.{parent_class}.{func_name}"
            else:
                qualified = f"{module.module_name}.{func_name}"

            # Extract parameters
            parameters = []
            params_node = node.child_by_field_name("parameters")
            if params_node:
                for child in params_node.children:
                    if child.type == "identifier":
                        parameters.append(self._node_text(child, source))
                    elif child.type in (
                        "typed_parameter",
                        "default_parameter",
                        "typed_default_parameter",
                    ):
                        param_name = child.child_by_field_name("name")
                        if param_name:
                            parameters.append(
                                self._node_text(param_name, source)
                            )

            # Extract function calls within the body
            calls = []
            body = node.child_by_field_name("body")
            if body:
                calls = self._extract_calls(body, source)

            # Extract decorators
            decorators = self._extract_decorators(node, source)

            # Extract docstring
            docstring = self._extract_docstring(node, source)

            # Get body text (for embedding/similarity)
            body_text = self._node_text(node, source) if node else ""

            parsed_func = ParsedFunction(
                name=func_name,
                filepath=module.filepath,
                start_line=node.start_point[0] + 1,
                end_line=node.end_point[0] + 1,
                start_byte=node.start_byte,
                end_byte=node.end_byte,
                qualified_name=qualified,
                parent_class=parent_class,
                decorators=decorators,
                parameters=parameters,
                calls=calls,
                is_method=parent_class is not None,
                docstring=docstring,
                body_text=body_text[:2000],  # Cap body text size
            )
            module.functions.append(parsed_func)

        except Exception as e:
            logger.debug(
                f"Failed to process function at line {node.start_point[0]}: {e}"
            )

    def _extract_calls(self, node: "Node", source: str) -> list[str]:
        """Extract all function/method call names within a node subtree."""
        calls = []
        for child in self._walk_tree(node):
            if child.type == "call":
                func_node = child.child_by_field_name("function")
                if func_node:
                    call_name = self._node_text(func_node, source)
                    # Clean up common patterns
                    call_name = call_name.strip()
                    if call_name and len(call_name) < 200:
                        calls.append(call_name)
        return calls

    def _extract_decorators(self, node: "Node", source: str) -> list[str]:
        """Extract decorator names from a function or class definition."""
        decorators = []
        # Check the parent for decorated_definition
        parent = node.parent
        if parent and parent.type == "decorated_definition":
            for child in parent.children:
                if child.type == "decorator":
                    dec_text = self._node_text(child, source).lstrip("@").strip()
                    decorators.append(dec_text)
        return decorators

    def _extract_docstring(self, node: "Node", source: str) -> Optional[str]:
        """Extract the docstring from a function or class body."""
        body = node.child_by_field_name("body")
        if body and body.child_count > 0:
            first_stmt = body.children[0]
            if first_stmt.type == "expression_statement":
                expr = first_stmt.children[0] if first_stmt.child_count > 0 else None
                if expr and expr.type == "string":
                    doc = self._node_text(expr, source)
                    # Strip triple quotes
                    for q in ('"""', "'''"):
                        if doc.startswith(q) and doc.endswith(q):
                            doc = doc[3:-3].strip()
                            break
                    return doc[:500]  # Cap docstring length
        return None

    def _extract_global_variables(
        self, root: "Node", source: str, module: ParsedModule
    ):
        """Extract top-level variable assignments."""
        for node in root.children:
            if node.type in ("expression_statement", "assignment"):
                target = node
                if node.type == "expression_statement" and node.child_count > 0:
                    target = node.children[0]
                if target.type == "assignment":
                    left = target.child_by_field_name("left")
                    if left and left.type == "identifier":
                        var_name = self._node_text(left, source)
                        if var_name and not var_name.startswith("_"):
                            module.global_variables.append(var_name)

    # ── Utility methods ───────────────────────────────────────────

    @staticmethod
    def _node_text(node: "Node", source: str) -> str:
        """Extract the text content of a tree-sitter node."""
        return source[node.start_byte:node.end_byte]

    @staticmethod
    def _walk_tree(node: "Node"):
        """
        Walk the tree-sitter AST in depth-first order.
        Yields every node in the subtree.
        """
        cursor = node.walk()
        visited_children = False

        while True:
            if not visited_children:
                yield cursor.node
                if not cursor.goto_first_child():
                    visited_children = True
            elif cursor.goto_next_sibling():
                visited_children = False
            elif not cursor.goto_parent():
                break

"""Code scanner for project context injection.

Walks a project directory respecting .gitignore patterns, identifies file
languages, and extracts structural information (classes, functions, imports)
from Python files via AST parsing. Non-Python files get a preview of their
first lines.
"""

from __future__ import annotations

import ast
import fnmatch
import logging
from pathlib import Path

from triad.schemas.context import FunctionSignature, ScannedFile

logger = logging.getLogger(__name__)

# Default directories/files to always ignore
_ALWAYS_IGNORE: set[str] = {
    ".git",
    ".hg",
    ".svn",
    "__pycache__",
    ".mypy_cache",
    ".ruff_cache",
    ".pytest_cache",
    "node_modules",
    ".venv",
    "venv",
    ".env",
    ".tox",
    "dist",
    "build",
    "*.egg-info",
    ".DS_Store",
    "Thumbs.db",
}

# File extension → language mapping
_EXT_LANG: dict[str, str] = {
    ".py": "python",
    ".js": "javascript",
    ".ts": "typescript",
    ".tsx": "typescript",
    ".jsx": "javascript",
    ".rs": "rust",
    ".go": "go",
    ".java": "java",
    ".kt": "kotlin",
    ".rb": "ruby",
    ".php": "php",
    ".c": "c",
    ".cpp": "cpp",
    ".h": "c",
    ".hpp": "cpp",
    ".cs": "csharp",
    ".swift": "swift",
    ".toml": "toml",
    ".yaml": "yaml",
    ".yml": "yaml",
    ".json": "json",
    ".md": "markdown",
    ".html": "html",
    ".css": "css",
    ".sql": "sql",
    ".sh": "shell",
    ".bash": "shell",
    ".dockerfile": "dockerfile",
}

# Max file size to scan (1 MB)
_MAX_FILE_BYTES = 1_048_576

# Lines to preview for non-Python files
_PREVIEW_LINES = 50


class CodeScanner:
    """Scans a project directory and produces ScannedFile entries.

    Respects .gitignore patterns and applies include/exclude glob filters.
    Python files are parsed with AST for structural extraction; other files
    get a text preview of their first lines.
    """

    def __init__(
        self,
        root: str | Path,
        include: list[str] | None = None,
        exclude: list[str] | None = None,
    ) -> None:
        self._root = Path(root).resolve()
        self._include = include or ["*"]
        self._exclude = exclude or []
        self._gitignore_patterns: list[str] = []
        self._load_gitignore()

    def _load_gitignore(self) -> None:
        """Load .gitignore patterns from the project root."""
        gitignore = self._root / ".gitignore"
        if gitignore.exists():
            for line in gitignore.read_text(encoding="utf-8").splitlines():
                stripped = line.strip()
                if stripped and not stripped.startswith("#"):
                    self._gitignore_patterns.append(stripped)

    def _is_ignored(self, rel_path: Path) -> bool:
        """Check if a path should be ignored based on gitignore + default rules."""
        path_str = str(rel_path).replace("\\", "/")
        parts = rel_path.parts

        # Check each path component against always-ignore set
        for part in parts:
            if part in _ALWAYS_IGNORE:
                return True
            for pattern in _ALWAYS_IGNORE:
                if fnmatch.fnmatch(part, pattern):
                    return True

        # Check gitignore patterns
        for pattern in self._gitignore_patterns:
            # Match against full relative path and individual components
            if fnmatch.fnmatch(path_str, pattern):
                return True
            if fnmatch.fnmatch(path_str, f"*/{pattern}"):
                return True
            for part in parts:
                if fnmatch.fnmatch(part, pattern):
                    return True

        return False

    def _matches_include(self, rel_path: Path) -> bool:
        """Check if a file matches any include glob pattern."""
        name = rel_path.name
        path_str = str(rel_path).replace("\\", "/")
        for pattern in self._include:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    def _matches_exclude(self, rel_path: Path) -> bool:
        """Check if a file matches any exclude glob pattern."""
        name = rel_path.name
        path_str = str(rel_path).replace("\\", "/")
        for pattern in self._exclude:
            if fnmatch.fnmatch(name, pattern):
                return True
            if fnmatch.fnmatch(path_str, pattern):
                return True
        return False

    def scan(self) -> list[ScannedFile]:
        """Walk the project and return a list of ScannedFile entries.

        Files are filtered by include/exclude patterns and .gitignore rules.
        Python files are parsed with AST; other files get a text preview.
        """
        if not self._root.exists():
            logger.warning("Context directory does not exist: %s", self._root)
            return []

        files: list[ScannedFile] = []

        for path in sorted(self._root.rglob("*")):
            if not path.is_file():
                continue

            rel = path.relative_to(self._root)

            if self._is_ignored(rel):
                continue

            if not self._matches_include(rel):
                continue

            if self._matches_exclude(rel):
                continue

            try:
                size = path.stat().st_size
            except OSError:
                continue

            if size > _MAX_FILE_BYTES:
                continue

            ext = path.suffix.lower()
            language = _EXT_LANG.get(ext, "unknown")
            rel_str = str(rel).replace("\\", "/")

            scanned = ScannedFile(
                path=rel_str,
                language=language,
                size_bytes=size,
            )

            if language == "python":
                self._parse_python(path, scanned)
            else:
                self._read_preview(path, scanned)

            files.append(scanned)

        logger.info("Scanned %d files in %s", len(files), self._root)
        return files

    def _parse_python(self, path: Path, scanned: ScannedFile) -> None:
        """Extract structural info from a Python file via AST parsing."""
        try:
            source = path.read_text(encoding="utf-8")
        except (UnicodeDecodeError, OSError):
            return

        try:
            tree = ast.parse(source, filename=str(path))
        except SyntaxError:
            # Fall back to preview for files with syntax errors
            self._read_preview(path, scanned)
            return

        # Module docstring
        scanned.docstring = ast.get_docstring(tree)

        # Imports
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.Import):
                for alias in node.names:
                    scanned.imports.append(alias.name)
            elif isinstance(node, ast.ImportFrom):
                module = node.module or ""
                for alias in node.names:
                    scanned.imports.append(f"{module}.{alias.name}")

        # Top-level classes and functions
        for node in ast.iter_child_nodes(tree):
            if isinstance(node, ast.ClassDef):
                scanned.classes.append(node.name)
                # Methods inside classes
                for item in ast.iter_child_nodes(node):
                    if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        scanned.functions.append(
                            _extract_function_sig(item)
                        )
            elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                scanned.functions.append(_extract_function_sig(node))

    def _read_preview(self, path: Path, scanned: ScannedFile) -> None:
        """Read the first N lines of a file as a preview."""
        try:
            text = path.read_text(encoding="utf-8")
            lines = text.splitlines()[:_PREVIEW_LINES]
            scanned.preview = "\n".join(lines)
        except (UnicodeDecodeError, OSError):
            pass


def _extract_function_sig(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> FunctionSignature:
    """Extract a FunctionSignature from an AST function node."""
    args: list[str] = []
    for arg in node.args.args:
        args.append(arg.arg)
    for arg in node.args.kwonlyargs:
        args.append(arg.arg)
    if node.args.vararg:
        args.append(f"*{node.args.vararg.arg}")
    if node.args.kwarg:
        args.append(f"**{node.args.kwarg.arg}")

    return_type: str | None = None
    if node.returns:
        try:
            return_type = ast.unparse(node.returns)
        except Exception:
            return_type = "..."

    decorators: list[str] = []
    for dec in node.decorator_list:
        try:
            decorators.append(ast.unparse(dec))
        except Exception:
            decorators.append("?")

    return FunctionSignature(
        name=node.name,
        args=args,
        return_type=return_type,
        is_async=isinstance(node, ast.AsyncFunctionDef),
        decorators=decorators,
    )

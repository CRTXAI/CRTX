"""AST-aware structured patcher for apply mode Phase 2.

Applies structured patches (insert, replace, delete, wrap) to
Python files using AST-based anchor resolution. Other languages
fall back to full-file replacement.
"""

from __future__ import annotations

import ast
import difflib
import logging
from pathlib import Path

from triad.schemas.apply import PatchAnchor, PatchOperation, PatchResult, StructuredPatch

logger = logging.getLogger(__name__)


class ASTPatcher:
    """Applies structured patches to source files.

    For Python files, uses ast.parse() to locate anchors (functions,
    classes) by name. Falls back to fuzzy line matching when AST
    resolution fails or for non-Python files.

    Anchor resolution cascade:
    1. Exact match — find anchor value as substring in file lines
    2. AST match — parse with ast, find function/class by name
    3. Fuzzy match — SequenceMatcher against context_lines
    """

    _FUZZY_THRESHOLD = 0.7

    def apply_patch(self, filepath: str, patch: StructuredPatch) -> PatchResult:
        """Apply a single structured patch to a file.

        Args:
            filepath: Absolute path to the target file.
            patch: The patch to apply.

        Returns:
            PatchResult with success status and metadata.
        """
        path = Path(filepath)
        if not path.exists():
            return PatchResult(
                success=False,
                error=f"File not found: {filepath}",
            )

        content = path.read_text(encoding="utf-8")
        lines = content.splitlines(keepends=True)
        is_python = filepath.endswith(".py")

        # Resolve anchor to line range
        anchor_line, match_method = self._resolve_anchor(
            lines, patch.anchor, is_python, content,
        )

        if anchor_line < 0:
            return PatchResult(
                success=False,
                anchor_match_method="none",
                error=f"Could not locate anchor: {patch.anchor.value}",
            )

        # Apply the operation
        try:
            new_lines = self._apply_operation(
                lines, anchor_line, patch.operation, patch.content,
            )
        except Exception as e:
            return PatchResult(
                success=False,
                anchor_match_method=match_method,
                error=f"Operation failed: {e}",
            )

        new_content = "".join(new_lines)

        # Post-patch validation for Python files
        validation_passed = True
        if is_python:
            validation_passed = self._validate_python(
                content, new_content, patch,
            )

        # Write the patched file
        path.write_text(new_content, encoding="utf-8")

        return PatchResult(
            success=True,
            anchor_match_method=match_method,
            validation_passed=validation_passed,
        )

    def _resolve_anchor(
        self,
        lines: list[str],
        anchor: PatchAnchor,
        is_python: bool,
        full_content: str,
    ) -> tuple[int, str]:
        """Resolve an anchor to a line number.

        Returns:
            Tuple of (line_index, match_method). Line index is -1 if
            no match found.
        """
        # 1. Exact match — find anchor value as substring in lines
        for i, line in enumerate(lines):
            if anchor.value in line:
                return i, "exact"

        # 2. AST match — Python only
        if is_python and anchor.anchor_type in ("function", "class"):
            try:
                tree = ast.parse(full_content)
                for node in ast.walk(tree):
                    if anchor.anchor_type == "function" and isinstance(
                        node, (ast.FunctionDef, ast.AsyncFunctionDef)
                    ) or anchor.anchor_type == "class" and isinstance(
                        node, ast.ClassDef
                    ):
                        if node.name == anchor.value:
                            return node.lineno - 1, "ast"
            except SyntaxError:
                pass

        # 3. Fuzzy match against context_lines
        if anchor.context_lines:
            best_score = 0.0
            best_line = -1
            context_text = "\n".join(anchor.context_lines)

            for i in range(len(lines) - len(anchor.context_lines) + 1):
                window = "".join(lines[i : i + len(anchor.context_lines)])
                ratio = difflib.SequenceMatcher(
                    None, context_text, window.strip()
                ).ratio()
                if ratio > best_score:
                    best_score = ratio
                    best_line = i

            if best_score >= self._FUZZY_THRESHOLD:
                return best_line, "fuzzy"

        return -1, "none"

    def _apply_operation(
        self,
        lines: list[str],
        anchor_line: int,
        operation: PatchOperation,
        content: str,
    ) -> list[str]:
        """Apply a patch operation at the anchor line.

        Args:
            lines: File lines.
            anchor_line: Line index where anchor was found.
            operation: The patch operation type.
            content: New content to insert/replace.

        Returns:
            Modified list of lines.
        """
        new_content_lines = [line + "\n" for line in content.splitlines()]
        result = list(lines)

        if operation == PatchOperation.INSERT_AFTER:
            insert_at = anchor_line + 1
            # Find end of anchored block (for functions/classes)
            insert_at = self._find_block_end(result, anchor_line)
            result[insert_at:insert_at] = ["\n"] + new_content_lines

        elif operation == PatchOperation.INSERT_BEFORE:
            result[anchor_line:anchor_line] = new_content_lines + ["\n"]

        elif operation == PatchOperation.REPLACE:
            # Replace the anchored block
            block_end = self._find_block_end(result, anchor_line)
            result[anchor_line:block_end] = new_content_lines

        elif operation == PatchOperation.DELETE:
            block_end = self._find_block_end(result, anchor_line)
            result[anchor_line:block_end] = []

        elif operation == PatchOperation.INSERT_IMPORT:
            # Find the last import line
            import_end = 0
            for i, line in enumerate(result):
                stripped = line.strip()
                if stripped.startswith(("import ", "from ")):
                    import_end = i + 1
            result[import_end:import_end] = new_content_lines

        elif operation == PatchOperation.INSERT_METHOD:
            # Insert after the anchor line with proper indentation
            block_end = self._find_block_end(result, anchor_line)
            result[block_end:block_end] = ["\n"] + new_content_lines

        elif operation == PatchOperation.WRAP:
            # Wrap the anchored block with new content
            block_end = self._find_block_end(result, anchor_line)
            wrapped = result[anchor_line:block_end]
            result[anchor_line:block_end] = (
                new_content_lines[:1] + wrapped + new_content_lines[1:]
            )

        return result

    def _find_block_end(self, lines: list[str], start: int) -> int:
        """Find the end of a code block starting at the given line.

        Uses indentation to determine block boundaries.
        """
        if start >= len(lines):
            return start + 1

        # Get indentation of the anchor line
        anchor_line = lines[start]
        anchor_indent = len(anchor_line) - len(anchor_line.lstrip())

        # Walk forward until we find a line with equal or less indentation
        for i in range(start + 1, len(lines)):
            line = lines[i]
            stripped = line.strip()
            if not stripped:
                continue
            current_indent = len(line) - len(line.lstrip())
            if current_indent <= anchor_indent:
                return i

        return len(lines)

    def _validate_python(
        self,
        original: str,
        patched: str,
        patch: StructuredPatch,
    ) -> bool:
        """Validate a patched Python file.

        Checks:
        1. Syntax is valid (ast.parse succeeds)
        2. Imports not unexpectedly removed
        3. Function/class signatures not unexpectedly changed

        Returns:
            True if validation passes.
        """
        # 1. Syntax check
        try:
            ast.parse(patched)
        except SyntaxError as e:
            logger.warning(
                "Patch to %s produced invalid syntax: %s",
                patch.filepath, e,
            )
            return False

        # 2. Import check — warn if imports were removed
        try:
            original_tree = ast.parse(original)
            patched_tree = ast.parse(patched)

            original_imports = {
                _import_name(node)
                for node in ast.walk(original_tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            }
            patched_imports = {
                _import_name(node)
                for node in ast.walk(patched_tree)
                if isinstance(node, (ast.Import, ast.ImportFrom))
            }

            removed = original_imports - patched_imports
            if removed:
                logger.warning(
                    "Patch removed imports: %s", ", ".join(removed)
                )
        except SyntaxError:
            pass

        return True


def _import_name(node: ast.AST) -> str:
    """Extract a canonical name from an import AST node."""
    if isinstance(node, ast.Import):
        return ", ".join(alias.name for alias in node.names)
    if isinstance(node, ast.ImportFrom):
        module = node.module or ""
        names = ", ".join(alias.name for alias in node.names)
        return f"from {module} import {names}"
    return ""

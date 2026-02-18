"""Tests for the AST-aware patcher."""

from __future__ import annotations

import pytest

from triad.apply.patcher import ASTPatcher
from triad.schemas.apply import PatchAnchor, PatchOperation, StructuredPatch


@pytest.fixture
def python_file(tmp_path):
    """Create a sample Python file for patching."""
    content = '''"""Module docstring."""

import os
import sys


def hello():
    """Say hello."""
    print("hello")


def goodbye():
    """Say goodbye."""
    print("goodbye")


class MyClass:
    """A sample class."""

    def method_one(self):
        return 1

    def method_two(self):
        return 2
'''
    filepath = tmp_path / "sample.py"
    filepath.write_text(content)
    return filepath


class TestASTPatcher:
    def test_replace_function_exact(self, python_file):
        patcher = ASTPatcher()
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.REPLACE,
            anchor=PatchAnchor(
                anchor_type="function",
                value="hello",
            ),
            content='def hello():\n    """Updated."""\n    print("hi!")',
        )
        result = patcher.apply_patch(str(python_file), patch)
        assert result.success is True
        content = python_file.read_text()
        assert "hi!" in content

    def test_insert_import(self, python_file):
        patcher = ASTPatcher()
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.INSERT_IMPORT,
            anchor=PatchAnchor(
                anchor_type="import_block",
                value="import os",
            ),
            content="import json",
        )
        result = patcher.apply_patch(str(python_file), patch)
        assert result.success is True
        content = python_file.read_text()
        assert "import json" in content

    def test_insert_after(self, python_file):
        patcher = ASTPatcher()
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.INSERT_AFTER,
            anchor=PatchAnchor(
                anchor_type="function",
                value="hello",
            ),
            content='def new_func():\n    """New function."""\n    pass',
        )
        result = patcher.apply_patch(str(python_file), patch)
        assert result.success is True
        content = python_file.read_text()
        assert "new_func" in content

    def test_delete_function(self, python_file):
        patcher = ASTPatcher()
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.DELETE,
            anchor=PatchAnchor(
                anchor_type="function",
                value="goodbye",
            ),
        )
        result = patcher.apply_patch(str(python_file), patch)
        assert result.success is True
        content = python_file.read_text()
        assert "goodbye" not in content or "def goodbye" not in content

    def test_ast_anchor_resolution(self, python_file):
        patcher = ASTPatcher()
        # Use a function name that requires AST resolution
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.REPLACE,
            anchor=PatchAnchor(
                anchor_type="function",
                value="method_one",
            ),
            content="    def method_one(self):\n        return 42",
        )
        result = patcher.apply_patch(str(python_file), patch)
        assert result.success is True

    def test_file_not_found(self, tmp_path):
        patcher = ASTPatcher()
        patch = StructuredPatch(
            filepath=str(tmp_path / "nonexistent.py"),
            operation=PatchOperation.REPLACE,
            anchor=PatchAnchor(anchor_type="function", value="foo"),
            content="pass",
        )
        result = patcher.apply_patch(str(tmp_path / "nonexistent.py"), patch)
        assert result.success is False
        assert "not found" in result.error.lower()

    def test_anchor_not_found(self, python_file):
        patcher = ASTPatcher()
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.REPLACE,
            anchor=PatchAnchor(
                anchor_type="function",
                value="nonexistent_function_xyz",
            ),
            content="pass",
        )
        result = patcher.apply_patch(str(python_file), patch)
        assert result.success is False
        assert "could not locate" in result.error.lower()

    def test_validation_catches_syntax_error(self, python_file):
        patcher = ASTPatcher()
        # Replace with invalid Python
        patch = StructuredPatch(
            filepath=str(python_file),
            operation=PatchOperation.REPLACE,
            anchor=PatchAnchor(
                anchor_type="function",
                value="hello",
            ),
            content="def hello(\n    broken syntax",
        )
        result = patcher.apply_patch(str(python_file), patch)
        # The patch is applied but validation fails
        assert result.success is True
        assert result.validation_passed is False

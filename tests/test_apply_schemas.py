"""Tests for apply mode schemas."""

from __future__ import annotations

from triad.schemas.apply import (
    ApplyConfig,
    ApplyResult,
    FileAction,
    GitState,
    PatchAnchor,
    PatchOperation,
    PatchResult,
    ResolvedFile,
    StructuredPatch,
)


class TestFileAction:
    def test_values(self):
        assert FileAction.CREATE == "create"
        assert FileAction.OVERWRITE == "overwrite"
        assert FileAction.SKIP == "skip"


class TestApplyConfig:
    def test_defaults(self):
        config = ApplyConfig()
        assert config.enabled is False
        assert config.confirm is True
        assert config.branch == ""
        assert config.apply_include == []
        assert config.apply_exclude == []
        assert config.rollback_on_fail is True
        assert config.test_command == ""

    def test_custom(self):
        config = ApplyConfig(
            enabled=True,
            confirm=False,
            branch="feat/apply",
            apply_include=["*.py"],
            apply_exclude=["test_*.py"],
            rollback_on_fail=False,
            test_command="pytest",
        )
        assert config.enabled is True
        assert config.branch == "feat/apply"


class TestResolvedFile:
    def test_create_action(self):
        rf = ResolvedFile(
            source_filepath="src/main.py",
            resolved_path="/project/src/main.py",
            action=FileAction.CREATE,
            content="print('hello')",
        )
        assert rf.action == FileAction.CREATE
        assert rf.match_confidence == 1.0
        assert rf.selected is True
        assert rf.existing_content is None

    def test_overwrite_action(self):
        rf = ResolvedFile(
            source_filepath="src/main.py",
            resolved_path="/project/src/main.py",
            action=FileAction.OVERWRITE,
            content="print('hello')",
            existing_content="print('old')",
            match_confidence=0.9,
        )
        assert rf.action == FileAction.OVERWRITE
        assert rf.existing_content == "print('old')"


class TestGitState:
    def test_defaults(self):
        state = GitState()
        assert state.is_git_repo is False
        assert state.is_clean is True
        assert state.current_branch == ""
        assert state.head_sha == ""

    def test_full(self):
        state = GitState(
            is_git_repo=True,
            is_clean=False,
            current_branch="main",
            head_sha="abc123",
            created_branch="feat/apply",
        )
        assert state.current_branch == "main"
        assert state.created_branch == "feat/apply"


class TestApplyResult:
    def test_defaults(self):
        result = ApplyResult()
        assert result.session_id == ""
        assert result.files_applied == []
        assert result.files_skipped == []
        assert result.commit_sha == ""
        assert result.test_passed is None
        assert result.rolled_back is False
        assert result.errors == []


class TestPatchOperation:
    def test_values(self):
        assert PatchOperation.INSERT_AFTER == "insert_after"
        assert PatchOperation.REPLACE == "replace"
        assert PatchOperation.DELETE == "delete"
        assert PatchOperation.INSERT_IMPORT == "insert_import"
        assert PatchOperation.INSERT_METHOD == "insert_method"
        assert PatchOperation.WRAP == "wrap"


class TestPatchAnchor:
    def test_basic(self):
        anchor = PatchAnchor(
            anchor_type="function",
            value="my_func",
            context_lines=["def my_func():", "    pass"],
        )
        assert anchor.anchor_type == "function"
        assert anchor.value == "my_func"
        assert len(anchor.context_lines) == 2


class TestStructuredPatch:
    def test_basic(self):
        patch = StructuredPatch(
            filepath="src/main.py",
            operation=PatchOperation.REPLACE,
            anchor=PatchAnchor(anchor_type="function", value="main"),
            content="def main():\n    print('new')",
            explanation="Updated main function",
        )
        assert patch.filepath == "src/main.py"
        assert patch.operation == PatchOperation.REPLACE


class TestPatchResult:
    def test_success(self):
        result = PatchResult(success=True, anchor_match_method="ast")
        assert result.success is True
        assert result.validation_passed is True

    def test_failure(self):
        result = PatchResult(
            success=False,
            error="Anchor not found",
        )
        assert result.success is False
        assert result.error == "Anchor not found"

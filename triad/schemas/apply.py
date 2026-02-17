"""Apply mode schemas for safe file writing and intelligent patching.

Defines configuration, resolution, and result models for applying
pipeline-generated code to disk. Includes Phase 1 (direct write)
and Phase 2 (structured patching) schemas.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


# ── Phase 1: Safe Direct Write ────────────────────────────────────


class FileAction(StrEnum):
    """Action to take for a resolved file."""

    CREATE = "create"
    OVERWRITE = "overwrite"
    SKIP = "skip"


class ApplyConfig(BaseModel):
    """CLI flags container for apply mode."""

    enabled: bool = Field(default=False, description="Enable apply mode")
    confirm: bool = Field(default=True, description="Interactive confirmation before writing")
    branch: str = Field(default="", description="Git branch name to create for apply")
    apply_include: list[str] = Field(
        default_factory=list, description="Glob patterns for files to include"
    )
    apply_exclude: list[str] = Field(
        default_factory=list, description="Glob patterns for files to exclude"
    )
    rollback_on_fail: bool = Field(
        default=True, description="Rollback changes if post-apply tests fail"
    )
    test_command: str = Field(default="", description="Test command to run after apply")


class ResolvedFile(BaseModel):
    """A code block mapped to a disk path for writing."""

    source_filepath: str = Field(description="Original filepath hint from code block")
    resolved_path: str = Field(description="Absolute path on disk")
    action: FileAction = Field(description="Whether to create, overwrite, or skip")
    content: str = Field(description="File content to write")
    language: str = Field(default="text", description="Programming language")
    existing_content: str | None = Field(
        default=None, description="Content of existing file (for diff)"
    )
    match_confidence: float = Field(
        default=1.0, ge=0.0, le=1.0, description="Path resolution confidence"
    )
    selected: bool = Field(default=True, description="Whether user selected this file for apply")


class GitState(BaseModel):
    """Snapshot of git repository state before apply."""

    is_git_repo: bool = Field(default=False, description="Whether the context dir is a git repo")
    is_clean: bool = Field(default=True, description="Whether the working tree is clean")
    current_branch: str = Field(default="", description="Current branch name")
    head_sha: str = Field(default="", description="HEAD commit SHA")
    created_branch: str = Field(default="", description="Branch created for apply")
    stash_created: bool = Field(default=False, description="Whether a stash was created")


class ApplyResult(BaseModel):
    """Audit record for an apply operation."""

    session_id: str = Field(default="", description="Pipeline session ID")
    files_applied: list[ResolvedFile] = Field(
        default_factory=list, description="Files that were written"
    )
    files_skipped: list[ResolvedFile] = Field(
        default_factory=list, description="Files that were skipped"
    )
    git_state: GitState = Field(
        default_factory=GitState, description="Git state snapshot"
    )
    commit_sha: str = Field(default="", description="Commit SHA after apply")
    test_passed: bool | None = Field(
        default=None, description="Post-apply test result (None if not run)"
    )
    test_output: str = Field(default="", description="Test command output")
    rolled_back: bool = Field(default=False, description="Whether changes were rolled back")
    errors: list[str] = Field(default_factory=list, description="Errors encountered during apply")


# ── Phase 2: Intelligent Patching ─────────────────────────────────


class PatchOperation(StrEnum):
    """Types of structured patch operations."""

    INSERT_AFTER = "insert_after"
    INSERT_BEFORE = "insert_before"
    REPLACE = "replace"
    DELETE = "delete"
    INSERT_IMPORT = "insert_import"
    INSERT_METHOD = "insert_method"
    WRAP = "wrap"


class PatchAnchor(BaseModel):
    """Semantic location for a patch operation."""

    anchor_type: str = Field(
        description="Type of anchor: function, class, line_pattern, import_block"
    )
    value: str = Field(description="Anchor value (e.g. function name, class name, pattern)")
    context_lines: list[str] = Field(
        default_factory=list,
        description="Surrounding lines for fuzzy matching",
    )


class StructuredPatch(BaseModel):
    """A single structured edit operation."""

    filepath: str = Field(description="Target file path")
    operation: PatchOperation = Field(description="What kind of edit to perform")
    anchor: PatchAnchor = Field(description="Where to apply the edit")
    content: str = Field(default="", description="New content to insert/replace")
    explanation: str = Field(default="", description="Why this edit is needed")


class PatchResult(BaseModel):
    """Result of applying a single patch."""

    success: bool = Field(description="Whether the patch was applied successfully")
    anchor_match_method: str = Field(
        default="exact", description="How the anchor was resolved: exact, ast, fuzzy"
    )
    validation_passed: bool = Field(
        default=True, description="Whether post-patch validation passed"
    )
    error: str = Field(default="", description="Error message if patch failed")

"""Context injection schemas.

Defines models for scanned files, function signatures, project profiles,
and context assembly results used by the context injection system.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class FunctionSignature(BaseModel):
    """Extracted function or method signature from AST parsing."""

    name: str = Field(description="Function or method name")
    args: list[str] = Field(default_factory=list, description="Argument names")
    return_type: str | None = Field(
        default=None, description="Return type annotation if present"
    )
    is_async: bool = Field(default=False, description="Whether the function is async")
    decorators: list[str] = Field(
        default_factory=list, description="Decorator names"
    )


class ScannedFile(BaseModel):
    """A file discovered and analyzed by the CodeScanner."""

    path: str = Field(description="Relative path from the project root")
    language: str = Field(default="unknown", description="Inferred programming language")
    size_bytes: int = Field(default=0, ge=0, description="File size in bytes")
    classes: list[str] = Field(
        default_factory=list, description="Class names found (Python AST)"
    )
    functions: list[FunctionSignature] = Field(
        default_factory=list, description="Function signatures (Python AST)"
    )
    imports: list[str] = Field(
        default_factory=list, description="Import statements (Python AST)"
    )
    docstring: str | None = Field(
        default=None, description="Module docstring (Python AST)"
    )
    preview: str | None = Field(
        default=None, description="First 50 lines for non-Python files"
    )
    relevance_score: float = Field(
        default=0.0, ge=0.0, description="Computed relevance score for this file"
    )


class ProjectProfile(BaseModel):
    """High-level profile of the scanned project."""

    root_path: str = Field(description="Absolute path to the project root")
    total_files: int = Field(default=0, ge=0, description="Total files scanned")
    total_lines: int = Field(default=0, ge=0, description="Approximate total lines")
    languages: dict[str, int] = Field(
        default_factory=dict, description="Language → file count mapping"
    )
    entry_points: list[str] = Field(
        default_factory=list, description="Likely entry point files"
    )
    key_patterns: list[str] = Field(
        default_factory=list, description="Key patterns detected (e.g. 'FastAPI', 'Django')"
    )


class ContextResult(BaseModel):
    """Result of the context building process."""

    profile: ProjectProfile = Field(description="Project profile summary")
    context_text: str = Field(default="", description="Assembled context string")
    files_included: int = Field(default=0, ge=0, description="Files included in context")
    files_scanned: int = Field(default=0, ge=0, description="Total files scanned")
    token_estimate: int = Field(default=0, ge=0, description="Estimated token count")
    truncated: bool = Field(
        default=False, description="Whether the context was truncated to fit budget"
    )

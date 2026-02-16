"""CI/CD review schemas.

Defines configuration, finding, assessment, and result models for the
multi-model parallel code review system used in CI pipelines.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class ReviewConfig(BaseModel):
    """Configuration for a CI review run."""

    models: list[str] | None = Field(
        default=None,
        description="Model keys to use (None = all registered models)",
    )
    focus_areas: list[str] = Field(
        default_factory=list,
        description="Focus areas: security, performance, correctness, etc.",
    )
    arbiter_enabled: bool = Field(
        default=True,
        description="Whether to cross-validate findings through the Arbiter",
    )
    context_dir: str | None = Field(
        default=None,
        description="Project directory for additional context",
    )
    max_cost: float | None = Field(
        default=None, ge=0.0,
        description="Maximum budget for the review in USD",
    )


class ReviewFinding(BaseModel):
    """A single finding from a code review."""

    severity: str = Field(
        description="Finding severity: critical, warning, or suggestion",
    )
    file: str = Field(
        description="File path where the finding was identified",
    )
    line: int | None = Field(
        default=None,
        description="Line number (None if not applicable)",
    )
    description: str = Field(
        description="Description of the issue found",
    )
    suggestion: str = Field(
        default="",
        description="Suggested fix or improvement",
    )
    reported_by: list[str] = Field(
        default_factory=list,
        description="Model keys that reported this finding",
    )
    confirmed: bool = Field(
        default=False,
        description="Whether the finding was confirmed by cross-validation",
    )
    confidence: str = Field(
        default="needs_verification",
        description="Confidence level: high or needs_verification",
    )


class ModelAssessment(BaseModel):
    """A single model's assessment of a diff."""

    model_key: str = Field(description="Model registry key")
    recommendation: str = Field(
        description="Overall recommendation: approve or request_changes",
    )
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="Findings from this model",
    )
    rationale: str = Field(
        default="",
        description="Reasoning for the recommendation",
    )
    cost: float = Field(
        default=0.0, ge=0.0,
        description="Cost of this model's review in USD",
    )


class ReviewResult(BaseModel):
    """Consolidated result from a multi-model code review."""

    consensus_recommendation: str = Field(
        description="Consensus: approve or request_changes",
    )
    findings: list[ReviewFinding] = Field(
        default_factory=list,
        description="All findings, deduplicated and cross-validated",
    )
    model_assessments: list[ModelAssessment] = Field(
        default_factory=list,
        description="Per-model assessment details",
    )
    total_findings: int = Field(
        default=0, ge=0,
        description="Total number of findings",
    )
    critical_count: int = Field(
        default=0, ge=0,
        description="Number of critical-severity findings",
    )
    models_used: list[str] = Field(
        default_factory=list,
        description="Model keys that participated in the review",
    )
    total_cost: float = Field(
        default=0.0, ge=0.0,
        description="Total cost of the review in USD",
    )
    duration_seconds: float = Field(
        default=0.0, ge=0.0,
        description="Wall-clock duration of the review",
    )

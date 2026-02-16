"""Arbiter review schemas for the independent adversarial review layer.

Defines the ArbiterReview output schema and all supporting types used by the
Arbiter to evaluate pipeline stage outputs and issue verdicts.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from triad.schemas.messages import PipelineStage


class Severity(StrEnum):
    """Severity levels for issues found during Arbiter review."""

    CRITICAL = "critical"
    WARNING = "warning"
    SUGGESTION = "suggestion"


class IssueCategory(StrEnum):
    """Categories of issues the Arbiter can identify in reviewed output."""

    LOGIC = "logic"
    PATTERN = "pattern"
    SECURITY = "security"
    PERFORMANCE = "performance"
    EDGE_CASE = "edge_case"
    HALLUCINATION = "hallucination"


class Verdict(StrEnum):
    """Arbiter verdict determining pipeline flow control.

    APPROVE: Output is sound, continue to next stage.
    FLAG: Issues detected but not blocking, continue with flags injected.
    REJECT: Significant problems, re-run current stage with feedback. Max 2 retries.
    HALT: Critical failure, stop pipeline for human review.
    """

    APPROVE = "approve"
    FLAG = "flag"
    REJECT = "reject"
    HALT = "halt"


class Issue(BaseModel):
    """A specific problem identified by the Arbiter during review."""

    severity: Severity = Field(description="How critical this issue is")
    category: IssueCategory = Field(description="Classification of the issue type")
    location: str = Field(
        default="", description="File path and line range where the issue occurs"
    )
    description: str = Field(description="What is wrong")
    suggestion: str = Field(default="", description="How to fix the issue")
    evidence: str = Field(default="", description="Why the Arbiter believes this is an issue")


class Alternative(BaseModel):
    """A better approach suggested by the Arbiter as a replacement."""

    description: str = Field(description="What to do differently")
    rationale: str = Field(description="Why this approach is better")
    code_sketch: str = Field(default="", description="Optional implementation hint")
    confidence: float = Field(
        ge=0.0, le=1.0, description="Arbiter's confidence in this alternative"
    )


class ArbiterReview(BaseModel):
    """Output from the Arbiter after reviewing a pipeline stage.

    The Arbiter is an independent referee that evaluates stage outputs using
    cross-model enforcement (arbiter model != generator model). It issues one
    of four verdicts that control pipeline flow.
    """

    stage_reviewed: PipelineStage = Field(description="Which pipeline stage was reviewed")
    reviewed_model: str = Field(description="Model identifier that produced the reviewed output")
    arbiter_model: str = Field(description="Model identifier performing the review")
    verdict: Verdict = Field(description="The Arbiter's verdict on this stage output")
    issues: list[Issue] = Field(
        default_factory=list, description="Structured list of problems found"
    )
    alternatives: list[Alternative] = Field(
        default_factory=list, description="Better approaches suggested by the Arbiter"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Overall confidence in this verdict"
    )
    reasoning: str = Field(description="Full chain-of-thought explanation for the verdict")
    token_cost: float = Field(ge=0.0, description="Cost of this review pass in USD")

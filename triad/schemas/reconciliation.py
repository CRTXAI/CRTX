"""Implementation Summary Reconciliation schemas.

Defines the ImplementationSummary (produced by the Verifier) and Deviation
types used by the reconciliation Arbiter pass to detect spec drift, missing
requirements, and silently dropped features.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from triad.schemas.messages import PipelineStage


class Deviation(BaseModel):
    """An intentional departure from the original task specification.

    Tracks what was changed or omitted, why, and which pipeline stage
    introduced the deviation.
    """

    what: str = Field(description="What was changed or omitted from the spec")
    reason: str = Field(description="Why the agent deviated from the spec")
    stage: PipelineStage = Field(description="Which pipeline stage introduced this deviation")


class ImplementationSummary(BaseModel):
    """Structured manifest of what was actually built by the pipeline.

    Produced by the Verifier after the Verify stage. When --reconcile is
    enabled, a cross-model Arbiter compares this summary against the original
    TaskSpec and Architect scaffold to catch spec drift.
    """

    task_echo: str = Field(description="Verifier's restatement of the original task")
    endpoints_implemented: list[str] = Field(
        default_factory=list, description="API routes or CLI commands delivered"
    )
    schemas_created: list[str] = Field(
        default_factory=list, description="Data models or Pydantic schemas created"
    )
    files_created: list[str] = Field(
        default_factory=list, description="File paths produced by the pipeline"
    )
    files_modified: list[str] = Field(
        default_factory=list, description="Existing files changed during the pipeline"
    )
    behaviors_implemented: list[str] = Field(
        default_factory=list, description="Functional behaviors and business logic delivered"
    )
    test_coverage: list[str] = Field(
        default_factory=list, description="What is tested, by name or description"
    )
    deviations: list[Deviation] = Field(
        default_factory=list, description="Intentional departures from the spec"
    )
    omissions: list[str] = Field(
        default_factory=list, description="Items from the spec NOT implemented"
    )

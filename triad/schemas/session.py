"""Session persistence schemas.

Defines the SessionRecord (full session data for storage/retrieval),
SessionSummary (lightweight listing), and SessionQuery (filter params).
"""

from __future__ import annotations

from datetime import datetime

from pydantic import BaseModel, Field

from triad.schemas.arbiter import ArbiterReview
from triad.schemas.pipeline import PipelineConfig, TaskSpec
from triad.schemas.routing import RoutingDecision


class StageRecord(BaseModel):
    """Persisted record of a single pipeline stage execution."""

    stage: str = Field(description="Pipeline stage name")
    model_key: str = Field(default="", description="Registry key of the model used")
    model_id: str = Field(default="", description="LiteLLM model identifier")
    content: str = Field(default="", description="Stage output content")
    confidence: float = Field(default=0.0, description="Confidence score")
    cost: float = Field(default=0.0, description="Cost in USD for this stage")
    tokens: int = Field(default=0, description="Total tokens for this stage")
    timestamp: str = Field(default="", description="ISO timestamp of execution")


class SessionRecord(BaseModel):
    """Full session record for storage and retrieval.

    Contains all data from a pipeline run: task, config, stage outputs,
    arbiter reviews, routing decisions, and aggregate metrics.
    """

    session_id: str = Field(description="Unique session identifier (UUID)")
    task: TaskSpec = Field(description="The original task specification")
    config: PipelineConfig = Field(description="Pipeline configuration used")
    stages: list[StageRecord] = Field(
        default_factory=list, description="Per-stage execution records",
    )
    arbiter_reviews: list[ArbiterReview] = Field(
        default_factory=list, description="All Arbiter reviews",
    )
    routing_decisions: list[RoutingDecision] = Field(
        default_factory=list, description="Routing decisions per stage",
    )
    started_at: datetime = Field(description="When the pipeline started")
    completed_at: datetime | None = Field(
        default=None, description="When the pipeline completed",
    )
    success: bool = Field(default=False, description="Whether the run succeeded")
    halted: bool = Field(default=False, description="Whether the run was halted")
    halt_reason: str = Field(default="", description="Halt reason if halted")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    total_tokens: int = Field(default=0, description="Total tokens used")
    duration_seconds: float = Field(default=0.0, description="Wall-clock duration")
    pipeline_mode: str = Field(default="sequential", description="Pipeline mode used")


class SessionSummary(BaseModel):
    """Lightweight session summary for listing."""

    session_id: str = Field(description="Unique session identifier")
    task_preview: str = Field(
        description="First 100 characters of the task description",
    )
    pipeline_mode: str = Field(default="sequential", description="Pipeline mode")
    started_at: datetime = Field(description="When the pipeline started")
    success: bool = Field(default=False, description="Whether the run succeeded")
    halted: bool = Field(default=False, description="Whether the run was halted")
    total_cost: float = Field(default=0.0, description="Total cost in USD")
    duration_seconds: float = Field(default=0.0, description="Duration in seconds")
    model_count: int = Field(default=0, description="Number of distinct models used")
    stage_count: int = Field(default=0, description="Number of stages executed")
    arbiter_verdict_summary: str = Field(
        default="", description="Summary of arbiter verdicts (e.g. '2 APPROVE, 1 REJECT')",
    )


class SessionQuery(BaseModel):
    """Query parameters for listing sessions."""

    limit: int = Field(default=20, ge=1, le=100, description="Max sessions to return")
    offset: int = Field(default=0, ge=0, description="Offset for pagination")
    task_filter: str | None = Field(
        default=None, description="Filter by task text (substring match)",
    )
    model_filter: str | None = Field(
        default=None, description="Filter by model key used",
    )
    verdict_filter: str | None = Field(
        default=None, description="Filter by arbiter verdict",
    )
    min_cost: float | None = Field(
        default=None, description="Minimum total cost filter",
    )
    max_cost: float | None = Field(
        default=None, description="Maximum total cost filter",
    )
    since: str | None = Field(
        default=None, description="Filter sessions after this ISO date",
    )

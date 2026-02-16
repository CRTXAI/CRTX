"""Routing decision schemas for the Smart Routing Engine.

Defines the routing strategy enum and the RoutingDecision model
that records which model was selected for each pipeline role and why.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from triad.schemas.messages import PipelineStage


class RoutingStrategy(StrEnum):
    """Available routing strategies for model-to-role assignment.

    Controls how the routing engine selects models for pipeline stages,
    trading off quality, cost, and speed.
    """

    QUALITY_FIRST = "quality_first"
    COST_OPTIMIZED = "cost_optimized"
    SPEED_FIRST = "speed_first"
    HYBRID = "hybrid"


class RoutingDecision(BaseModel):
    """Record of a routing decision for a single pipeline stage.

    Captures which model was selected, by which strategy, and the
    rationale and metrics behind the choice.
    """

    model_key: str = Field(description="Registry key of the selected model")
    model_id: str = Field(description="LiteLLM model identifier")
    role: PipelineStage = Field(description="Pipeline stage this model was assigned to")
    strategy: RoutingStrategy = Field(description="Routing strategy that made this decision")
    rationale: str = Field(description="Human-readable explanation of the selection")
    fitness_score: float = Field(
        ge=0.0, le=1.0, description="Fitness score of the selected model for this role"
    )
    estimated_cost: float = Field(
        ge=0.0, description="Estimated cost in USD for this stage"
    )


class CostEstimate(BaseModel):
    """Cost estimate for a full pipeline run under a given routing strategy.

    Returned by estimate_cost() to help developers compare strategies
    before committing to a run.
    """

    strategy: RoutingStrategy = Field(description="Routing strategy used for this estimate")
    decisions: list[RoutingDecision] = Field(
        default_factory=list, description="Per-stage routing decisions"
    )
    total_estimated_cost: float = Field(
        ge=0.0, description="Total estimated cost in USD for the full pipeline"
    )

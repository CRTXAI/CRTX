"""Task planner schemas.

Defines the result model for the task planner, which expands rough
descriptions into structured task specifications.
"""

from __future__ import annotations

from pydantic import BaseModel, Field

from triad.schemas.messages import TokenUsage
from triad.schemas.pipeline import TaskSpec


class PlannerResult(BaseModel):
    """Output from the task planner.

    Contains the original description, expanded specification, parsed
    TaskSpec, inferred tech stack, model and cost data, and optional
    interactive-mode artifacts (clarifying questions and user answers).
    """

    original_description: str = Field(
        description="The rough description provided by the user",
    )
    expanded_spec: str = Field(
        description="The full structured task specification text",
    )
    task_spec: TaskSpec = Field(
        description="Parsed TaskSpec ready for pipeline execution",
    )
    tech_stack_inferred: list[str] = Field(
        default_factory=list,
        description="Technologies inferred or suggested by the planner",
    )
    model_used: str = Field(
        description="Model key used for planning",
    )
    token_usage: TokenUsage = Field(
        description="Token usage for the planning call(s)",
    )
    cost: float = Field(
        ge=0.0,
        description="Total USD cost of the planning call(s)",
    )
    interactive: bool = Field(
        default=False,
        description="Whether interactive mode was used",
    )
    clarifying_questions: list[str] | None = Field(
        default=None,
        description="Questions asked in interactive mode (phase 1)",
    )
    user_answers: str | None = Field(
        default=None,
        description="User answers to clarifying questions",
    )

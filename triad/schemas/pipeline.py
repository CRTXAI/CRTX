"""Pipeline configuration and task specification schemas.

Defines the configuration models for the pipeline engine, model registry,
role fitness scoring, task specifications, and pipeline results.
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field

from triad.schemas.messages import AgentMessage, PipelineStage, Suggestion


class ArbiterMode(StrEnum):
    """Configurable Arbiter review depth.

    Controls which pipeline stages the Arbiter reviews, trading off
    quality assurance against cost and speed.
    """

    OFF = "off"
    FINAL_ONLY = "final_only"
    BOOKEND = "bookend"
    FULL = "full"


class RoleFitness(BaseModel):
    """Fitness scores for a model across all pipeline roles.

    Scores range from 0.0 to 1.0 and are updated by periodic benchmarks.
    The model with the highest fitness for a role gets default assignment.
    """

    architect: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fitness for the Architect role"
    )
    implementer: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fitness for the Implementer role"
    )
    refactorer: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fitness for the Refactorer role"
    )
    verifier: float = Field(
        default=0.0, ge=0.0, le=1.0, description="Fitness for the Verifier role"
    )


class ModelConfig(BaseModel):
    """Configuration for a single LLM model in the registry.

    Loaded from models.toml. Each entry provides the LiteLLM routing
    information, capability flags, cost data, and fitness scores.
    """

    provider: str = Field(description="Provider identifier (e.g. 'anthropic', 'openai', 'xai')")
    model: str = Field(description="LiteLLM model identifier (e.g. 'claude-opus-4-5-20250929')")
    display_name: str = Field(description="Human-friendly model name for CLI output")
    api_key_env: str = Field(description="Environment variable name holding the API key")
    api_base: str = Field(default="", description="Custom API base URL (empty = provider default)")
    context_window: int = Field(gt=0, description="Maximum context window size in tokens")
    supports_tools: bool = Field(default=False, description="Whether the model supports tool use")
    supports_structured: bool = Field(
        default=False, description="Whether the model supports structured output"
    )
    supports_vision: bool = Field(
        default=False, description="Whether the model supports image inputs"
    )
    supports_thinking: bool = Field(
        default=False, description="Whether the model supports extended thinking"
    )
    cost_input: float = Field(ge=0.0, description="Cost per 1M input tokens in USD")
    cost_output: float = Field(ge=0.0, description="Cost per 1M output tokens in USD")
    fitness: RoleFitness = Field(
        default_factory=RoleFitness, description="Role fitness scores from benchmarks"
    )


class TaskSpec(BaseModel):
    """Specification for a pipeline task submitted by the developer.

    Contains the task description, optional context, and any domain-specific
    rules to inject into agent prompts.
    """

    task: str = Field(description="The task description — what to build")
    context: str = Field(default="", description="Additional context for the pipeline agents")
    domain_rules: str = Field(
        default="", description="Domain-specific rules injected into agent prompts"
    )
    output_dir: str = Field(
        default="output", description="Directory for pipeline output files"
    )


class StageConfig(BaseModel):
    """Configuration overrides for a specific pipeline stage."""

    model: str = Field(default="", description="Model override for this stage (empty = auto)")
    timeout: int = Field(
        default=120, gt=0, description="Timeout in seconds for this stage's model call"
    )
    max_retries: int = Field(
        default=2, ge=0, le=5, description="Max Arbiter-triggered retries for this stage"
    )


class PipelineConfig(BaseModel):
    """Top-level configuration for a pipeline run.

    Loaded from defaults.toml and overridden by CLI flags. Controls Arbiter
    mode, reconciliation, timeouts, and per-stage model assignments.
    """

    arbiter_mode: ArbiterMode = Field(
        default=ArbiterMode.BOOKEND, description="Arbiter review depth"
    )
    reconciliation_enabled: bool = Field(
        default=False, description="Whether to run Implementation Summary Reconciliation"
    )
    default_timeout: int = Field(
        default=120, gt=0, description="Default timeout in seconds per model call"
    )
    max_retries: int = Field(
        default=2, ge=0, le=5, description="Default max Arbiter-triggered retries per stage"
    )
    reconciliation_retries: int = Field(
        default=1, ge=0, le=2, description="Max retries for reconciliation REJECT"
    )
    stages: dict[PipelineStage, StageConfig] = Field(
        default_factory=dict, description="Per-stage configuration overrides"
    )
    arbiter_model: str = Field(
        default="", description="Global Arbiter model override (empty = auto)"
    )
    reconcile_model: str = Field(
        default="", description="Reconciliation Arbiter model override (empty = auto)"
    )


class PipelineResult(BaseModel):
    """Result of a complete pipeline run.

    Contains all stage outputs, collected suggestions, cost/token totals,
    and timing data. Used for output rendering and session logging.
    """

    task: TaskSpec = Field(description="The original task specification")
    config: PipelineConfig = Field(description="Pipeline configuration used for this run")
    stages: dict[PipelineStage, AgentMessage] = Field(
        default_factory=dict, description="AgentMessage output from each pipeline stage"
    )
    suggestions: list[Suggestion] = Field(
        default_factory=list, description="All cross-domain suggestions collected during the run"
    )
    total_cost: float = Field(
        default=0.0, ge=0.0, description="Total USD cost across all model calls"
    )
    total_tokens: int = Field(
        default=0, ge=0, description="Total tokens (prompt + completion) across all calls"
    )
    duration_seconds: float = Field(
        default=0.0, ge=0.0, description="Wall-clock duration of the pipeline run in seconds"
    )
    success: bool = Field(default=False, description="Whether the pipeline completed successfully")

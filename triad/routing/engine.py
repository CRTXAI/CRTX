"""Smart Routing Engine for model-to-role assignment.

Selects models for pipeline stages based on configurable strategies
(quality-first, cost-optimized, speed-first, hybrid). Supports per-stage
overrides and provides cost estimation before a run.
"""

from __future__ import annotations

import logging

from triad.routing.strategies import (
    cost_optimized,
    hybrid,
    quality_first,
    speed_first,
)
from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ModelConfig, PipelineConfig, StageConfig
from triad.schemas.routing import CostEstimate, RoutingDecision, RoutingStrategy

logger = logging.getLogger(__name__)

# Conservative token estimates per stage (input, output)
_TOKEN_ESTIMATES: dict[PipelineStage, tuple[int, int]] = {
    PipelineStage.ARCHITECT: (50_000, 8_000),
    PipelineStage.IMPLEMENT: (40_000, 12_000),
    PipelineStage.REFACTOR: (60_000, 15_000),
    PipelineStage.VERIFY: (60_000, 15_000),
}

# Strategy dispatcher
_STRATEGY_FN = {
    RoutingStrategy.QUALITY_FIRST: lambda reg, role, mf: quality_first(reg, role),
    RoutingStrategy.COST_OPTIMIZED: lambda reg, role, mf: cost_optimized(
        reg, role, mf,
    ),
    RoutingStrategy.SPEED_FIRST: lambda reg, role, mf: speed_first(reg, role),
    RoutingStrategy.HYBRID: lambda reg, role, mf: hybrid(reg, role, mf),
}

# Pipeline stages in execution order
_STAGES: list[PipelineStage] = [
    PipelineStage.ARCHITECT,
    PipelineStage.IMPLEMENT,
    PipelineStage.REFACTOR,
    PipelineStage.VERIFY,
]


def _estimate_stage_cost(config: ModelConfig, stage: PipelineStage) -> float:
    """Estimate the USD cost for a single stage based on token estimates."""
    input_tokens, output_tokens = _TOKEN_ESTIMATES[stage]
    input_cost = (input_tokens / 1_000_000) * config.cost_input
    output_cost = (output_tokens / 1_000_000) * config.cost_output
    return input_cost + output_cost


class RoutingEngine:
    """Smart model-to-role routing engine.

    Selects models for each pipeline stage using the configured strategy.
    Respects per-stage overrides from PipelineConfig.stages. Records all
    routing decisions for audit trail.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
    ) -> None:
        self._config = config
        self._registry = registry

    def select_model(
        self,
        role: PipelineStage,
        strategy: RoutingStrategy | None = None,
    ) -> RoutingDecision:
        """Select a model for a single pipeline stage.

        Checks for per-stage overrides first, then applies the routing
        strategy. Returns a RoutingDecision with full rationale.

        Args:
            role: The pipeline stage to select a model for.
            strategy: Override strategy (defaults to config.routing_strategy).

        Returns:
            RoutingDecision with the selected model and rationale.

        Raises:
            RuntimeError: If the configured override model isn't in the
                          registry or no models are available.
        """
        effective_strategy = strategy or self._config.routing_strategy

        # Check for per-stage model override
        stage_config: StageConfig | None = self._config.stages.get(role)
        if stage_config and stage_config.model:
            override_key = stage_config.model
            if override_key not in self._registry:
                raise RuntimeError(
                    f"Stage {role.value} configured with model "
                    f"'{override_key}' but it is not in the model registry"
                )
            model_cfg = self._registry[override_key]
            fitness = getattr(
                model_cfg.fitness,
                _get_fitness_field(role),
            )
            return RoutingDecision(
                model_key=override_key,
                model_id=model_cfg.model,
                role=role,
                strategy=effective_strategy,
                rationale=f"Per-stage override: {override_key}",
                fitness_score=fitness,
                estimated_cost=_estimate_stage_cost(model_cfg, role),
            )

        # Apply strategy
        dispatch = _STRATEGY_FN[effective_strategy]
        model_key, rationale = dispatch(
            self._registry, role, self._config.min_fitness,
        )

        model_cfg = self._registry[model_key]
        fitness = getattr(model_cfg.fitness, _get_fitness_field(role))

        decision = RoutingDecision(
            model_key=model_key,
            model_id=model_cfg.model,
            role=role,
            strategy=effective_strategy,
            rationale=rationale,
            fitness_score=fitness,
            estimated_cost=_estimate_stage_cost(model_cfg, role),
        )

        logger.info(
            "Routing %s → %s (%s, fitness=%.2f, est=$%.4f)",
            role.value,
            model_key,
            effective_strategy.value,
            fitness,
            decision.estimated_cost,
        )
        return decision

    def select_pipeline_models(
        self,
        strategy: RoutingStrategy | None = None,
    ) -> list[RoutingDecision]:
        """Select models for all 4 pipeline stages.

        Returns:
            List of RoutingDecisions, one per stage in execution order.
        """
        return [self.select_model(stage, strategy) for stage in _STAGES]


def estimate_cost(
    config: PipelineConfig,
    registry: dict[str, ModelConfig],
    strategy: RoutingStrategy | None = None,
) -> CostEstimate:
    """Estimate the total cost of a pipeline run under a given strategy.

    Uses conservative token estimates per stage and the model's published
    cost rates. Does not include Arbiter review costs.

    Args:
        config: Pipeline configuration.
        registry: Model registry.
        strategy: Override strategy (defaults to config.routing_strategy).

    Returns:
        CostEstimate with per-stage decisions and total cost.
    """
    effective_strategy = strategy or config.routing_strategy
    engine = RoutingEngine(config, registry)
    decisions = engine.select_pipeline_models(effective_strategy)
    total = sum(d.estimated_cost for d in decisions)

    return CostEstimate(
        strategy=effective_strategy,
        decisions=decisions,
        total_estimated_cost=total,
    )


def _get_fitness_field(role: PipelineStage) -> str:
    """Map a pipeline stage to its RoleFitness field name."""
    return {
        PipelineStage.ARCHITECT: "architect",
        PipelineStage.IMPLEMENT: "implementer",
        PipelineStage.REFACTOR: "refactorer",
        PipelineStage.VERIFY: "verifier",
    }[role]

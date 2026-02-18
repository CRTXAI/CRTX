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

# Output token estimates per stage — based on observed averages across
# real pipeline runs.  Previous values (3k-6k) over-estimated by 3-7x
# because expensive models charge $15-75/1M output tokens.
_OUTPUT_ESTIMATES: dict[PipelineStage, int] = {
    PipelineStage.ARCHITECT: 1_500,
    PipelineStage.IMPLEMENT: 3_000,
    PipelineStage.REFACTOR: 2_000,
    PipelineStage.VERIFY: 1_000,
}

# Input tokens accumulated from prior stage outputs
_STAGE_ACCUMULATION: dict[PipelineStage, int] = {
    PipelineStage.ARCHITECT: 0,        # No prior output
    PipelineStage.IMPLEMENT: 2_000,    # Architect output
    PipelineStage.REFACTOR: 4_000,     # Architect + Implement
    PipelineStage.VERIFY: 5_000,       # All prior outputs
}

# Base overhead: system prompt + role prompt + output schema
_BASE_INPUT_OVERHEAD = 1_500

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


def _estimate_stage_cost(
    config: ModelConfig,
    stage: PipelineStage,
    context_tokens: int = 0,
    task_tokens: int = 500,
) -> float:
    """Estimate the USD cost for a single stage based on actual context size.

    Args:
        config: Model configuration with cost rates.
        stage: Pipeline stage being estimated.
        context_tokens: Injected project context tokens (0 when no --context-dir).
        task_tokens: Approximate tokens from the task description.
    """
    input_tokens = (
        _BASE_INPUT_OVERHEAD
        + task_tokens
        + context_tokens
        + _STAGE_ACCUMULATION[stage]
    )
    output_tokens = _OUTPUT_ESTIMATES[stage]
    input_cost = (input_tokens / 1_000_000) * config.cost_input
    output_cost = (output_tokens / 1_000_000) * config.cost_output
    return input_cost + output_cost


class RoutingEngine:
    """Smart model-to-role routing engine.

    Selects models for each pipeline stage using the configured strategy.
    Respects per-stage overrides from PipelineConfig.stages. Records all
    routing decisions for audit trail. Integrates with ProviderHealth to
    skip models known to be down.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
        health: object | None = None,
        context_tokens: int = 0,
        task_tokens: int = 500,
    ) -> None:
        self._config = config
        self._registry = registry
        self._health = health  # ProviderHealth instance (optional)
        self._context_tokens = context_tokens
        self._task_tokens = task_tokens

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
                estimated_cost=_estimate_stage_cost(
                    model_cfg, role, self._context_tokens, self._task_tokens,
                ),
            )

        # Filter to healthy models, fall back to full registry if all unhealthy
        active = self._healthy_registry()

        # Apply strategy
        dispatch = _STRATEGY_FN[effective_strategy]
        model_key, rationale = dispatch(
            active, role, self._config.min_fitness,
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
            estimated_cost=_estimate_stage_cost(
                model_cfg, role, self._context_tokens, self._task_tokens,
            ),
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

    # Maximum number of stages a single model can be assigned to
    _MAX_STAGES_PER_MODEL = 1

    def select_pipeline_models(
        self,
        strategy: RoutingStrategy | None = None,
    ) -> list[RoutingDecision]:
        """Select models for all 4 pipeline stages.

        Applies diversity enforcement so no single model is assigned
        to more than ``_MAX_STAGES_PER_MODEL`` stages.

        Returns:
            List of RoutingDecisions, one per stage in execution order.
        """
        decisions = [self.select_model(stage, strategy) for stage in _STAGES]
        return self._enforce_diversity(decisions)

    def _enforce_diversity(
        self,
        decisions: list[RoutingDecision],
    ) -> list[RoutingDecision]:
        """Ensure no single model is used for more than _MAX_STAGES_PER_MODEL stages.

        Keeps the first group of stages (architect, implement) and reassigns
        the second group (refactor, verify) to the next-best model via
        ``get_fallback()``. Skips enforcement when the registry has fewer
        than 2 models.
        """
        if len(self._registry) < 2:
            return decisions

        from collections import Counter

        counts: Counter[str] = Counter(d.model_key for d in decisions)
        overused = {k for k, c in counts.items() if c > self._MAX_STAGES_PER_MODEL}

        if not overused:
            return decisions

        # Stages to reassign: prefer later stages (refactor, verify)
        reassign_stages = {PipelineStage.REFACTOR, PipelineStage.VERIFY}
        result: list[RoutingDecision] = []

        for decision in decisions:
            if (
                decision.model_key in overused
                and decision.role in reassign_stages
            ):
                fallback = self.get_fallback(
                    decision.role, exclude_models=[decision.model_key],
                )
                if fallback is not None:
                    fallback = RoutingDecision(
                        model_key=fallback.model_key,
                        model_id=fallback.model_id,
                        role=decision.role,
                        strategy=decision.strategy,
                        rationale=(
                            f"Diversity enforcement: reassigned from "
                            f"{decision.model_key} (>{self._MAX_STAGES_PER_MODEL} stages)"
                        ),
                        fitness_score=fallback.fitness_score,
                        estimated_cost=fallback.estimated_cost,
                    )
                    result.append(fallback)
                    continue
            result.append(decision)

        return result

    def get_fallback(
        self,
        role: PipelineStage,
        exclude_models: list[str],
    ) -> RoutingDecision | None:
        """Select a fallback model for a role, excluding already-tried models.

        Filters out unhealthy models and ``exclude_models``, then picks
        the highest-fitness model for the role. Returns ``None`` when no
        model is available (the stage truly cannot be completed).
        """
        candidates = {
            k: v for k, v in self._registry.items()
            if k not in exclude_models
            and (not self._health or self._health.is_healthy(k))
        }
        if not candidates:
            return None

        best_key = max(
            candidates,
            key=lambda k: getattr(
                candidates[k].fitness, _get_fitness_field(role),
            ),
        )
        model_cfg = candidates[best_key]
        fitness = getattr(model_cfg.fitness, _get_fitness_field(role))

        decision = RoutingDecision(
            model_key=best_key,
            model_id=model_cfg.model,
            role=role,
            strategy=self._config.routing_strategy,
            rationale=f"Fallback: best available after excluding {exclude_models}",
            fitness_score=fitness,
            estimated_cost=_estimate_stage_cost(
                model_cfg, role, self._context_tokens, self._task_tokens,
            ),
        )
        logger.info(
            "Fallback routing %s → %s (fitness=%.2f)",
            role.value, best_key, fitness,
        )
        return decision

    def _healthy_registry(self) -> dict[str, ModelConfig]:
        """Return the registry filtered to healthy models.

        Falls back to the full registry if all models are unhealthy
        (better to retry a recovering model than to fail immediately).
        """
        if not self._health:
            return self._registry
        healthy = {
            k: v for k, v in self._registry.items()
            if self._health.is_healthy(k)
        }
        return healthy if healthy else self._registry


def estimate_cost(
    config: PipelineConfig,
    registry: dict[str, ModelConfig],
    strategy: RoutingStrategy | None = None,
    task_text: str = "",
) -> CostEstimate:
    """Estimate the total cost of a pipeline run under a given strategy.

    Uses context-aware token estimates per stage and the model's published
    cost rates. Does not include Arbiter review costs.

    Args:
        config: Pipeline configuration.
        registry: Model registry.
        strategy: Override strategy (defaults to config.routing_strategy).
        task_text: The task description (used to estimate task tokens).

    Returns:
        CostEstimate with per-stage decisions and total cost.
    """
    effective_strategy = strategy or config.routing_strategy
    context_tokens = config.context_token_budget if config.context_dir else 0
    task_tokens = max(100, len(task_text) // 4) if task_text else 500
    engine = RoutingEngine(
        config, registry,
        context_tokens=context_tokens,
        task_tokens=task_tokens,
    )
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

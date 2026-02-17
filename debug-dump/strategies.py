"""Routing strategy functions for model-to-role assignment.

Each strategy selects a model from the registry for a given pipeline stage
based on different optimization criteria: quality, cost, speed, or a hybrid.
"""

from __future__ import annotations

from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ModelConfig, RoleFitness

# Map pipeline stages to RoleFitness field names
_ROLE_FIELD: dict[PipelineStage, str] = {
    PipelineStage.ARCHITECT: "architect",
    PipelineStage.IMPLEMENT: "implementer",
    PipelineStage.REFACTOR: "refactorer",
    PipelineStage.VERIFY: "verifier",
}


def _fitness_for_role(fitness: RoleFitness, role: PipelineStage) -> float:
    """Get a model's fitness score for a given pipeline role."""
    return getattr(fitness, _ROLE_FIELD[role])


def _avg_cost_per_token(config: ModelConfig) -> float:
    """Weighted average cost per token (input-heavy approximation).

    Uses a 4:1 input-to-output ratio to reflect typical pipeline usage.
    """
    return (4 * config.cost_input + config.cost_output) / 5


def quality_first(
    registry: dict[str, ModelConfig],
    role: PipelineStage,
) -> tuple[str, str]:
    """Select the highest-fitness model for a role regardless of cost.

    Returns:
        Tuple of (model_key, rationale).

    Raises:
        RuntimeError: If the registry is empty.
    """
    if not registry:
        raise RuntimeError("No models available in registry")

    best_key = max(
        registry,
        key=lambda k: _fitness_for_role(registry[k].fitness, role),
    )
    score = _fitness_for_role(registry[best_key].fitness, role)
    return best_key, f"Highest fitness for {role.value} ({score:.2f})"


def cost_optimized(
    registry: dict[str, ModelConfig],
    role: PipelineStage,
    min_fitness: float = 0.70,
) -> tuple[str, str]:
    """Select the cheapest model above a minimum fitness threshold.

    Falls back to quality_first if no model meets the threshold.

    Returns:
        Tuple of (model_key, rationale).

    Raises:
        RuntimeError: If the registry is empty.
    """
    if not registry:
        raise RuntimeError("No models available in registry")

    eligible = {
        k: v for k, v in registry.items()
        if _fitness_for_role(v.fitness, role) >= min_fitness
    }

    if not eligible:
        key, _ = quality_first(registry, role)
        score = _fitness_for_role(registry[key].fitness, role)
        return key, (
            f"No model meets min_fitness={min_fitness:.2f} for "
            f"{role.value}; fell back to quality_first ({score:.2f})"
        )

    cheapest = min(eligible, key=lambda k: _avg_cost_per_token(eligible[k]))
    score = _fitness_for_role(registry[cheapest].fitness, role)
    cost = _avg_cost_per_token(registry[cheapest])
    return cheapest, (
        f"Cheapest model above {min_fitness:.2f} threshold for "
        f"{role.value} (fitness={score:.2f}, avg_cost=${cost:.2f}/MTok)"
    )


def speed_first(
    registry: dict[str, ModelConfig],
    role: PipelineStage,
) -> tuple[str, str]:
    """Select the fastest model using context_window as a latency proxy.

    Smaller context windows correlate with lighter, faster models.
    Ties broken by lower cost.

    Returns:
        Tuple of (model_key, rationale).

    Raises:
        RuntimeError: If the registry is empty.
    """
    if not registry:
        raise RuntimeError("No models available in registry")

    fastest = min(
        registry,
        key=lambda k: (
            registry[k].context_window,
            _avg_cost_per_token(registry[k]),
        ),
    )
    window = registry[fastest].context_window
    return fastest, (
        f"Smallest context_window for speed "
        f"({window:,} tokens) — {role.value}"
    )


def hybrid(
    registry: dict[str, ModelConfig],
    role: PipelineStage,
    min_fitness: float = 0.70,
) -> tuple[str, str]:
    """Quality-first for critical stages, best-above-threshold for others.

    Critical stages (refactor, verify): absolute best model regardless of cost.
    Other stages (architect, implement): highest-fitness model above the
    min_fitness threshold, with cost as tiebreaker.

    Returns:
        Tuple of (model_key, rationale).

    Raises:
        RuntimeError: If the registry is empty.
    """
    if not registry:
        raise RuntimeError("No models available in registry")

    critical_stages = {PipelineStage.REFACTOR, PipelineStage.VERIFY}

    if role in critical_stages:
        key, rationale = quality_first(registry, role)
        return key, f"[hybrid/quality] {rationale}"

    # Non-critical stages: best model above fitness threshold
    eligible = {
        k: v for k, v in registry.items()
        if _fitness_for_role(v.fitness, role) >= min_fitness
    }

    if not eligible:
        key, rationale = quality_first(registry, role)
        return key, f"[hybrid/quality] {rationale} (none above threshold)"

    # Highest fitness, with cheapest cost as tiebreaker
    best_key = max(
        eligible,
        key=lambda k: (
            _fitness_for_role(eligible[k].fitness, role),
            -_avg_cost_per_token(eligible[k]),
        ),
    )
    score = _fitness_for_role(registry[best_key].fitness, role)
    return best_key, (
        f"[hybrid/balanced] Best above {min_fitness:.2f} for "
        f"{role.value} (fitness={score:.2f})"
    )

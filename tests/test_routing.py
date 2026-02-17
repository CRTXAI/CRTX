"""Tests for the Smart Routing Engine (Day 7).

Covers routing strategies, RoutingEngine, cost estimation,
per-stage overrides, and orchestrator integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.routing.engine import RoutingEngine, _estimate_stage_cost, estimate_cost
from triad.routing.strategies import (
    cost_optimized,
    hybrid,
    quality_first,
    speed_first,
)
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    TokenUsage,
)
from triad.schemas.pipeline import (
    ModelConfig,
    PipelineConfig,
    RoleFitness,
    StageConfig,
    TaskSpec,
)
from triad.schemas.routing import CostEstimate, RoutingDecision, RoutingStrategy

# Patch targets
_PROVIDER = "triad.orchestrator.LiteLLMProvider"
_ARBITER_REVIEW = "triad.orchestrator.ArbiterEngine.review"


# ── Factories ──────────────────────────────────────────────────────


def _make_model_config(
    model: str = "model-a-v1", **overrides,
) -> ModelConfig:
    defaults = {
        "provider": "test",
        "model": model,
        "display_name": "Test Model",
        "api_key_env": "TEST_KEY",
        "context_window": 128000,
        "cost_input": 3.0,
        "cost_output": 15.0,
        "fitness": RoleFitness(
            architect=0.9, implementer=0.8,
            refactorer=0.7, verifier=0.85,
        ),
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_diverse_registry() -> dict[str, ModelConfig]:
    """Registry with 3 models spanning different cost/fitness profiles."""
    return {
        "premium": _make_model_config(
            model="premium-v1",
            cost_input=15.0,
            cost_output=75.0,
            context_window=200000,
            fitness=RoleFitness(
                architect=0.95, implementer=0.90,
                refactorer=0.95, verifier=0.92,
            ),
        ),
        "mid-tier": _make_model_config(
            model="mid-tier-v1",
            cost_input=3.0,
            cost_output=15.0,
            context_window=128000,
            fitness=RoleFitness(
                architect=0.80, implementer=0.85,
                refactorer=0.80, verifier=0.80,
            ),
        ),
        "budget": _make_model_config(
            model="budget-v1",
            cost_input=0.15,
            cost_output=0.60,
            context_window=200000,
            fitness=RoleFitness(
                architect=0.70, implementer=0.72,
                refactorer=0.68, verifier=0.70,
            ),
        ),
    }


def _make_task(**overrides) -> TaskSpec:
    defaults = {"task": "Build a REST API"}
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_agent_message(
    content: str = "output", cost: float = 0.01,
) -> AgentMessage:
    return AgentMessage(
        from_agent=PipelineStage.ARCHITECT,
        to_agent=PipelineStage.IMPLEMENT,
        msg_type=MessageType.IMPLEMENTATION,
        content=f"{content}\n\nCONFIDENCE: 0.85",
        confidence=0.0,
        token_usage=TokenUsage(
            prompt_tokens=100, completion_tokens=50, cost=cost,
        ),
        model="model-a-v1",
    )


def _mock_provider(responses: list[AgentMessage]):
    mock_cls = MagicMock()
    mock_inst = MagicMock()
    mock_cls.return_value = mock_inst
    mock_inst.complete = AsyncMock(side_effect=responses)
    return mock_cls, mock_inst


# ── Strategy: quality_first ──────────────────────────────────────


class TestQualityFirst:
    def test_selects_highest_fitness_architect(self):
        registry = _make_diverse_registry()
        key, rationale = quality_first(registry, PipelineStage.ARCHITECT)
        assert key == "premium"  # 0.95 architect fitness
        assert "0.95" in rationale

    def test_selects_highest_fitness_implementer(self):
        registry = _make_diverse_registry()
        key, _ = quality_first(registry, PipelineStage.IMPLEMENT)
        assert key == "premium"  # 0.90 implementer fitness

    def test_selects_highest_fitness_refactorer(self):
        registry = _make_diverse_registry()
        key, _ = quality_first(registry, PipelineStage.REFACTOR)
        assert key == "premium"  # 0.95 refactorer fitness

    def test_selects_highest_fitness_verifier(self):
        registry = _make_diverse_registry()
        key, _ = quality_first(registry, PipelineStage.VERIFY)
        assert key == "premium"  # 0.92 verifier fitness

    def test_empty_registry_raises(self):
        with pytest.raises(RuntimeError, match="No models available"):
            quality_first({}, PipelineStage.ARCHITECT)


# ── Strategy: cost_optimized ─────────────────────────────────────


class TestCostOptimized:
    def test_selects_cheapest_above_threshold(self):
        registry = _make_diverse_registry()
        # budget has architect=0.70, meets 0.70 threshold
        key, rationale = cost_optimized(
            registry, PipelineStage.ARCHITECT, min_fitness=0.70,
        )
        assert key == "budget"
        assert "0.70" in rationale

    def test_falls_back_to_quality_when_none_meet_threshold(self):
        registry = _make_diverse_registry()
        # refactorer fitness: premium=0.95, mid=0.80, budget=0.68
        # threshold 0.99 means none qualify
        key, rationale = cost_optimized(
            registry, PipelineStage.REFACTOR, min_fitness=0.99,
        )
        assert key == "premium"  # quality_first fallback
        assert "fell back to quality_first" in rationale

    def test_respects_custom_threshold(self):
        registry = _make_diverse_registry()
        # With threshold 0.80, only premium (0.95) and mid (0.80) qualify
        key, _ = cost_optimized(
            registry, PipelineStage.REFACTOR, min_fitness=0.80,
        )
        assert key == "mid-tier"  # cheapest of the two eligible

    def test_empty_registry_raises(self):
        with pytest.raises(RuntimeError, match="No models available"):
            cost_optimized({}, PipelineStage.ARCHITECT)

    def test_single_model_above_threshold(self):
        registry = {
            "only": _make_model_config(
                model="only-v1",
                cost_input=5.0, cost_output=20.0,
                fitness=RoleFitness(architect=0.80),
            ),
        }
        key, _ = cost_optimized(registry, PipelineStage.ARCHITECT, 0.70)
        assert key == "only"


# ── Strategy: speed_first ────────────────────────────────────────


class TestSpeedFirst:
    def test_selects_smallest_context_window(self):
        registry = _make_diverse_registry()
        # mid-tier has 128K (smallest)
        key, rationale = speed_first(registry, PipelineStage.ARCHITECT)
        assert key == "mid-tier"
        assert "128,000" in rationale

    def test_breaks_ties_by_cost(self):
        registry = {
            "a": _make_model_config(
                model="a-v1", context_window=128000,
                cost_input=5.0, cost_output=20.0,
            ),
            "b": _make_model_config(
                model="b-v1", context_window=128000,
                cost_input=1.0, cost_output=4.0,
            ),
        }
        key, _ = speed_first(registry, PipelineStage.ARCHITECT)
        assert key == "b"  # same window, cheaper

    def test_empty_registry_raises(self):
        with pytest.raises(RuntimeError, match="No models available"):
            speed_first({}, PipelineStage.ARCHITECT)


# ── Strategy: hybrid ─────────────────────────────────────────────


class TestHybrid:
    def test_uses_quality_for_refactor(self):
        registry = _make_diverse_registry()
        key, rationale = hybrid(registry, PipelineStage.REFACTOR)
        assert key == "premium"  # quality_first for critical stage
        assert "[hybrid/quality]" in rationale

    def test_uses_quality_for_verify(self):
        registry = _make_diverse_registry()
        key, rationale = hybrid(registry, PipelineStage.VERIFY)
        assert key == "premium"
        assert "[hybrid/quality]" in rationale

    def test_uses_best_for_architect(self):
        registry = _make_diverse_registry()
        key, rationale = hybrid(
            registry, PipelineStage.ARCHITECT, min_fitness=0.70,
        )
        assert key == "premium"  # best above threshold for early stage
        assert "[hybrid/balanced]" in rationale

    def test_uses_best_for_implement(self):
        registry = _make_diverse_registry()
        key, rationale = hybrid(
            registry, PipelineStage.IMPLEMENT, min_fitness=0.70,
        )
        assert key == "premium"  # best above threshold for early stage
        assert "[hybrid/balanced]" in rationale


# ── RoutingEngine ────────────────────────────────────────────────


class TestRoutingEngine:
    def test_select_model_returns_decision(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )
        engine = RoutingEngine(config, registry)
        decision = engine.select_model(PipelineStage.ARCHITECT)

        assert isinstance(decision, RoutingDecision)
        assert decision.model_key == "premium"
        assert decision.model_id == "premium-v1"
        assert decision.role == PipelineStage.ARCHITECT
        assert decision.strategy == RoutingStrategy.QUALITY_FIRST
        assert decision.fitness_score == 0.95
        assert decision.estimated_cost > 0

    def test_select_model_respects_override(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
            stages={
                PipelineStage.ARCHITECT: StageConfig(model="budget"),
            },
        )
        engine = RoutingEngine(config, registry)
        decision = engine.select_model(PipelineStage.ARCHITECT)

        assert decision.model_key == "budget"
        assert "Per-stage override" in decision.rationale

    def test_select_model_missing_override_raises(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            stages={
                PipelineStage.ARCHITECT: StageConfig(model="nonexistent"),
            },
        )
        engine = RoutingEngine(config, registry)
        with pytest.raises(RuntimeError, match="not in the model registry"):
            engine.select_model(PipelineStage.ARCHITECT)

    def test_select_model_override_strategy(self):
        """select_model accepts an explicit strategy override."""
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )
        engine = RoutingEngine(config, registry)
        decision = engine.select_model(
            PipelineStage.ARCHITECT,
            strategy=RoutingStrategy.SPEED_FIRST,
        )
        assert decision.strategy == RoutingStrategy.SPEED_FIRST
        assert decision.model_key == "mid-tier"

    def test_select_pipeline_models_returns_four(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )
        engine = RoutingEngine(config, registry)
        decisions = engine.select_pipeline_models()

        assert len(decisions) == 4
        roles = {d.role for d in decisions}
        assert roles == {
            PipelineStage.ARCHITECT,
            PipelineStage.IMPLEMENT,
            PipelineStage.REFACTOR,
            PipelineStage.VERIFY,
        }

    def test_hybrid_mixed_selection(self):
        """Hybrid should pick best model for all stages above threshold."""
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.HYBRID,
            min_fitness=0.70,
        )
        engine = RoutingEngine(config, registry)
        decisions = engine.select_pipeline_models()

        decision_map = {d.role: d for d in decisions}
        # All stages: best model (premium has highest fitness everywhere)
        assert decision_map[PipelineStage.ARCHITECT].model_key == "premium"
        assert decision_map[PipelineStage.IMPLEMENT].model_key == "premium"
        assert decision_map[PipelineStage.REFACTOR].model_key == "premium"
        assert decision_map[PipelineStage.VERIFY].model_key == "premium"

    def test_min_fitness_from_config(self):
        """Engine should use min_fitness from PipelineConfig."""
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.COST_OPTIMIZED,
            min_fitness=0.85,
        )
        engine = RoutingEngine(config, registry)
        decision = engine.select_model(PipelineStage.ARCHITECT)
        # Only premium (0.95) meets 0.85 threshold for architect
        assert decision.model_key == "premium"


# ── Cost Estimation ──────────────────────────────────────────────


class TestCostEstimation:
    def test_estimate_stage_cost(self):
        config = _make_model_config(
            model="test-v1", cost_input=3.0, cost_output=15.0,
        )
        # Architect: 50K input * $3/MTok + 8K output * $15/MTok
        cost = _estimate_stage_cost(config, PipelineStage.ARCHITECT)
        expected = (50_000 / 1_000_000) * 3.0 + (8_000 / 1_000_000) * 15.0
        assert abs(cost - expected) < 1e-10

    def test_estimate_cost_returns_all_stages(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )
        estimate = estimate_cost(config, registry)

        assert isinstance(estimate, CostEstimate)
        assert len(estimate.decisions) == 4
        assert estimate.total_estimated_cost > 0
        assert estimate.strategy == RoutingStrategy.QUALITY_FIRST

    def test_estimate_cost_total_matches_sum(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )
        estimate = estimate_cost(config, registry)
        calculated_total = sum(d.estimated_cost for d in estimate.decisions)
        assert abs(estimate.total_estimated_cost - calculated_total) < 1e-10

    def test_cost_optimized_cheaper_than_quality(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(arbiter_mode="off", min_fitness=0.70)

        quality_est = estimate_cost(
            config, registry, RoutingStrategy.QUALITY_FIRST,
        )
        cost_est = estimate_cost(
            config, registry, RoutingStrategy.COST_OPTIMIZED,
        )
        assert cost_est.total_estimated_cost < quality_est.total_estimated_cost

    def test_estimate_cost_with_strategy_override(self):
        registry = _make_diverse_registry()
        config = PipelineConfig(
            arbiter_mode="off",
            routing_strategy=RoutingStrategy.QUALITY_FIRST,
        )
        # Override to cost_optimized
        estimate = estimate_cost(
            config, registry, RoutingStrategy.COST_OPTIMIZED,
        )
        assert estimate.strategy == RoutingStrategy.COST_OPTIMIZED


# ── Schema Validation ────────────────────────────────────────────


class TestRoutingSchemas:
    def test_routing_strategy_values(self):
        assert RoutingStrategy.QUALITY_FIRST == "quality_first"
        assert RoutingStrategy.COST_OPTIMIZED == "cost_optimized"
        assert RoutingStrategy.SPEED_FIRST == "speed_first"
        assert RoutingStrategy.HYBRID == "hybrid"

    def test_routing_decision_from_dict(self):
        data = {
            "model_key": "test",
            "model_id": "test-v1",
            "role": "architect",
            "strategy": "quality_first",
            "rationale": "Best fit",
            "fitness_score": 0.9,
            "estimated_cost": 0.05,
        }
        decision = RoutingDecision(**data)
        assert decision.model_key == "test"
        assert decision.strategy == RoutingStrategy.QUALITY_FIRST

    def test_cost_estimate_schema(self):
        estimate = CostEstimate(
            strategy=RoutingStrategy.HYBRID,
            decisions=[],
            total_estimated_cost=0.0,
        )
        assert estimate.strategy == RoutingStrategy.HYBRID

    def test_pipeline_config_routing_defaults(self):
        config = PipelineConfig(arbiter_mode="off")
        assert config.routing_strategy == RoutingStrategy.HYBRID
        assert config.min_fitness == 0.70


# ── Orchestrator Integration ─────────────────────────────────────


class TestOrchestratorRouting:
    async def test_routing_decisions_in_result(self):
        """PipelineResult should contain routing decisions."""
        from triad.orchestrator import PipelineOrchestrator
        from triad.schemas.arbiter import ArbiterReview, Verdict

        registry = _make_diverse_registry()
        responses = [_make_agent_message(f"stage-{i}") for i in range(4)]
        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=ArbiterReview(
                    stage_reviewed=PipelineStage.VERIFY,
                    reviewed_model="premium-v1",
                    arbiter_model="mid-tier-v1",
                    verdict=Verdict.APPROVE,
                    confidence=0.9,
                    reasoning="VERDICT: APPROVE",
                    token_cost=0.005,
                )),
            ),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="off",
                    routing_strategy=RoutingStrategy.QUALITY_FIRST,
                ),
                registry=registry,
            )
            result = await orch.run()

        assert len(result.routing_decisions) == 4
        for decision in result.routing_decisions:
            assert isinstance(decision, RoutingDecision)
            assert decision.model_key == "premium"

    async def test_hybrid_uses_best_models(self):
        """Hybrid routing should use best model above threshold for all stages."""
        from triad.orchestrator import PipelineOrchestrator

        registry = _make_diverse_registry()
        responses = [_make_agent_message(f"stage-{i}") for i in range(4)]
        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(),
            ),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="off",
                    routing_strategy=RoutingStrategy.HYBRID,
                    min_fitness=0.70,
                ),
                registry=registry,
            )
            result = await orch.run()

        decision_map = {d.role: d for d in result.routing_decisions}
        assert decision_map[PipelineStage.ARCHITECT].model_key == "premium"
        assert decision_map[PipelineStage.IMPLEMENT].model_key == "premium"
        assert decision_map[PipelineStage.REFACTOR].model_key == "premium"
        assert decision_map[PipelineStage.VERIFY].model_key == "premium"

    async def test_stage_override_takes_precedence(self):
        """Per-stage model override should override routing strategy."""
        from triad.orchestrator import PipelineOrchestrator

        registry = _make_diverse_registry()
        responses = [_make_agent_message(f"stage-{i}") for i in range(4)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock()),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="off",
                    routing_strategy=RoutingStrategy.QUALITY_FIRST,
                    stages={
                        PipelineStage.ARCHITECT: StageConfig(
                            model="mid-tier",
                        ),
                    },
                ),
                registry=registry,
            )
            result = await orch.run()

        decision_map = {d.role: d for d in result.routing_decisions}
        assert decision_map[PipelineStage.ARCHITECT].model_key == "mid-tier"
        assert "Per-stage override" in decision_map[PipelineStage.ARCHITECT].rationale
        # Other stages use quality_first -> premium
        assert decision_map[PipelineStage.IMPLEMENT].model_key == "premium"

    async def test_cost_optimized_route_uses_cheapest(self):
        """Cost-optimized routing should select cheapest above threshold."""
        from triad.orchestrator import PipelineOrchestrator

        registry = _make_diverse_registry()
        responses = [_make_agent_message(f"stage-{i}") for i in range(4)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock()),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="off",
                    routing_strategy=RoutingStrategy.COST_OPTIMIZED,
                    min_fitness=0.70,
                ),
                registry=registry,
            )
            result = await orch.run()

        decision_map = {d.role: d for d in result.routing_decisions}
        # budget qualifies for architect (0.70), implement (0.72), verify (0.70)
        assert decision_map[PipelineStage.ARCHITECT].model_key == "budget"
        assert decision_map[PipelineStage.IMPLEMENT].model_key == "budget"
        assert decision_map[PipelineStage.VERIFY].model_key == "budget"
        # budget refactorer=0.68 < 0.70 threshold, so mid-tier (0.80, cheapest eligible)
        assert decision_map[PipelineStage.REFACTOR].model_key == "mid-tier"

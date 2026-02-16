"""Tests for the debate pipeline mode."""

from unittest.mock import AsyncMock, MagicMock, patch

from triad.orchestrator import DebateOrchestrator
from triad.schemas.arbiter import ArbiterReview, Verdict
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
    TaskSpec,
)

# Patch targets
_PROVIDER = "triad.orchestrator.LiteLLMProvider"
_ARBITER_REVIEW = "triad.orchestrator.ArbiterEngine.review"


# ── Factories ──────────────────────────────────────────────────────

def _make_model_config(model: str = "model-a-v1", **overrides) -> ModelConfig:
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


def _make_three_model_registry() -> dict[str, ModelConfig]:
    return {
        "model-a": _make_model_config(
            model="model-a-v1",
            fitness=RoleFitness(
                architect=0.9, implementer=0.8,
                refactorer=0.7, verifier=0.85,
            ),
        ),
        "model-b": _make_model_config(
            model="model-b-v1",
            fitness=RoleFitness(
                architect=0.7, implementer=0.9,
                refactorer=0.8, verifier=0.75,
            ),
        ),
        "model-c": _make_model_config(
            model="model-c-v1",
            fitness=RoleFitness(
                architect=0.6, implementer=0.7,
                refactorer=0.9, verifier=0.90,
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


def _make_approve_review(**overrides) -> ArbiterReview:
    defaults = {
        "stage_reviewed": PipelineStage.VERIFY,
        "reviewed_model": "model-a-v1",
        "arbiter_model": "model-b-v1",
        "verdict": Verdict.APPROVE,
        "confidence": 0.9,
        "reasoning": "VERDICT: APPROVE",
        "token_cost": 0.005,
    }
    defaults.update(overrides)
    return ArbiterReview(**defaults)


def _mock_provider(responses: list[AgentMessage]):
    mock_cls = MagicMock()
    mock_inst = MagicMock()
    mock_cls.return_value = mock_inst
    mock_inst.complete = AsyncMock(side_effect=responses)
    return mock_cls, mock_inst


# For 3 models, debate needs:
# Phase 1: 3 proposals
# Phase 2: 3 rebuttals
# Phase 3: 3 final arguments
# Phase 4: 1 judgment
# Total: 10 provider calls

def _debate_responses(n_models: int = 3) -> list[AgentMessage]:
    """Create canned responses for all debate phases."""
    responses = []
    # Phase 1: proposals
    for i in range(n_models):
        responses.append(_make_agent_message(f"Proposal from model {i}"))
    # Phase 2: rebuttals (with parseable headers)
    for i in range(n_models):
        others = [f"model-{chr(97+j)}" for j in range(n_models) if j != i]
        rebuttal_text = ""
        for other in others:
            rebuttal_text += (
                f"## Rebuttal: {other}\n\n"
                f"**Strengths**: Good ideas.\n"
                f"**Weaknesses**: Could be better.\n\n"
            )
        responses.append(_make_agent_message(rebuttal_text))
    # Phase 3: final arguments
    for i in range(n_models):
        responses.append(_make_agent_message(f"Final argument from model {i}"))
    # Phase 4: judgment
    responses.append(_make_agent_message("Judge's decision: model-a wins"))
    return responses


# ── Full Debate Execution ─────────────────────────────────────────

class TestDebateExecution:
    async def test_all_four_phases_execute(self):
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.debate_result is not None
        dr = result.debate_result
        assert len(dr.proposals) == 3
        assert len(dr.rebuttals) == 3
        assert len(dr.final_arguments) == 3
        assert dr.judgment != ""
        assert dr.judge_model != ""
        assert result.success is True

    async def test_judge_is_tiebreaker_model(self):
        """Judge should be the highest-fitness verifier model."""
        registry = _make_three_model_registry()
        # model-c has verifier=0.90 (highest)
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.debate_result.judge_model == "model-c"

    async def test_provider_called_correct_times(self):
        """3 proposals + 3 rebuttals + 3 finals + 1 judgment = 10."""
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            await orch.run()

        assert mock_inst.complete.call_count == 10


# ── Debate Rebuttals ─────────────────────────────────────────────

class TestDebateRebuttals:
    async def test_each_model_sees_others_proposals(self):
        """In the rebuttal phase, each model receives the other proposals."""
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            await orch.run()

        # Rebuttal calls are indices 3, 4, 5 (after 3 proposals)
        calls = mock_inst.complete.call_args_list
        # Each rebuttal system prompt should contain "Proposal from model"
        for i in range(3, 6):
            system = calls[i].kwargs["system"]
            assert "Proposal from model" in system

    async def test_rebuttals_parsed_per_target(self):
        """Rebuttals should be parsed into per-target dictionaries."""
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        dr = result.debate_result
        # Each model has rebuttals for the other 2
        for key, targets in dr.rebuttals.items():
            other_keys = [k for k in dr.proposals if k != key]
            for other in other_keys:
                assert other in targets


# ── Debate Arbiter ────────────────────────────────────────────────

class TestDebateArbiter:
    async def test_arbiter_reviews_judgment(self):
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        mock_arbiter = AsyncMock(return_value=_make_approve_review())

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="bookend",
                ),
                registry=registry,
            )
            result = await orch.run()

        mock_arbiter.assert_called_once()
        assert len(result.arbiter_reviews) == 1

    async def test_arbiter_off_skips_review(self):
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        mock_arbiter = AsyncMock()

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        mock_arbiter.assert_not_called()
        assert result.arbiter_reviews == []

    async def test_arbiter_halt_on_judgment(self):
        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        halt_review = _make_approve_review(
            verdict=Verdict.HALT,
            reasoning="Judgment is fundamentally flawed",
        )
        mock_arbiter = AsyncMock(return_value=halt_review)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="bookend",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.halted is True
        assert result.success is False
        assert "fundamentally flawed" in result.halt_reason


# ── Debate Costs ──────────────────────────────────────────────────

class TestDebateCosts:
    async def test_costs_aggregated(self):
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}", cost=0.10)
                     for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        # 10 calls * $0.10 = $1.00
        assert abs(result.total_cost - 1.0) < 1e-10


# ── Two-Model Debate ──────────────────────────────────────────────

class TestTwoModelDebate:
    async def test_works_with_two_models(self):
        """Debate should work with just 2 models."""
        registry = {
            "model-a": _make_model_config(
                model="a-v1",
                fitness=RoleFitness(verifier=0.9),
            ),
            "model-b": _make_model_config(
                model="b-v1",
                fitness=RoleFitness(verifier=0.8),
            ),
        }
        # 2 proposals + 2 rebuttals + 2 finals + 1 judgment = 7
        responses = []
        for i in range(2):
            responses.append(_make_agent_message(f"Proposal {i}"))
        for i in range(2):
            other = "model-b" if i == 0 else "model-a"
            responses.append(_make_agent_message(
                f"## Rebuttal: {other}\nCritique here."
            ))
        for i in range(2):
            responses.append(_make_agent_message(f"Final {i}"))
        responses.append(_make_agent_message("Judgment"))

        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(
                _ARBITER_REVIEW,
                AsyncMock(return_value=_make_approve_review()),
            ),
        ):
            orch = DebateOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.success is True
        assert result.debate_result is not None
        assert len(result.debate_result.proposals) == 2
        assert mock_inst.complete.call_count == 7

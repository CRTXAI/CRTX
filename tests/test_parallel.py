"""Tests for the parallel exploration pipeline mode."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.orchestrator import (
    ParallelOrchestrator,
    _extract_scores,
    _get_tiebreaker_key,
    run_pipeline,
)
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
_CONSENSUS_PROVIDER = "triad.consensus.protocol.LiteLLMProvider"
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


def _make_review_content(arch: int, impl: int, quality: int) -> str:
    return (
        f"ARCHITECTURE: {arch}\nGood structure.\n\n"
        f"IMPLEMENTATION: {impl}\nSolid code.\n\n"
        f"QUALITY: {quality}\nClean.\n\n"
        f"CONFIDENCE: 0.85"
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


# ── Score Extraction ──────────────────────────────────────────────

class TestExtractScores:
    def test_extracts_all_three(self):
        content = _make_review_content(8, 7, 9)
        scores = _extract_scores(content)
        assert scores == {"architecture": 8, "implementation": 7, "quality": 9}

    def test_clamps_above_ten(self):
        scores = _extract_scores("ARCHITECTURE: 15\nIMPLEMENTATION: 5\nQUALITY: 5")
        assert scores["architecture"] == 10

    def test_clamps_below_one(self):
        scores = _extract_scores("ARCHITECTURE: 0\nIMPLEMENTATION: 5\nQUALITY: 5")
        assert scores["architecture"] == 1

    def test_defaults_on_missing(self):
        scores = _extract_scores("No scores here")
        assert scores == {"architecture": 5, "implementation": 5, "quality": 5}


# ── Tiebreaker ────────────────────────────────────────────────────

class TestGetTiebreaker:
    def test_selects_highest_verifier_fitness(self):
        registry = _make_three_model_registry()
        # model-c has verifier=0.90 (highest)
        assert _get_tiebreaker_key(registry) == "model-c"


# ── Parallel Fan-Out ──────────────────────────────────────────────

class TestParallelFanOut:
    async def test_calls_all_models(self):
        """Fan-out should call all models in the registry."""
        registry = _make_three_model_registry()
        # 3 fan-out + 6 reviews (3*2) + 1 synthesis = 10 calls
        responses = [_make_agent_message(f"output-{i}") for i in range(10)]
        mock_cls, mock_inst = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=_make_approve_review())),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.parallel_result is not None
        assert len(result.parallel_result.individual_outputs) == 3
        assert result.success is True

    async def test_fewer_than_two_models_raises(self):
        """Parallel mode needs at least 2 successful outputs."""
        registry = {"model-a": _make_model_config()}
        responses = [_make_agent_message("only output")]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            pytest.raises(RuntimeError, match="at least 2"),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            await orch.run()


# ── Cross-Review + Voting ─────────────────────────────────────────

class TestParallelVoting:
    async def test_scores_and_votes_populated(self):
        registry = _make_three_model_registry()
        # 3 fan-out + 6 reviews + 1 synthesis = 10
        fan_out = [
            _make_agent_message("solution A"),
            _make_agent_message("solution B"),
            _make_agent_message("solution C"),
        ]
        # Reviews: a→b, a→c, b→a, b→c, c→a, c→b
        # model-a scores model-b highest
        reviews = [
            _make_agent_message(_make_review_content(8, 9, 8)),  # a→b
            _make_agent_message(_make_review_content(6, 5, 6)),  # a→c
            _make_agent_message(_make_review_content(7, 7, 7)),  # b→a
            _make_agent_message(_make_review_content(5, 6, 5)),  # b→c
            _make_agent_message(_make_review_content(8, 8, 7)),  # c→a
            _make_agent_message(_make_review_content(6, 7, 6)),  # c→b
        ]
        synthesis = [_make_agent_message("synthesized output")]

        mock_cls, _ = _mock_provider(fan_out + reviews + synthesis)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=_make_approve_review())),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        pr = result.parallel_result
        assert pr is not None
        assert len(pr.scores) == 3
        assert len(pr.votes) == 3
        assert pr.winner != ""
        assert pr.synthesized_output != ""

    async def test_majority_vote_wins(self):
        """If 2 of 3 vote for the same model, that model wins."""
        registry = _make_three_model_registry()
        fan_out = [
            _make_agent_message("solution A"),
            _make_agent_message("solution B"),
            _make_agent_message("solution C"),
        ]
        # a votes for b (b scores higher), b votes for a, c votes for a
        # → a wins with 2 votes
        reviews = [
            _make_agent_message(_make_review_content(9, 9, 9)),  # a→b
            _make_agent_message(_make_review_content(5, 5, 5)),  # a→c
            _make_agent_message(_make_review_content(9, 9, 9)),  # b→a
            _make_agent_message(_make_review_content(5, 5, 5)),  # b→c
            _make_agent_message(_make_review_content(9, 9, 9)),  # c→a
            _make_agent_message(_make_review_content(5, 5, 5)),  # c→b
        ]
        synthesis = [_make_agent_message("final")]

        mock_cls, _ = _mock_provider(fan_out + reviews + synthesis)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=_make_approve_review())),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        pr = result.parallel_result
        assert pr is not None
        # b and a get voted for; a→b, b→a, c→a → a wins with 2 votes
        assert pr.winner == "model-a"

    async def test_tie_goes_to_tiebreaker(self):
        """If votes are tied, consensus tiebreaker resolves."""
        registry = {
            "model-a": _make_model_config(
                model="a-v1",
                fitness=RoleFitness(verifier=0.7),
            ),
            "model-b": _make_model_config(
                model="b-v1",
                fitness=RoleFitness(verifier=0.95),
            ),
        }
        fan_out = [
            _make_agent_message("solution A"),
            _make_agent_message("solution B"),
        ]
        # Each votes for the other → tie
        reviews = [
            _make_agent_message(_make_review_content(9, 9, 9)),  # a→b
            _make_agent_message(_make_review_content(9, 9, 9)),  # b→a
        ]
        synthesis = [_make_agent_message("final")]

        mock_cls, _ = _mock_provider(fan_out + reviews + synthesis)

        # Tiebreaker response via consensus protocol
        tiebreak_msg = _make_agent_message(
            "WINNER: model-b\n\nREASONING: Better approach.",
        )
        consensus_mock_cls = MagicMock()
        consensus_mock_inst = MagicMock()
        consensus_mock_cls.return_value = consensus_mock_inst
        consensus_mock_inst.complete = AsyncMock(return_value=tiebreak_msg)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_CONSENSUS_PROVIDER, consensus_mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=_make_approve_review())),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        # model-b selected by consensus tiebreaker
        assert result.parallel_result.winner == "model-b"


# ── Synthesis ─────────────────────────────────────────────────────

class TestParallelSynthesis:
    async def test_synthesized_output_in_result(self):
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=_make_approve_review())),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.parallel_result.synthesized_output != ""


# ── Arbiter Review ────────────────────────────────────────────────

class TestParallelArbiter:
    async def test_arbiter_reviews_synthesized_output(self):
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)
        mock_arbiter = AsyncMock(return_value=_make_approve_review())

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="bookend",
                ),
                registry=registry,
            )
            result = await orch.run()

        mock_arbiter.assert_called_once()
        assert len(result.arbiter_reviews) == 1
        assert result.success is True

    async def test_arbiter_off_skips_review(self):
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)
        mock_arbiter = AsyncMock()

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        mock_arbiter.assert_not_called()
        assert result.arbiter_reviews == []

    async def test_arbiter_halt_stops_pipeline(self):
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)
        halt_review = _make_approve_review(
            verdict=Verdict.HALT,
            reasoning="Critical issues in synthesis",
        )
        mock_arbiter = AsyncMock(return_value=halt_review)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="bookend",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.halted is True
        assert result.success is False


# ── Cost Tracking ─────────────────────────────────────────────────

class TestParallelCosts:
    async def test_costs_aggregated(self):
        registry = _make_three_model_registry()
        responses = [
            _make_agent_message(f"out-{i}", cost=0.10) for i in range(10)
        ]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=_make_approve_review())),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        # 10 calls * $0.10 = $1.00
        assert abs(result.total_cost - 1.0) < 1e-10


# ── Mode Dispatch ─────────────────────────────────────────────────

class TestModeDispatch:
    async def test_sequential_mode_uses_pipeline_orchestrator(self):
        """run_pipeline with sequential mode uses PipelineOrchestrator."""
        with patch(
            "triad.orchestrator.PipelineOrchestrator"
        ) as mock_orch_cls:
            mock_inst = MagicMock()
            mock_inst.run = AsyncMock(return_value=MagicMock())
            mock_orch_cls.return_value = mock_inst

            await run_pipeline(
                _make_task(),
                PipelineConfig(
                    pipeline_mode="sequential", arbiter_mode="off",
                    persist_sessions=False,
                ),
                _make_three_model_registry(),
            )
            mock_orch_cls.assert_called_once()

    async def test_parallel_mode_uses_parallel_orchestrator(self):
        with patch(
            "triad.orchestrator.ParallelOrchestrator"
        ) as mock_orch_cls:
            mock_inst = MagicMock()
            mock_inst.run = AsyncMock(return_value=MagicMock())
            mock_orch_cls.return_value = mock_inst

            await run_pipeline(
                _make_task(),
                PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                    persist_sessions=False,
                ),
                _make_three_model_registry(),
            )
            mock_orch_cls.assert_called_once()

    async def test_debate_mode_uses_debate_orchestrator(self):
        with patch(
            "triad.orchestrator.DebateOrchestrator"
        ) as mock_orch_cls:
            mock_inst = MagicMock()
            mock_inst.run = AsyncMock(return_value=MagicMock())
            mock_orch_cls.return_value = mock_inst

            await run_pipeline(
                _make_task(),
                PipelineConfig(
                    pipeline_mode="debate", arbiter_mode="off",
                    persist_sessions=False,
                ),
                _make_three_model_registry(),
            )
            mock_orch_cls.assert_called_once()

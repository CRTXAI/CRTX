"""Tests for the Consensus Protocol (Day 8).

Covers suggestion evaluation, escalation, vote tallying, tiebreaker,
ConsensusResult schema, and orchestrator integration.
"""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.consensus.protocol import _ESCALATION_THRESHOLD, ConsensusEngine
from triad.consensus.suggestions import (
    SuggestionEvaluator,
    format_suggestion_for_evaluation,
    parse_evaluation,
)
from triad.consensus.voting import (
    build_consensus_result,
    extract_winner,
    select_tiebreaker,
    tally_votes,
)
from triad.schemas.consensus import (
    ConsensusResult,
    EscalationResult,
    SuggestionDecision,
    SuggestionVerdict,
    VoteTally,
)
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    Suggestion,
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
_SUGGEST_PROVIDER = "triad.consensus.suggestions.LiteLLMProvider"
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


def _make_registry() -> dict[str, ModelConfig]:
    return {
        "model-a": _make_model_config(
            model="a-v1",
            fitness=RoleFitness(
                architect=0.9, implementer=0.8,
                refactorer=0.7, verifier=0.90,
            ),
        ),
        "model-b": _make_model_config(
            model="b-v1",
            fitness=RoleFitness(
                architect=0.7, implementer=0.9,
                refactorer=0.8, verifier=0.75,
            ),
        ),
        "model-c": _make_model_config(
            model="c-v1",
            fitness=RoleFitness(
                architect=0.6, implementer=0.7,
                refactorer=0.9, verifier=0.80,
            ),
        ),
    }


def _make_suggestion(
    domain: PipelineStage = PipelineStage.IMPLEMENT,
    confidence: float = 0.8,
    **overrides,
) -> Suggestion:
    defaults = {
        "domain": domain,
        "rationale": "Use dependency injection for better testability",
        "confidence": confidence,
        "code_sketch": "class Service:\n    def __init__(self, repo: Repo): ...",
        "impact_assessment": "Improves test isolation",
    }
    defaults.update(overrides)
    return Suggestion(**defaults)


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


def _make_task(**overrides) -> TaskSpec:
    defaults = {"task": "Build a REST API"}
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _mock_provider(responses):
    mock_cls = MagicMock()
    mock_inst = MagicMock()
    mock_cls.return_value = mock_inst
    if isinstance(responses, list):
        mock_inst.complete = AsyncMock(side_effect=responses)
    else:
        mock_inst.complete = AsyncMock(return_value=responses)
    return mock_cls, mock_inst


# ── Suggestion Formatting ────────────────────────────────────────


class TestSuggestionFormatting:
    def test_format_includes_all_fields(self):
        s = _make_suggestion()
        text = format_suggestion_for_evaluation(s)
        assert "implement" in text
        assert "dependency injection" in text
        assert "0.80" in text
        assert "class Service" in text
        assert "test isolation" in text

    def test_format_without_optional_fields(self):
        s = _make_suggestion(code_sketch="", impact_assessment="")
        text = format_suggestion_for_evaluation(s)
        assert "implement" in text
        assert "Code sketch" not in text
        assert "Impact" not in text


# ── Evaluation Parsing ───────────────────────────────────────────


class TestEvaluationParsing:
    def test_parse_accept(self):
        content = "DECISION: ACCEPT\nRATIONALE: Good idea."
        verdict, rationale = parse_evaluation(content)
        assert verdict == SuggestionVerdict.ACCEPT
        assert "Good idea" in rationale

    def test_parse_reject(self):
        content = "DECISION: REJECT\nRATIONALE: Too complex."
        verdict, rationale = parse_evaluation(content)
        assert verdict == SuggestionVerdict.REJECT
        assert "Too complex" in rationale

    def test_parse_case_insensitive(self):
        content = "Decision: accept\nRationale: Fine."
        verdict, _ = parse_evaluation(content)
        assert verdict == SuggestionVerdict.ACCEPT

    def test_parse_no_decision_defaults_reject(self):
        content = "This is some rambling output without a clear decision."
        verdict, _ = parse_evaluation(content)
        assert verdict == SuggestionVerdict.REJECT

    def test_parse_rationale_fallback(self):
        content = "DECISION: ACCEPT\nSome reasoning here."
        verdict, rationale = parse_evaluation(content)
        assert verdict == SuggestionVerdict.ACCEPT
        # Falls back to full content since RATIONALE: isn't present
        assert "Some reasoning" in rationale


# ── Suggestion Evaluator ─────────────────────────────────────────


class TestSuggestionEvaluator:
    async def test_evaluate_accept(self):
        registry = _make_registry()
        msg = _make_agent_message(
            "DECISION: ACCEPT\nRATIONALE: Great idea.",
        )
        mock_cls, _ = _mock_provider(msg)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            evaluator = SuggestionEvaluator(registry, timeout=30)
            decision = await evaluator.evaluate(
                suggestion=_make_suggestion(),
                evaluator_key="model-b",
                task="Build an API",
            )

        assert decision.decision == SuggestionVerdict.ACCEPT
        assert decision.evaluator_model == "model-b"
        assert "Great idea" in decision.rationale

    async def test_evaluate_reject(self):
        registry = _make_registry()
        msg = _make_agent_message(
            "DECISION: REJECT\nRATIONALE: Unnecessary complexity.",
        )
        mock_cls, _ = _mock_provider(msg)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            evaluator = SuggestionEvaluator(registry, timeout=30)
            decision = await evaluator.evaluate(
                suggestion=_make_suggestion(),
                evaluator_key="model-b",
                task="Build an API",
            )

        assert decision.decision == SuggestionVerdict.REJECT

    async def test_escalate_majority_accept(self):
        registry = _make_registry()
        # 3 models: 2 accept, 1 reject → group accepts
        responses = [
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Good."),
            _make_agent_message("DECISION: REJECT\nRATIONALE: Bad."),
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Fine."),
        ]
        mock_cls, _ = _mock_provider(responses)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            evaluator = SuggestionEvaluator(registry, timeout=30)
            result = await evaluator.escalate(
                suggestion=_make_suggestion(),
                pipeline_models={
                    "architect": "model-a",
                    "implement": "model-b",
                    "refactor": "model-c",
                },
                task="Build an API",
            )

        assert result.decision == SuggestionVerdict.ACCEPT
        assert "2-1" in result.rationale

    async def test_escalate_majority_reject(self):
        registry = _make_registry()
        responses = [
            _make_agent_message("DECISION: REJECT\nRATIONALE: No."),
            _make_agent_message("DECISION: REJECT\nRATIONALE: Nah."),
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
        ]
        mock_cls, _ = _mock_provider(responses)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            evaluator = SuggestionEvaluator(registry, timeout=30)
            result = await evaluator.escalate(
                suggestion=_make_suggestion(),
                pipeline_models={
                    "architect": "model-a",
                    "implement": "model-b",
                    "refactor": "model-c",
                },
                task="Build an API",
            )

        assert result.decision == SuggestionVerdict.REJECT

    async def test_escalate_tie_rejects(self):
        """Tie in escalation vote → conservative reject."""
        registry = {
            "model-a": _make_model_config(model="a-v1"),
            "model-b": _make_model_config(model="b-v1"),
        }
        responses = [
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
            _make_agent_message("DECISION: REJECT\nRATIONALE: No."),
        ]
        mock_cls, _ = _mock_provider(responses)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            evaluator = SuggestionEvaluator(registry, timeout=30)
            result = await evaluator.escalate(
                suggestion=_make_suggestion(),
                pipeline_models={
                    "architect": "model-a",
                    "implement": "model-b",
                },
                task="Build an API",
            )

        assert result.decision == SuggestionVerdict.REJECT

    async def test_escalate_deduplicates_models(self):
        """Same model assigned to multiple roles only votes once."""
        registry = _make_registry()
        responses = [
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
        ]
        mock_cls, mock_inst = _mock_provider(responses)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            evaluator = SuggestionEvaluator(registry, timeout=30)
            result = await evaluator.escalate(
                suggestion=_make_suggestion(),
                pipeline_models={
                    "architect": "model-a",
                    "implement": "model-a",  # same model, different role
                    "refactor": "model-a",
                },
                task="Build an API",
            )

        # Only 1 call since all roles use the same model
        assert mock_inst.complete.call_count == 1
        assert len(result.votes) == 1


# ── Vote Tallying ────────────────────────────────────────────────


class TestVoteTallying:
    def test_clear_winner(self):
        votes = {"a": "X", "b": "X", "c": "Y"}
        tally = tally_votes(votes)
        assert tally.winner == "X"
        assert not tally.is_tie
        assert tally.counts == {"X": 2, "Y": 1}

    def test_tie_detected(self):
        votes = {"a": "X", "b": "Y"}
        tally = tally_votes(votes)
        assert tally.winner is None
        assert tally.is_tie
        assert set(tally.tied_options) == {"X", "Y"}

    def test_three_way_tie(self):
        votes = {"a": "X", "b": "Y", "c": "Z"}
        tally = tally_votes(votes)
        assert tally.is_tie
        assert len(tally.tied_options) == 3

    def test_single_voter(self):
        votes = {"a": "X"}
        tally = tally_votes(votes)
        assert tally.winner == "X"
        assert not tally.is_tie

    def test_unanimous(self):
        votes = {"a": "X", "b": "X", "c": "X"}
        tally = tally_votes(votes)
        assert tally.winner == "X"
        assert not tally.is_tie
        assert tally.counts == {"X": 3}

    def test_empty_votes(self):
        tally = tally_votes({})
        assert tally.winner is None
        assert tally.is_tie


# ── Tiebreaker ───────────────────────────────────────────────────


class TestTiebreaker:
    def test_select_tiebreaker_highest_verifier(self):
        registry = _make_registry()
        # model-a has verifier=0.90 (highest)
        key = select_tiebreaker(registry)
        assert key == "model-a"

    def test_select_tiebreaker_empty_raises(self):
        with pytest.raises(RuntimeError, match="No models available"):
            select_tiebreaker({})

    def test_extract_winner_valid(self):
        content = "WINNER: model-b\n\nREASONING: Better approach."
        assert extract_winner(content) == "model-b"

    def test_extract_winner_missing(self):
        content = "I think model-b is better."
        assert extract_winner(content) is None

    def test_extract_winner_embedded(self):
        content = "After careful review:\n\nWINNER: model-c\n\nDone."
        assert extract_winner(content) == "model-c"


# ── ConsensusResult Builder ──────────────────────────────────────


class TestConsensusResultBuilder:
    def test_majority_result(self):
        votes = {"a": "X", "b": "X", "c": "Y"}
        tally = tally_votes(votes)
        result = build_consensus_result(votes, tally)
        assert result.method == "majority"
        assert result.winner == "X"
        assert not result.tiebreaker_used
        assert result.tiebreaker_model is None

    def test_unanimous_result(self):
        votes = {"a": "X", "b": "X", "c": "X"}
        tally = tally_votes(votes)
        result = build_consensus_result(votes, tally)
        assert result.method == "unanimous"
        assert result.winner == "X"

    def test_tiebreak_result(self):
        votes = {"a": "X", "b": "Y"}
        tally = tally_votes(votes)
        result = build_consensus_result(
            votes, tally,
            tiebreaker_model="model-c",
            tiebreaker_winner="Y",
        )
        assert result.method == "tiebreak"
        assert result.winner == "Y"
        assert result.tiebreaker_used
        assert result.tiebreaker_model == "model-c"


# ── Schema Validation ────────────────────────────────────────────


class TestConsensusSchemas:
    def test_suggestion_verdict_values(self):
        assert SuggestionVerdict.ACCEPT == "accept"
        assert SuggestionVerdict.REJECT == "reject"

    def test_suggestion_decision_from_dict(self):
        data = {
            "suggestion_id": "implement-123",
            "evaluator_model": "model-a",
            "decision": "accept",
            "rationale": "Good idea",
        }
        d = SuggestionDecision(**data)
        assert d.decision == SuggestionVerdict.ACCEPT

    def test_escalation_result_from_dict(self):
        data = {
            "suggestion_id": "refactor-456",
            "votes": {"a": "accept", "b": "reject", "c": "accept"},
            "decision": "accept",
            "rationale": "2-1 accept",
        }
        r = EscalationResult(**data)
        assert r.decision == SuggestionVerdict.ACCEPT
        assert len(r.votes) == 3

    def test_vote_tally_from_dict(self):
        data = {
            "counts": {"X": 2, "Y": 1},
            "winner": "X",
            "is_tie": False,
            "tied_options": [],
        }
        t = VoteTally(**data)
        assert t.winner == "X"

    def test_consensus_result_from_dict(self):
        data = {
            "method": "majority",
            "winner": "model-a",
            "votes": {"b": "model-a", "c": "model-a"},
            "tiebreaker_used": False,
            "tiebreaker_model": None,
            "rationale": "2-0 majority",
        }
        r = ConsensusResult(**data)
        assert r.method == "majority"
        assert r.winner == "model-a"


# ── ConsensusEngine ──────────────────────────────────────────────


class TestConsensusEngine:
    async def test_evaluate_suggestion_delegates(self):
        registry = _make_registry()
        config = PipelineConfig(arbiter_mode="off")
        msg = _make_agent_message("DECISION: ACCEPT\nRATIONALE: OK.")
        mock_cls, _ = _mock_provider(msg)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            engine = ConsensusEngine(config, registry)
            decision = await engine.evaluate_suggestion(
                suggestion=_make_suggestion(),
                evaluator_key="model-a",
                task="Build an API",
            )

        assert decision.decision == SuggestionVerdict.ACCEPT

    async def test_escalate_suggestion_delegates(self):
        registry = _make_registry()
        config = PipelineConfig(arbiter_mode="off")
        responses = [
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
            _make_agent_message("DECISION: REJECT\nRATIONALE: No."),
        ]
        mock_cls, _ = _mock_provider(responses)

        with patch(_SUGGEST_PROVIDER, mock_cls):
            engine = ConsensusEngine(config, registry)
            result = await engine.escalate_suggestion(
                suggestion=_make_suggestion(),
                pipeline_models={
                    "architect": "model-a",
                    "implement": "model-b",
                    "refactor": "model-c",
                },
                task="Build an API",
            )

        assert result.decision == SuggestionVerdict.ACCEPT

    def test_should_escalate_high_confidence(self):
        config = PipelineConfig(arbiter_mode="off")
        engine = ConsensusEngine(config, _make_registry())
        s = _make_suggestion(confidence=0.85)
        assert engine.should_escalate(s) is True

    def test_should_not_escalate_low_confidence(self):
        config = PipelineConfig(arbiter_mode="off")
        engine = ConsensusEngine(config, _make_registry())
        s = _make_suggestion(confidence=0.5)
        assert engine.should_escalate(s) is False

    def test_should_not_escalate_at_threshold(self):
        config = PipelineConfig(arbiter_mode="off")
        engine = ConsensusEngine(config, _make_registry())
        s = _make_suggestion(confidence=_ESCALATION_THRESHOLD)
        assert engine.should_escalate(s) is False

    async def test_resolve_consensus_majority(self):
        registry = _make_registry()
        config = PipelineConfig(arbiter_mode="off")
        engine = ConsensusEngine(config, registry)

        result = await engine.resolve_consensus(
            votes={"a": "model-a", "b": "model-a", "c": "model-b"},
            task="Build an API",
        )

        assert result.method == "majority"
        assert result.winner == "model-a"
        assert not result.tiebreaker_used

    async def test_resolve_consensus_tie_invokes_tiebreaker(self):
        registry = _make_registry()
        config = PipelineConfig(arbiter_mode="off")

        tiebreak_msg = _make_agent_message(
            "WINNER: model-b\n\nREASONING: Better.",
        )
        mock_cls, _ = _mock_provider(tiebreak_msg)

        with patch(_CONSENSUS_PROVIDER, mock_cls):
            engine = ConsensusEngine(config, registry)
            result = await engine.resolve_consensus(
                votes={"a": "model-a", "b": "model-b"},
                task="Build an API",
                candidate_outputs={
                    "model-a": "solution A",
                    "model-b": "solution B",
                },
            )

        assert result.method == "tiebreak"
        assert result.winner == "model-b"
        assert result.tiebreaker_used
        assert result.tiebreaker_model == "model-a"  # highest verifier fitness

    async def test_resolve_consensus_tiebreaker_fallback(self):
        """If tiebreaker output is unparseable, falls back to first option."""
        registry = _make_registry()
        config = PipelineConfig(arbiter_mode="off")

        tiebreak_msg = _make_agent_message("I can't decide.")
        mock_cls, _ = _mock_provider(tiebreak_msg)

        with patch(_CONSENSUS_PROVIDER, mock_cls):
            engine = ConsensusEngine(config, registry)
            result = await engine.resolve_consensus(
                votes={"a": "model-a", "b": "model-b"},
                task="Build an API",
            )

        assert result.method == "tiebreak"
        assert result.tiebreaker_used
        # Falls back to first tied option (sorted alphabetically)
        assert result.winner in ("model-a", "model-b")


# ── Orchestrator Integration ─────────────────────────────────────


class TestOrchestratorConsensusIntegration:
    async def test_suggestions_evaluated_in_sequential(self):
        """Suggestions from a stage are evaluated via consensus."""
        from triad.orchestrator import PipelineOrchestrator

        registry = {"test-model": _make_model_config(model="test-v1")}

        suggestion = _make_suggestion(
            domain=PipelineStage.IMPLEMENT, confidence=0.5,
        )
        # Stage responses: architect has suggestions, the rest don't
        arch_msg = _make_agent_message("architecture output")
        arch_msg.suggestions = [suggestion]
        other_msgs = [
            _make_agent_message("impl output"),
            _make_agent_message("refactor output"),
            _make_agent_message("verify output"),
        ]

        mock_cls, _ = _mock_provider([arch_msg] + other_msgs)

        # Mock suggestion evaluation → accept
        eval_msg = _make_agent_message(
            "DECISION: ACCEPT\nRATIONALE: Good idea.",
        )
        suggest_mock_cls, _ = _mock_provider(eval_msg)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_SUGGEST_PROVIDER, suggest_mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock()),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=registry,
            )
            result = await orch.run()

        assert result.success is True
        # Accepted suggestion was included
        assert len(result.suggestions) >= 1

    async def test_high_confidence_rejected_suggestion_escalates(self):
        """High-confidence rejected suggestions trigger escalation."""
        from triad.orchestrator import PipelineOrchestrator

        registry = {
            "model-a": _make_model_config(
                model="a-v1",
                fitness=RoleFitness(
                    architect=0.9, implementer=0.8,
                    refactorer=0.7, verifier=0.85,
                ),
            ),
            "model-b": _make_model_config(
                model="b-v1",
                fitness=RoleFitness(
                    architect=0.7, implementer=0.9,
                    refactorer=0.8, verifier=0.75,
                ),
            ),
        }

        suggestion = _make_suggestion(
            domain=PipelineStage.IMPLEMENT,
            confidence=0.9,  # above escalation threshold
        )
        arch_msg = _make_agent_message("architecture output")
        arch_msg.suggestions = [suggestion]
        other_msgs = [
            _make_agent_message("impl"),
            _make_agent_message("refactor"),
            _make_agent_message("verify"),
        ]
        mock_cls, _ = _mock_provider([arch_msg] + other_msgs)

        # First call: primary holder rejects
        # Then escalation: 2 models vote (both accept)
        suggest_responses = [
            _make_agent_message("DECISION: REJECT\nRATIONALE: No."),
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
            _make_agent_message("DECISION: ACCEPT\nRATIONALE: Yes."),
        ]
        suggest_mock_cls, _ = _mock_provider(suggest_responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_SUGGEST_PROVIDER, suggest_mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock()),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=registry,
            )
            result = await orch.run()

        assert result.success is True
        # Suggestion was escalated and accepted by group vote
        assert len(result.suggestions) >= 1

    async def test_parallel_uses_consensus_voting(self):
        """Parallel mode uses ConsensusEngine for voting."""
        from triad.orchestrator import ParallelOrchestrator

        registry = _make_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock()),
        ):
            orch = ParallelOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="parallel", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.success is True
        assert result.parallel_result is not None
        assert result.parallel_result.winner != ""

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


# ── Judge Selection ──────────────────────────────────────────────


class TestJudgeSelection:
    """Tests for the _select_judge() method on DebateOrchestrator."""

    def test_prefers_external_judge(self):
        """Judge should come from outside the debater set when possible."""
        # 3 debaters, plus an external model in the full registry
        debater_registry = _make_three_model_registry()
        full_registry = {
            **debater_registry,
            "external-judge": _make_model_config(
                model="judge-v1",
                fitness=RoleFitness(
                    architect=0.5, implementer=0.5,
                    refactorer=0.5, verifier=0.95,
                ),
            ),
        }

        orch = DebateOrchestrator(
            task=_make_task(),
            config=PipelineConfig(
                pipeline_mode="debate", arbiter_mode="off",
            ),
            registry=full_registry,
        )
        debater_keys = set(debater_registry.keys())
        judge = orch._select_judge(debater_keys)
        assert judge == "external-judge"

    def test_selects_highest_verifier_external(self):
        """Among multiple external models, pick highest verifier fitness."""
        debater_registry = {
            "model-a": _make_model_config(
                model="a-v1",
                fitness=RoleFitness(verifier=0.7),
            ),
            "model-b": _make_model_config(
                model="b-v1",
                fitness=RoleFitness(verifier=0.8),
            ),
        }
        full_registry = {
            **debater_registry,
            "ext-low": _make_model_config(
                model="ext-low-v1",
                fitness=RoleFitness(verifier=0.75),
            ),
            "ext-high": _make_model_config(
                model="ext-high-v1",
                fitness=RoleFitness(verifier=0.99),
            ),
        }

        orch = DebateOrchestrator(
            task=_make_task(),
            config=PipelineConfig(
                pipeline_mode="debate", arbiter_mode="off",
            ),
            registry=full_registry,
        )
        judge = orch._select_judge(set(debater_registry.keys()))
        assert judge == "ext-high"

    def test_falls_back_to_debater_when_no_external(self):
        """If no external models, fall back to best debater (verifier)."""
        registry = _make_three_model_registry()
        # model-c has verifier=0.90 (highest)

        orch = DebateOrchestrator(
            task=_make_task(),
            config=PipelineConfig(
                pipeline_mode="debate", arbiter_mode="off",
            ),
            registry=registry,
        )
        judge = orch._select_judge(set(registry.keys()))
        assert judge == "model-c"

    def test_all_models_are_debaters_uses_tiebreaker(self):
        """When registry == debaters, use tiebreaker (best verifier)."""
        registry = {
            "model-x": _make_model_config(
                model="x-v1",
                fitness=RoleFitness(verifier=0.60),
            ),
            "model-y": _make_model_config(
                model="y-v1",
                fitness=RoleFitness(verifier=0.85),
            ),
        }

        orch = DebateOrchestrator(
            task=_make_task(),
            config=PipelineConfig(
                pipeline_mode="debate", arbiter_mode="off",
            ),
            registry=registry,
        )
        judge = orch._select_judge(set(registry.keys()))
        assert judge == "model-y"


# ── Rebuttal Parsing ─────────────────────────────────────────────


class TestRebuttalParsing:
    """Tests for the improved _parse_rebuttals method."""

    def _make_orch(self):
        """Create a minimal DebateOrchestrator for calling _parse_rebuttals."""
        return DebateOrchestrator(
            task=_make_task(),
            config=PipelineConfig(
                pipeline_mode="debate", arbiter_mode="off",
            ),
            registry=_make_three_model_registry(),
        )

    def test_parses_standard_rebuttal_header(self):
        """Standard '## Rebuttal: key' header should be parsed."""
        orch = self._make_orch()
        content = (
            "## Rebuttal: model-b\n\n"
            "Good but has issues.\n\n"
            "## Rebuttal: model-c\n\n"
            "Interesting approach."
        )
        result = orch._parse_rebuttals(content, ["model-b", "model-c"])
        assert "model-b" in result
        assert "model-c" in result
        assert "Good but has issues." in result["model-b"]
        assert "Interesting approach." in result["model-c"]

    def test_parses_response_to_header(self):
        """'## Response to key' header should be parsed."""
        orch = self._make_orch()
        content = (
            "## Response to model-b\n\n"
            "My response.\n\n"
            "## Response to model-c\n\n"
            "Another response."
        )
        result = orch._parse_rebuttals(content, ["model-b", "model-c"])
        assert "My response." in result["model-b"]
        assert "Another response." in result["model-c"]

    def test_parses_bare_model_key_header(self):
        """'## model-key' header should be parsed."""
        orch = self._make_orch()
        content = (
            "## model-b\n\n"
            "Critique of B.\n\n"
            "## model-c\n\n"
            "Critique of C."
        )
        result = orch._parse_rebuttals(content, ["model-b", "model-c"])
        assert "Critique of B." in result["model-b"]
        assert "Critique of C." in result["model-c"]

    def test_parses_h3_rebuttal_header(self):
        """'### Rebuttal: key' header should be parsed."""
        orch = self._make_orch()
        content = (
            "### Rebuttal: model-b\n\n"
            "H3 rebuttal.\n\n"
            "### Rebuttal: model-c\n\n"
            "Another H3 rebuttal."
        )
        result = orch._parse_rebuttals(content, ["model-b", "model-c"])
        assert "H3 rebuttal." in result["model-b"]

    def test_falls_back_to_full_content_on_no_match(self):
        """If no headers match, entire content is assigned to all targets."""
        orch = self._make_orch()
        content = "No headers here, just raw rebuttal text."
        result = orch._parse_rebuttals(content, ["model-b", "model-c"])
        assert result["model-b"] == content
        assert result["model-c"] == content

    def test_case_insensitive_matching(self):
        """Header matching should be case-insensitive."""
        orch = self._make_orch()
        content = (
            "## REBUTTAL: model-b\n\n"
            "Uppercase rebuttal."
        )
        result = orch._parse_rebuttals(content, ["model-b"])
        assert "Uppercase rebuttal." in result["model-b"]

    def test_mixed_heading_patterns(self):
        """Different heading patterns for different targets should work."""
        orch = self._make_orch()
        content = (
            "## Rebuttal: model-b\n\n"
            "Standard rebuttal.\n\n"
            "## Response to model-c\n\n"
            "Response-style rebuttal."
        )
        result = orch._parse_rebuttals(content, ["model-b", "model-c"])
        assert "Standard rebuttal." in result["model-b"]
        assert "Response-style rebuttal." in result["model-c"]


# ── Debate Event Emissions ───────────────────────────────────────


class TestDebateEventEmissions:
    """Tests that the debate orchestrator emits correct events."""

    async def test_emits_pipeline_started_and_completed(self):
        """Debate mode should emit pipeline_started and pipeline_completed."""
        from triad.dashboard.events import PipelineEventEmitter

        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        emitter = PipelineEventEmitter()

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
                event_emitter=emitter,
            )
            await orch.run()

        event_types = [e.type for e in emitter.history]
        assert "pipeline_started" in event_types
        assert "pipeline_completed" in event_types

    async def test_emits_stage_events_for_all_phases(self):
        """Debate mode should emit stage_started/completed for each phase."""
        from triad.dashboard.events import PipelineEventEmitter

        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        emitter = PipelineEventEmitter()

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
                event_emitter=emitter,
            )
            await orch.run()

        event_types = [e.type for e in emitter.history]
        # Should have at least 4 stage_started (proposals, rebuttals,
        # final_arguments, judgment) and 4 stage_completed
        assert event_types.count("stage_started") >= 4
        assert event_types.count("stage_completed") >= 4

    async def test_pipeline_started_event_has_mode_debate(self):
        """pipeline_started event should have mode='debate'."""
        from triad.dashboard.events import PipelineEventEmitter

        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        emitter = PipelineEventEmitter()

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
                event_emitter=emitter,
            )
            await orch.run()

        started_events = [
            e for e in emitter.history if e.type == "pipeline_started"
        ]
        assert len(started_events) == 1
        assert started_events[0].data["mode"] == "debate"

    async def test_stage_names_include_debate_phases(self):
        """Emitted stage events should reference debate phase names."""
        from triad.dashboard.events import PipelineEventEmitter

        registry = _make_three_model_registry()
        responses = _debate_responses(3)
        mock_cls, _ = _mock_provider(responses)
        emitter = PipelineEventEmitter()

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
                event_emitter=emitter,
            )
            await orch.run()

        stage_names = [
            e.data.get("stage", "")
            for e in emitter.history
            if e.type in ("stage_started", "stage_completed")
        ]
        assert "debate_proposals" in stage_names
        assert "debate_rebuttals" in stage_names
        assert "debate_final_arguments" in stage_names
        assert "debate_judgment" in stage_names

    async def test_no_events_when_no_emitter(self):
        """When no emitter is provided, no events are emitted (no error)."""
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
                event_emitter=None,
            )
            result = await orch.run()

        assert result.success is True

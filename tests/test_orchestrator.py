"""Tests for triad.orchestrator — Sequential pipeline engine."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from pydantic import ValidationError

from triad.orchestrator import (
    PipelineOrchestrator,
    _extract_confidence,
    _format_suggestions,
)
from triad.schemas.arbiter import ArbiterReview, Verdict
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
    PipelineResult,
    RoleFitness,
    StageConfig,
    TaskSpec,
)

# Patch targets
_PROVIDER = "triad.orchestrator.LiteLLMProvider"
_ARBITER_REVIEW = "triad.orchestrator.ArbiterEngine.review"
_RECONCILER_RECONCILE = "triad.orchestrator.ReconciliationEngine.reconcile"


# ── Factories ──────────────────────────────────────────────────────

def _make_model_config(**overrides) -> ModelConfig:
    defaults = {
        "provider": "test",
        "model": "test-model-v1",
        "display_name": "Test Model",
        "api_key_env": "TEST_KEY",
        "context_window": 128000,
        "cost_input": 3.0,
        "cost_output": 15.0,
        "fitness": RoleFitness(
            architect=0.9, implementer=0.8, refactorer=0.7, verifier=0.85,
        ),
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_registry() -> dict[str, ModelConfig]:
    return {"test-model-v1": _make_model_config()}


def _make_task(**overrides) -> TaskSpec:
    defaults = {"task": "Build a REST API with authentication"}
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_agent_message(
    content: str = "stage output",
    confidence: float = 0.85,
    suggestions: list | None = None,
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    cost: float = 0.001,
) -> AgentMessage:
    """Create a canned AgentMessage for mock provider responses."""
    return AgentMessage(
        from_agent=PipelineStage.ARCHITECT,
        to_agent=PipelineStage.IMPLEMENT,
        msg_type=MessageType.IMPLEMENTATION,
        content=f"{content}\n\nCONFIDENCE: {confidence}",
        confidence=0.0,
        suggestions=suggestions or [],
        token_usage=TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        ),
        model="test-model-v1",
    )


def _make_approve_review(**overrides) -> ArbiterReview:
    """Create a canned APPROVE ArbiterReview."""
    defaults = {
        "stage_reviewed": PipelineStage.VERIFY,
        "reviewed_model": "test-model-v1",
        "arbiter_model": "test-model-v2",
        "verdict": Verdict.APPROVE,
        "confidence": 0.95,
        "reasoning": "All looks good. VERDICT: APPROVE\nCONFIDENCE: 0.95",
        "token_cost": 0.01,
    }
    defaults.update(overrides)
    return ArbiterReview(**defaults)


def _mock_provider_factory(responses: list[AgentMessage]):
    """Create a mock LiteLLMProvider class that returns canned responses."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.complete = AsyncMock(side_effect=responses)
    return mock_cls, mock_instance


# ── ExtractConfidence ──────────────────────────────────────────────

class TestExtractConfidence:
    def test_extracts_valid_confidence(self):
        assert _extract_confidence("Some output\n\nCONFIDENCE: 0.85") == 0.85

    def test_extracts_integer_confidence(self):
        assert _extract_confidence("CONFIDENCE: 1") == 1.0

    def test_clamps_above_one(self):
        assert _extract_confidence("CONFIDENCE: 1.5") == 1.0

    def test_clamps_below_zero(self):
        assert _extract_confidence("CONFIDENCE: -0.3") == 0.0

    def test_returns_zero_when_missing(self):
        assert _extract_confidence("No confidence here") == 0.0

    def test_returns_zero_for_empty(self):
        assert _extract_confidence("") == 0.0

    def test_extracts_from_middle_of_text(self):
        text = "## Report\nLooks good.\nCONFIDENCE: 0.92\n## Suggestions"
        assert _extract_confidence(text) == 0.92


# ── FormatSuggestions ──────────────────────────────────────────────

class TestFormatSuggestions:
    def test_formats_relevant_suggestions(self):
        suggestions = [
            Suggestion(
                domain=PipelineStage.REFACTOR,
                rationale="Consider adding input validation",
                confidence=0.8,
            ),
        ]
        result = _format_suggestions(suggestions, PipelineStage.REFACTOR)
        assert "Consider adding input validation" in result
        assert "0.8" in result

    def test_filters_by_target_stage(self):
        suggestions = [
            Suggestion(
                domain=PipelineStage.REFACTOR,
                rationale="Refactor suggestion",
                confidence=0.8,
            ),
            Suggestion(
                domain=PipelineStage.VERIFY,
                rationale="Verify suggestion",
                confidence=0.7,
            ),
        ]
        result = _format_suggestions(suggestions, PipelineStage.REFACTOR)
        assert "Refactor suggestion" in result
        assert "Verify suggestion" not in result

    def test_returns_empty_when_no_matches(self):
        suggestions = [
            Suggestion(
                domain=PipelineStage.ARCHITECT,
                rationale="Arch suggestion",
                confidence=0.9,
            ),
        ]
        assert _format_suggestions(suggestions, PipelineStage.REFACTOR) == ""

    def test_returns_empty_for_empty_list(self):
        assert _format_suggestions([], PipelineStage.ARCHITECT) == ""

    def test_includes_code_sketch(self):
        suggestions = [
            Suggestion(
                domain=PipelineStage.IMPLEMENT,
                rationale="Use async generators",
                confidence=0.75,
                code_sketch="async def stream(): yield item",
            ),
        ]
        result = _format_suggestions(suggestions, PipelineStage.IMPLEMENT)
        assert "async def stream(): yield item" in result

    def test_includes_impact_assessment(self):
        suggestions = [
            Suggestion(
                domain=PipelineStage.VERIFY,
                rationale="Add edge case tests",
                confidence=0.6,
                impact_assessment="Catches null input bugs",
            ),
        ]
        result = _format_suggestions(suggestions, PipelineStage.VERIFY)
        assert "Catches null input bugs" in result


# ── PipelineResult Schema ─────────────────────────────────────────

class TestPipelineResult:
    def test_defaults(self):
        result = PipelineResult(
            task=_make_task(),
            config=PipelineConfig(arbiter_mode="off"),
        )
        assert result.stages == {}
        assert result.suggestions == []
        assert result.total_cost == 0.0
        assert result.total_tokens == 0
        assert result.duration_seconds == 0.0
        assert result.success is False

    def test_full_result(self):
        msg = _make_agent_message()
        result = PipelineResult(
            task=_make_task(),
            config=PipelineConfig(arbiter_mode="off"),
            stages={PipelineStage.ARCHITECT: msg},
            suggestions=[
                Suggestion(
                    domain=PipelineStage.IMPLEMENT,
                    rationale="test",
                    confidence=0.5,
                ),
            ],
            total_cost=1.23,
            total_tokens=5000,
            duration_seconds=45.6,
            success=True,
        )
        assert result.success is True
        assert result.total_cost == 1.23
        assert PipelineStage.ARCHITECT in result.stages

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            PipelineResult(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                total_cost=-1.0,
            )

    def test_serialization_roundtrip(self):
        result = PipelineResult(
            task=_make_task(),
            config=PipelineConfig(arbiter_mode="off"),
            total_cost=0.5,
            total_tokens=1000,
            duration_seconds=10.0,
            success=True,
        )
        data = result.model_dump()
        restored = PipelineResult.model_validate(data)
        assert restored.total_cost == result.total_cost
        assert restored.success is True


# ── PipelineOrchestrator ──────────────────────────────────────────

class TestPipelineOrchestratorInit:
    def test_session_starts_empty(self):
        orch = PipelineOrchestrator(
            task=_make_task(),
            config=PipelineConfig(arbiter_mode="off"),
            registry=_make_registry(),
        )
        assert orch.session == []


class TestPipelineExecution:
    """Tests for full pipeline runs with mocked providers."""

    @pytest.fixture()
    def four_stage_responses(self):
        """Canned AgentMessages for all 4 stages."""
        return [
            _make_agent_message("Architect scaffold output", 0.92),
            _make_agent_message("Implementer wired code", 0.88),
            _make_agent_message("Refactorer improved code", 0.90),
            _make_agent_message("Verifier report", 0.95),
        ]

    async def test_full_pipeline_runs_four_stages(self, four_stage_responses):
        mock_cls, mock_inst = _mock_provider_factory(four_stage_responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        assert result.success is True
        assert len(result.stages) == 4
        assert PipelineStage.ARCHITECT in result.stages
        assert PipelineStage.IMPLEMENT in result.stages
        assert PipelineStage.REFACTOR in result.stages
        assert PipelineStage.VERIFY in result.stages

    async def test_provider_called_four_times(self, four_stage_responses):
        mock_cls, mock_inst = _mock_provider_factory(four_stage_responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        assert mock_inst.complete.call_count == 4

    async def test_stage_routing_metadata(self, four_stage_responses):
        mock_cls, mock_inst = _mock_provider_factory(four_stage_responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        arch_msg = result.stages[PipelineStage.ARCHITECT]
        assert arch_msg.from_agent == PipelineStage.ARCHITECT
        assert arch_msg.to_agent == PipelineStage.IMPLEMENT
        assert arch_msg.msg_type == MessageType.PROPOSAL

        impl_msg = result.stages[PipelineStage.IMPLEMENT]
        assert impl_msg.from_agent == PipelineStage.IMPLEMENT
        assert impl_msg.to_agent == PipelineStage.REFACTOR
        assert impl_msg.msg_type == MessageType.IMPLEMENTATION

        verify_msg = result.stages[PipelineStage.VERIFY]
        assert verify_msg.from_agent == PipelineStage.VERIFY
        assert verify_msg.msg_type == MessageType.VERIFICATION

    async def test_confidence_extracted_from_content(self, four_stage_responses):
        mock_cls, mock_inst = _mock_provider_factory(four_stage_responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        assert result.stages[PipelineStage.ARCHITECT].confidence == 0.92
        assert result.stages[PipelineStage.VERIFY].confidence == 0.95

    async def test_session_audit_trail(self, four_stage_responses):
        mock_cls, mock_inst = _mock_provider_factory(four_stage_responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        assert len(orch.session) == 4
        assert orch.session[0].from_agent == PipelineStage.ARCHITECT
        assert orch.session[3].from_agent == PipelineStage.VERIFY


class TestOutputPassing:
    """Tests that output flows correctly between stages."""

    async def test_architect_output_in_implementer_prompt(self):
        responses = [
            _make_agent_message("SCAFFOLD: file structure here", 0.9),
            _make_agent_message("Implementation done", 0.85),
            _make_agent_message("Refactored", 0.88),
            _make_agent_message("Verified", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        assert "SCAFFOLD: file structure here" in calls[1].kwargs["system"]

    async def test_implementer_output_in_refactorer_prompt(self):
        responses = [
            _make_agent_message("Architect output", 0.9),
            _make_agent_message("IMPL: business logic wired", 0.85),
            _make_agent_message("Refactored", 0.88),
            _make_agent_message("Verified", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        assert "IMPL: business logic wired" in calls[2].kwargs["system"]

    async def test_verifier_receives_architect_and_refactorer_output(self):
        responses = [
            _make_agent_message("ARCH: scaffold", 0.9),
            _make_agent_message("IMPL: code", 0.85),
            _make_agent_message("REFACT: improved code", 0.88),
            _make_agent_message("Verified", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        verifier_system = calls[3].kwargs["system"]
        # Verifier should see the refactorer output as previous_output
        assert "REFACT: improved code" in verifier_system
        # Verifier should also see the architect output separately
        assert "ARCH: scaffold" in verifier_system

    async def test_task_appears_in_all_prompts(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(task="Build a CLI tool with arg parsing"),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        for call in calls:
            assert "Build a CLI tool with arg parsing" in call.kwargs["system"]

    async def test_domain_rules_injected(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(domain_rules="Always use snake_case"),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        assert "Always use snake_case" in calls[0].kwargs["system"]


class TestSuggestions:
    """Tests for cross-domain suggestion collection and downstream passing."""

    async def test_suggestions_collected_from_all_stages(self):
        arch_sugg = Suggestion(
            domain=PipelineStage.IMPLEMENT,
            rationale="Use async generators",
            confidence=0.8,
        )
        impl_sugg = Suggestion(
            domain=PipelineStage.REFACTOR,
            rationale="Add input validation",
            confidence=0.7,
        )
        responses = [
            _make_agent_message("arch", 0.9, suggestions=[arch_sugg]),
            _make_agent_message("impl", 0.85, suggestions=[impl_sugg]),
            _make_agent_message("refact", 0.88),
            _make_agent_message("verify", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        assert len(result.suggestions) == 2
        assert result.suggestions[0].rationale == "Use async generators"
        assert result.suggestions[1].rationale == "Add input validation"

    async def test_suggestions_passed_downstream(self):
        """Suggestions from architect should appear in refactorer's prompt
        if they target the refactor domain."""
        arch_sugg = Suggestion(
            domain=PipelineStage.REFACTOR,
            rationale="Consider adding type narrowing",
            confidence=0.75,
        )
        responses = [
            _make_agent_message("arch", 0.9, suggestions=[arch_sugg]),
            _make_agent_message("impl", 0.85),
            _make_agent_message("refact", 0.88),
            _make_agent_message("verify", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        refactorer_system = calls[2].kwargs["system"]
        assert "Consider adding type narrowing" in refactorer_system


class TestTimeoutConfig:
    """Tests for per-stage timeout configuration."""

    async def test_default_timeout_used(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)
        config = PipelineConfig(default_timeout=90, arbiter_mode="off")

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=config,
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        for call in calls:
            assert call.kwargs["timeout"] == 90

    async def test_per_stage_timeout_override(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)
        config = PipelineConfig(
            default_timeout=120,
            arbiter_mode="off",
            stages={
                PipelineStage.ARCHITECT: StageConfig(timeout=60),
                PipelineStage.IMPLEMENT: StageConfig(timeout=180),
            },
        )

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=config,
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        assert calls[0].kwargs["timeout"] == 60   # architect override
        assert calls[1].kwargs["timeout"] == 180  # implement override
        assert calls[2].kwargs["timeout"] == 120  # refactor default
        assert calls[3].kwargs["timeout"] == 120  # verify default


class TestPipelineResultAggregation:
    """Tests for cost, token, and duration aggregation."""

    async def test_total_cost_aggregated(self):
        responses = [
            _make_agent_message("out", 0.9, cost=0.50),
            _make_agent_message("out", 0.85, cost=1.20),
            _make_agent_message("out", 0.88, cost=0.80),
            _make_agent_message("out", 0.92, cost=0.30),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        assert abs(result.total_cost - 2.80) < 1e-10

    async def test_total_tokens_aggregated(self):
        responses = [
            _make_agent_message("out", 0.9, prompt_tokens=1000, completion_tokens=500),
            _make_agent_message("out", 0.85, prompt_tokens=2000, completion_tokens=800),
            _make_agent_message("out", 0.88, prompt_tokens=1500, completion_tokens=600),
            _make_agent_message("out", 0.92, prompt_tokens=1200, completion_tokens=400),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        expected = (1000 + 500) + (2000 + 800) + (1500 + 600) + (1200 + 400)
        assert result.total_tokens == expected

    async def test_duration_measured(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            result = await orch.run()

        assert result.duration_seconds >= 0.0


class TestModelSelection:
    """Tests for model resolution per stage."""

    async def test_stage_override_model_used(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        fast_model = _make_model_config(
            model="fast-model-v1",
            display_name="Fast Model",
            fitness=RoleFitness(architect=0.5, implementer=0.5),
        )
        registry = {
            "test-model-v1": _make_model_config(),
            "fast-model-v1": fast_model,
        }
        config = PipelineConfig(
            arbiter_mode="off",
            stages={PipelineStage.ARCHITECT: StageConfig(model="fast-model-v1")},
        )

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=config,
                registry=registry,
            )
            await orch.run()

        # First call should use the override model config
        first_call_config = mock_cls.call_args_list[0][0][0]
        assert first_call_config.model == "fast-model-v1"

    async def test_missing_override_model_raises(self):
        responses = [_make_agent_message("out", 0.9)]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        config = PipelineConfig(
            arbiter_mode="off",
            stages={PipelineStage.ARCHITECT: StageConfig(model="nonexistent-model")},
        )

        with (
            patch(_PROVIDER, mock_cls),
            pytest.raises(RuntimeError, match="not in the model registry"),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=config,
                registry=_make_registry(),
            )
            await orch.run()

    async def test_empty_registry_raises(self):
        responses = [_make_agent_message("out", 0.9)]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with (
            patch(_PROVIDER, mock_cls),
            pytest.raises(RuntimeError, match="No models available"),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry={},
            )
            await orch.run()


class TestReconciliationFlag:
    """Tests that the reconciliation flag is passed to the verifier."""

    async def test_reconciliation_disabled_by_default(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        verifier_system = calls[3].kwargs["system"]
        # The reconciliation-specific section should NOT appear
        assert "Implementation Summary" not in verifier_system

    async def test_reconciliation_enabled_injects_section(self):
        responses = [
            _make_agent_message("out", 0.9),
            _make_agent_message("out", 0.85),
            _make_agent_message("out", 0.88),
            _make_agent_message("out", 0.92),
        ]
        mock_cls, mock_inst = _mock_provider_factory(responses)
        config = PipelineConfig(
            reconciliation_enabled=True, arbiter_mode="off",
        )

        with (
            patch(_PROVIDER, mock_cls),
            patch(_RECONCILER_RECONCILE, new_callable=AsyncMock) as mock_recon,
        ):
            mock_recon.return_value = _make_approve_review()
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=config,
                registry=_make_registry(),
            )
            await orch.run()

        calls = mock_inst.complete.call_args_list
        verifier_system = calls[3].kwargs["system"]
        assert "Implementation Summary" in verifier_system

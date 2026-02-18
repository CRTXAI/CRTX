"""Tests for the Arbiter layer — ArbiterEngine, feedback, reconciler."""

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.arbiter.arbiter import (
    ArbiterEngine,
    _extract_confidence,
    _extract_verdict,
)
from triad.arbiter.feedback import format_arbiter_feedback
from triad.arbiter.reconciler import ReconciliationEngine
from triad.schemas.arbiter import (
    Alternative,
    ArbiterReview,
    Issue,
    IssueCategory,
    Severity,
    Verdict,
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
    TaskSpec,
)

# Patch targets
_ARBITER_PROVIDER = "triad.arbiter.arbiter.LiteLLMProvider"
_RECONCILER_PROVIDER = "triad.arbiter.reconciler.LiteLLMProvider"


# ── Factories ──────────────────────────────────────────────────────

def _make_model_config(
    model: str = "test-model-v1",
    display_name: str = "Test Model",
    **overrides,
) -> ModelConfig:
    defaults = {
        "provider": "test",
        "model": model,
        "display_name": display_name,
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


def _make_two_model_registry() -> dict[str, ModelConfig]:
    """Registry with two distinct models for cross-model enforcement."""
    return {
        "model-a": _make_model_config(
            model="model-a-v1",
            display_name="Model A",
            fitness=RoleFitness(
                architect=0.9, implementer=0.8,
                refactorer=0.7, verifier=0.85,
            ),
        ),
        "model-b": _make_model_config(
            model="model-b-v1",
            display_name="Model B",
            fitness=RoleFitness(
                architect=0.7, implementer=0.9,
                refactorer=0.8, verifier=0.75,
            ),
        ),
    }


def _make_task(**overrides) -> TaskSpec:
    defaults = {"task": "Build a REST API with authentication"}
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_agent_message(
    content: str = "VERDICT: APPROVE\nCONFIDENCE: 0.90",
    model: str = "model-a-v1",
) -> AgentMessage:
    return AgentMessage(
        from_agent=PipelineStage.ARCHITECT,
        to_agent=PipelineStage.IMPLEMENT,
        msg_type=MessageType.IMPLEMENTATION,
        content=content,
        confidence=0.0,
        token_usage=TokenUsage(
            prompt_tokens=500, completion_tokens=200, cost=0.01,
        ),
        model=model,
    )


def _make_review(**overrides) -> ArbiterReview:
    defaults = {
        "stage_reviewed": PipelineStage.ARCHITECT,
        "reviewed_model": "model-a-v1",
        "arbiter_model": "model-b-v1",
        "verdict": Verdict.REJECT,
        "confidence": 0.85,
        "reasoning": "Issues found.",
        "token_cost": 0.01,
        "issues": [
            Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.LOGIC,
                description="Missing error handling for auth failures",
                suggestion="Add try/except around auth calls",
            ),
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.SECURITY,
                description="API key exposed in logs",
                location="src/auth.py:42-45",
            ),
        ],
        "alternatives": [
            Alternative(
                description="Use OAuth2 instead of API keys",
                rationale="More secure and standard",
                confidence=0.8,
            ),
        ],
    }
    defaults.update(overrides)
    return ArbiterReview(**defaults)


def _mock_arbiter_provider(content: str):
    """Create a mock provider class that returns an AgentMessage."""
    mock_cls = MagicMock()
    mock_instance = MagicMock()
    mock_cls.return_value = mock_instance
    mock_instance.complete = AsyncMock(
        return_value=_make_agent_message(content),
    )
    return mock_cls, mock_instance


# ── Verdict Extraction ─────────────────────────────────────────────

class TestExtractVerdict:
    def test_extracts_approve(self):
        assert _extract_verdict("VERDICT: APPROVE") == Verdict.APPROVE

    def test_extracts_flag(self):
        assert _extract_verdict("VERDICT: FLAG") == Verdict.FLAG

    def test_extracts_reject(self):
        assert _extract_verdict("VERDICT: REJECT") == Verdict.REJECT

    def test_extracts_halt(self):
        assert _extract_verdict("VERDICT: HALT") == Verdict.HALT

    def test_case_insensitive(self):
        assert _extract_verdict("VERDICT: approve") == Verdict.APPROVE

    def test_bold_markdown(self):
        assert _extract_verdict("**VERDICT: REJECT**") == Verdict.REJECT

    def test_defaults_to_flag_on_missing(self):
        assert _extract_verdict("No verdict here") == Verdict.FLAG

    def test_defaults_to_flag_on_empty(self):
        assert _extract_verdict("") == Verdict.FLAG

    def test_extracts_from_longer_text(self):
        text = "## Review\nLooks mostly good.\nVERDICT: FLAG\n## Issues"
        assert _extract_verdict(text) == Verdict.FLAG


# ── Confidence Extraction (Arbiter) ────────────────────────────────

class TestArbiterExtractConfidence:
    def test_extracts_valid(self):
        assert _extract_confidence("CONFIDENCE: 0.85") == 0.85

    def test_defaults_to_half_on_missing(self):
        assert _extract_confidence("No confidence") == 0.5

    def test_clamps_above_one(self):
        assert _extract_confidence("CONFIDENCE: 1.5") == 1.0

    def test_negative_returns_default(self):
        # Regex doesn't match negative sign, so returns default 0.5
        assert _extract_confidence("CONFIDENCE: -0.1") == 0.5


# ── ArbiterEngine ─────────────────────────────────────────────────

class TestArbiterEngine:
    async def test_review_returns_structured_review(self):
        mock_cls, _ = _mock_arbiter_provider(
            "Good output.\nVERDICT: APPROVE\nCONFIDENCE: 0.92"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="Scaffold output",
                task=_make_task(),
            )

        assert review.verdict == Verdict.APPROVE
        assert review.confidence == 0.92
        assert review.stage_reviewed == PipelineStage.ARCHITECT
        assert review.reviewed_model == "model-a-v1"
        assert review.arbiter_model == "model-b-v1"
        assert review.token_cost == 0.01

    async def test_cross_model_enforcement(self):
        """Arbiter model must differ from stage model."""
        mock_cls, mock_inst = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="output",
                task=_make_task(),
            )

        # The provider should have been called with model-b, not model-a
        provider_config = mock_cls.call_args[0][0]
        assert provider_config.model != "model-a-v1"

    async def test_global_arbiter_override(self):
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig(arbiter_model="model-b")
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="output",
                task=_make_task(),
            )

        assert review.arbiter_model == "model-b-v1"

    async def test_global_override_same_as_stage_returns_none(self):
        """When the override is the same model as the stage, arbiter degrades gracefully."""
        registry = _make_two_model_registry()
        config = PipelineConfig(arbiter_model="model-a")
        engine = ArbiterEngine(config, registry)

        review = await engine.review(
            stage=PipelineStage.ARCHITECT,
            stage_model="model-a-v1",
            stage_output="output",
            task=_make_task(),
        )
        # Graceful degradation: returns None when no arbiter model available
        assert review is None

    async def test_single_model_registry_returns_none(self):
        """Single-model registry cannot do cross-model review — degrades gracefully."""
        registry = {"only": _make_model_config(model="only-model")}
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        review = await engine.review(
            stage=PipelineStage.ARCHITECT,
            stage_model="only-model",
            stage_output="output",
            task=_make_task(),
        )
        assert review is None

    async def test_nonexistent_override_falls_back(self):
        """When arbiter_model override isn't in registry, falls back to best available."""
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig(arbiter_model="nonexistent")
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="output",
                task=_make_task(),
            )

        # Falls back to model-b (the only cross-model candidate)
        assert review is not None
        assert review.arbiter_model == "model-b-v1"

    async def test_low_confidence_approve_downgraded_to_flag(self):
        """APPROVE with confidence below 0.50 is downgraded to FLAG."""
        mock_cls, _ = _mock_arbiter_provider(
            "Looks ok.\nVERDICT: APPROVE\nCONFIDENCE: 0.44"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="Scaffold output",
                task=_make_task(),
            )

        assert review.verdict == Verdict.FLAG
        assert review.confidence == 0.44

    async def test_approve_at_threshold_not_downgraded(self):
        """APPROVE with confidence exactly at 0.50 stays APPROVE."""
        mock_cls, _ = _mock_arbiter_provider(
            "Looks good.\nVERDICT: APPROVE\nCONFIDENCE: 0.50"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="Scaffold output",
                task=_make_task(),
            )

        assert review.verdict == Verdict.APPROVE
        assert review.confidence == 0.50

    async def test_high_confidence_approve_not_downgraded(self):
        """APPROVE with high confidence stays APPROVE."""
        mock_cls, _ = _mock_arbiter_provider(
            "Excellent.\nVERDICT: APPROVE\nCONFIDENCE: 0.92"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="Scaffold output",
                task=_make_task(),
            )

        assert review.verdict == Verdict.APPROVE
        assert review.confidence == 0.92

    async def test_low_confidence_reject_not_affected(self):
        """REJECT with low confidence is NOT changed — floor only applies to APPROVE."""
        mock_cls, _ = _mock_arbiter_provider(
            "Bad output.\nVERDICT: REJECT\nCONFIDENCE: 0.30"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="model-a-v1",
                stage_output="Scaffold output",
                task=_make_task(),
            )

        assert review.verdict == Verdict.REJECT
        assert review.confidence == 0.30

    async def test_selects_highest_fitness_candidate(self):
        """When auto-selecting, picks the model with highest avg fitness."""
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        high_fitness = _make_model_config(
            model="high-v1",
            display_name="High",
            fitness=RoleFitness(
                architect=0.95, implementer=0.95,
                refactorer=0.95, verifier=0.95,
            ),
        )
        low_fitness = _make_model_config(
            model="low-v1",
            display_name="Low",
            fitness=RoleFitness(
                architect=0.3, implementer=0.3,
                refactorer=0.3, verifier=0.3,
            ),
        )
        stage_model = _make_model_config(
            model="stage-v1",
            display_name="Stage",
        )
        registry = {
            "high": high_fitness,
            "low": low_fitness,
            "stage": stage_model,
        }
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="stage-v1",
                stage_output="output",
                task=_make_task(),
            )

        assert review.arbiter_model == "high-v1"

    async def test_selects_by_verifier_fitness_not_average(self):
        """Arbiter should pick the highest verifier fitness, not avg."""
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        # high_avg has high average but LOW verifier
        high_avg = _make_model_config(
            model="high-avg-v1",
            display_name="High Avg",
            fitness=RoleFitness(
                architect=0.95, implementer=0.95,
                refactorer=0.95, verifier=0.70,
            ),
        )
        # high_verifier has low average but HIGH verifier
        high_verifier = _make_model_config(
            model="high-ver-v1",
            display_name="High Verifier",
            fitness=RoleFitness(
                architect=0.50, implementer=0.50,
                refactorer=0.50, verifier=0.98,
            ),
        )
        stage_model = _make_model_config(
            model="stage-v1",
            display_name="Stage",
        )
        registry = {
            "high-avg": high_avg,
            "high-ver": high_verifier,
            "stage": stage_model,
        }
        config = PipelineConfig()
        engine = ArbiterEngine(config, registry)

        with patch(_ARBITER_PROVIDER, mock_cls):
            review = await engine.review(
                stage=PipelineStage.ARCHITECT,
                stage_model="stage-v1",
                stage_output="output",
                task=_make_task(),
            )

        # Should pick the high-verifier model, not the high-average one
        assert review.arbiter_model == "high-ver-v1"


# ── Feedback Injection ─────────────────────────────────────────────

class TestFormatArbiterFeedback:
    def test_includes_retry_number(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "Retry 1 of 2" in feedback

    def test_includes_arbiter_model(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "model-b-v1" in feedback

    def test_critical_issues_section(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "CRITICAL ISSUES" in feedback
        assert "Missing error handling" in feedback

    def test_warnings_section(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "WARNINGS" in feedback
        assert "API key exposed" in feedback

    def test_alternatives_section(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "ALTERNATIVES TO CONSIDER" in feedback
        assert "OAuth2" in feedback

    def test_no_issues_still_produces_feedback(self):
        review = _make_review(issues=[], alternatives=[])
        feedback = format_arbiter_feedback(review, 2)
        assert "Retry 2 of 2" in feedback
        assert "REJECTED" in feedback

    def test_issue_location_included(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "src/auth.py:42-45" in feedback

    def test_issue_suggestion_included(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        assert "Add try/except around auth calls" in feedback

    def test_critical_before_warnings(self):
        review = _make_review()
        feedback = format_arbiter_feedback(review, 1)
        crit_pos = feedback.index("CRITICAL ISSUES")
        warn_pos = feedback.index("WARNINGS")
        assert crit_pos < warn_pos


# ── ReconciliationEngine ──────────────────────────────────────────

class TestReconciliationEngine:
    async def test_reconcile_returns_review(self):
        mock_cls, _ = _mock_arbiter_provider(
            "All requirements met.\nVERDICT: APPROVE\nCONFIDENCE: 0.95"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ReconciliationEngine(config, registry)

        with patch(_RECONCILER_PROVIDER, mock_cls):
            review = await engine.reconcile(
                task=_make_task(),
                architect_output="scaffold",
                implementation_summary="summary",
                verifier_model="model-a-v1",
            )

        assert review.verdict == Verdict.APPROVE
        assert review.stage_reviewed == PipelineStage.VERIFY

    async def test_cross_model_enforcement(self):
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig()
        engine = ReconciliationEngine(config, registry)

        with patch(_RECONCILER_PROVIDER, mock_cls):
            review = await engine.reconcile(
                task=_make_task(),
                architect_output="scaffold",
                implementation_summary="summary",
                verifier_model="model-a-v1",
            )

        assert review.arbiter_model != "model-a-v1"

    async def test_explicit_reconcile_model_override(self):
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig(reconcile_model="model-b")
        engine = ReconciliationEngine(config, registry)

        with patch(_RECONCILER_PROVIDER, mock_cls):
            review = await engine.reconcile(
                task=_make_task(),
                architect_output="scaffold",
                implementation_summary="summary",
                verifier_model="model-a-v1",
            )

        assert review.arbiter_model == "model-b-v1"

    async def test_reconcile_model_same_as_verifier_raises(self):
        registry = _make_two_model_registry()
        config = PipelineConfig(reconcile_model="model-a")
        engine = ReconciliationEngine(config, registry)

        with pytest.raises(RuntimeError, match="same model"):
            await engine.reconcile(
                task=_make_task(),
                architect_output="scaffold",
                implementation_summary="summary",
                verifier_model="model-a-v1",
            )

    async def test_falls_back_to_global_arbiter_model(self):
        mock_cls, _ = _mock_arbiter_provider(
            "VERDICT: APPROVE\nCONFIDENCE: 0.9"
        )
        registry = _make_two_model_registry()
        config = PipelineConfig(arbiter_model="model-b")
        engine = ReconciliationEngine(config, registry)

        with patch(_RECONCILER_PROVIDER, mock_cls):
            review = await engine.reconcile(
                task=_make_task(),
                architect_output="scaffold",
                implementation_summary="summary",
                verifier_model="model-a-v1",
            )

        assert review.arbiter_model == "model-b-v1"

    async def test_single_model_raises(self):
        registry = {"only": _make_model_config(model="only-model")}
        config = PipelineConfig()
        engine = ReconciliationEngine(config, registry)

        with pytest.raises(RuntimeError, match="No reconciliation model"):
            await engine.reconcile(
                task=_make_task(),
                architect_output="scaffold",
                implementation_summary="summary",
                verifier_model="only-model",
            )

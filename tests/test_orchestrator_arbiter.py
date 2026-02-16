"""Tests for Arbiter integration in the pipeline orchestrator.

Tests arbiter_mode behavior (off, final_only, bookend, full), REJECT retry
loops, HALT stopping, FLAG propagation, and ISR reconciliation flow.
"""

from unittest.mock import AsyncMock, MagicMock, patch

from triad.orchestrator import PipelineOrchestrator
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
_RECONCILER = "triad.orchestrator.ReconciliationEngine.reconcile"


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


def _make_two_model_registry() -> dict[str, ModelConfig]:
    return {
        "model-a": _make_model_config(model="model-a-v1"),
        "model-b": _make_model_config(
            model="model-b-v1",
            fitness=RoleFitness(
                architect=0.7, implementer=0.9,
                refactorer=0.8, verifier=0.75,
            ),
        ),
    }


def _make_task(**overrides) -> TaskSpec:
    defaults = {"task": "Build a REST API"}
    defaults.update(overrides)
    return TaskSpec(**defaults)


def _make_agent_message(
    content: str = "output", confidence: float = 0.85,
    cost: float = 0.01,
) -> AgentMessage:
    return AgentMessage(
        from_agent=PipelineStage.ARCHITECT,
        to_agent=PipelineStage.IMPLEMENT,
        msg_type=MessageType.IMPLEMENTATION,
        content=f"{content}\n\nCONFIDENCE: {confidence}",
        confidence=0.0,
        token_usage=TokenUsage(
            prompt_tokens=100, completion_tokens=50, cost=cost,
        ),
        model="model-a-v1",
    )


def _make_review(
    verdict: Verdict = Verdict.APPROVE, **overrides,
) -> ArbiterReview:
    defaults = {
        "stage_reviewed": PipelineStage.ARCHITECT,
        "reviewed_model": "model-a-v1",
        "arbiter_model": "model-b-v1",
        "verdict": verdict,
        "confidence": 0.9,
        "reasoning": f"VERDICT: {verdict.value.upper()}",
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


def _four_stage_responses():
    return [
        _make_agent_message("architect output", 0.92),
        _make_agent_message("implementer output", 0.88),
        _make_agent_message("refactorer output", 0.90),
        _make_agent_message("verifier output", 0.95),
    ]


# ── Arbiter Mode: OFF ────────────────────────────────────────────

class TestArbiterModeOff:
    async def test_no_arbiter_reviews_when_off(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())

        with patch(_PROVIDER, mock_cls):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="off"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert result.arbiter_reviews == []
        assert result.success is True


# ── Arbiter Mode: BOOKEND ────────────────────────────────────────

class TestArbiterModeBookend:
    async def test_reviews_architect_and_verify(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        reviews = [
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert len(result.arbiter_reviews) == 2
        assert result.arbiter_reviews[0].stage_reviewed == PipelineStage.ARCHITECT
        assert result.arbiter_reviews[1].stage_reviewed == PipelineStage.VERIFY
        assert result.success is True

    async def test_arbiter_cost_included_in_total(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        reviews = [
            _make_review(Verdict.APPROVE, token_cost=0.05),
            _make_review(Verdict.APPROVE, token_cost=0.03),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        # 4 stages * 0.01 = 0.04, plus arbiter 0.05 + 0.03 = 0.08
        expected = 4 * 0.01 + 0.05 + 0.03
        assert abs(result.total_cost - expected) < 1e-10


# ── Arbiter Mode: FINAL_ONLY ────────────────────────────────────

class TestArbiterModeFinalOnly:
    async def test_only_reviews_verify(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        mock_arbiter = AsyncMock(
            return_value=_make_review(
                Verdict.APPROVE, stage_reviewed=PipelineStage.VERIFY,
            ),
        )

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="final_only"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert len(result.arbiter_reviews) == 1
        assert result.arbiter_reviews[0].stage_reviewed == PipelineStage.VERIFY


# ── Arbiter Mode: FULL ──────────────────────────────────────────

class TestArbiterModeFull:
    async def test_reviews_all_four_stages(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        reviews = [
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.IMPLEMENT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.REFACTOR),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="full"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert len(result.arbiter_reviews) == 4
        assert result.success is True


# ── HALT Verdict ─────────────────────────────────────────────────

class TestHaltVerdict:
    async def test_halt_stops_pipeline(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        mock_arbiter = AsyncMock(
            return_value=_make_review(
                Verdict.HALT,
                stage_reviewed=PipelineStage.ARCHITECT,
                reasoning="Critical failure in architect output",
            ),
        )

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert result.halted is True
        assert result.success is False
        assert "Critical failure" in result.halt_reason
        # Only architect stage completed before halt
        assert PipelineStage.ARCHITECT in result.stages
        assert PipelineStage.IMPLEMENT not in result.stages

    async def test_halt_during_retry_stops_pipeline(self):
        """If arbiter issues HALT during a retry, pipeline stops."""
        # 4 stage responses + 1 retry response
        responses = _four_stage_responses() + [
            _make_agent_message("retry output", 0.80),
        ]
        mock_cls, _ = _mock_provider(responses)

        # First review: REJECT, second review (of retry): HALT
        reviews = [
            _make_review(Verdict.REJECT,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.HALT,
                         stage_reviewed=PipelineStage.ARCHITECT,
                         reasoning="Still broken after retry"),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert result.halted is True
        assert result.success is False


# ── REJECT + Retry ───────────────────────────────────────────────

class TestRejectRetry:
    async def test_reject_triggers_retry(self):
        """REJECT on architect triggers re-run, then APPROVE continues."""
        # 4 stage responses + 1 retry response
        responses = _four_stage_responses() + [
            _make_agent_message("improved architect", 0.95),
        ]
        mock_cls, mock_inst = _mock_provider(responses)

        reviews = [
            # First: REJECT architect
            _make_review(Verdict.REJECT,
                         stage_reviewed=PipelineStage.ARCHITECT),
            # Second: APPROVE retry
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            # Third: APPROVE verify (bookend)
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert result.success is True
        # 3 reviews: initial REJECT + retry APPROVE + verify APPROVE
        assert len(result.arbiter_reviews) == 3

    async def test_retry_injects_feedback_into_prompt(self):
        """The retry call should include arbiter feedback in the system prompt."""
        responses = _four_stage_responses() + [
            _make_agent_message("improved architect", 0.95),
        ]
        mock_cls, mock_inst = _mock_provider(responses)

        reject_review = _make_review(
            Verdict.REJECT,
            stage_reviewed=PipelineStage.ARCHITECT,
            reasoning="Missing error handling",
        )
        reviews = [
            reject_review,
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            await orch.run()

        # The retry call is the 5th provider call (4 stages + 1 retry)
        # But the retry replaces the architect stage, so call order is:
        # arch(1), impl(2), ref(3), ver(4), arch-retry(5)
        # Wait — retry happens inline before continuing to next stage.
        # So: arch(1) -> REJECT -> arch-retry(2) -> APPROVE ->
        #     impl(3) -> ref(4) -> ver(5)
        calls = mock_inst.complete.call_args_list
        # The retry call is index 1 (second provider call)
        retry_system = calls[1].kwargs["system"]
        assert "REJECTED" in retry_system
        assert "Retry 1 of 2" in retry_system

    async def test_exhausted_retries_continues(self):
        """After max retries, pipeline continues with last output."""
        # 4 stages + 2 retries
        responses = _four_stage_responses() + [
            _make_agent_message("retry 1", 0.80),
            _make_agent_message("retry 2", 0.82),
        ]
        mock_cls, _ = _mock_provider(responses)

        reviews = [
            _make_review(Verdict.REJECT,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.REJECT,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.REJECT,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ]
        mock_arbiter = AsyncMock(side_effect=reviews)

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="bookend", max_retries=2,
                ),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        # Pipeline should still complete (not halted)
        assert result.success is True


# ── FLAG Propagation ─────────────────────────────────────────────

class TestFlagPropagation:
    async def test_flag_injects_warnings_downstream(self):
        """FLAG on architect should inject flagged_issues into impl prompt."""
        mock_cls, mock_inst = _mock_provider(_four_stage_responses())

        from triad.schemas.arbiter import Issue, IssueCategory, Severity
        flag_review = _make_review(
            Verdict.FLAG,
            stage_reviewed=PipelineStage.ARCHITECT,
            issues=[
                Issue(
                    severity=Severity.WARNING,
                    category=IssueCategory.SECURITY,
                    description="Consider rate limiting on auth endpoint",
                ),
            ],
        )
        approve_review = _make_review(
            Verdict.APPROVE,
            stage_reviewed=PipelineStage.VERIFY,
        )
        mock_arbiter = AsyncMock(side_effect=[flag_review, approve_review])

        with patch(_PROVIDER, mock_cls), patch(_ARBITER_REVIEW, mock_arbiter):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(arbiter_mode="bookend"),
                registry=_make_two_model_registry(),
            )
            await orch.run()

        # The implementer prompt (2nd call) should contain flagged issues
        calls = mock_inst.complete.call_args_list
        impl_system = calls[1].kwargs["system"]
        assert "rate limiting" in impl_system


# ── ISR Reconciliation ──────────────────────────────────────────

class TestISRReconciliation:
    async def test_reconciliation_runs_when_enabled(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        mock_arbiter = AsyncMock(side_effect=[
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ])
        mock_recon = AsyncMock(return_value=_make_review(
            Verdict.APPROVE,
            stage_reviewed=PipelineStage.VERIFY,
            reasoning="All requirements met",
        ))

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, mock_arbiter),
            patch(_RECONCILER, mock_recon),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="bookend",
                    reconciliation_enabled=True,
                ),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        mock_recon.assert_called_once()
        # 2 arbiter reviews + 1 reconciliation review
        assert len(result.arbiter_reviews) == 3
        assert result.success is True

    async def test_reconciliation_not_run_when_disabled(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        mock_arbiter = AsyncMock(side_effect=[
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ])
        mock_recon = AsyncMock()

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, mock_arbiter),
            patch(_RECONCILER, mock_recon),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="bookend",
                    reconciliation_enabled=False,
                ),
                registry=_make_two_model_registry(),
            )
            await orch.run()

        mock_recon.assert_not_called()

    async def test_reconciliation_halt_sets_halted(self):
        mock_cls, _ = _mock_provider(_four_stage_responses())
        mock_arbiter = AsyncMock(side_effect=[
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.ARCHITECT),
            _make_review(Verdict.APPROVE,
                         stage_reviewed=PipelineStage.VERIFY),
        ])
        mock_recon = AsyncMock(return_value=_make_review(
            Verdict.HALT,
            reasoning="Fundamental spec drift detected",
        ))

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, mock_arbiter),
            patch(_RECONCILER, mock_recon),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="bookend",
                    reconciliation_enabled=True,
                ),
                registry=_make_two_model_registry(),
            )
            result = await orch.run()

        assert result.halted is True
        assert result.success is False
        assert "spec drift" in result.halt_reason

    async def test_reconciliation_skipped_on_halt(self):
        """If pipeline halted during stages, reconciliation should not run."""
        mock_cls, _ = _mock_provider(_four_stage_responses())
        mock_arbiter = AsyncMock(
            return_value=_make_review(
                Verdict.HALT,
                stage_reviewed=PipelineStage.ARCHITECT,
            ),
        )
        mock_recon = AsyncMock()

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, mock_arbiter),
            patch(_RECONCILER, mock_recon),
        ):
            orch = PipelineOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    arbiter_mode="bookend",
                    reconciliation_enabled=True,
                ),
                registry=_make_two_model_registry(),
            )
            await orch.run()

        mock_recon.assert_not_called()

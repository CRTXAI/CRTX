"""Tests for CI/CD Integration (Day 13).

Covers: CI schemas (ReviewConfig, ReviewFinding, ModelAssessment, ReviewResult),
ReviewRunner (parallel review, findings extraction, cross-validation, consensus),
formatters (GitHub comments, summary Markdown, exit codes), CLI integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.ci.formatter import format_exit_code, format_github_comments, format_summary
from triad.ci.reviewer import ReviewRunner, _parse_findings
from triad.schemas.ci import ModelAssessment, ReviewConfig, ReviewFinding, ReviewResult
from triad.schemas.messages import AgentMessage, MessageType, PipelineStage, TokenUsage
from triad.schemas.pipeline import ModelConfig, RoleFitness

# ── Helpers ──────────────────────────────────────────────────────


def _make_model(
    provider: str,
    model: str,
    cost_in: float = 3.0,
    cost_out: float = 15.0,
) -> ModelConfig:
    return ModelConfig(
        provider=provider,
        model=model,
        display_name=model,
        api_key_env=f"{provider.upper()}_API_KEY",
        context_window=200_000,
        cost_input=cost_in,
        cost_output=cost_out,
        fitness=RoleFitness(
            architect=0.85,
            implementer=0.75,
            refactorer=0.70,
            verifier=0.70,
        ),
    )


@pytest.fixture()
def registry() -> dict[str, ModelConfig]:
    """Three-model registry for tests."""
    return {
        "claude-sonnet": _make_model("anthropic", "claude-sonnet"),
        "gpt-4o": _make_model("openai", "gpt-4o"),
        "gemini-pro": _make_model("google", "gemini-pro"),
    }


@pytest.fixture()
def single_registry() -> dict[str, ModelConfig]:
    """Single-model registry for edge-case tests."""
    return {
        "claude-sonnet": _make_model("anthropic", "claude-sonnet"),
    }


def _make_finding(
    severity: str = "warning",
    file: str = "src/main.py",
    line: int | None = 42,
    description: str = "Unused import",
    suggestion: str = "Remove the import",
    reported_by: list[str] | None = None,
    confirmed: bool = False,
) -> ReviewFinding:
    return ReviewFinding(
        severity=severity,
        file=file,
        line=line,
        description=description,
        suggestion=suggestion,
        reported_by=reported_by or ["claude-sonnet"],
        confirmed=confirmed,
    )


def _make_assessment(
    model_key: str = "claude-sonnet",
    recommendation: str = "approve",
    findings: list[ReviewFinding] | None = None,
    cost: float = 0.005,
) -> ModelAssessment:
    return ModelAssessment(
        model_key=model_key,
        recommendation=recommendation,
        findings=findings or [],
        cost=cost,
    )


SAMPLE_DIFF = """\
diff --git a/src/main.py b/src/main.py
index abc123..def456 100644
--- a/src/main.py
+++ b/src/main.py
@@ -10,6 +10,8 @@ def process(data):
     result = transform(data)
+    # BUG: no null check
+    result.save()
     return result
"""

MODEL_OUTPUT_TWO_FINDINGS = """\
FINDING:
SEVERITY: critical
FILE: src/main.py
LINE: 12
DESCRIPTION: Missing null check before calling save() — will crash on None data
SUGGESTION: Add `if result is not None:` guard before save()

FINDING:
SEVERITY: suggestion
FILE: src/main.py
LINE: 10
DESCRIPTION: Variable name 'data' is too generic
SUGGESTION: Use a more descriptive name like 'input_payload'

ASSESSMENT: REQUEST_CHANGES
RATIONALE: Critical null safety issue found that will cause runtime errors.
"""

MODEL_OUTPUT_APPROVE = """\
ASSESSMENT: APPROVE
RATIONALE: Code looks clean, no significant issues found.
"""


# ══════════════════════════════════════════════════════════════════
# Schema Validation
# ══════════════════════════════════════════════════════════════════


class TestReviewConfig:
    """ReviewConfig schema tests."""

    def test_defaults(self):
        config = ReviewConfig()
        assert config.models is None
        assert config.focus_areas == []
        assert config.arbiter_enabled is True
        assert config.context_dir is None
        assert config.max_cost is None

    def test_custom_values(self):
        config = ReviewConfig(
            models=["claude-sonnet", "gpt-4o"],
            focus_areas=["security", "performance"],
            arbiter_enabled=False,
            max_cost=1.0,
        )
        assert config.models == ["claude-sonnet", "gpt-4o"]
        assert len(config.focus_areas) == 2
        assert config.arbiter_enabled is False
        assert config.max_cost == 1.0

    def test_max_cost_must_be_non_negative(self):
        with pytest.raises(ValueError):
            ReviewConfig(max_cost=-1.0)


class TestReviewFinding:
    """ReviewFinding schema tests."""

    def test_minimal_finding(self):
        f = ReviewFinding(
            severity="warning",
            file="app.py",
            description="Issue found",
        )
        assert f.severity == "warning"
        assert f.line is None
        assert f.suggestion == ""
        assert f.reported_by == []
        assert f.confirmed is False
        assert f.confidence == "needs_verification"

    def test_full_finding(self):
        f = _make_finding(confirmed=True)
        assert f.severity == "warning"
        assert f.line == 42
        assert f.confirmed is True


class TestModelAssessment:
    """ModelAssessment schema tests."""

    def test_minimal_assessment(self):
        a = ModelAssessment(
            model_key="gpt-4o",
            recommendation="approve",
        )
        assert a.findings == []
        assert a.rationale == ""
        assert a.cost == 0.0

    def test_assessment_with_findings(self):
        findings = [_make_finding(), _make_finding(severity="critical")]
        a = _make_assessment(findings=findings)
        assert len(a.findings) == 2

    def test_cost_must_be_non_negative(self):
        with pytest.raises(ValueError):
            ModelAssessment(model_key="x", recommendation="approve", cost=-0.01)


class TestReviewResult:
    """ReviewResult schema tests."""

    def test_empty_result(self):
        r = ReviewResult(consensus_recommendation="approve")
        assert r.total_findings == 0
        assert r.critical_count == 0
        assert r.models_used == []
        assert r.total_cost == 0.0
        assert r.duration_seconds == 0.0

    def test_full_result(self):
        findings = [_make_finding(severity="critical"), _make_finding()]
        r = ReviewResult(
            consensus_recommendation="request_changes",
            findings=findings,
            model_assessments=[_make_assessment()],
            total_findings=2,
            critical_count=1,
            models_used=["claude-sonnet"],
            total_cost=0.01,
            duration_seconds=5.3,
        )
        assert r.critical_count == 1
        assert r.total_findings == 2


# ══════════════════════════════════════════════════════════════════
# Findings Parsing
# ══════════════════════════════════════════════════════════════════


class TestParseFIndings:
    """Tests for _parse_findings regex parsing."""

    def test_parse_two_findings(self):
        findings = _parse_findings(MODEL_OUTPUT_TWO_FINDINGS, "claude-sonnet")
        assert len(findings) == 2
        assert findings[0].severity == "critical"
        assert findings[0].file == "src/main.py"
        assert findings[0].line == 12
        assert "null check" in findings[0].description.lower()
        assert findings[0].reported_by == ["claude-sonnet"]
        assert findings[1].severity == "suggestion"
        assert findings[1].line == 10

    def test_parse_no_findings(self):
        findings = _parse_findings(MODEL_OUTPUT_APPROVE, "gpt-4o")
        assert findings == []

    def test_parse_unknown_severity_defaults_to_suggestion(self):
        content = """\
FINDING:
SEVERITY: blocker
FILE: foo.py
LINE: 1
DESCRIPTION: Something bad
SUGGESTION: Fix it
"""
        findings = _parse_findings(content, "model-a")
        assert len(findings) == 1
        assert findings[0].severity == "suggestion"

    def test_parse_non_numeric_line_becomes_none(self):
        content = """\
FINDING:
SEVERITY: warning
FILE: bar.py
LINE: N/A
DESCRIPTION: Potential issue
SUGGESTION: Check it
"""
        findings = _parse_findings(content, "model-a")
        assert len(findings) == 1
        assert findings[0].line is None


# ══════════════════════════════════════════════════════════════════
# ReviewRunner
# ══════════════════════════════════════════════════════════════════


def _mock_provider_response(content: str, cost: float = 0.005) -> AgentMessage:
    """Create a mock AgentMessage response."""
    return AgentMessage(
        from_agent=PipelineStage.VERIFY,
        to_agent=PipelineStage.VERIFY,
        msg_type=MessageType.REVIEW,
        model="mock-model",
        content=content,
        code_blocks=[],
        suggestions=[],
        confidence=0.9,
        token_usage=TokenUsage(
            prompt_tokens=100,
            completion_tokens=200,
            cost=cost,
        ),
    )


class TestReviewRunner:
    """ReviewRunner parallel review tests."""

    @pytest.mark.asyncio()
    async def test_empty_diff_returns_clean_approve(self, registry):
        runner = ReviewRunner(registry)
        config = ReviewConfig()
        result = await runner.review("", config)
        assert result.consensus_recommendation == "approve"
        assert result.findings == []
        assert result.total_findings == 0
        assert result.models_used == []

    @pytest.mark.asyncio()
    async def test_whitespace_only_diff_returns_approve(self, registry):
        runner = ReviewRunner(registry)
        result = await runner.review("   \n\n  ", ReviewConfig())
        assert result.consensus_recommendation == "approve"

    @pytest.mark.asyncio()
    async def test_no_valid_models_raises(self):
        runner = ReviewRunner({})
        config = ReviewConfig(models=["nonexistent"])
        with pytest.raises(RuntimeError, match="No valid models"):
            await runner.review(SAMPLE_DIFF, config)

    @pytest.mark.asyncio()
    async def test_parallel_review_all_approve(self, registry):
        mock_response = _mock_provider_response(MODEL_OUTPUT_APPROVE)

        with patch("triad.ci.reviewer.LiteLLMProvider") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.complete = AsyncMock(return_value=mock_response)

            runner = ReviewRunner(registry)
            result = await runner.review(SAMPLE_DIFF, ReviewConfig())

        assert result.consensus_recommendation == "approve"
        assert len(result.models_used) == 3
        assert result.total_findings == 0

    @pytest.mark.asyncio()
    async def test_parallel_review_with_findings(self, registry):
        mock_response = _mock_provider_response(MODEL_OUTPUT_TWO_FINDINGS, cost=0.01)

        with patch("triad.ci.reviewer.LiteLLMProvider") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.complete = AsyncMock(return_value=mock_response)

            runner = ReviewRunner(registry)
            result = await runner.review(SAMPLE_DIFF, ReviewConfig())

        assert result.consensus_recommendation == "request_changes"
        assert result.total_cost == pytest.approx(0.03, abs=0.001)
        assert len(result.models_used) == 3

    @pytest.mark.asyncio()
    async def test_single_model_review(self, single_registry):
        mock_response = _mock_provider_response(MODEL_OUTPUT_TWO_FINDINGS)

        with patch("triad.ci.reviewer.LiteLLMProvider") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.complete = AsyncMock(return_value=mock_response)

            runner = ReviewRunner(single_registry)
            result = await runner.review(SAMPLE_DIFF, ReviewConfig())

        assert len(result.models_used) == 1
        assert result.total_findings == 2

    @pytest.mark.asyncio()
    async def test_model_failure_is_handled_gracefully(self, registry):
        """If one model fails, others still produce results."""
        call_count = 0

        async def side_effect(*args, **kwargs):
            nonlocal call_count
            call_count += 1
            if call_count == 1:
                raise RuntimeError("Model unavailable")
            return _mock_provider_response(MODEL_OUTPUT_APPROVE)

        with patch("triad.ci.reviewer.LiteLLMProvider") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.complete = AsyncMock(side_effect=side_effect)

            runner = ReviewRunner(registry)
            result = await runner.review(SAMPLE_DIFF, ReviewConfig())

        assert len(result.models_used) == 2
        assert result.consensus_recommendation == "approve"

    @pytest.mark.asyncio()
    async def test_focus_areas_passed_to_prompt(self, single_registry):
        mock_response = _mock_provider_response(MODEL_OUTPUT_APPROVE)

        with patch("triad.ci.reviewer.LiteLLMProvider") as mock_cls, \
             patch("triad.ci.reviewer.render_prompt") as mock_render:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.complete = AsyncMock(return_value=mock_response)
            mock_render.return_value = "system prompt"

            runner = ReviewRunner(single_registry)
            config = ReviewConfig(focus_areas=["security", "performance"])
            await runner.review(SAMPLE_DIFF, config)

        mock_render.assert_called_once()
        call_kwargs = mock_render.call_args
        assert "focus_areas" in call_kwargs.kwargs or (
            len(call_kwargs.args) > 1 and "security" in str(call_kwargs)
        )

    @pytest.mark.asyncio()
    async def test_config_selects_specific_models(self, registry):
        mock_response = _mock_provider_response(MODEL_OUTPUT_APPROVE)

        with patch("triad.ci.reviewer.LiteLLMProvider") as mock_cls:
            mock_inst = MagicMock()
            mock_cls.return_value = mock_inst
            mock_inst.complete = AsyncMock(return_value=mock_response)

            runner = ReviewRunner(registry)
            config = ReviewConfig(models=["claude-sonnet", "gpt-4o"])
            result = await runner.review(SAMPLE_DIFF, config)

        assert len(result.models_used) == 2
        assert set(result.models_used) == {"claude-sonnet", "gpt-4o"}


# ══════════════════════════════════════════════════════════════════
# Cross-Validation
# ══════════════════════════════════════════════════════════════════


class TestCrossValidation:
    """Tests for cross-validation / agreement mapping."""

    def test_arbiter_disabled_returns_all_findings(self, registry):
        runner = ReviewRunner(registry)
        f1 = _make_finding(reported_by=["claude-sonnet"])
        f2 = _make_finding(severity="critical", reported_by=["gpt-4o"])
        assessments = [
            _make_assessment(model_key="claude-sonnet", findings=[f1]),
            _make_assessment(model_key="gpt-4o", findings=[f2]),
        ]
        result = runner._cross_validate(assessments, arbiter_enabled=False)
        assert len(result) == 2
        assert not any(f.confirmed for f in result)

    def test_single_assessment_no_cross_validation(self, registry):
        runner = ReviewRunner(registry)
        f1 = _make_finding()
        assessments = [_make_assessment(findings=[f1])]
        result = runner._cross_validate(assessments, arbiter_enabled=True)
        assert len(result) == 1
        assert not result[0].confirmed

    def test_matching_findings_get_confirmed(self, registry):
        runner = ReviewRunner(registry)
        # Same file, severity, and identical description[:50] from two models
        f1 = _make_finding(
            file="src/main.py", severity="critical",
            description="Missing null check before save — will crash on None",
            reported_by=["claude-sonnet"],
        )
        f2 = _make_finding(
            file="src/main.py", severity="critical",
            description="Missing null check before save — will crash on None",
            reported_by=["gpt-4o"],
        )
        assessments = [
            _make_assessment(model_key="claude-sonnet", findings=[f1]),
            _make_assessment(model_key="gpt-4o", findings=[f2]),
        ]
        result = runner._cross_validate(assessments, arbiter_enabled=True)
        confirmed = [f for f in result if f.confirmed]
        assert len(confirmed) >= 1
        assert confirmed[0].confidence == "high"
        assert set(confirmed[0].reported_by) == {"claude-sonnet", "gpt-4o"}

    def test_different_findings_stay_unconfirmed(self, registry):
        runner = ReviewRunner(registry)
        f1 = _make_finding(
            file="src/a.py", severity="warning",
            description="Unused import os",
            reported_by=["claude-sonnet"],
        )
        f2 = _make_finding(
            file="src/b.py", severity="critical",
            description="SQL injection vulnerability",
            reported_by=["gpt-4o"],
        )
        assessments = [
            _make_assessment(model_key="claude-sonnet", findings=[f1]),
            _make_assessment(model_key="gpt-4o", findings=[f2]),
        ]
        result = runner._cross_validate(assessments, arbiter_enabled=True)
        assert len(result) == 2
        assert not any(f.confirmed for f in result)


# ══════════════════════════════════════════════════════════════════
# Consensus
# ══════════════════════════════════════════════════════════════════


class TestConsensus:
    """Tests for consensus recommendation logic."""

    def test_empty_assessments_approve(self, registry):
        runner = ReviewRunner(registry)
        result = runner._compute_consensus([], [])
        assert result == "approve"

    def test_all_approve_consensus(self, registry):
        runner = ReviewRunner(registry)
        assessments = [
            _make_assessment(model_key="claude-sonnet", recommendation="approve"),
            _make_assessment(model_key="gpt-4o", recommendation="approve"),
            _make_assessment(model_key="gemini-pro", recommendation="approve"),
        ]
        result = runner._compute_consensus(assessments, [])
        assert result == "approve"

    def test_majority_request_changes(self, registry):
        runner = ReviewRunner(registry)
        assessments = [
            _make_assessment(model_key="claude-sonnet", recommendation="request_changes"),
            _make_assessment(model_key="gpt-4o", recommendation="request_changes"),
            _make_assessment(model_key="gemini-pro", recommendation="approve"),
        ]
        result = runner._compute_consensus(assessments, [])
        assert result == "request_changes"

    def test_confirmed_critical_forces_request_changes(self, registry):
        """Even if all models approve, a confirmed critical finding overrides."""
        runner = ReviewRunner(registry)
        assessments = [
            _make_assessment(model_key="claude-sonnet", recommendation="approve"),
            _make_assessment(model_key="gpt-4o", recommendation="approve"),
        ]
        findings = [
            _make_finding(severity="critical", confirmed=True),
        ]
        result = runner._compute_consensus(assessments, findings)
        assert result == "request_changes"

    def test_unconfirmed_critical_does_not_override(self, registry):
        runner = ReviewRunner(registry)
        assessments = [
            _make_assessment(model_key="claude-sonnet", recommendation="approve"),
            _make_assessment(model_key="gpt-4o", recommendation="approve"),
        ]
        findings = [
            _make_finding(severity="critical", confirmed=False),
        ]
        result = runner._compute_consensus(assessments, findings)
        assert result == "approve"

    def test_minority_request_changes_still_approves(self, registry):
        runner = ReviewRunner(registry)
        assessments = [
            _make_assessment(model_key="claude-sonnet", recommendation="request_changes"),
            _make_assessment(model_key="gpt-4o", recommendation="approve"),
            _make_assessment(model_key="gemini-pro", recommendation="approve"),
        ]
        result = runner._compute_consensus(assessments, [])
        assert result == "approve"


# ══════════════════════════════════════════════════════════════════
# Formatter — GitHub Comments
# ══════════════════════════════════════════════════════════════════


class TestFormatGitHubComments:
    """Tests for format_github_comments."""

    def test_no_findings_empty_list(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[],
        )
        comments = format_github_comments(result)
        assert comments == []

    def test_finding_with_line(self):
        result = ReviewResult(
            consensus_recommendation="request_changes",
            findings=[_make_finding(line=42)],
        )
        comments = format_github_comments(result)
        assert len(comments) == 1
        assert comments[0]["path"] == "src/main.py"
        assert comments[0]["line"] == 42
        assert comments[0]["side"] == "RIGHT"
        assert "WARNING" in comments[0]["body"]

    def test_finding_without_line(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(line=None)],
        )
        comments = format_github_comments(result)
        assert len(comments) == 1
        assert "line" not in comments[0]

    def test_suggestion_included_in_body(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(suggestion="Use pathlib instead")],
        )
        comments = format_github_comments(result)
        assert "pathlib" in comments[0]["body"]

    def test_confirmed_finding_shows_checkmark(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(confirmed=True)],
        )
        comments = format_github_comments(result)
        assert "Confirmed" in comments[0]["body"]

    def test_reporter_names_in_body(self):
        f = _make_finding(reported_by=["claude-sonnet", "gpt-4o"])
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[f],
        )
        comments = format_github_comments(result)
        assert "claude-sonnet" in comments[0]["body"]
        assert "gpt-4o" in comments[0]["body"]


# ══════════════════════════════════════════════════════════════════
# Formatter — Summary Markdown
# ══════════════════════════════════════════════════════════════════


class TestFormatSummary:
    """Tests for format_summary Markdown output."""

    def test_approve_banner(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            total_findings=0,
            critical_count=0,
            models_used=["claude-sonnet"],
            total_cost=0.005,
            duration_seconds=2.1,
        )
        summary = format_summary(result)
        assert "Approved" in summary
        assert "0 findings" in summary

    def test_request_changes_banner(self):
        result = ReviewResult(
            consensus_recommendation="request_changes",
            findings=[_make_finding(severity="critical")],
            total_findings=1,
            critical_count=1,
            models_used=["claude-sonnet", "gpt-4o"],
            total_cost=0.02,
            duration_seconds=5.0,
        )
        summary = format_summary(result)
        assert "Changes Requested" in summary
        assert "1 critical" in summary

    def test_findings_section_sorted_by_severity(self):
        findings = [
            _make_finding(severity="suggestion", description="Minor style"),
            _make_finding(severity="critical", description="Null pointer"),
            _make_finding(severity="warning", description="Unused var"),
        ]
        result = ReviewResult(
            consensus_recommendation="request_changes",
            findings=findings,
            total_findings=3,
            critical_count=1,
            models_used=["claude-sonnet"],
            total_cost=0.01,
            duration_seconds=3.0,
        )
        summary = format_summary(result)
        # Critical should appear before warning, warning before suggestion
        crit_pos = summary.index("CRITICAL")
        warn_pos = summary.index("WARNING")
        sugg_pos = summary.index("SUGGESTION")
        assert crit_pos < warn_pos < sugg_pos

    def test_model_assessments_table(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            model_assessments=[
                _make_assessment(model_key="claude-sonnet", recommendation="approve"),
                _make_assessment(model_key="gpt-4o", recommendation="request_changes"),
            ],
            models_used=["claude-sonnet", "gpt-4o"],
            total_cost=0.01,
            duration_seconds=4.0,
        )
        summary = format_summary(result)
        assert "Model Assessments" in summary
        assert "claude-sonnet" in summary
        assert "gpt-4o" in summary

    def test_footer_contains_crtx_link(self):
        result = ReviewResult(consensus_recommendation="approve")
        summary = format_summary(result)
        assert "CRTX" in summary

    def test_cost_formatting(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            models_used=["claude-sonnet"],
            total_cost=0.1234,
            duration_seconds=1.0,
        )
        summary = format_summary(result)
        assert "$0.1234" in summary


# ══════════════════════════════════════════════════════════════════
# Formatter — Exit Codes
# ══════════════════════════════════════════════════════════════════


class TestFormatExitCode:
    """Tests for format_exit_code."""

    def test_no_assessments_returns_error(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            model_assessments=[],
        )
        assert format_exit_code(result) == 2

    def test_approve_no_findings_returns_zero(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            model_assessments=[_make_assessment()],
        )
        assert format_exit_code(result) == 0

    def test_fail_on_critical_with_critical_finding(self):
        result = ReviewResult(
            consensus_recommendation="request_changes",
            findings=[_make_finding(severity="critical")],
            model_assessments=[_make_assessment()],
            critical_count=1,
        )
        assert format_exit_code(result, fail_on="critical") == 1

    def test_fail_on_critical_with_only_warnings(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(severity="warning")],
            model_assessments=[_make_assessment()],
            total_findings=1,
            critical_count=0,
        )
        assert format_exit_code(result, fail_on="critical") == 0

    def test_fail_on_warning_with_warning(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(severity="warning")],
            model_assessments=[_make_assessment()],
            total_findings=1,
        )
        assert format_exit_code(result, fail_on="warning") == 1

    def test_fail_on_warning_with_critical(self):
        result = ReviewResult(
            consensus_recommendation="request_changes",
            findings=[_make_finding(severity="critical")],
            model_assessments=[_make_assessment()],
            critical_count=1,
        )
        assert format_exit_code(result, fail_on="warning") == 1

    def test_fail_on_warning_with_only_suggestions(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(severity="suggestion")],
            model_assessments=[_make_assessment()],
            total_findings=1,
        )
        assert format_exit_code(result, fail_on="warning") == 0

    def test_fail_on_any_with_suggestion(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            findings=[_make_finding(severity="suggestion")],
            model_assessments=[_make_assessment()],
            total_findings=1,
        )
        assert format_exit_code(result, fail_on="any") == 1

    def test_fail_on_any_with_no_findings(self):
        result = ReviewResult(
            consensus_recommendation="approve",
            model_assessments=[_make_assessment()],
            total_findings=0,
        )
        assert format_exit_code(result, fail_on="any") == 0


# ══════════════════════════════════════════════════════════════════
# CLI Integration
# ══════════════════════════════════════════════════════════════════


class TestCLIReview:
    """CLI review command tests."""

    def test_review_help(self):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["review", "--help"])
        assert result.exit_code == 0
        assert "diff" in result.output.lower()

    def test_review_no_input_shows_error(self):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["review"])
        assert result.exit_code == 1

    def test_review_missing_diff_file(self, tmp_path):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["review", "--diff", str(tmp_path / "nope.txt")])
        assert result.exit_code == 1
        assert "not found" in result.output.lower()

    def test_review_empty_diff_file(self, tmp_path):
        from typer.testing import CliRunner

        from triad.cli import app

        diff_file = tmp_path / "empty.diff"
        diff_file.write_text("")
        runner = CliRunner()
        result = runner.invoke(app, ["review", "--diff", str(diff_file)])
        assert result.exit_code == 0
        assert "empty" in result.output.lower()

    def test_review_diff_file_invokes_runner(self, tmp_path):
        from typer.testing import CliRunner

        from triad.cli import app

        diff_file = tmp_path / "test.diff"
        diff_file.write_text(SAMPLE_DIFF)

        mock_result = ReviewResult(
            consensus_recommendation="approve",
            findings=[],
            model_assessments=[_make_assessment()],
            total_findings=0,
            critical_count=0,
            models_used=["claude-sonnet"],
            total_cost=0.005,
            duration_seconds=1.0,
        )

        with patch("triad.cli.load_models") as mock_load, \
             patch("triad.ci.reviewer.ReviewRunner.review", new_callable=AsyncMock) as mock_review:
            mock_load.return_value = {
                "claude-sonnet": _make_model("anthropic", "claude-sonnet"),
            }
            mock_review.return_value = mock_result

            runner = CliRunner()
            result = runner.invoke(app, ["review", "--diff", str(diff_file)])

        assert result.exit_code == 0
        mock_review.assert_called_once()

    def test_review_json_format(self, tmp_path):
        from typer.testing import CliRunner

        from triad.cli import app

        diff_file = tmp_path / "test.diff"
        diff_file.write_text(SAMPLE_DIFF)

        mock_result = ReviewResult(
            consensus_recommendation="approve",
            findings=[],
            model_assessments=[_make_assessment()],
            total_findings=0,
            models_used=["claude-sonnet"],
            total_cost=0.005,
            duration_seconds=1.0,
        )

        with patch("triad.cli.load_models") as mock_load, \
             patch("triad.ci.reviewer.ReviewRunner.review", new_callable=AsyncMock) as mock_review:
            mock_load.return_value = {
                "claude-sonnet": _make_model("anthropic", "claude-sonnet"),
            }
            mock_review.return_value = mock_result

            runner = CliRunner()
            result = runner.invoke(app, [
                "review", "--diff", str(diff_file), "--format", "json",
            ])

        assert result.exit_code == 0
        assert "consensus_recommendation" in result.output

    def test_review_exit_code_on_critical(self, tmp_path):
        from typer.testing import CliRunner

        from triad.cli import app

        diff_file = tmp_path / "test.diff"
        diff_file.write_text(SAMPLE_DIFF)

        mock_result = ReviewResult(
            consensus_recommendation="request_changes",
            findings=[_make_finding(severity="critical")],
            model_assessments=[_make_assessment(recommendation="request_changes")],
            total_findings=1,
            critical_count=1,
            models_used=["claude-sonnet"],
            total_cost=0.01,
            duration_seconds=2.0,
        )

        with patch("triad.cli.load_models") as mock_load, \
             patch("triad.ci.reviewer.ReviewRunner.review", new_callable=AsyncMock) as mock_review:
            mock_load.return_value = {
                "claude-sonnet": _make_model("anthropic", "claude-sonnet"),
            }
            mock_review.return_value = mock_result

            runner = CliRunner()
            result = runner.invoke(app, [
                "review", "--diff", str(diff_file), "--fail-on", "critical",
            ])

        assert result.exit_code == 1


# ══════════════════════════════════════════════════════════════════
# Module Re-exports
# ══════════════════════════════════════════════════════════════════


class TestModuleExports:
    """Verify CI module re-exports."""

    def test_ci_module_exports(self):
        from triad.ci import (
            ReviewRunner,
            format_exit_code,
            format_github_comments,
            format_summary,
        )
        assert ReviewRunner is not None
        assert format_exit_code is not None
        assert format_github_comments is not None
        assert format_summary is not None

    def test_schemas_init_exports_ci_types(self):
        from triad.schemas import (
            ModelAssessment,
            ReviewConfig,
            ReviewFinding,
            ReviewResult,
        )
        assert ReviewConfig is not None
        assert ReviewFinding is not None
        assert ModelAssessment is not None
        assert ReviewResult is not None

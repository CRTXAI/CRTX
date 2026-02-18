"""Tests for the crtx demo module.

Covers model selection logic and CLI integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from triad.cli import app
from triad.demo import select_demo_models
from triad.schemas.arbiter import (
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
from triad.schemas.pipeline import ModelConfig, RoleFitness

runner = CliRunner()


# ── Factories ──────────────────────────────────────────────────────


def _cfg(
    provider: str,
    model: str = "",
    display_name: str = "",
    api_key_env: str = "",
    cost_input: float = 1.0,
    cost_output: float = 2.0,
) -> ModelConfig:
    """Build a minimal ModelConfig for testing."""
    return ModelConfig(
        provider=provider,
        model=model or f"{provider}/test-model",
        display_name=display_name or f"Test {provider}",
        api_key_env=api_key_env or f"{provider.upper()}_API_KEY",
        context_window=128000,
        cost_input=cost_input,
        cost_output=cost_output,
        fitness=RoleFitness(
            architect=0.5, implementer=0.5, refactorer=0.5, verifier=0.5,
        ),
    )


def _registry_two_providers() -> dict[str, ModelConfig]:
    """Registry with models from two providers (openai + anthropic)."""
    return {
        "gpt-4o-mini": _cfg(
            "openai", "gpt-4o-mini", "GPT-4o Mini",
            "OPENAI_API_KEY", 0.15, 0.60,
        ),
        "claude-sonnet": _cfg(
            "anthropic", "anthropic/claude-sonnet-4-5", "Claude Sonnet 4.5",
            "ANTHROPIC_API_KEY", 3.0, 15.0,
        ),
        "claude-haiku": _cfg(
            "anthropic", "anthropic/claude-haiku-4-5", "Claude Haiku 4.5",
            "ANTHROPIC_API_KEY", 0.80, 4.0,
        ),
    }


def _registry_single_provider() -> dict[str, ModelConfig]:
    """Registry with models from only one provider."""
    return {
        "claude-sonnet": _cfg(
            "anthropic", "anthropic/claude-sonnet", "Sonnet",
            "ANTHROPIC_API_KEY",
        ),
        "claude-haiku": _cfg(
            "anthropic", "anthropic/claude-haiku", "Haiku",
            "ANTHROPIC_API_KEY",
        ),
    }


def _make_agent_message(content: str = "def validate_email():\n    pass") -> AgentMessage:
    """Build a mock AgentMessage returned by LiteLLMProvider.complete()."""
    return AgentMessage(
        from_agent=PipelineStage.IMPLEMENT,
        to_agent=PipelineStage.VERIFY,
        msg_type=MessageType.IMPLEMENTATION,
        content=content,
        confidence=0.9,
        token_usage=TokenUsage(
            prompt_tokens=500, completion_tokens=800, cost=0.03,
        ),
    )


def _make_review(verdict: Verdict = Verdict.FLAG) -> ArbiterReview:
    """Build a mock ArbiterReview."""
    issues = []
    if verdict != Verdict.APPROVE:
        issues = [
            Issue(
                severity=Severity.CRITICAL,
                category=IssueCategory.EDGE_CASE,
                description="Regex doesn't handle Unicode domains",
                suggestion="Use IDNA-aware validation",
            ),
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.LOGIC,
                description="Plus-addressing test is wrong",
            ),
        ]
    return ArbiterReview(
        stage_reviewed=PipelineStage.IMPLEMENT,
        reviewed_model="openai/gpt-4o-mini",
        arbiter_model="anthropic/claude-sonnet-4-5",
        verdict=verdict,
        issues=issues,
        confidence=0.87,
        reasoning="The email validator has several edge-case issues.",
        token_cost=0.04,
    )


# ── select_demo_models tests ──────────────────────────────────────


class TestSelectDemoModels:

    def test_cross_provider(self, monkeypatch):
        """Two providers available -> picks gen and arb from different ones."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        registry = _registry_two_providers()
        (gen_key, gen_cfg), (arb_key, arb_cfg) = select_demo_models(registry)

        assert gen_cfg.provider != arb_cfg.provider

    def test_single_provider_raises(self, monkeypatch):
        """Only one provider reachable -> RuntimeError."""
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        registry = _registry_single_provider()
        with pytest.raises(RuntimeError, match="at least 2 providers"):
            select_demo_models(registry)

    def test_no_reachable_raises(self, monkeypatch):
        """No API keys set -> RuntimeError."""
        # Ensure no relevant keys are set
        monkeypatch.delenv("OPENAI_API_KEY", raising=False)
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

        registry = _registry_two_providers()
        with pytest.raises(RuntimeError, match="at least 2 providers"):
            select_demo_models(registry)

    def test_preference_order(self, monkeypatch):
        """Prefers gpt-4o-mini as gen + claude-sonnet as arb when both available."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        registry = _registry_two_providers()
        (gen_key, _gen_cfg), (arb_key, _arb_cfg) = select_demo_models(registry)

        assert gen_key == "gpt-4o-mini"
        assert arb_key == "claude-sonnet"

    def test_fallback_to_cheapest(self, monkeypatch):
        """Non-preferred keys still select cheapest per provider."""
        monkeypatch.setenv("PROVIDER_A_KEY", "key-a")
        monkeypatch.setenv("PROVIDER_B_KEY", "key-b")

        registry = {
            "model-a-expensive": _cfg(
                "providerA", "providerA/expensive", "Expensive A",
                "PROVIDER_A_KEY", 10.0, 20.0,
            ),
            "model-a-cheap": _cfg(
                "providerA", "providerA/cheap", "Cheap A",
                "PROVIDER_A_KEY", 0.1, 0.2,
            ),
            "model-b-only": _cfg(
                "providerB", "providerB/only", "Only B",
                "PROVIDER_B_KEY", 5.0, 10.0,
            ),
        }

        (gen_key, gen_cfg), (arb_key, arb_cfg) = select_demo_models(registry)

        # Generator should be the cheapest overall -> model-a-cheap
        assert gen_key == "model-a-cheap"
        # Arbiter must be from a different provider
        assert arb_cfg.provider != gen_cfg.provider

    def test_skips_unreachable(self, monkeypatch):
        """Models without env var set are excluded."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)
        monkeypatch.setenv("GOOGLE_API_KEY", "goog-test")

        registry = _registry_two_providers()
        # Add a google model so we have 2 reachable providers
        registry["gemini-flash"] = _cfg(
            "google", "google/gemini-flash", "Gemini Flash",
            "GOOGLE_API_KEY", 0.075, 0.30,
        )

        (gen_key, gen_cfg), (arb_key, arb_cfg) = select_demo_models(registry)

        # Anthropic models should be excluded (no key)
        assert gen_cfg.provider != "anthropic"
        assert arb_cfg.provider != "anthropic"
        # Should still pick from different providers
        assert gen_cfg.provider != arb_cfg.provider


# ── CLI integration tests ─────────────────────────────────────────


class TestDemoCLI:

    def test_demo_command_help(self):
        """crtx demo --help shows the command description."""
        result = runner.invoke(app, ["demo", "--help"])
        assert result.exit_code == 0
        assert "demo" in result.output.lower()
        assert "--yes" in result.output

    def test_run_no_task_offers_demo(self):
        """crtx run with empty task prompts for demo, declining exits cleanly."""
        with (
            patch("triad.cli_display.is_interactive", return_value=True),
            patch("typer.confirm", return_value=False),
            patch("triad.keys.has_any_key", return_value=True),
        ):
            result = runner.invoke(app, ["run"])
            # Should exit 0 after declining
            assert result.exit_code == 0
            assert "demo" in result.output.lower() or "Tip" in result.output

    def test_demo_skip_confirm(self, monkeypatch):
        """--yes flag skips the Continue? prompt."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_msg = _make_agent_message()
        mock_review = _make_review(Verdict.FLAG)

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_msg)

        mock_engine = MagicMock()
        mock_engine.review = AsyncMock(return_value=mock_review)

        with (
            patch("triad.keys.load_keys_env"),
            patch("triad.providers.registry.load_models", return_value=_registry_two_providers()),
            patch("triad.providers.litellm_provider.LiteLLMProvider", return_value=mock_provider),
            patch("triad.arbiter.arbiter.ArbiterEngine", return_value=mock_engine),
            patch("triad.output.writer.write_pipeline_output", return_value="crtx-output/test"),
        ):
            result = runner.invoke(app, ["demo", "--yes"])
            # typer.confirm should NOT have been called
            assert result.exit_code == 0
            assert "Demo Complete" in result.output

    def test_demo_reject_path(self, monkeypatch):
        """Mock arbiter returning FLAG shows issues display."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_msg = _make_agent_message(
            "def validate_email(email):\n"
            "    import re\n"
            "    return bool(re.match(r'^[a-zA-Z0-9]+@[a-zA-Z0-9]+\\.[a-z]+$', email))\n"
            "\ndef test_basic():\n    assert validate_email('a@b.com')\n"
        )
        mock_review = _make_review(Verdict.FLAG)

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_msg)

        mock_engine = MagicMock()
        mock_engine.review = AsyncMock(return_value=mock_review)

        with (
            patch("triad.keys.load_keys_env"),
            patch("triad.providers.registry.load_models", return_value=_registry_two_providers()),
            patch("triad.providers.litellm_provider.LiteLLMProvider", return_value=mock_provider),
            patch("triad.arbiter.arbiter.ArbiterEngine", return_value=mock_engine),
            patch("triad.output.writer.write_pipeline_output", return_value="crtx-output/test"),
        ):
            result = runner.invoke(app, ["demo", "--yes"])
            assert result.exit_code == 0
            assert "FOUND ISSUES" in result.output
            assert "Unicode" in result.output

    def test_demo_approved_path(self, monkeypatch):
        """Mock arbiter returning APPROVE shows approval display."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_msg = _make_agent_message()
        mock_review = _make_review(Verdict.APPROVE)

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_msg)

        mock_engine = MagicMock()
        mock_engine.review = AsyncMock(return_value=mock_review)

        with (
            patch("triad.keys.load_keys_env"),
            patch("triad.providers.registry.load_models", return_value=_registry_two_providers()),
            patch("triad.providers.litellm_provider.LiteLLMProvider", return_value=mock_provider),
            patch("triad.arbiter.arbiter.ArbiterEngine", return_value=mock_engine),
            patch("triad.output.writer.write_pipeline_output", return_value="crtx-output/test"),
        ):
            result = runner.invoke(app, ["demo", "--yes"])
            assert result.exit_code == 0
            assert "APPROVED" in result.output

    def test_demo_output_written(self, monkeypatch, tmp_path):
        """Verify output files are written via write_pipeline_output."""
        monkeypatch.setenv("OPENAI_API_KEY", "sk-test")
        monkeypatch.setenv("ANTHROPIC_API_KEY", "sk-ant-test")

        mock_msg = _make_agent_message()
        mock_review = _make_review(Verdict.FLAG)
        output_dir = str(tmp_path / "demo-output")

        mock_provider = MagicMock()
        mock_provider.complete = AsyncMock(return_value=mock_msg)

        mock_engine = MagicMock()
        mock_engine.review = AsyncMock(return_value=mock_review)

        mock_write = MagicMock(return_value=output_dir)

        with (
            patch("triad.keys.load_keys_env"),
            patch("triad.providers.registry.load_models", return_value=_registry_two_providers()),
            patch("triad.providers.litellm_provider.LiteLLMProvider", return_value=mock_provider),
            patch("triad.arbiter.arbiter.ArbiterEngine", return_value=mock_engine),
            patch("triad.output.writer.write_pipeline_output", mock_write),
        ):
            result = runner.invoke(app, ["demo", "--yes"])
            assert result.exit_code == 0

        # write_pipeline_output was called with a PipelineResult
        mock_write.assert_called_once()
        call_args = mock_write.call_args
        pipeline_result = call_args[0][0]
        assert pipeline_result.success is True
        assert pipeline_result.total_cost > 0
        assert len(pipeline_result.arbiter_reviews) == 1
        assert PipelineStage.IMPLEMENT in pipeline_result.stages

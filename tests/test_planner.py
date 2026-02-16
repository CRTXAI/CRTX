"""Tests for the Task Planner (Day 12).

Covers: PlannerResult schema, TaskPlanner class (quick + interactive modes),
model selection, tech stack extraction, CLI integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.planner import TaskPlanner, _extract_tech_stack, _parse_questions
from triad.schemas.messages import TokenUsage
from triad.schemas.pipeline import ModelConfig, PipelineConfig, RoleFitness, TaskSpec
from triad.schemas.planner import PlannerResult

# ── Test Registry ─────────────────────────────────────────────────


def _make_model(
    provider: str,
    model: str,
    arch_fitness: float = 0.80,
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
            architect=arch_fitness,
            implementer=0.75,
            refactorer=0.70,
            verifier=0.70,
        ),
    )


@pytest.fixture()
def registry() -> dict[str, ModelConfig]:
    """Model registry with varied costs and fitness."""
    return {
        "claude-opus": _make_model(
            "anthropic", "claude-opus", arch_fitness=0.95, cost_in=15.0, cost_out=75.0,
        ),
        "claude-sonnet": _make_model(
            "anthropic", "claude-sonnet", arch_fitness=0.88, cost_in=3.0, cost_out=15.0,
        ),
        "gemini-flash": _make_model(
            "google", "gemini-flash", arch_fitness=0.75, cost_in=0.15, cost_out=0.60,
        ),
        "gpt-4o": _make_model(
            "openai", "gpt-4o", arch_fitness=0.85, cost_in=2.50, cost_out=10.0,
        ),
        "cheap-low": _make_model(
            "xai", "cheap-low", arch_fitness=0.60, cost_in=0.10, cost_out=0.30,
        ),
    }


def _mock_llm_response(
    content: str,
    prompt_tokens: int = 500,
    completion_tokens: int = 1000,
    cost: float = 0.005,
) -> MagicMock:
    """Create a mock AgentMessage-like response."""
    msg = MagicMock()
    msg.content = content
    msg.token_usage = TokenUsage(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        cost=cost,
    )
    return msg


# ── Schema Tests ──────────────────────────────────────────────────


class TestPlannerResultSchema:
    """Test PlannerResult schema validation."""

    def test_all_fields(self):
        result = PlannerResult(
            original_description="build an API",
            expanded_spec="### Requirements\n- REST API\n\nTECH_STACK: Python, FastAPI",
            task_spec=TaskSpec(task="build an API"),
            tech_stack_inferred=["Python", "FastAPI"],
            model_used="claude-sonnet",
            token_usage=TokenUsage(
                prompt_tokens=500,
                completion_tokens=1000,
                cost=0.005,
            ),
            cost=0.005,
            interactive=False,
        )
        assert result.original_description == "build an API"
        assert result.cost == 0.005
        assert len(result.tech_stack_inferred) == 2

    def test_interactive_fields(self):
        result = PlannerResult(
            original_description="build an API",
            expanded_spec="spec",
            task_spec=TaskSpec(task="spec"),
            model_used="claude-sonnet",
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, cost=0.0),
            cost=0.0,
            interactive=True,
            clarifying_questions=["1. What scale?", "2. Auth method?"],
            user_answers="Q1: Small\nQ2: JWT",
        )
        assert result.interactive is True
        assert len(result.clarifying_questions) == 2
        assert result.user_answers is not None

    def test_optional_fields_default_none(self):
        result = PlannerResult(
            original_description="test",
            expanded_spec="spec",
            task_spec=TaskSpec(task="test"),
            model_used="m",
            token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, cost=0.0),
            cost=0.0,
        )
        assert result.clarifying_questions is None
        assert result.user_answers is None
        assert result.interactive is False

    def test_serialization_roundtrip(self):
        result = PlannerResult(
            original_description="build X",
            expanded_spec="full spec",
            task_spec=TaskSpec(task="build X"),
            tech_stack_inferred=["Python"],
            model_used="test-model",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=200, cost=0.01),
            cost=0.01,
        )
        data = result.model_dump()
        restored = PlannerResult(**data)
        assert restored.original_description == "build X"
        assert restored.cost == 0.01

    def test_cost_validation(self):
        with pytest.raises(ValueError):
            PlannerResult(
                original_description="test",
                expanded_spec="spec",
                task_spec=TaskSpec(task="test"),
                model_used="m",
                token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, cost=0.0),
                cost=-1.0,
            )


# ── Model Selection Tests ─────────────────────────────────────────


class TestModelSelection:
    """Test the planner's model selection logic."""

    def test_selects_cheapest_above_threshold(self, registry):
        planner = TaskPlanner(registry)
        selected = planner.select_model()
        # gemini-flash is cheapest above 0.70 (fitness=0.75, cost_in=0.15)
        assert selected == "gemini-flash"

    def test_skips_below_threshold(self, registry):
        planner = TaskPlanner(registry)
        selected = planner.select_model()
        # cheap-low has fitness 0.60 < 0.70, should not be selected
        assert selected != "cheap-low"

    def test_override_model(self, registry):
        planner = TaskPlanner(registry)
        selected = planner.select_model("claude-opus")
        assert selected == "claude-opus"

    def test_override_invalid_model(self, registry):
        planner = TaskPlanner(registry)
        with pytest.raises(RuntimeError, match="not found"):
            planner.select_model("nonexistent")

    def test_empty_registry(self):
        planner = TaskPlanner({})
        with pytest.raises(RuntimeError, match="No models"):
            planner.select_model()

    def test_fallback_when_none_above_threshold(self):
        """When no model meets 0.70, falls back to highest fitness."""
        reg = {
            "weak-a": _make_model("a", "a", arch_fitness=0.50, cost_in=1.0, cost_out=5.0),
            "weak-b": _make_model("b", "b", arch_fitness=0.65, cost_in=0.5, cost_out=2.0),
        }
        planner = TaskPlanner(reg)
        selected = planner.select_model()
        assert selected == "weak-b"  # highest architect fitness


# ── Quick Mode Tests ──────────────────────────────────────────────


class TestQuickMode:
    """Test quick mode (single LLM call)."""

    @pytest.mark.asyncio
    async def test_quick_mode_produces_spec(self, registry):
        planner = TaskPlanner(registry)
        expanded_text = (
            "### Requirements\n- REST API with auth\n\n"
            "### Tech Stack\n- Python 3.12\n- FastAPI\n\n"
            "### Architecture Approach\nREST with JWT\n\n"
            "### File Structure\n- app.py\n- models.py\n\n"
            "### Acceptance Criteria\n1. Auth works\n\n"
            "### Edge Cases to Handle\n- Invalid token\n\n"
            "### Suggested Tests\n- test_auth.py\n\n"
            "TECH_STACK: Python, FastAPI, JWT"
        )
        mock_response = _mock_llm_response(expanded_text)

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            result = await planner.plan("build a REST API with auth")

        assert result.expanded_spec == expanded_text
        assert result.interactive is False
        assert result.clarifying_questions is None
        assert result.cost == 0.005

    @pytest.mark.asyncio
    async def test_quick_mode_single_llm_call(self, registry):
        planner = TaskPlanner(registry)
        mock_response = _mock_llm_response("spec\nTECH_STACK: Python")

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            await planner.plan("build something")

        # Only one call to complete()
        assert mock_provider.complete.await_count == 1

    @pytest.mark.asyncio
    async def test_quick_mode_extracts_tech_stack(self, registry):
        planner = TaskPlanner(registry)
        mock_response = _mock_llm_response(
            "Some spec text\nTECH_STACK: Python, FastAPI, PostgreSQL, Pydantic"
        )

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            result = await planner.plan("build an API")

        assert result.tech_stack_inferred == ["Python", "FastAPI", "PostgreSQL", "Pydantic"]

    @pytest.mark.asyncio
    async def test_quick_mode_task_spec_populated(self, registry):
        planner = TaskPlanner(registry)
        mock_response = _mock_llm_response("Full expanded spec here")

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            result = await planner.plan("build a CLI tool")

        assert result.task_spec.task == "Full expanded spec here"
        assert isinstance(result.task_spec, TaskSpec)

    @pytest.mark.asyncio
    async def test_quick_mode_tracks_token_usage(self, registry):
        planner = TaskPlanner(registry)
        mock_response = _mock_llm_response(
            "spec", prompt_tokens=800, completion_tokens=2000, cost=0.012,
        )

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            result = await planner.plan("build something")

        assert result.token_usage.prompt_tokens == 800
        assert result.token_usage.completion_tokens == 2000
        assert result.cost == 0.012


# ── Interactive Mode Tests ────────────────────────────────────────


class TestInteractiveMode:
    """Test interactive mode (two-phase planning)."""

    @pytest.mark.asyncio
    async def test_phase1_returns_questions(self, registry):
        planner = TaskPlanner(registry)
        questions_text = (
            "1. What scale do you need? (impacts database choice)\n"
            "2. Do you need real-time features? (WebSocket vs polling)\n"
            "3. What auth method? (JWT, OAuth, session-based)"
        )
        mock_response = _mock_llm_response(questions_text)

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            result = await planner.plan("build a task management API", interactive=True)

        assert result.interactive is True
        assert result.clarifying_questions is not None
        assert len(result.clarifying_questions) == 3
        assert result.user_answers is None

    @pytest.mark.asyncio
    async def test_phase2_with_answers(self, registry):
        planner = TaskPlanner(registry)
        expanded = "Full spec with answers\nTECH_STACK: Python, FastAPI"
        mock_response = _mock_llm_response(expanded)

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            result = await planner.plan(
                "build a task API",
                interactive=True,
                user_answers="Q1: Small scale\nQ2: JWT auth",
            )

        assert result.interactive is True
        assert result.user_answers == "Q1: Small scale\nQ2: JWT auth"
        assert "Full spec" in result.expanded_spec

    @pytest.mark.asyncio
    async def test_phase2_injects_answers_into_prompt(self, registry):
        planner = TaskPlanner(registry)
        mock_response = _mock_llm_response("spec")

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            with patch("triad.planner.render_prompt") as mock_render:
                mock_render.return_value = "rendered prompt"

                await planner.plan(
                    "build X",
                    interactive=True,
                    user_answers="My answers here",
                )

                # The planner prompt should be called with user_answers
                mock_render.assert_called_with(
                    "planner",
                    description="build X",
                    user_answers="My answers here",
                )

    @pytest.mark.asyncio
    async def test_phase1_single_llm_call(self, registry):
        planner = TaskPlanner(registry)
        mock_response = _mock_llm_response("1. Scale?\n2. Auth?")

        with patch(
            "triad.planner.LiteLLMProvider"
        ) as mock_provider_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_provider_cls.return_value = mock_provider

            await planner.plan("build something", interactive=True)

        assert mock_provider.complete.await_count == 1


# ── Utility Function Tests ────────────────────────────────────────


class TestUtilityFunctions:
    """Test helper functions."""

    def test_extract_tech_stack(self):
        text = "Some text\nTECH_STACK: Python, FastAPI, PostgreSQL\nMore text"
        result = _extract_tech_stack(text)
        assert result == ["Python", "FastAPI", "PostgreSQL"]

    def test_extract_tech_stack_case_insensitive(self):
        text = "tech_stack: React, Node.js"
        result = _extract_tech_stack(text)
        assert result == ["React", "Node.js"]

    def test_extract_tech_stack_missing(self):
        text = "No tech stack line here"
        result = _extract_tech_stack(text)
        assert result == []

    def test_parse_questions_numbered(self):
        text = (
            "1. What scale?\n"
            "2. What auth method?\n"
            "3. Deploy where?"
        )
        result = _parse_questions(text)
        assert len(result) == 3
        assert result[0].startswith("1.")

    def test_parse_questions_with_parens(self):
        text = (
            "1) What scale? (impacts DB choice)\n"
            "2) Auth method? (JWT vs OAuth)\n"
        )
        result = _parse_questions(text)
        assert len(result) == 2

    def test_parse_questions_empty(self):
        result = _parse_questions("")
        assert result == []

    def test_parse_questions_no_numbers(self):
        text = "Just some text without numbered questions"
        result = _parse_questions(text)
        assert result == []


# ── CLI Integration Tests ─────────────────────────────────────────


class TestCLIPlan:
    """Test the CLI plan command."""

    def test_plan_help(self):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["plan", "--help"])
        assert result.exit_code == 0
        assert "--interactive" in result.output
        assert "--run" in result.output
        assert "--save" in result.output
        assert "--edit" in result.output
        assert "--model" in result.output

    def test_plan_quick_mode(self, registry):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        expanded = "### Requirements\n- API\n\nTECH_STACK: Python, FastAPI"
        mock_response = _mock_llm_response(expanded)

        with patch("triad.cli._load_registry", return_value=registry), \
             patch("triad.planner.LiteLLMProvider") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_provider

            result = runner.invoke(
                app, ["plan", "build an API"],
                input="q\n",
            )

        assert result.exit_code == 0
        assert "Task Planner" in result.output

    def test_plan_with_save(self, registry, tmp_path):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        expanded = "Full spec\nTECH_STACK: Python"
        mock_response = _mock_llm_response(expanded)
        save_path = tmp_path / "spec.md"

        with patch("triad.cli._load_registry", return_value=registry), \
             patch("triad.planner.LiteLLMProvider") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_provider

            result = runner.invoke(
                app, ["plan", "build an API", "--save", str(save_path)],
            )

        assert result.exit_code == 0
        assert save_path.exists()
        assert "Full spec" in save_path.read_text(encoding="utf-8")

    def test_plan_with_model_override(self, registry):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        expanded = "Spec\nTECH_STACK: Python"
        mock_response = _mock_llm_response(expanded)

        with patch("triad.cli._load_registry", return_value=registry), \
             patch("triad.planner.LiteLLMProvider") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_provider

            result = runner.invoke(
                app, ["plan", "build X", "--model", "claude-opus"],
                input="q\n",
            )

        assert result.exit_code == 0
        assert "claude-opus" in result.output

    def test_plan_shows_tech_stack(self, registry):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        expanded = "Spec content\nTECH_STACK: Python, FastAPI, PostgreSQL"
        mock_response = _mock_llm_response(expanded)

        with patch("triad.cli._load_registry", return_value=registry), \
             patch("triad.planner.LiteLLMProvider") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_provider

            result = runner.invoke(
                app, ["plan", "build an API"],
                input="q\n",
            )

        assert result.exit_code == 0
        assert "Tech Stack" in result.output

    def test_plan_shows_cost(self, registry):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        mock_response = _mock_llm_response(
            "Spec\nTECH_STACK: Python",
            prompt_tokens=500,
            completion_tokens=1000,
            cost=0.0123,
        )

        with patch("triad.cli._load_registry", return_value=registry), \
             patch("triad.planner.LiteLLMProvider") as mock_cls:
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_provider

            result = runner.invoke(
                app, ["plan", "build X"],
                input="q\n",
            )

        assert result.exit_code == 0
        assert "$0.0123" in result.output

    def test_plan_with_run_flag(self, registry):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        expanded = "Full task spec\nTECH_STACK: Python"
        mock_response = _mock_llm_response(expanded)

        mock_pipeline_result = MagicMock()
        mock_pipeline_result.halted = False
        mock_pipeline_result.success = True
        mock_pipeline_result.total_cost = 0.05
        mock_pipeline_result.total_tokens = 2000
        mock_pipeline_result.duration_seconds = 5.0
        mock_pipeline_result.session_id = "plan-test-123"
        mock_pipeline_result.arbiter_reviews = []
        mock_pipeline_result.routing_decisions = []
        mock_pipeline_result.task = TaskSpec(task=expanded)
        mock_pipeline_result.config = PipelineConfig()

        with patch("triad.cli._load_registry", return_value=registry), \
             patch("triad.cli._load_config", return_value=PipelineConfig()), \
             patch("triad.planner.LiteLLMProvider") as mock_cls, \
             patch("triad.orchestrator.run_pipeline") as mock_run, \
             patch("triad.output.writer.write_pipeline_output"):
            mock_provider = MagicMock()
            mock_provider.complete = AsyncMock(return_value=mock_response)
            mock_cls.return_value = mock_provider
            mock_run.return_value = mock_pipeline_result

            result = runner.invoke(
                app, ["plan", "build an API", "--run"],
            )

        assert result.exit_code == 0
        # Should show both plan and pipeline result
        assert "Task Planner" in result.output

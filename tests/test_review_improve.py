"""Tests for review and improve pipeline modes."""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from triad.orchestrator import (
    ImproveOrchestrator,
    ReviewOrchestrator,
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
    defaults = {"task": "Review this code", "context": "def hello(): pass"}
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


# ── ReviewOrchestrator Tests ──────────────────────────────────────


class TestReviewOrchestrator:

    async def test_fan_out_calls_all_models(self):
        """Review fan-out should call all models with review_analyze template."""
        registry = _make_three_model_registry()
        # 3 fan-out + 6 cross-reviews + 1 synthesis = 10
        responses = [_make_agent_message(f"analysis-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ReviewOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="review", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.review_result is not None
        assert len(result.review_result.individual_analyses) == 3
        assert result.success is True

    async def test_cross_review_populated(self):
        """Cross-reviews should be populated for all reviewer pairs."""
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ReviewOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="review", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        cr = result.review_result.cross_reviews
        assert len(cr) > 0

    async def test_synthesis_populates_synthesized_review(self):
        """Synthesized review should be populated."""
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ReviewOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="review", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.review_result.synthesized_review != ""

    async def test_result_structure(self):
        """Result should have review_result but not parallel/debate/improve."""
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ReviewOrchestrator(
                task=_make_task(),
                config=PipelineConfig(
                    pipeline_mode="review", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.review_result is not None
        assert result.parallel_result is None
        assert result.debate_result is None
        assert result.improve_result is None


# ── ImproveOrchestrator Tests ─────────────────────────────────────


class TestImproveOrchestrator:

    async def test_fan_out_calls_all_models(self):
        """Improve fan-out should call all models."""
        registry = _make_three_model_registry()
        # 3 fan-out + 6 cross-reviews + 1 synthesis = 10
        responses = [_make_agent_message(f"improvement-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ImproveOrchestrator(
                task=_make_task(task="Improve this code"),
                config=PipelineConfig(
                    pipeline_mode="improve", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.improve_result is not None
        assert len(result.improve_result.individual_outputs) == 3
        assert result.success is True

    async def test_cross_review_and_voting(self):
        """Cross-review scores and votes should be populated."""
        registry = _make_three_model_registry()
        # 3 fan-out + 6 cross-reviews (with scores) + 1 synthesis = 10
        fan_out = [_make_agent_message(f"imp-{i}") for i in range(3)]
        reviews = [
            _make_agent_message(
                "ARCHITECTURE: 8\nGood.\nIMPLEMENTATION: 7\nSolid.\n"
                "QUALITY: 9\nClean.\nVOTE: YES\nCONFIDENCE: 0.85"
            )
            for _ in range(6)
        ]
        synthesis = [_make_agent_message("synthesized improvement")]
        responses = fan_out + reviews + synthesis
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ImproveOrchestrator(
                task=_make_task(task="Improve this code"),
                config=PipelineConfig(
                    pipeline_mode="improve", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        ir = result.improve_result
        assert len(ir.scores) > 0
        assert len(ir.votes) > 0
        assert ir.winner != ""

    async def test_synthesis_populates_output(self):
        """Synthesized output should be populated."""
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ImproveOrchestrator(
                task=_make_task(task="Improve this code"),
                config=PipelineConfig(
                    pipeline_mode="improve", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.improve_result.synthesized_output != ""

    async def test_result_structure(self):
        """Result should have improve_result but not parallel/debate/review."""
        registry = _make_three_model_registry()
        responses = [_make_agent_message(f"out-{i}") for i in range(10)]
        mock_cls, _ = _mock_provider(responses)

        with (
            patch(_PROVIDER, mock_cls),
            patch(_ARBITER_REVIEW, AsyncMock(return_value=None)),
        ):
            orch = ImproveOrchestrator(
                task=_make_task(task="Improve this code"),
                config=PipelineConfig(
                    pipeline_mode="improve", arbiter_mode="off",
                ),
                registry=registry,
            )
            result = await orch.run()

        assert result.improve_result is not None
        assert result.parallel_result is None
        assert result.debate_result is None
        assert result.review_result is None


# ── File Reader Tests ─────────────────────────────────────────────


class TestFileReader:

    def test_reads_single_file(self, tmp_path):
        """_read_source_files should read a single file."""
        from triad.cli import _read_source_files

        f = tmp_path / "test.py"
        f.write_text("print('hello')", encoding="utf-8")

        result = _read_source_files([str(f)])

        assert "# file:" in result
        assert "print('hello')" in result

    def test_reads_multiple_files(self, tmp_path):
        """_read_source_files should read multiple files."""
        from triad.cli import _read_source_files

        f1 = tmp_path / "a.py"
        f1.write_text("x = 1", encoding="utf-8")
        f2 = tmp_path / "b.py"
        f2.write_text("y = 2", encoding="utf-8")

        result = _read_source_files([str(f1), str(f2)])

        assert "x = 1" in result
        assert "y = 2" in result

    def test_reads_directory(self, tmp_path):
        """_read_source_files should expand directories with *.py."""
        from triad.cli import _read_source_files

        subdir = tmp_path / "src"
        subdir.mkdir()
        (subdir / "main.py").write_text("main()", encoding="utf-8")
        (subdir / "util.py").write_text("util()", encoding="utf-8")
        (subdir / "readme.txt").write_text("not python", encoding="utf-8")

        result = _read_source_files([str(subdir)])

        assert "main()" in result
        assert "util()" in result
        # .txt should not be included
        assert "not python" not in result

    def test_nonexistent_file_raises(self, tmp_path):
        """_read_source_files should exit on missing file."""
        from click.exceptions import Exit

        from triad.cli import _read_source_files

        with pytest.raises(Exit):
            _read_source_files([str(tmp_path / "nonexistent.py")])


# ── CLI Command Tests ─────────────────────────────────────────────


class TestReviewCodeCLI:

    def test_review_code_command_exists(self):
        """The review-code command should be registered."""
        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["review-code", "--help"])
        assert result.exit_code == 0
        assert "review" in result.output.lower()

    def test_review_code_help_shows_options(self):
        """review-code --help should show expected options."""
        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["review-code", "--help"])
        assert "--focus" in result.output
        assert "--preset" in result.output
        assert "--timeout" in result.output


class TestImproveCLI:

    def test_improve_command_exists(self):
        """The improve command should be registered."""
        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["improve", "--help"])
        assert result.exit_code == 0
        assert "improve" in result.output.lower()

    def test_improve_help_shows_options(self):
        """improve --help should show expected options."""
        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["improve", "--help"])
        assert "--focus" in result.output
        assert "--apply" in result.output
        assert "--branch" in result.output

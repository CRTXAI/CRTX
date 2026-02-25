"""Tests for the CLI Interface (Day 10).

Covers all commands, --help output, invalid args, output modules,
and orchestrator integration via CliRunner.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime
from pathlib import Path
from unittest.mock import patch

from typer.testing import CliRunner

from triad.cli import app
from triad.output.renderer import render_summary
from triad.output.writer import write_pipeline_output
from triad.schemas.arbiter import (
    ArbiterReview,
    Issue,
    IssueCategory,
    Severity,
)
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    TokenUsage,
)
from triad.schemas.pipeline import (
    PipelineConfig,
    PipelineResult,
    TaskSpec,
)
from triad.schemas.routing import RoutingDecision, RoutingStrategy

# NO_COLOR=1 prevents Rich from injecting ANSI codes inside option names,
# which breaks substring matching in CI (headless, no TTY).
# COLUMNS=200 prevents wrapping that could split a flag across lines.
runner = CliRunner(env={"NO_COLOR": "1", "COLUMNS": "200"})


# ── Factories ──────────────────────────────────────────────────────


def _make_task() -> TaskSpec:
    return TaskSpec(task="Build a REST API", context="Python FastAPI")


def _make_config(**overrides) -> PipelineConfig:
    defaults = {"persist_sessions": False}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_agent_message(stage: str = "architect") -> AgentMessage:
    return AgentMessage(
        from_agent=stage,
        to_agent="implement",
        msg_type=MessageType.PROPOSAL,
        content=f"Output from {stage}\n\nCONFIDENCE: 0.85",
        confidence=0.85,
        model="test-model-v1",
        token_usage=TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            cost=0.02,
        ),
    )


def _make_review(verdict: str = "approve") -> ArbiterReview:
    return ArbiterReview(
        stage_reviewed="architect",
        reviewed_model="test-model-v1",
        arbiter_model="arbiter-model-v1",
        verdict=verdict,
        issues=[
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.PATTERN,
                description="Test issue",
            ),
        ],
        confidence=0.9,
        reasoning="LGTM",
        token_cost=0.01,
    )


def _make_routing_decision(role: str = "architect") -> RoutingDecision:
    return RoutingDecision(
        model_key="test-model",
        model_id="test-model-v1",
        role=role,
        strategy=RoutingStrategy.HYBRID,
        rationale="Best fitness",
        fitness_score=0.9,
        estimated_cost=0.05,
    )


def _make_pipeline_result(**overrides) -> PipelineResult:
    defaults = {
        "session_id": "test-session-id",
        "task": _make_task(),
        "config": _make_config(),
        "stages": {
            PipelineStage.ARCHITECT: _make_agent_message("architect"),
            PipelineStage.IMPLEMENT: _make_agent_message("implement"),
            PipelineStage.REFACTOR: _make_agent_message("refactor"),
            PipelineStage.VERIFY: _make_agent_message("verify"),
        },
        "arbiter_reviews": [_make_review()],
        "routing_decisions": [
            _make_routing_decision("architect"),
            _make_routing_decision("implement"),
        ],
        "total_cost": 0.25,
        "total_tokens": 6000,
        "duration_seconds": 45.0,
        "success": True,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


# ── Help Tests ────────────────────────────────────────────────────


class TestHelp:
    def test_main_help(self):
        """Main app --help shows all commands."""
        result = runner.invoke(app, ["--help"])
        assert result.exit_code == 0
        assert "run" in result.output
        assert "plan" in result.output
        assert "estimate" in result.output
        assert "models" in result.output
        assert "config" in result.output
        assert "sessions" in result.output

    def test_run_help(self):
        """run --help shows all options."""
        result = runner.invoke(app, ["run", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output
        assert "--route" in result.output
        assert "--arbiter" in result.output
        assert "--reconcile" in result.output
        assert "--no-persist" in result.output

    def test_estimate_help(self):
        result = runner.invoke(app, ["estimate", "--help"])
        assert result.exit_code == 0
        assert "--mode" in result.output

    def test_models_help(self):
        result = runner.invoke(app, ["models", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "test" in result.output

    def test_config_help(self):
        result = runner.invoke(app, ["config", "--help"])
        assert result.exit_code == 0
        assert "show" in result.output
        assert "path" in result.output

    def test_sessions_help(self):
        result = runner.invoke(app, ["sessions", "--help"])
        assert result.exit_code == 0
        assert "list" in result.output
        assert "show" in result.output
        assert "export" in result.output
        assert "delete" in result.output


# ── crtx run Tests ───────────────────────────────────────────────


class TestRun:
    def test_invalid_mode(self):
        """Invalid mode produces a helpful error."""
        result = runner.invoke(app, ["run", "test task", "--mode", "bad"])
        assert result.exit_code == 1
        assert "Invalid mode" in result.output

    def test_invalid_route(self):
        """Invalid routing strategy produces a helpful error."""
        result = runner.invoke(app, ["run", "test task", "--route", "bad"])
        assert result.exit_code == 1
        assert "Invalid routing strategy" in result.output

    def test_invalid_arbiter(self):
        """Invalid arbiter mode produces a helpful error."""
        result = runner.invoke(app, ["run", "test task", "--arbiter", "bad"])
        assert result.exit_code == 1
        assert "Invalid arbiter mode" in result.output

    def test_run_invokes_pipeline(self, tmp_path):
        """crtx run invokes the pipeline and writes output."""
        mock_result = _make_pipeline_result()

        with (
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output") as mock_write,
        ):
            result = runner.invoke(app, [
                "run", "Build a test API",
                "--mode", "sequential",
                "--arbiter", "off",
                "--no-persist",
                "--output-dir", str(tmp_path / "out"),
            ])

        assert result.exit_code == 0
        assert "PIPELINE COMPLETED SUCCESSFULLY" in result.output
        mock_write.assert_called_once()

    def test_run_shows_halted_result(self):
        """crtx run displays halt info when pipeline halts."""
        mock_result = _make_pipeline_result(
            success=False, halted=True, halt_reason="Critical bug found",
        )

        with (
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "HALTED" in result.output

    def test_run_shows_cost_summary(self):
        """crtx run displays cost and token summary."""
        mock_result = _make_pipeline_result()

        with (
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "$0.25" in result.output
        assert "6,000" in result.output

    def test_run_domain_rules_missing_file(self):
        """crtx run errors when domain rules file doesn't exist."""
        result = runner.invoke(app, [
            "run", "test", "--domain-rules", "/nonexistent/file.toml",
        ])
        assert result.exit_code == 1
        assert "Domain rules file not found" in result.output


# ── crtx plan Tests ──────────────────────────────────────────────


class TestPlan:
    def test_plan_requires_description(self):
        """crtx plan without args shows usage error."""
        result = runner.invoke(app, ["plan"])
        assert result.exit_code == 2  # missing required argument


# ── crtx estimate Tests ──────────────────────────────────────────


class TestEstimate:
    def test_estimate_shows_strategies(self):
        """crtx estimate shows all 4 routing strategies."""
        result = runner.invoke(app, ["estimate", "Build an API"])
        assert result.exit_code == 0
        assert "quality_first" in result.output
        assert "cost_optimized" in result.output
        assert "speed_first" in result.output
        assert "hybrid" in result.output

    def test_estimate_shows_costs(self):
        """crtx estimate shows dollar amounts."""
        result = runner.invoke(app, ["estimate", "Build an API"])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_estimate_shows_model_assignments(self):
        """crtx estimate shows which models are assigned."""
        result = runner.invoke(app, ["estimate", "Build an API"])
        assert result.exit_code == 0
        assert "Model Assignments" in result.output

    def test_estimate_invalid_mode(self):
        """crtx estimate rejects invalid mode."""
        result = runner.invoke(app, ["estimate", "test", "--mode", "bad"])
        assert result.exit_code == 1
        assert "Invalid mode" in result.output


# ── crtx models Tests ────────────────────────────────────────────


class TestModels:
    def test_models_list_shows_all_models(self):
        """crtx models list shows all registered models."""
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        # Rich may truncate in narrow CliRunner terminal
        assert "Registered Models" in result.output
        assert "models registered" in result.output
        # Check for providers (short enough to not be truncated)
        assert "xai" in result.output.lower()

    def test_models_list_shows_fitness(self):
        """crtx models list shows fitness scores."""
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "Arch" in result.output
        assert "Impl" in result.output

    def test_models_show_valid_key(self):
        """crtx models show displays full model details."""
        result = runner.invoke(app, ["models", "show", "claude-opus"])
        assert result.exit_code == 0
        assert "Claude Opus" in result.output
        assert "anthropic" in result.output
        assert "200,000" in result.output

    def test_models_show_invalid_key(self):
        """crtx models show errors on unknown key."""
        result = runner.invoke(app, ["models", "show", "nonexistent"])
        assert result.exit_code == 1
        assert "Model not found" in result.output

    def test_models_test_no_api_key(self):
        """crtx models test errors when API key not set."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure ANTHROPIC_API_KEY is not set
            env = dict(os.environ)
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                result = runner.invoke(app, ["models", "test", "claude-opus"])
        assert result.exit_code == 1
        assert "API key not set" in result.output

    def test_models_test_invalid_key(self):
        """crtx models test errors on unknown model key."""
        result = runner.invoke(app, ["models", "test", "nonexistent"])
        assert result.exit_code == 1
        assert "Model not found" in result.output


# ── crtx config Tests ────────────────────────────────────────────


class TestConfig:
    def test_config_show(self):
        """crtx config show displays pipeline configuration."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "bookend" in result.output
        assert "120s" in result.output
        assert "hybrid" in result.output
        assert "0.70" in result.output

    def test_config_path(self):
        """crtx config path shows config file locations."""
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        # Rich may truncate long paths — check for labels
        assert "Models" in result.output
        assert "Defaults" in result.output
        assert "Routing" in result.output
        assert "found" in result.output


# ── crtx sessions Tests ──────────────────────────────────────────


class TestSessions:
    def test_sessions_list_empty(self, tmp_path):
        """crtx sessions list with no sessions shows message."""
        db_path = str(tmp_path / "test.db")
        config = _make_config(
            persist_sessions=True,
            session_db_path=db_path,
        )

        with patch("triad.cli._load_config", return_value=config):
            result = runner.invoke(app, ["sessions", "list"])

        assert result.exit_code == 0
        assert "No sessions found" in result.output

    def test_sessions_show_not_found(self, tmp_path):
        """crtx sessions show errors on unknown session."""
        db_path = str(tmp_path / "test.db")
        config = _make_config(
            persist_sessions=True,
            session_db_path=db_path,
        )

        with patch("triad.cli._load_config", return_value=config):
            result = runner.invoke(app, ["sessions", "show", "nonexistent"])

        assert result.exit_code == 1
        assert "Session not found" in result.output

    def test_sessions_export_invalid_format(self, tmp_path):
        """crtx sessions export errors on invalid format."""
        db_path = str(tmp_path / "test.db")
        config = _make_config(
            persist_sessions=True,
            session_db_path=db_path,
        )

        # Save a session first
        import asyncio

        from triad.persistence.database import close_db, init_db
        from triad.persistence.session import SessionStore
        from triad.schemas.session import SessionRecord

        record = SessionRecord(
            session_id="test-export-session",
            task=_make_task(),
            config=_make_config(),
            started_at=datetime(2026, 2, 16, 10, 0, 0, tzinfo=UTC),
            success=True,
        )

        async def _setup():
            db = await init_db(db_path)
            store = SessionStore(db)
            await store.save_session(record)
            await close_db(db)

        asyncio.run(_setup())

        with patch("triad.cli._load_config", return_value=config):
            result = runner.invoke(app, [
                "sessions", "export", "test-export-session", "--format", "xml",
            ])

        assert result.exit_code == 1
        assert "Invalid format" in result.output

    def test_sessions_delete_not_found(self, tmp_path):
        """crtx sessions delete errors on unknown session."""
        db_path = str(tmp_path / "test.db")
        config = _make_config(
            persist_sessions=True,
            session_db_path=db_path,
        )

        with patch("triad.cli._load_config", return_value=config):
            result = runner.invoke(
                app, ["sessions", "delete", "nonexistent", "--yes"],
            )

        assert result.exit_code == 1
        assert "Session not found" in result.output


# ── Output Module Tests ───────────────────────────────────────────


class TestRenderer:
    def test_render_summary_contains_sections(self):
        """render_summary produces all expected sections."""
        result = _make_pipeline_result()
        md = render_summary(result)

        assert "# CRTX Pipeline Summary" in md
        assert "## Task" in md
        assert "Build a REST API" in md
        assert "## Pipeline Configuration" in md
        assert "## Stage Summaries" in md
        assert "## Arbiter Verdicts" in md
        assert "## Cost Summary" in md
        assert "$0.2500" in md

    def test_render_summary_session_id(self):
        """render_summary includes session ID."""
        result = _make_pipeline_result()
        md = render_summary(result)
        assert "test-session-id" in md

    def test_render_summary_routing_table(self):
        """render_summary includes routing decisions table."""
        result = _make_pipeline_result()
        md = render_summary(result)
        assert "## Models Used" in md
        assert "test-model" in md

    def test_render_summary_halted(self):
        """render_summary shows HALTED status."""
        result = _make_pipeline_result(
            success=False, halted=True, halt_reason="Bug found",
        )
        md = render_summary(result)
        assert "HALTED" in md
        assert "Bug found" in md


class TestWriter:
    def test_write_pipeline_output_creates_structure(self, tmp_path):
        """write_pipeline_output creates all expected directories and files."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        base = Path(actual_path)
        assert base.name == "test-ses"  # first 8 chars of session_id
        assert (base / "code").is_dir()
        assert (base / "tests").is_dir()
        assert (base / "reviews").is_dir()
        assert (base / "summary.md").is_file()
        assert (base / "session.json").is_file()

    def test_write_pipeline_output_returns_path(self, tmp_path):
        """write_pipeline_output returns the session-namespaced path."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        assert actual_path == str(Path(out_dir) / "test-ses")

    def test_write_pipeline_output_summary_content(self, tmp_path):
        """write_pipeline_output creates a valid summary.md."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        summary = (Path(actual_path) / "summary.md").read_text()
        assert "CRTX Pipeline Summary" in summary
        assert "Build a REST API" in summary

    def test_write_pipeline_output_session_json(self, tmp_path):
        """write_pipeline_output creates valid session.json."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        session_data = json.loads(
            (Path(actual_path) / "session.json").read_text()
        )
        assert session_data["session_id"] == "test-session-id"
        assert session_data["success"] is True

    def test_write_pipeline_output_reviews(self, tmp_path):
        """write_pipeline_output creates arbiter review files."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        reviews = list(Path(actual_path, "reviews").glob("*.md"))
        assert len(reviews) == 1
        content = reviews[0].read_text()
        assert "APPROVE" in content

    def test_write_extracts_code_blocks(self, tmp_path):
        """write_pipeline_output extracts code blocks from output."""
        msg = _make_agent_message("verify")
        msg.content = (
            "Here is the code:\n\n"
            "```python\n"
            "# file: api/main.py\n"
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "```\n\n"
            "And the test:\n\n"
            "```python\n"
            "# file: test_main.py\n"
            "def test_app(): pass\n"
            "```"
        )

        result = _make_pipeline_result(
            stages={PipelineStage.VERIFY: msg},
            arbiter_reviews=[],
        )
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        code_files = list(Path(actual_path, "code").glob("*"))
        test_files = list(Path(actual_path, "tests").glob("*"))
        assert len(code_files) >= 1
        assert len(test_files) >= 1

    def test_write_no_code_blocks(self, tmp_path):
        """write_pipeline_output works with no code blocks."""
        msg = _make_agent_message("verify")
        msg.content = "Plain text output with no code blocks"
        result = _make_pipeline_result(
            stages={PipelineStage.VERIFY: msg},
            arbiter_reviews=[],
        )
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        # Should still create the directory structure
        assert (Path(actual_path) / "summary.md").is_file()


# ── CLI Parallel Display Tests ───────────────────────────────────


_FAKE_PROVIDER_KEYS = {
    "ANTHROPIC_API_KEY": "sk-test-fake",
    "OPENAI_API_KEY": "sk-test-fake",
    "GEMINI_API_KEY": "fake-key",
    "XAI_API_KEY": "fake-key",
}


class TestCliParallelDisplay:
    """Tests for _display_result showing parallel mode results."""

    def _make_parallel_result(self, **overrides):
        """Create a PipelineResult with parallel_result populated."""
        from triad.schemas.consensus import ParallelResult

        pr = ParallelResult(
            individual_outputs={
                "model-a": "Solution A",
                "model-b": "Solution B",
                "model-c": "Solution C",
            },
            scores={
                "model-a": {
                    "model-b": {"architecture": 8, "implementation": 7, "quality": 9},
                    "model-c": {"architecture": 6, "implementation": 5, "quality": 6},
                },
                "model-b": {
                    "model-a": {"architecture": 7, "implementation": 7, "quality": 7},
                    "model-c": {"architecture": 5, "implementation": 6, "quality": 5},
                },
                "model-c": {
                    "model-a": {"architecture": 8, "implementation": 8, "quality": 7},
                    "model-b": {"architecture": 6, "implementation": 7, "quality": 6},
                },
            },
            votes={"model-a": "model-b", "model-b": "model-a", "model-c": "model-a"},
            winner="model-a",
            synthesized_output="Final synthesized code output",
        )

        defaults = {
            "session_id": "test-parallel-session",
            "task": _make_task(),
            "config": _make_config(pipeline_mode="parallel"),
            "stages": {},
            "arbiter_reviews": [],
            "routing_decisions": [],
            "total_cost": 0.50,
            "total_tokens": 12000,
            "duration_seconds": 30.0,
            "success": True,
            "parallel_result": pr,
        }
        defaults.update(overrides)
        return PipelineResult(**defaults)

    def test_parallel_run_shows_winner(self):
        """crtx run with parallel result shows winner banner."""
        mock_result = self._make_parallel_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "parallel",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "model-a" in result.output

    def test_parallel_run_shows_votes(self):
        """crtx run with parallel result shows voting table."""
        mock_result = self._make_parallel_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "parallel",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        # The output should contain voter and voted-for
        assert "Consensus Votes" in result.output or "model-b" in result.output

    def test_parallel_run_shows_cost(self):
        """crtx run with parallel result shows cost summary."""
        mock_result = self._make_parallel_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "parallel",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "$0.50" in result.output

    def test_parallel_run_shows_stage_and_model_counts(self):
        """Parallel mode shows non-zero stage and model counts."""
        mock_result = self._make_parallel_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "parallel",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        # 3 individual outputs + 1 synthesis = 4 stages
        assert "Stages:" in result.output
        assert "4" in result.output
        # 3 models
        assert "3 providers" in result.output

    def test_parallel_run_shows_completed_with_rejections(self):
        """Parallel result with REJECT verdicts shows warning status."""
        mock_result = self._make_parallel_result(
            arbiter_reviews=[_make_review(verdict="reject")],
        )

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "parallel",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "COMPLETED WITH REJECTIONS" in result.output
        assert "PIPELINE COMPLETED SUCCESSFULLY" not in result.output


# ── CLI Completion Status Tests ──────────────────────────────────


class TestCliCompletionStatus:
    """Tests for _display_completion status line logic."""

    def test_success_without_rejects_shows_success(self):
        """Normal success shows green SUCCESS."""
        mock_result = _make_pipeline_result()

        with (
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "PIPELINE COMPLETED SUCCESSFULLY" in result.output

    def test_success_with_reject_shows_warning(self):
        """Success with REJECT verdicts shows warning status."""
        mock_result = _make_pipeline_result(
            arbiter_reviews=[_make_review(verdict="reject")],
        )

        with (
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--arbiter", "bookend",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "COMPLETED WITH REJECTIONS" in result.output


# ── CLI Debate Display Tests ─────────────────────────────────────


class TestCliDebateDisplay:
    """Tests for _display_result showing debate mode results."""

    def _make_debate_result(self, **overrides):
        """Create a PipelineResult with debate_result populated."""
        from triad.schemas.consensus import DebateResult

        dr = DebateResult(
            proposals={
                "model-a": "Proposal from A: use microservices",
                "model-b": "Proposal from B: use monolith",
                "model-c": "Proposal from C: use serverless",
            },
            rebuttals={
                "model-a": {"model-b": "B is too simple", "model-c": "C is too complex"},
                "model-b": {"model-a": "A is overengineered", "model-c": "C has cold starts"},
                "model-c": {"model-a": "A is fragile", "model-b": "B doesn't scale"},
            },
            final_arguments={
                "model-a": "Final from A",
                "model-b": "Final from B",
                "model-c": "Final from C",
            },
            judgment="After careful consideration, model-a's approach wins.",
            judge_model="model-c",
        )

        defaults = {
            "session_id": "test-debate-session",
            "task": _make_task(),
            "config": _make_config(pipeline_mode="debate"),
            "stages": {},
            "arbiter_reviews": [],
            "routing_decisions": [],
            "total_cost": 0.75,
            "total_tokens": 18000,
            "duration_seconds": 60.0,
            "success": True,
            "debate_result": dr,
        }
        defaults.update(overrides)
        return PipelineResult(**defaults)

    def test_debate_run_shows_judge(self):
        """crtx run with debate result shows judge model."""
        mock_result = self._make_debate_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "debate",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "model-c" in result.output

    def test_debate_run_shows_judgment(self):
        """crtx run with debate result shows judgment preview."""
        mock_result = self._make_debate_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "debate",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "careful consideration" in result.output

    def test_debate_run_shows_cost(self):
        """crtx run with debate result shows cost summary."""
        mock_result = self._make_debate_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "debate",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "$0.75" in result.output

    def test_debate_halted_shows_halt_reason(self):
        """crtx run with halted debate shows halt reason."""
        mock_result = self._make_debate_result(
            success=False, halted=True,
            halt_reason="Debate judgment fundamentally flawed",
        )

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "debate",
                "--arbiter", "bookend",
                "--no-persist",
            ])

        assert result.exit_code == 0
        assert "HALTED" in result.output

    def test_debate_run_shows_stage_and_model_counts(self):
        """Debate mode shows non-zero stage and model counts."""
        mock_result = self._make_debate_result()

        with (
            patch.dict(os.environ, _FAKE_PROVIDER_KEYS),
            patch("triad.cli.asyncio.run", return_value=mock_result),
            patch("triad.output.writer.write_pipeline_output"),
        ):
            result = runner.invoke(app, [
                "run", "Build API",
                "--mode", "debate",
                "--arbiter", "off",
                "--no-persist",
            ])

        assert result.exit_code == 0
        # 4 phases (proposals + rebuttals + final_args + judgment)
        assert "4" in result.output
        # 3 debaters + judge (model-c is both debater and judge, so 3 providers)
        assert "3 providers" in result.output

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

runner = CliRunner()


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


# ── triad run Tests ───────────────────────────────────────────────


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
        """triad run invokes the pipeline and writes output."""
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
        assert "Pipeline completed successfully" in result.output
        mock_write.assert_called_once()

    def test_run_shows_halted_result(self):
        """triad run displays halt info when pipeline halts."""
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
        """triad run displays cost and token summary."""
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
        assert "$0.2500" in result.output
        assert "6,000" in result.output

    def test_run_domain_rules_missing_file(self):
        """triad run errors when domain rules file doesn't exist."""
        result = runner.invoke(app, [
            "run", "test", "--domain-rules", "/nonexistent/file.toml",
        ])
        assert result.exit_code == 1
        assert "Domain rules file not found" in result.output


# ── triad plan Tests ──────────────────────────────────────────────


class TestPlan:
    def test_plan_shows_stub(self):
        """triad plan shows the v0.1 stub message."""
        result = runner.invoke(app, ["plan"])
        assert result.exit_code == 0
        assert "Coming in v0.1" in result.output


# ── triad estimate Tests ──────────────────────────────────────────


class TestEstimate:
    def test_estimate_shows_strategies(self):
        """triad estimate shows all 4 routing strategies."""
        result = runner.invoke(app, ["estimate", "Build an API"])
        assert result.exit_code == 0
        assert "quality_first" in result.output
        assert "cost_optimized" in result.output
        assert "speed_first" in result.output
        assert "hybrid" in result.output

    def test_estimate_shows_costs(self):
        """triad estimate shows dollar amounts."""
        result = runner.invoke(app, ["estimate", "Build an API"])
        assert result.exit_code == 0
        assert "$" in result.output

    def test_estimate_shows_model_assignments(self):
        """triad estimate shows which models are assigned."""
        result = runner.invoke(app, ["estimate", "Build an API"])
        assert result.exit_code == 0
        assert "Model Assignments" in result.output

    def test_estimate_invalid_mode(self):
        """triad estimate rejects invalid mode."""
        result = runner.invoke(app, ["estimate", "test", "--mode", "bad"])
        assert result.exit_code == 1
        assert "Invalid mode" in result.output


# ── triad models Tests ────────────────────────────────────────────


class TestModels:
    def test_models_list_shows_all_models(self):
        """triad models list shows all registered models."""
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        # Rich may truncate in narrow CliRunner terminal
        assert "Registered Models" in result.output
        assert "9 models registered" in result.output
        # Check for providers (short enough to not be truncated)
        assert "xai" in result.output.lower()

    def test_models_list_shows_fitness(self):
        """triad models list shows fitness scores."""
        result = runner.invoke(app, ["models", "list"])
        assert result.exit_code == 0
        assert "Arch" in result.output
        assert "Impl" in result.output

    def test_models_show_valid_key(self):
        """triad models show displays full model details."""
        result = runner.invoke(app, ["models", "show", "claude-opus"])
        assert result.exit_code == 0
        assert "Claude Opus" in result.output
        assert "anthropic" in result.output
        assert "200,000" in result.output

    def test_models_show_invalid_key(self):
        """triad models show errors on unknown key."""
        result = runner.invoke(app, ["models", "show", "nonexistent"])
        assert result.exit_code == 1
        assert "Model not found" in result.output

    def test_models_test_no_api_key(self):
        """triad models test errors when API key not set."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure ANTHROPIC_API_KEY is not set
            env = dict(os.environ)
            env.pop("ANTHROPIC_API_KEY", None)
            with patch.dict("os.environ", env, clear=True):
                result = runner.invoke(app, ["models", "test", "claude-opus"])
        assert result.exit_code == 1
        assert "API key not set" in result.output

    def test_models_test_invalid_key(self):
        """triad models test errors on unknown model key."""
        result = runner.invoke(app, ["models", "test", "nonexistent"])
        assert result.exit_code == 1
        assert "Model not found" in result.output


# ── triad config Tests ────────────────────────────────────────────


class TestConfig:
    def test_config_show(self):
        """triad config show displays pipeline configuration."""
        result = runner.invoke(app, ["config", "show"])
        assert result.exit_code == 0
        assert "bookend" in result.output
        assert "120s" in result.output
        assert "hybrid" in result.output
        assert "0.70" in result.output

    def test_config_path(self):
        """triad config path shows config file locations."""
        result = runner.invoke(app, ["config", "path"])
        assert result.exit_code == 0
        # Rich may truncate long paths — check for labels
        assert "Models" in result.output
        assert "Defaults" in result.output
        assert "Routing" in result.output
        assert "found" in result.output


# ── triad sessions Tests ──────────────────────────────────────────


class TestSessions:
    def test_sessions_list_empty(self, tmp_path):
        """triad sessions list with no sessions shows message."""
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
        """triad sessions show errors on unknown session."""
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
        """triad sessions export errors on invalid format."""
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
        """triad sessions delete errors on unknown session."""
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

        assert "# Triad Pipeline Summary" in md
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

        write_pipeline_output(result, out_dir)

        base = Path(out_dir)
        assert (base / "code").is_dir()
        assert (base / "tests").is_dir()
        assert (base / "reviews").is_dir()
        assert (base / "summary.md").is_file()
        assert (base / "session.json").is_file()

    def test_write_pipeline_output_summary_content(self, tmp_path):
        """write_pipeline_output creates a valid summary.md."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        write_pipeline_output(result, out_dir)

        summary = (Path(out_dir) / "summary.md").read_text()
        assert "Triad Pipeline Summary" in summary
        assert "Build a REST API" in summary

    def test_write_pipeline_output_session_json(self, tmp_path):
        """write_pipeline_output creates valid session.json."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        write_pipeline_output(result, out_dir)

        session_data = json.loads(
            (Path(out_dir) / "session.json").read_text()
        )
        assert session_data["session_id"] == "test-session-id"
        assert session_data["success"] is True

    def test_write_pipeline_output_reviews(self, tmp_path):
        """write_pipeline_output creates arbiter review files."""
        result = _make_pipeline_result()
        out_dir = str(tmp_path / "output")

        write_pipeline_output(result, out_dir)

        reviews = list(Path(out_dir, "reviews").glob("*.md"))
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

        write_pipeline_output(result, out_dir)

        code_files = list(Path(out_dir, "code").glob("*"))
        test_files = list(Path(out_dir, "tests").glob("*"))
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

        write_pipeline_output(result, out_dir)

        # Should still create the directory structure
        assert (Path(out_dir) / "summary.md").is_file()

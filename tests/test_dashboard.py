"""Tests for Dashboard (Day 14).

Covers: PipelineEventEmitter, EventType, PipelineEvent schema,
server app creation and routes, orchestrator event integration.
"""

from __future__ import annotations

from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.dashboard.events import EventType, PipelineEvent, PipelineEventEmitter

# ══════════════════════════════════════════════════════════════════
# PipelineEvent Schema
# ══════════════════════════════════════════════════════════════════


class TestPipelineEvent:
    """PipelineEvent schema tests."""

    def test_create_event_with_defaults(self):
        event = PipelineEvent(type=EventType.PIPELINE_STARTED)
        assert event.type == EventType.PIPELINE_STARTED
        assert event.timestamp > 0
        assert event.data == {}

    def test_create_event_with_data(self):
        event = PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "model": "claude-sonnet", "duration": 3.5},
        )
        assert event.data["stage"] == "architect"
        assert event.data["duration"] == 3.5

    def test_event_serializes_to_dict(self):
        event = PipelineEvent(type=EventType.ERROR, data={"message": "fail"})
        d = event.model_dump()
        assert d["type"] == "error"
        assert d["data"]["message"] == "fail"
        assert "timestamp" in d


# ══════════════════════════════════════════════════════════════════
# EventType Enum
# ══════════════════════════════════════════════════════════════════


class TestEventType:
    """EventType enum tests."""

    def test_all_event_types_exist(self):
        expected = [
            "pipeline_started", "stage_started", "stage_completed",
            "arbiter_started", "arbiter_verdict", "suggestion_filed",
            "consensus_vote", "retry_triggered", "pipeline_completed",
            "pipeline_halted", "error",
        ]
        for val in expected:
            assert val in [e.value for e in EventType]

    def test_event_type_is_string(self):
        assert EventType.PIPELINE_STARTED == "pipeline_started"
        assert str(EventType.ERROR) == "error"


# ══════════════════════════════════════════════════════════════════
# PipelineEventEmitter
# ══════════════════════════════════════════════════════════════════


class TestPipelineEventEmitter:
    """PipelineEventEmitter tests."""

    @pytest.mark.asyncio()
    async def test_emit_calls_sync_listener(self):
        emitter = PipelineEventEmitter()
        received = []
        emitter.add_listener(lambda e: received.append(e))

        await emitter.emit(EventType.PIPELINE_STARTED, mode="sequential")

        assert len(received) == 1
        assert received[0].type == EventType.PIPELINE_STARTED
        assert received[0].data["mode"] == "sequential"

    @pytest.mark.asyncio()
    async def test_emit_calls_async_listener(self):
        emitter = PipelineEventEmitter()
        received = []

        async def async_listener(event):
            received.append(event)

        emitter.add_listener(async_listener)
        await emitter.emit(EventType.STAGE_STARTED, stage="architect")

        assert len(received) == 1
        assert received[0].data["stage"] == "architect"

    @pytest.mark.asyncio()
    async def test_emit_with_no_listeners_no_error(self):
        emitter = PipelineEventEmitter()
        # Should not raise
        await emitter.emit(EventType.PIPELINE_COMPLETED, total_cost=0.05)

    @pytest.mark.asyncio()
    async def test_emit_multiple_listeners(self):
        emitter = PipelineEventEmitter()
        count = {"a": 0, "b": 0}
        emitter.add_listener(lambda e: count.__setitem__("a", count["a"] + 1))
        emitter.add_listener(lambda e: count.__setitem__("b", count["b"] + 1))

        await emitter.emit(EventType.STAGE_COMPLETED)

        assert count["a"] == 1
        assert count["b"] == 1

    @pytest.mark.asyncio()
    async def test_listener_exception_does_not_propagate(self):
        emitter = PipelineEventEmitter()
        emitter.add_listener(lambda e: 1 / 0)  # ZeroDivisionError
        received = []
        emitter.add_listener(lambda e: received.append(e))

        await emitter.emit(EventType.ERROR, message="test")

        # Second listener still called despite first raising
        assert len(received) == 1

    @pytest.mark.asyncio()
    async def test_remove_listener(self):
        emitter = PipelineEventEmitter()
        received = []
        listener = lambda e: received.append(e)  # noqa: E731
        emitter.add_listener(listener)
        emitter.remove_listener(listener)

        await emitter.emit(EventType.PIPELINE_STARTED)

        assert received == []

    @pytest.mark.asyncio()
    async def test_history_tracks_all_events(self):
        emitter = PipelineEventEmitter()
        await emitter.emit(EventType.PIPELINE_STARTED)
        await emitter.emit(EventType.STAGE_STARTED, stage="architect")
        await emitter.emit(EventType.STAGE_COMPLETED, stage="architect")

        assert len(emitter.history) == 3
        assert emitter.history[0].type == EventType.PIPELINE_STARTED
        assert emitter.history[2].type == EventType.STAGE_COMPLETED

    @pytest.mark.asyncio()
    async def test_history_returns_copy(self):
        emitter = PipelineEventEmitter()
        await emitter.emit(EventType.PIPELINE_STARTED)
        history = emitter.history
        history.clear()
        assert len(emitter.history) == 1  # Original unchanged


# ══════════════════════════════════════════════════════════════════
# Server
# ══════════════════════════════════════════════════════════════════


class TestServer:
    """Server creation and route tests."""

    def test_create_app_returns_fastapi_instance(self):
        pytest.importorskip("fastapi")
        from triad.dashboard.server import create_app

        app = create_app()
        assert app is not None
        assert hasattr(app, "routes")

    def test_app_has_websocket_route(self):
        pytest.importorskip("fastapi")
        from triad.dashboard.server import create_app

        app = create_app()
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/ws" in paths

    def test_app_has_api_routes(self):
        pytest.importorskip("fastapi")
        from triad.dashboard.server import create_app

        app = create_app()
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/api/sessions" in paths
        assert "/api/sessions/{session_id}" in paths
        assert "/api/models" in paths
        assert "/api/config" in paths

    def test_app_has_index_route(self):
        pytest.importorskip("fastapi")
        from triad.dashboard.server import create_app

        app = create_app()
        paths = [r.path for r in app.routes if hasattr(r, "path")]
        assert "/" in paths

    def test_get_emitter_returns_singleton(self):
        pytest.importorskip("fastapi")
        from triad.dashboard.server import get_emitter

        emitter1 = get_emitter()
        emitter2 = get_emitter()
        assert emitter1 is emitter2
        assert isinstance(emitter1, PipelineEventEmitter)


# ══════════════════════════════════════════════════════════════════
# Event Serialization
# ══════════════════════════════════════════════════════════════════


class TestEventSerialization:
    """Events serialize to JSON correctly."""

    def test_event_to_json_roundtrip(self):
        import json

        event = PipelineEvent(
            type=EventType.ARBITER_VERDICT,
            data={
                "stage": "architect",
                "verdict": "approve",
                "confidence": 0.95,
                "issues_count": 0,
            },
        )
        json_str = json.dumps(event.model_dump(), default=str)
        parsed = json.loads(json_str)
        assert parsed["type"] == "arbiter_verdict"
        assert parsed["data"]["confidence"] == 0.95

    def test_all_event_types_create_valid_events(self):
        for event_type in EventType:
            event = PipelineEvent(type=event_type, data={"key": "value"})
            d = event.model_dump()
            assert d["type"] == event_type.value


# ══════════════════════════════════════════════════════════════════
# Orchestrator Integration
# ══════════════════════════════════════════════════════════════════


class TestOrchestratorEmitter:
    """Verify orchestrator emits events at correct milestones."""

    def _make_registry(self):
        from triad.schemas.pipeline import ModelConfig, RoleFitness

        return {
            "claude-sonnet": ModelConfig(
                provider="anthropic",
                model="claude-sonnet",
                display_name="Claude Sonnet",
                api_key_env="ANTHROPIC_API_KEY",
                context_window=200_000,
                cost_input=3.0,
                cost_output=15.0,
                fitness=RoleFitness(
                    architect=0.88,
                    implementer=0.80,
                    refactorer=0.75,
                    verifier=0.80,
                ),
            ),
        }

    def _make_config(self):
        from triad.schemas.pipeline import ArbiterMode, PipelineConfig

        return PipelineConfig(arbiter_mode=ArbiterMode.OFF)

    def _make_task(self):
        from triad.schemas.pipeline import TaskSpec

        return TaskSpec(task="Build a hello world API")

    @pytest.mark.asyncio()
    async def test_sequential_orchestrator_accepts_emitter(self):
        from triad.orchestrator import PipelineOrchestrator

        emitter = PipelineEventEmitter()
        received = []
        emitter.add_listener(lambda e: received.append(e))

        orch = PipelineOrchestrator(
            task=self._make_task(),
            config=self._make_config(),
            registry=self._make_registry(),
            event_emitter=emitter,
        )
        assert orch._emitter is emitter

    @pytest.mark.asyncio()
    async def test_parallel_orchestrator_accepts_emitter(self):
        from triad.orchestrator import ParallelOrchestrator

        emitter = PipelineEventEmitter()
        orch = ParallelOrchestrator(
            task=self._make_task(),
            config=self._make_config(),
            registry=self._make_registry(),
            event_emitter=emitter,
        )
        assert orch._emitter is emitter

    @pytest.mark.asyncio()
    async def test_debate_orchestrator_accepts_emitter(self):
        from triad.orchestrator import DebateOrchestrator

        emitter = PipelineEventEmitter()
        orch = DebateOrchestrator(
            task=self._make_task(),
            config=self._make_config(),
            registry=self._make_registry(),
            event_emitter=emitter,
        )
        assert orch._emitter is emitter

    @pytest.mark.asyncio()
    async def test_run_pipeline_passes_emitter(self):
        from triad.orchestrator import run_pipeline
        from triad.schemas.pipeline import PipelineResult

        emitter = PipelineEventEmitter()
        config = self._make_config()
        config.persist_sessions = False
        task = self._make_task()

        mock_result = MagicMock(spec=PipelineResult)
        mock_result.session_id = ""
        mock_result.config = config

        with patch("triad.orchestrator.PipelineOrchestrator") as mock_cls:
            mock_orch = MagicMock()
            mock_orch.run = AsyncMock(return_value=mock_result)
            mock_cls.return_value = mock_orch

            await run_pipeline(
                task=task,
                config=config,
                registry=self._make_registry(),
                event_emitter=emitter,
            )

            # Verify emitter was passed to the orchestrator constructor
            args, kwargs = mock_cls.call_args
            # Emitter is 4th positional arg
            assert args[3] is emitter

    @pytest.mark.asyncio()
    async def test_none_emitter_no_error(self):
        """Orchestrator works fine without emitter (normal CLI path)."""
        from triad.orchestrator import PipelineOrchestrator

        orch = PipelineOrchestrator(
            task=self._make_task(),
            config=self._make_config(),
            registry=self._make_registry(),
        )
        assert orch._emitter is None
        # _emit should be a no-op
        await orch._emit("pipeline_started", mode="sequential")


# ══════════════════════════════════════════════════════════════════
# CLI Dashboard Command
# ══════════════════════════════════════════════════════════════════


class TestCLIDashboard:
    """CLI dashboard command tests."""

    def test_dashboard_help(self):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["dashboard", "--help"])
        assert result.exit_code == 0
        assert "dashboard" in result.output.lower()
        assert "port" in result.output.lower()


# ══════════════════════════════════════════════════════════════════
# Module Re-exports
# ══════════════════════════════════════════════════════════════════


class TestModuleExports:
    """Verify dashboard module re-exports."""

    def test_dashboard_module_exports(self):
        from triad.dashboard import EventType, PipelineEvent, PipelineEventEmitter

        assert EventType is not None
        assert PipelineEvent is not None
        assert PipelineEventEmitter is not None

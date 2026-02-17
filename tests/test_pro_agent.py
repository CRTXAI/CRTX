"""Tests for triad.pro.agent — ProAgent event forwarder."""

from __future__ import annotations

import asyncio
import json
from unittest.mock import MagicMock, patch

import pytest

from triad.dashboard.events import PipelineEvent, PipelineEventEmitter
from triad.pro.agent import ProAgent


class TestProAgentInit:
    """ProAgent initialization and configuration tests."""

    def test_from_config_returns_none_without_key(self):
        """ProAgent.from_config() returns None when no key is configured."""
        with patch.dict("os.environ", {}, clear=False):
            # Ensure TRIAD_PRO_KEY is not set
            import os
            os.environ.pop("TRIAD_PRO_KEY", None)
            agent = ProAgent.from_config()
            assert agent is None

    def test_from_config_returns_agent_with_key(self):
        """ProAgent.from_config() returns an agent when key is set."""
        with patch.dict("os.environ", {"TRIAD_PRO_KEY": "sk_triad_test123"}):
            agent = ProAgent.from_config()
            assert agent is not None
            assert isinstance(agent, ProAgent)

    def test_custom_api_url_from_env(self):
        """ProAgent picks up custom API URL from env var."""
        with patch.dict("os.environ", {
            "TRIAD_PRO_KEY": "sk_triad_test123",
            "TRIAD_PRO_URL": "https://custom.api.example.com/ingest",
        }):
            agent = ProAgent.from_config()
            assert agent is not None
            assert agent._api_url == "https://custom.api.example.com/ingest"


class TestProAgentListener:
    """ProAgent event listener tests."""

    def test_create_listener_returns_callable(self):
        """create_listener() returns a callable event handler."""
        agent = ProAgent(api_key="test")
        listener = agent.create_listener()
        assert callable(listener)

    def test_listener_queues_events(self):
        """Listener puts events into the internal queue."""
        agent = ProAgent(api_key="test")
        listener = agent.create_listener()

        event = PipelineEvent(type="stage_started", data={"stage": "architect"})
        listener(event)

        assert agent._queue.qsize() == 1

    def test_listener_captures_session_id(self):
        """Listener extracts session_id from pipeline_started events."""
        agent = ProAgent(api_key="test")
        listener = agent.create_listener()

        event = PipelineEvent(
            type="pipeline_started",
            data={"session_id": "abc-123", "task": "Build API"},
        )
        listener(event)

        assert agent._session_id == "abc-123"

    def test_listener_handles_multiple_events(self):
        """Listener queues multiple events correctly."""
        agent = ProAgent(api_key="test")
        listener = agent.create_listener()

        for i in range(5):
            event = PipelineEvent(type="stage_started", data={"stage": f"stage_{i}"})
            listener(event)

        assert agent._queue.qsize() == 5


class TestProAgentIntegration:
    """ProAgent integration with PipelineEventEmitter."""

    def test_registers_as_listener(self):
        """ProAgent can register with PipelineEventEmitter."""
        emitter = PipelineEventEmitter()
        agent = ProAgent(api_key="test")
        emitter.add_listener(agent.create_listener())

        assert len(emitter._listeners) == 1

    @pytest.mark.asyncio
    async def test_receives_emitted_events(self):
        """ProAgent receives events emitted by PipelineEventEmitter."""
        emitter = PipelineEventEmitter()
        agent = ProAgent(api_key="test")
        emitter.add_listener(agent.create_listener())

        await emitter.emit("pipeline_started", session_id="sess-1", task="test")
        await emitter.emit("stage_started", stage="architect")

        assert agent._queue.qsize() == 2
        assert agent._session_id == "sess-1"


class TestProAgentBatching:
    """ProAgent event batching and flush tests."""

    @pytest.mark.asyncio
    async def test_flush_drains_queue(self):
        """_flush() drains events from the queue."""
        agent = ProAgent(api_key="test")
        agent._session_id = "sess-1"

        # Add events to queue
        for i in range(3):
            agent._queue.put_nowait({
                "type": "stage_started",
                "timestamp": 1000 + i,
                "data": {"stage": f"stage_{i}"},
            })

        # Mock the HTTP send to prevent actual network calls
        with patch.object(agent, "_send_request") as mock_send:
            await agent._flush()

        assert agent._queue.qsize() == 0
        mock_send.assert_called_once()

        # Verify the request body
        call_args = mock_send.call_args[0][0]
        body = json.loads(call_args.data)
        assert body["session_id"] == "sess-1"
        assert len(body["events"]) == 3

    @pytest.mark.asyncio
    async def test_flush_noop_when_empty(self):
        """_flush() does nothing when queue is empty."""
        agent = ProAgent(api_key="test")

        with patch.object(agent, "_send_request") as mock_send:
            await agent._flush()

        mock_send.assert_not_called()

    @pytest.mark.asyncio
    async def test_graceful_degradation_on_network_error(self):
        """ProAgent doesn't raise when API is unreachable."""
        agent = ProAgent(api_key="test")
        agent._session_id = "sess-1"

        agent._queue.put_nowait({
            "type": "stage_started",
            "timestamp": 1000,
            "data": {},
        })

        # Simulate network error
        with patch.object(agent, "_send_request", side_effect=Exception("Connection refused")):
            # Should not raise
            await agent._flush()

        # Queue should be drained even on error
        assert agent._queue.qsize() == 0

    @pytest.mark.asyncio
    async def test_stop_flushes_remaining(self):
        """stop() flushes remaining events before stopping."""
        agent = ProAgent(api_key="test")
        agent._session_id = "sess-1"

        agent._queue.put_nowait({
            "type": "pipeline_completed",
            "timestamp": 2000,
            "data": {},
        })

        with patch.object(agent, "_send_request"):
            await agent.stop()

        assert agent._stopped is True
        assert agent._queue.qsize() == 0

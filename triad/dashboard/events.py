"""Pipeline event emitter for real-time dashboard updates.

Emits structured events during pipeline execution that the WebSocket
server broadcasts to connected clients. Events cover every pipeline
milestone: stage transitions, arbiter verdicts, consensus votes,
retries, and pipeline lifecycle.
"""

from __future__ import annotations

import asyncio
import logging
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from pydantic import BaseModel, Field

logger = logging.getLogger(__name__)


class EventType(StrEnum):
    """Types of pipeline events emitted to the dashboard."""

    PIPELINE_STARTED = "pipeline_started"
    STAGE_STARTED = "stage_started"
    STAGE_COMPLETED = "stage_completed"
    ARBITER_STARTED = "arbiter_started"
    ARBITER_VERDICT = "arbiter_verdict"
    SUGGESTION_FILED = "suggestion_filed"
    CONSENSUS_VOTE = "consensus_vote"
    RETRY_TRIGGERED = "retry_triggered"
    PIPELINE_COMPLETED = "pipeline_completed"
    PIPELINE_HALTED = "pipeline_halted"
    ERROR = "error"


class PipelineEvent(BaseModel):
    """A single pipeline event for dashboard consumption."""

    type: EventType = Field(description="Event type")
    timestamp: float = Field(
        default_factory=time.time,
        description="Unix timestamp when the event occurred",
    )
    data: dict[str, Any] = Field(
        default_factory=dict,
        description="Event payload — varies by event type",
    )


# Type alias for event listener callbacks
EventListener = Callable[[PipelineEvent], Any]


class PipelineEventEmitter:
    """Broadcasts pipeline events to registered listeners.

    Listeners can be sync or async callables. The emitter is passed into
    orchestrator classes as an optional dependency. When no emitter is
    provided (normal CLI usage), all emit calls are skipped — zero overhead.
    """

    def __init__(self) -> None:
        self._listeners: list[EventListener] = []
        self._history: list[PipelineEvent] = []

    @property
    def history(self) -> list[PipelineEvent]:
        """All events emitted so far (for late-connecting clients)."""
        return list(self._history)

    def add_listener(self, listener: EventListener) -> None:
        """Register a listener to receive pipeline events."""
        self._listeners.append(listener)

    def remove_listener(self, listener: EventListener) -> None:
        """Remove a previously registered listener."""
        self._listeners = [ln for ln in self._listeners if ln is not listener]

    async def emit(self, event_type: EventType, **data: Any) -> None:
        """Emit a pipeline event to all registered listeners.

        Creates a PipelineEvent and dispatches it to every listener.
        Sync listeners are called directly; async listeners are awaited.
        Listener exceptions are logged but never propagate.
        """
        event = PipelineEvent(type=event_type, data=data)
        self._history.append(event)

        for listener in self._listeners:
            try:
                result = listener(event)
                if asyncio.iscoroutine(result):
                    await result
            except Exception:
                logger.exception("Event listener error for %s", event_type)

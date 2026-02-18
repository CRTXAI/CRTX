"""Real-time pipeline dashboard — optional dependency.

Provides a FastAPI WebSocket server that broadcasts pipeline events
to a React-based browser UI. Install with: pip install crtx[dashboard]
"""

from triad.dashboard.events import EventType, PipelineEvent, PipelineEventEmitter

__all__ = [
    "EventType",
    "PipelineEvent",
    "PipelineEventEmitter",
]

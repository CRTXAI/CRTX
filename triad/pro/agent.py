"""ProAgent — lightweight event forwarder for Triad Pro dashboard.

Registers as a PipelineEventEmitter listener and batches events
for delivery to the Pro ingestion API. Uses only stdlib (no new
dependencies). Graceful degradation: if the API is unreachable,
events are dropped silently — the pipeline is never blocked.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import urllib.error
import urllib.request
from collections.abc import Callable
from typing import Any

logger = logging.getLogger(__name__)

# Default API endpoint — can be overridden via env var
_DEFAULT_API_URL = "https://api.triad-orchestrator.com/api/v1/events/ingest"

# Batch window: collect events for this many seconds before flushing
_BATCH_WINDOW_S = 0.1

# Maximum events per batch (safety cap)
_MAX_BATCH_SIZE = 200


class ProAgent:
    """Forwards pipeline events to the Triad Pro ingestion API.

    Usage:
        pro = ProAgent.from_config()
        if pro:
            emitter.add_listener(pro.create_listener())
    """

    def __init__(self, api_key: str, api_url: str | None = None) -> None:
        self._api_key = api_key
        self._api_url = api_url or os.environ.get("TRIAD_PRO_URL", _DEFAULT_API_URL)
        self._session_id: str | None = None
        self._queue: asyncio.Queue[dict[str, Any]] = asyncio.Queue()
        self._flush_task: asyncio.Task[None] | None = None
        self._stopped = False

    @classmethod
    def from_config(cls) -> ProAgent | None:
        """Create a ProAgent if a Pro API key is configured.

        Returns None if no key is found (no-op).
        """
        from triad.keys import load_keys_env

        load_keys_env()
        key = os.environ.get("TRIAD_PRO_KEY", "")
        if not key:
            return None
        return cls(api_key=key)

    def create_listener(self) -> Callable[..., Any]:
        """Return an event listener callback for PipelineEventEmitter."""

        def on_event(event: Any) -> None:
            payload = {
                "type": str(event.type),
                "timestamp": event.timestamp,
                "data": event.data,
            }

            # Capture session_id from pipeline_started events
            if event.type == "pipeline_started":
                self._session_id = event.data.get("session_id")

            self._queue.put_nowait(payload)
            self._ensure_flush_task()

        return on_event

    def _ensure_flush_task(self) -> None:
        """Start the background flush loop if not already running."""
        if self._flush_task is None or self._flush_task.done():
            try:
                loop = asyncio.get_running_loop()
                self._flush_task = loop.create_task(self._flush_loop())
            except RuntimeError:
                # No running event loop — skip background flushing
                pass

    async def _flush_loop(self) -> None:
        """Batch events and send them periodically."""
        while not self._stopped:
            await asyncio.sleep(_BATCH_WINDOW_S)
            await self._flush()

    async def _flush(self) -> None:
        """Drain the queue and POST events to the ingestion API."""
        batch: list[dict[str, Any]] = []
        while not self._queue.empty() and len(batch) < _MAX_BATCH_SIZE:
            try:
                batch.append(self._queue.get_nowait())
            except asyncio.QueueEmpty:
                break

        if not batch:
            return

        body = json.dumps({
            "session_id": self._session_id or "",
            "events": batch,
        }).encode("utf-8")

        req = urllib.request.Request(
            self._api_url,
            data=body,
            headers={
                "Content-Type": "application/json",
                "Authorization": f"Bearer {self._api_key}",
            },
            method="POST",
        )

        try:
            # Run blocking HTTP call in thread pool to avoid blocking the loop
            loop = asyncio.get_running_loop()
            await loop.run_in_executor(None, self._send_request, req)
        except Exception:
            logger.debug("ProAgent: failed to send %d events (API unreachable)", len(batch))

    @staticmethod
    def _send_request(req: urllib.request.Request) -> None:
        """Synchronous HTTP send (runs in executor)."""
        try:
            with urllib.request.urlopen(req, timeout=5) as resp:
                resp.read()
        except (urllib.error.URLError, OSError, TimeoutError):
            pass  # Graceful degradation — drop silently

    async def stop(self) -> None:
        """Flush remaining events and stop the background task."""
        self._stopped = True
        await self._flush()
        if self._flush_task and not self._flush_task.done():
            self._flush_task.cancel()

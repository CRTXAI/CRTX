"""FastAPI WebSocket server for the real-time pipeline dashboard.

Serves the React UI as a static HTML file and broadcasts pipeline
events over WebSocket to all connected clients. Also provides REST
endpoints for session history, model registry, and configuration.

Requires the 'dashboard' optional dependency group:
    pip install crtx[dashboard]
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any

from triad.dashboard.events import PipelineEvent, PipelineEventEmitter

logger = logging.getLogger(__name__)

# Resolve the static directory relative to this file
_STATIC_DIR = Path(__file__).parent / "static"

# Global emitter instance shared between the server and pipeline runs
_emitter = PipelineEventEmitter()


def get_emitter() -> PipelineEventEmitter:
    """Return the global PipelineEventEmitter for this server process."""
    return _emitter


def create_app() -> Any:
    """Create and configure the FastAPI application.

    Returns the app instance. FastAPI is imported inside this function
    so the module can be imported without dashboard deps installed.
    """
    try:
        from fastapi import FastAPI, WebSocket, WebSocketDisconnect
        from fastapi.middleware.cors import CORSMiddleware
        from fastapi.responses import FileResponse, HTMLResponse
    except ImportError as exc:
        raise ImportError(
            "Dashboard requires extra dependencies. "
            "Install with: pip install crtx[dashboard]"
        ) from exc

    app = FastAPI(
        title="Triad Dashboard",
        description="Real-time pipeline visualization",
    )

    app.add_middleware(
        CORSMiddleware,
        allow_origins=["*"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Track connected WebSocket clients
    connected_clients: list[WebSocket] = []

    async def _broadcast(event: PipelineEvent) -> None:
        """Broadcast a pipeline event to all connected WebSocket clients."""
        payload = json.dumps(event.model_dump(), default=str)
        disconnected: list[WebSocket] = []
        for ws in connected_clients:
            try:
                await ws.send_text(payload)
            except Exception:
                disconnected.append(ws)
        for ws in disconnected:
            connected_clients.remove(ws)

    # Register the broadcast callback on the global emitter
    _emitter.add_listener(_broadcast)

    # ── Static / UI ──────────────────────────────────────────────

    @app.get("/", response_class=HTMLResponse)
    async def index() -> FileResponse:
        """Serve the React dashboard."""
        index_path = _STATIC_DIR / "index.html"
        if not index_path.exists():
            return HTMLResponse("<h1>Dashboard not found</h1>", status_code=404)
        return FileResponse(index_path, media_type="text/html")

    # ── WebSocket ────────────────────────────────────────────────

    @app.websocket("/ws")
    async def websocket_endpoint(ws: WebSocket) -> None:
        """WebSocket endpoint for real-time event streaming.

        On connect, sends all historical events so the client can
        reconstruct the current pipeline state. Then streams new
        events as they occur.
        """
        await ws.accept()
        connected_clients.append(ws)
        logger.info("Dashboard client connected (%d total)", len(connected_clients))

        try:
            # Send event history for late-connecting clients
            for event in _emitter.history:
                await ws.send_text(
                    json.dumps(event.model_dump(), default=str)
                )

            # Keep connection alive, listen for client messages
            while True:
                await ws.receive_text()
        except WebSocketDisconnect:
            pass
        finally:
            if ws in connected_clients:
                connected_clients.remove(ws)
            logger.info(
                "Dashboard client disconnected (%d remaining)",
                len(connected_clients),
            )

    # ── REST API ─────────────────────────────────────────────────

    @app.get("/api/sessions")
    async def list_sessions() -> list[dict]:
        """List recent pipeline sessions."""
        try:
            from triad.persistence.database import close_db, init_db
            from triad.persistence.session import SessionStore
            from triad.schemas.session import SessionQuery

            db = await init_db()
            store = SessionStore(db)
            summaries = await store.list_sessions(SessionQuery(limit=50))
            await close_db(db)
            return [s.model_dump() for s in summaries]
        except Exception:
            logger.exception("Failed to list sessions")
            return []

    @app.get("/api/sessions/{session_id}")
    async def get_session(session_id: str) -> dict:
        """Get full details for a specific session."""
        try:
            from triad.persistence.database import close_db, init_db
            from triad.persistence.session import SessionStore

            db = await init_db()
            store = SessionStore(db)
            record = await store.get_session(session_id)
            await close_db(db)
            if record is None:
                return {"error": "Session not found"}
            return record.model_dump()
        except Exception:
            logger.exception("Failed to get session %s", session_id)
            return {"error": "Failed to retrieve session"}

    @app.get("/api/models")
    async def list_models() -> list[dict]:
        """List registered models with fitness scores."""
        try:
            from triad.providers.registry import load_models

            registry = load_models()
            return [
                {
                    "key": key,
                    "provider": cfg.provider,
                    "model": cfg.model,
                    "display_name": cfg.display_name,
                    "context_window": cfg.context_window,
                    "cost_input": cfg.cost_input,
                    "cost_output": cfg.cost_output,
                    "fitness": cfg.fitness.model_dump(),
                }
                for key, cfg in registry.items()
            ]
        except Exception:
            logger.exception("Failed to list models")
            return []

    @app.get("/api/config")
    async def get_config() -> dict:
        """Return current pipeline configuration."""
        try:
            from triad.providers.registry import load_pipeline_config

            config = load_pipeline_config()
            return config.model_dump()
        except Exception:
            logger.exception("Failed to load config")
            return {"error": "Failed to load configuration"}

    return app

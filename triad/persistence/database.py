"""SQLite database layer for session persistence.

Manages the SQLite database connection, schema creation, and low-level
CRUD operations. Uses aiosqlite for async access with WAL mode for
concurrent read performance.
"""

from __future__ import annotations

import logging
from pathlib import Path

import aiosqlite

logger = logging.getLogger(__name__)

# SQL schema for the sessions database
_SCHEMA = """
CREATE TABLE IF NOT EXISTS sessions (
    session_id   TEXT PRIMARY KEY,
    task_json    TEXT NOT NULL,
    config_json  TEXT NOT NULL,
    started_at   TEXT NOT NULL,
    completed_at TEXT,
    success      INTEGER NOT NULL DEFAULT 0,
    halted       INTEGER NOT NULL DEFAULT 0,
    halt_reason  TEXT NOT NULL DEFAULT '',
    total_cost   REAL NOT NULL DEFAULT 0.0,
    total_tokens INTEGER NOT NULL DEFAULT 0,
    duration_seconds REAL NOT NULL DEFAULT 0.0,
    pipeline_mode TEXT NOT NULL DEFAULT 'sequential'
);

CREATE TABLE IF NOT EXISTS stages (
    id          INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id  TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    stage       TEXT NOT NULL,
    model_key   TEXT NOT NULL DEFAULT '',
    model_id    TEXT NOT NULL DEFAULT '',
    content     TEXT NOT NULL DEFAULT '',
    confidence  REAL NOT NULL DEFAULT 0.0,
    cost        REAL NOT NULL DEFAULT 0.0,
    tokens      INTEGER NOT NULL DEFAULT 0,
    timestamp   TEXT NOT NULL DEFAULT ''
);

CREATE TABLE IF NOT EXISTS arbiter_reviews (
    id              INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id      TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    stage_reviewed  TEXT NOT NULL,
    reviewed_model  TEXT NOT NULL,
    arbiter_model   TEXT NOT NULL,
    verdict         TEXT NOT NULL,
    issues_json     TEXT NOT NULL DEFAULT '[]',
    alternatives_json TEXT NOT NULL DEFAULT '[]',
    confidence      REAL NOT NULL DEFAULT 0.0,
    reasoning       TEXT NOT NULL DEFAULT '',
    token_cost      REAL NOT NULL DEFAULT 0.0
);

CREATE TABLE IF NOT EXISTS routing_decisions (
    id             INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id     TEXT NOT NULL REFERENCES sessions(session_id) ON DELETE CASCADE,
    model_key      TEXT NOT NULL,
    model_id       TEXT NOT NULL,
    role           TEXT NOT NULL,
    strategy       TEXT NOT NULL,
    rationale      TEXT NOT NULL DEFAULT '',
    fitness_score  REAL NOT NULL DEFAULT 0.0,
    estimated_cost REAL NOT NULL DEFAULT 0.0
);

CREATE INDEX IF NOT EXISTS idx_stages_session ON stages(session_id);
CREATE INDEX IF NOT EXISTS idx_arbiter_session ON arbiter_reviews(session_id);
CREATE INDEX IF NOT EXISTS idx_routing_session ON routing_decisions(session_id);
CREATE INDEX IF NOT EXISTS idx_sessions_started ON sessions(started_at);
"""


async def init_db(db_path: str) -> aiosqlite.Connection:
    """Initialize the database connection and create tables if needed.

    Creates parent directories if they don't exist, enables WAL mode
    and foreign keys, then runs the schema DDL.

    Args:
        db_path: Path to the SQLite database file. Supports ~ expansion.

    Returns:
        An open aiosqlite connection ready for use.
    """
    resolved = Path(db_path).expanduser()
    resolved.parent.mkdir(parents=True, exist_ok=True)

    db = await aiosqlite.connect(str(resolved))
    await db.execute("PRAGMA journal_mode=WAL")
    await db.execute("PRAGMA foreign_keys=ON")
    await db.executescript(_SCHEMA)
    await db.commit()

    logger.info("Session database initialized at %s", resolved)
    return db


async def close_db(db: aiosqlite.Connection) -> None:
    """Close the database connection."""
    await db.close()

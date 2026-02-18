"""Session store for saving, retrieving, listing, and deleting sessions.

Provides the SessionStore class that wraps low-level database operations
with Pydantic schema serialization/deserialization.
"""

from __future__ import annotations

import json
import logging
from datetime import datetime

import aiosqlite

from triad.schemas.arbiter import ArbiterReview
from triad.schemas.routing import RoutingDecision
from triad.schemas.session import (
    SessionQuery,
    SessionRecord,
    SessionSummary,
    StageRecord,
)

logger = logging.getLogger(__name__)


class SessionStore:
    """Persistent session store backed by SQLite.

    All methods are async and operate on an aiosqlite connection
    initialized by database.init_db().
    """

    def __init__(self, db: aiosqlite.Connection) -> None:
        self._db = db

    async def save_session(self, record: SessionRecord) -> None:
        """Save a complete session record to the database.

        Inserts the session row and all related stage, arbiter review,
        and routing decision rows in a single transaction.
        """
        await self._db.execute(
            """
            INSERT OR REPLACE INTO sessions
                (session_id, task_json, config_json, started_at, completed_at,
                 success, halted, halt_reason, total_cost, total_tokens,
                 duration_seconds, pipeline_mode)
            VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                record.session_id,
                record.task.model_dump_json(),
                record.config.model_dump_json(),
                record.started_at.isoformat(),
                record.completed_at.isoformat() if record.completed_at else None,
                int(record.success),
                int(record.halted),
                record.halt_reason,
                record.total_cost,
                record.total_tokens,
                record.duration_seconds,
                record.pipeline_mode,
            ),
        )

        # Clear existing child rows (for upsert)
        for table in ("stages", "arbiter_reviews", "routing_decisions"):
            await self._db.execute(
                f"DELETE FROM {table} WHERE session_id = ?",  # noqa: S608
                (record.session_id,),
            )

        # Insert stages
        for stage in record.stages:
            await self._db.execute(
                """
                INSERT INTO stages
                    (session_id, stage, model_key, model_id, content,
                     confidence, cost, tokens, timestamp)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    stage.stage,
                    stage.model_key,
                    stage.model_id,
                    stage.content,
                    stage.confidence,
                    stage.cost,
                    stage.tokens,
                    stage.timestamp,
                ),
            )

        # Insert arbiter reviews
        for review in record.arbiter_reviews:
            await self._db.execute(
                """
                INSERT INTO arbiter_reviews
                    (session_id, stage_reviewed, reviewed_model, arbiter_model,
                     verdict, issues_json, alternatives_json, confidence,
                     reasoning, token_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    review.stage_reviewed.value,
                    review.reviewed_model,
                    review.arbiter_model,
                    review.verdict.value,
                    json.dumps([i.model_dump() for i in review.issues]),
                    json.dumps([a.model_dump() for a in review.alternatives]),
                    review.confidence,
                    review.reasoning,
                    review.token_cost,
                ),
            )

        # Insert routing decisions
        for decision in record.routing_decisions:
            await self._db.execute(
                """
                INSERT INTO routing_decisions
                    (session_id, model_key, model_id, role, strategy,
                     rationale, fitness_score, estimated_cost)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    record.session_id,
                    decision.model_key,
                    decision.model_id,
                    decision.role.value,
                    decision.strategy.value,
                    decision.rationale,
                    decision.fitness_score,
                    decision.estimated_cost,
                ),
            )

        await self._db.commit()
        logger.info("Saved session %s", record.session_id)

    async def get_session(self, session_id: str) -> SessionRecord | None:
        """Retrieve a full session record by ID or unique ID prefix.

        Tries an exact match first.  If that fails and the input is at
        least 4 characters, falls back to a prefix match.  Returns None
        when no match is found or the prefix is ambiguous (matches more
        than one session).
        """
        self._db.row_factory = aiosqlite.Row
        async with self._db.execute(
            "SELECT * FROM sessions WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            row = await cursor.fetchone()

        # Fall back to prefix match
        if not row and len(session_id) >= 4:
            async with self._db.execute(
                "SELECT * FROM sessions WHERE session_id LIKE ?"
                " ORDER BY started_at DESC LIMIT 2",
                (session_id + "%",),
            ) as cursor:
                rows = await cursor.fetchall()
            if len(rows) == 1:
                row = rows[0]

        if not row:
            return None

        return await self._row_to_record(row)

    async def list_sessions(self, query: SessionQuery) -> list[SessionSummary]:
        """List sessions matching the given query filters.

        Returns lightweight SessionSummary objects sorted by started_at
        descending (most recent first).
        """
        conditions: list[str] = []
        params: list[object] = []

        if query.task_filter:
            conditions.append("task_json LIKE ?")
            params.append(f"%{query.task_filter}%")

        if query.model_filter:
            conditions.append(
                "session_id IN ("
                "SELECT DISTINCT session_id FROM stages WHERE model_key = ?)"
            )
            params.append(query.model_filter)

        if query.verdict_filter:
            conditions.append(
                "session_id IN ("
                "SELECT DISTINCT session_id FROM arbiter_reviews WHERE verdict = ?)"
            )
            params.append(query.verdict_filter)

        if query.min_cost is not None:
            conditions.append("total_cost >= ?")
            params.append(query.min_cost)

        if query.max_cost is not None:
            conditions.append("total_cost <= ?")
            params.append(query.max_cost)

        if query.since:
            conditions.append("started_at >= ?")
            params.append(query.since)

        where = f"WHERE {' AND '.join(conditions)}" if conditions else ""
        sql = f"""
            SELECT * FROM sessions
            {where}
            ORDER BY started_at DESC
            LIMIT ? OFFSET ?
        """
        params.extend([query.limit, query.offset])

        self._db.row_factory = aiosqlite.Row
        summaries: list[SessionSummary] = []
        async with self._db.execute(sql, params) as cursor:
            async for row in cursor:
                summary = await self._row_to_summary(row)
                summaries.append(summary)

        return summaries

    async def resolve_session_id(self, prefix: str) -> str | None:
        """Resolve a session ID prefix to a full session ID.

        Returns the full ID if exactly one match is found, None otherwise.
        Accepts full IDs as well (exact match always wins).
        """
        self._db.row_factory = aiosqlite.Row
        # Exact match first
        async with self._db.execute(
            "SELECT session_id FROM sessions WHERE session_id = ?",
            (prefix,),
        ) as cursor:
            row = await cursor.fetchone()
        if row:
            return row["session_id"]
        # Prefix match
        if len(prefix) >= 4:
            async with self._db.execute(
                "SELECT session_id FROM sessions WHERE session_id LIKE ?"
                " LIMIT 2",
                (prefix + "%",),
            ) as cursor:
                rows = await cursor.fetchall()
            if len(rows) == 1:
                return rows[0]["session_id"]
        return None

    async def delete_session(self, session_id: str) -> bool:
        """Delete a session and all related records.

        Supports both full IDs and unique prefixes (>= 4 chars).
        Returns True if a session was deleted, False if it didn't exist
        or the prefix was ambiguous.
        """
        full_id = await self.resolve_session_id(session_id)
        if not full_id:
            return False
        cursor = await self._db.execute(
            "DELETE FROM sessions WHERE session_id = ?",
            (full_id,),
        )
        await self._db.commit()
        deleted = cursor.rowcount > 0
        if deleted:
            logger.info("Deleted session %s", full_id)
        return deleted

    async def export_session(self, session_id: str) -> dict | None:
        """Export a session as a JSON-serializable dictionary.

        Returns None if the session does not exist.
        """
        record = await self.get_session(session_id)
        if not record:
            return None
        return record.model_dump(mode="json")

    async def _row_to_record(self, row: aiosqlite.Row) -> SessionRecord:
        """Convert a sessions row + child rows into a SessionRecord."""
        from triad.schemas.pipeline import PipelineConfig, TaskSpec

        session_id = row["session_id"]

        # Load stages
        stages: list[StageRecord] = []
        async with self._db.execute(
            "SELECT * FROM stages WHERE session_id = ? ORDER BY id",
            (session_id,),
        ) as cursor:
            async for srow in cursor:
                stages.append(StageRecord(
                    stage=srow["stage"],
                    model_key=srow["model_key"],
                    model_id=srow["model_id"],
                    content=srow["content"],
                    confidence=srow["confidence"],
                    cost=srow["cost"],
                    tokens=srow["tokens"],
                    timestamp=srow["timestamp"],
                ))

        # Load arbiter reviews
        reviews: list[ArbiterReview] = []
        async with self._db.execute(
            "SELECT * FROM arbiter_reviews WHERE session_id = ? ORDER BY id",
            (session_id,),
        ) as cursor:
            async for arow in cursor:
                reviews.append(ArbiterReview(
                    stage_reviewed=arow["stage_reviewed"],
                    reviewed_model=arow["reviewed_model"],
                    arbiter_model=arow["arbiter_model"],
                    verdict=arow["verdict"],
                    issues=json.loads(arow["issues_json"]),
                    alternatives=json.loads(arow["alternatives_json"]),
                    confidence=arow["confidence"],
                    reasoning=arow["reasoning"],
                    token_cost=arow["token_cost"],
                ))

        # Load routing decisions
        decisions: list[RoutingDecision] = []
        async with self._db.execute(
            "SELECT * FROM routing_decisions WHERE session_id = ? ORDER BY id",
            (session_id,),
        ) as cursor:
            async for drow in cursor:
                decisions.append(RoutingDecision(
                    model_key=drow["model_key"],
                    model_id=drow["model_id"],
                    role=drow["role"],
                    strategy=drow["strategy"],
                    rationale=drow["rationale"],
                    fitness_score=drow["fitness_score"],
                    estimated_cost=drow["estimated_cost"],
                ))

        completed_at = (
            datetime.fromisoformat(row["completed_at"])
            if row["completed_at"]
            else None
        )

        return SessionRecord(
            session_id=session_id,
            task=TaskSpec.model_validate_json(row["task_json"]),
            config=PipelineConfig.model_validate_json(row["config_json"]),
            stages=stages,
            arbiter_reviews=reviews,
            routing_decisions=decisions,
            started_at=datetime.fromisoformat(row["started_at"]),
            completed_at=completed_at,
            success=bool(row["success"]),
            halted=bool(row["halted"]),
            halt_reason=row["halt_reason"],
            total_cost=row["total_cost"],
            total_tokens=row["total_tokens"],
            duration_seconds=row["duration_seconds"],
            pipeline_mode=row["pipeline_mode"],
        )

    async def _row_to_summary(self, row: aiosqlite.Row) -> SessionSummary:
        """Convert a sessions row into a SessionSummary with aggregated data."""
        session_id = row["session_id"]

        # Task preview: first 100 chars of the task description
        task_data = json.loads(row["task_json"])
        task_text = task_data.get("task", "")
        task_preview = task_text[:100]

        # Count distinct models
        async with self._db.execute(
            "SELECT COUNT(DISTINCT model_key) FROM stages WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            model_row = await cursor.fetchone()
            model_count = model_row[0] if model_row else 0

        # Count stages
        async with self._db.execute(
            "SELECT COUNT(*) FROM stages WHERE session_id = ?",
            (session_id,),
        ) as cursor:
            stage_row = await cursor.fetchone()
            stage_count = stage_row[0] if stage_row else 0

        # Arbiter verdict summary
        async with self._db.execute(
            "SELECT verdict, COUNT(*) FROM arbiter_reviews "
            "WHERE session_id = ? GROUP BY verdict",
            (session_id,),
        ) as cursor:
            verdict_parts = []
            async for vrow in cursor:
                verdict_parts.append(f"{vrow[1]} {vrow[0].upper()}")
            verdict_summary = ", ".join(verdict_parts)

        return SessionSummary(
            session_id=session_id,
            task_preview=task_preview,
            pipeline_mode=row["pipeline_mode"],
            started_at=datetime.fromisoformat(row["started_at"]),
            success=bool(row["success"]),
            halted=bool(row["halted"]),
            total_cost=row["total_cost"],
            duration_seconds=row["duration_seconds"],
            model_count=model_count,
            stage_count=stage_count,
            arbiter_verdict_summary=verdict_summary,
        )

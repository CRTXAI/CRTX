"""Tests for Session Persistence (Day 9).

Covers database initialization, SessionStore CRUD operations,
export formatters (JSON/Markdown), query filtering, orchestrator
integration, and schema validation.
"""

from __future__ import annotations

import json
from datetime import UTC, datetime
from unittest.mock import AsyncMock, patch

import pytest

from triad.persistence.database import close_db, init_db
from triad.persistence.export import export_json, export_markdown
from triad.persistence.session import SessionStore
from triad.schemas.arbiter import (
    ArbiterReview,
    Issue,
    IssueCategory,
    Severity,
    Verdict,
)
from triad.schemas.pipeline import (
    ModelConfig,
    PipelineConfig,
    PipelineResult,
    RoleFitness,
    TaskSpec,
)
from triad.schemas.routing import RoutingDecision, RoutingStrategy
from triad.schemas.session import (
    SessionQuery,
    SessionRecord,
    SessionSummary,
    StageRecord,
)

# Patch targets
_PROVIDER = "triad.orchestrator.LiteLLMProvider"
_INIT_DB = "triad.orchestrator.init_db"
_CLOSE_DB = "triad.orchestrator.close_db"
_SESSION_STORE = "triad.orchestrator.SessionStore"


# ── Factories ──────────────────────────────────────────────────────


def _make_task() -> TaskSpec:
    return TaskSpec(task="Build a REST API", context="Python FastAPI")


def _make_config(**overrides) -> PipelineConfig:
    defaults = {
        "persist_sessions": True,
        "session_db_path": ":memory:",
    }
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_stage_record(
    stage: str = "architect",
    model_key: str = "claude",
    **overrides,
) -> StageRecord:
    defaults = {
        "stage": stage,
        "model_key": model_key,
        "model_id": "claude-opus-4-5-20250929",
        "content": f"Output from {stage}",
        "confidence": 0.85,
        "cost": 0.05,
        "tokens": 5000,
        "timestamp": "2026-02-16T10:00:00+00:00",
    }
    defaults.update(overrides)
    return StageRecord(**defaults)


def _make_review(
    stage: str = "architect",
    verdict: str = "approve",
) -> ArbiterReview:
    return ArbiterReview(
        stage_reviewed=stage,
        reviewed_model="claude-opus-4-5-20250929",
        arbiter_model="gpt-4o",
        verdict=verdict,
        issues=[
            Issue(
                severity=Severity.WARNING,
                category=IssueCategory.PATTERN,
                location="api/main.py:10-20",
                description="Missing error handling",
                suggestion="Add try/except blocks",
            ),
        ],
        confidence=0.9,
        reasoning="Output looks good overall.",
        token_cost=0.02,
    )


def _make_routing_decision(
    role: str = "architect",
    model_key: str = "claude",
) -> RoutingDecision:
    return RoutingDecision(
        model_key=model_key,
        model_id="claude-opus-4-5-20250929",
        role=role,
        strategy=RoutingStrategy.HYBRID,
        rationale="Best fitness for role",
        fitness_score=0.9,
        estimated_cost=0.05,
    )


def _make_session_record(
    session_id: str = "test-session-001",
    success: bool = True,
    halted: bool = False,
) -> SessionRecord:
    return SessionRecord(
        session_id=session_id,
        task=_make_task(),
        config=_make_config(),
        stages=[
            _make_stage_record("architect", "claude"),
            _make_stage_record("implement", "gpt4"),
            _make_stage_record("refactor", "claude"),
            _make_stage_record("verify", "gemini"),
        ],
        arbiter_reviews=[_make_review("architect", "approve")],
        routing_decisions=[
            _make_routing_decision("architect", "claude"),
            _make_routing_decision("implement", "gpt4"),
        ],
        started_at=datetime(2026, 2, 16, 10, 0, 0, tzinfo=UTC),
        completed_at=datetime(2026, 2, 16, 10, 2, 30, tzinfo=UTC),
        success=success,
        halted=halted,
        halt_reason="Critical issue" if halted else "",
        total_cost=0.25,
        total_tokens=20000,
        duration_seconds=150.0,
        pipeline_mode="sequential",
    )


# ── Database Initialization Tests ─────────────────────────────────


@pytest.mark.asyncio
async def test_init_db_creates_tables(tmp_path):
    """init_db creates all 4 tables in a new database."""
    db_path = str(tmp_path / "test.db")
    db = await init_db(db_path)

    async with db.execute(
        "SELECT name FROM sqlite_master WHERE type='table' ORDER BY name"
    ) as cursor:
        tables = [row[0] for row in await cursor.fetchall()]

    assert "sessions" in tables
    assert "stages" in tables
    assert "arbiter_reviews" in tables
    assert "routing_decisions" in tables

    await close_db(db)


@pytest.mark.asyncio
async def test_init_db_wal_mode(tmp_path):
    """init_db enables WAL journal mode."""
    db_path = str(tmp_path / "test.db")
    db = await init_db(db_path)

    async with db.execute("PRAGMA journal_mode") as cursor:
        row = await cursor.fetchone()
        assert row[0] == "wal"

    await close_db(db)


@pytest.mark.asyncio
async def test_init_db_foreign_keys(tmp_path):
    """init_db enables foreign key enforcement."""
    db_path = str(tmp_path / "test.db")
    db = await init_db(db_path)

    async with db.execute("PRAGMA foreign_keys") as cursor:
        row = await cursor.fetchone()
        assert row[0] == 1

    await close_db(db)


@pytest.mark.asyncio
async def test_init_db_creates_parent_dirs(tmp_path):
    """init_db creates parent directories if they don't exist."""
    db_path = str(tmp_path / "nested" / "deep" / "test.db")
    db = await init_db(db_path)
    assert db is not None
    await close_db(db)


@pytest.mark.asyncio
async def test_init_db_idempotent(tmp_path):
    """init_db can be called multiple times on the same database."""
    db_path = str(tmp_path / "test.db")
    db1 = await init_db(db_path)
    await close_db(db1)

    db2 = await init_db(db_path)
    # Should succeed without errors
    async with db2.execute(
        "SELECT name FROM sqlite_master WHERE type='table'"
    ) as cursor:
        tables = [row[0] for row in await cursor.fetchall()]
    assert "sessions" in tables
    await close_db(db2)


# ── SessionStore: save_session Tests ──────────────────────────────


@pytest.mark.asyncio
async def test_save_session_inserts_record(tmp_path):
    """save_session inserts a full session record with all child rows."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    record = _make_session_record()

    await store.save_session(record)

    # Verify session row
    async with db.execute("SELECT COUNT(*) FROM sessions") as cursor:
        assert (await cursor.fetchone())[0] == 1

    # Verify stage rows
    async with db.execute("SELECT COUNT(*) FROM stages") as cursor:
        assert (await cursor.fetchone())[0] == 4

    # Verify arbiter review rows
    async with db.execute("SELECT COUNT(*) FROM arbiter_reviews") as cursor:
        assert (await cursor.fetchone())[0] == 1

    # Verify routing decision rows
    async with db.execute("SELECT COUNT(*) FROM routing_decisions") as cursor:
        assert (await cursor.fetchone())[0] == 2

    await close_db(db)


@pytest.mark.asyncio
async def test_save_session_upsert(tmp_path):
    """save_session replaces existing record on re-save (upsert)."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    record = _make_session_record()

    await store.save_session(record)

    # Modify and re-save
    record.total_cost = 0.50
    await store.save_session(record)

    # Should still be 1 session
    async with db.execute("SELECT COUNT(*) FROM sessions") as cursor:
        assert (await cursor.fetchone())[0] == 1

    # Verify updated cost
    async with db.execute(
        "SELECT total_cost FROM sessions WHERE session_id = ?",
        (record.session_id,),
    ) as cursor:
        row = await cursor.fetchone()
        assert row[0] == pytest.approx(0.50)

    await close_db(db)


# ── SessionStore: get_session Tests ───────────────────────────────


@pytest.mark.asyncio
async def test_get_session_returns_full_record(tmp_path):
    """get_session returns a complete SessionRecord with all child data."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    original = _make_session_record()

    await store.save_session(original)
    retrieved = await store.get_session(original.session_id)

    assert retrieved is not None
    assert retrieved.session_id == original.session_id
    assert retrieved.task.task == "Build a REST API"
    assert retrieved.success is True
    assert retrieved.total_cost == pytest.approx(0.25)
    assert retrieved.total_tokens == 20000
    assert retrieved.duration_seconds == pytest.approx(150.0)
    assert retrieved.pipeline_mode == "sequential"

    # Stages
    assert len(retrieved.stages) == 4
    assert retrieved.stages[0].stage == "architect"
    assert retrieved.stages[0].model_key == "claude"

    # Arbiter reviews
    assert len(retrieved.arbiter_reviews) == 1
    assert retrieved.arbiter_reviews[0].verdict == Verdict.APPROVE
    assert len(retrieved.arbiter_reviews[0].issues) == 1

    # Routing decisions
    assert len(retrieved.routing_decisions) == 2
    assert retrieved.routing_decisions[0].model_key == "claude"

    await close_db(db)


@pytest.mark.asyncio
async def test_get_session_nonexistent(tmp_path):
    """get_session returns None for a non-existent session ID."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    result = await store.get_session("nonexistent-id")
    assert result is None

    await close_db(db)


@pytest.mark.asyncio
async def test_get_session_preserves_arbiter_issues(tmp_path):
    """get_session correctly deserializes nested arbiter issue objects."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    original = _make_session_record()
    await store.save_session(original)

    retrieved = await store.get_session(original.session_id)
    assert retrieved is not None
    issue = retrieved.arbiter_reviews[0].issues[0]
    assert issue.severity == Severity.WARNING
    assert issue.category == IssueCategory.PATTERN
    assert issue.location == "api/main.py:10-20"
    assert issue.description == "Missing error handling"

    await close_db(db)


@pytest.mark.asyncio
async def test_get_session_halted_record(tmp_path):
    """get_session correctly retrieves a halted session."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    record = _make_session_record(
        session_id="halted-session",
        success=False,
        halted=True,
    )
    await store.save_session(record)

    retrieved = await store.get_session("halted-session")
    assert retrieved is not None
    assert retrieved.halted is True
    assert retrieved.success is False
    assert retrieved.halt_reason == "Critical issue"

    await close_db(db)


# ── SessionStore: list_sessions Tests ─────────────────────────────


@pytest.mark.asyncio
async def test_list_sessions_default(tmp_path):
    """list_sessions returns all sessions sorted by started_at descending."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    # Save 3 sessions with different timestamps
    for i in range(3):
        record = _make_session_record(session_id=f"session-{i}")
        record.started_at = datetime(2026, 2, 16, 10 + i, 0, 0, tzinfo=UTC)
        await store.save_session(record)

    summaries = await store.list_sessions(SessionQuery())
    assert len(summaries) == 3
    # Most recent first
    assert summaries[0].session_id == "session-2"
    assert summaries[2].session_id == "session-0"

    await close_db(db)


@pytest.mark.asyncio
async def test_list_sessions_limit_offset(tmp_path):
    """list_sessions respects limit and offset."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    for i in range(5):
        record = _make_session_record(session_id=f"session-{i}")
        record.started_at = datetime(2026, 2, 16, 10 + i, 0, 0, tzinfo=UTC)
        await store.save_session(record)

    # Page 1
    page1 = await store.list_sessions(SessionQuery(limit=2, offset=0))
    assert len(page1) == 2
    assert page1[0].session_id == "session-4"

    # Page 2
    page2 = await store.list_sessions(SessionQuery(limit=2, offset=2))
    assert len(page2) == 2
    assert page2[0].session_id == "session-2"

    await close_db(db)


@pytest.mark.asyncio
async def test_list_sessions_task_filter(tmp_path):
    """list_sessions filters by task text substring."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    record1 = _make_session_record(session_id="s1")
    record1.task = TaskSpec(task="Build a REST API")
    await store.save_session(record1)

    record2 = _make_session_record(session_id="s2")
    record2.task = TaskSpec(task="Create a CLI tool")
    await store.save_session(record2)

    results = await store.list_sessions(
        SessionQuery(task_filter="REST API")
    )
    assert len(results) == 1
    assert results[0].session_id == "s1"

    await close_db(db)


@pytest.mark.asyncio
async def test_list_sessions_verdict_filter(tmp_path):
    """list_sessions filters by arbiter verdict."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    record1 = _make_session_record(session_id="s1")
    record1.arbiter_reviews = [_make_review("architect", "approve")]
    await store.save_session(record1)

    record2 = _make_session_record(session_id="s2")
    record2.arbiter_reviews = [_make_review("architect", "reject")]
    await store.save_session(record2)

    results = await store.list_sessions(
        SessionQuery(verdict_filter="reject")
    )
    assert len(results) == 1
    assert results[0].session_id == "s2"

    await close_db(db)


@pytest.mark.asyncio
async def test_list_sessions_cost_filter(tmp_path):
    """list_sessions filters by min/max cost."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    for i, cost in enumerate([0.10, 0.50, 1.00]):
        record = _make_session_record(session_id=f"s{i}")
        record.total_cost = cost
        await store.save_session(record)

    results = await store.list_sessions(
        SessionQuery(min_cost=0.40, max_cost=0.60)
    )
    assert len(results) == 1
    assert results[0].total_cost == pytest.approx(0.50)

    await close_db(db)


@pytest.mark.asyncio
async def test_list_sessions_summary_fields(tmp_path):
    """list_sessions returns correct SessionSummary aggregated data."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    record = _make_session_record()
    await store.save_session(record)

    summaries = await store.list_sessions(SessionQuery())
    assert len(summaries) == 1
    s = summaries[0]

    assert s.session_id == "test-session-001"
    assert s.task_preview == "Build a REST API"
    assert s.pipeline_mode == "sequential"
    assert s.success is True
    assert s.model_count >= 1  # distinct model keys in stages
    assert s.stage_count == 4
    assert "APPROVE" in s.arbiter_verdict_summary

    await close_db(db)


# ── SessionStore: delete_session Tests ────────────────────────────


@pytest.mark.asyncio
async def test_delete_session(tmp_path):
    """delete_session removes session and all child rows."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    record = _make_session_record()
    await store.save_session(record)

    result = await store.delete_session(record.session_id)
    assert result is True

    # Session gone
    assert await store.get_session(record.session_id) is None

    # Child rows also gone (cascade)
    async with db.execute("SELECT COUNT(*) FROM stages") as cursor:
        assert (await cursor.fetchone())[0] == 0
    async with db.execute("SELECT COUNT(*) FROM arbiter_reviews") as cursor:
        assert (await cursor.fetchone())[0] == 0
    async with db.execute("SELECT COUNT(*) FROM routing_decisions") as cursor:
        assert (await cursor.fetchone())[0] == 0

    await close_db(db)


@pytest.mark.asyncio
async def test_delete_nonexistent_session(tmp_path):
    """delete_session returns False for a non-existent session."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    result = await store.delete_session("nonexistent")
    assert result is False

    await close_db(db)


# ── SessionStore: export_session Tests ────────────────────────────


@pytest.mark.asyncio
async def test_export_session_returns_dict(tmp_path):
    """export_session returns a JSON-serializable dictionary."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)
    record = _make_session_record()
    await store.save_session(record)

    exported = await store.export_session(record.session_id)
    assert exported is not None
    assert exported["session_id"] == record.session_id
    # Should be JSON-serializable
    json_str = json.dumps(exported)
    assert len(json_str) > 0

    await close_db(db)


@pytest.mark.asyncio
async def test_export_session_nonexistent(tmp_path):
    """export_session returns None for a non-existent session."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    result = await store.export_session("nonexistent")
    assert result is None

    await close_db(db)


# ── Export Formatters Tests ───────────────────────────────────────


def test_export_json_format():
    """export_json produces valid formatted JSON."""
    record = _make_session_record()
    json_str = export_json(record)

    parsed = json.loads(json_str)
    assert parsed["session_id"] == "test-session-001"
    assert parsed["task"]["task"] == "Build a REST API"
    assert len(parsed["stages"]) == 4
    assert len(parsed["arbiter_reviews"]) == 1


def test_export_markdown_contains_sections():
    """export_markdown produces a Markdown report with all sections."""
    record = _make_session_record()
    md = export_markdown(record)

    assert "# Session Report:" in md
    assert "## Metadata" in md
    assert "## Pipeline Stages" in md
    assert "## Arbiter Reviews" in md
    assert "## Routing Decisions" in md
    assert "Build a REST API" in md
    assert "$0.2500" in md
    assert "Success" in md


def test_export_markdown_halted_session():
    """export_markdown shows halt info for halted sessions."""
    record = _make_session_record(
        session_id="halted-1", success=False, halted=True,
    )
    md = export_markdown(record)

    assert "Halted" in md
    assert "Critical issue" in md


def test_export_markdown_routing_table():
    """export_markdown includes a routing decisions table."""
    record = _make_session_record()
    md = export_markdown(record)

    assert "| Role | Model | Strategy | Fitness | Est. Cost |" in md
    assert "claude" in md
    assert "hybrid" in md


def test_export_markdown_arbiter_issues():
    """export_markdown includes arbiter issues in the review section."""
    record = _make_session_record()
    md = export_markdown(record)

    assert "Missing error handling" in md
    assert "WARNING" in md
    assert "pattern" in md


def test_export_json_roundtrip():
    """export_json output can be parsed back into a SessionRecord."""
    original = _make_session_record()
    json_str = export_json(original)
    parsed = json.loads(json_str)
    restored = SessionRecord.model_validate(parsed)

    assert restored.session_id == original.session_id
    assert len(restored.stages) == len(original.stages)
    assert len(restored.arbiter_reviews) == len(original.arbiter_reviews)


# ── Schema Validation Tests ───────────────────────────────────────


def test_stage_record_defaults():
    """StageRecord has correct defaults for optional fields."""
    record = StageRecord(stage="architect")
    assert record.model_key == ""
    assert record.model_id == ""
    assert record.content == ""
    assert record.confidence == 0.0
    assert record.cost == 0.0
    assert record.tokens == 0


def test_session_record_required_fields():
    """SessionRecord requires session_id, task, config, and started_at."""
    with pytest.raises(ValueError):
        SessionRecord()  # type: ignore[call-arg]


def test_session_query_defaults():
    """SessionQuery has sensible defaults."""
    query = SessionQuery()
    assert query.limit == 20
    assert query.offset == 0
    assert query.task_filter is None
    assert query.model_filter is None
    assert query.verdict_filter is None
    assert query.min_cost is None
    assert query.max_cost is None
    assert query.since is None


def test_session_query_limit_bounds():
    """SessionQuery enforces limit bounds (1-100)."""
    with pytest.raises(ValueError):
        SessionQuery(limit=0)
    with pytest.raises(ValueError):
        SessionQuery(limit=101)


def test_session_summary_fields():
    """SessionSummary captures aggregated session data."""
    summary = SessionSummary(
        session_id="test-id",
        task_preview="Build a REST API",
        pipeline_mode="sequential",
        started_at=datetime(2026, 2, 16, 10, 0, 0, tzinfo=UTC),
        success=True,
        total_cost=0.25,
        duration_seconds=150.0,
        model_count=3,
        stage_count=4,
        arbiter_verdict_summary="2 APPROVE, 1 FLAG",
    )
    assert summary.session_id == "test-id"
    assert summary.model_count == 3


def test_pipeline_config_persist_defaults():
    """PipelineConfig has persist_sessions=True and default db path."""
    config = PipelineConfig()
    assert config.persist_sessions is True
    assert config.session_db_path == "~/.triad/sessions.db"


def test_pipeline_result_session_id_default():
    """PipelineResult session_id defaults to empty string."""
    result = PipelineResult(
        task=_make_task(),
        config=_make_config(),
    )
    assert result.session_id == ""


# ── Multiple Sessions Tests ───────────────────────────────────────


@pytest.mark.asyncio
async def test_multiple_sessions_independent(tmp_path):
    """Multiple sessions can be saved and retrieved independently."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    for i in range(3):
        record = _make_session_record(session_id=f"multi-{i}")
        await store.save_session(record)

    # Each can be retrieved independently
    for i in range(3):
        r = await store.get_session(f"multi-{i}")
        assert r is not None
        assert r.session_id == f"multi-{i}"

    # Delete one doesn't affect others
    await store.delete_session("multi-1")
    assert await store.get_session("multi-0") is not None
    assert await store.get_session("multi-1") is None
    assert await store.get_session("multi-2") is not None

    await close_db(db)


@pytest.mark.asyncio
async def test_session_with_no_child_rows(tmp_path):
    """A session with no stages/reviews/decisions saves and loads correctly."""
    db = await init_db(str(tmp_path / "test.db"))
    store = SessionStore(db)

    record = SessionRecord(
        session_id="empty-session",
        task=_make_task(),
        config=_make_config(),
        started_at=datetime(2026, 2, 16, 10, 0, 0, tzinfo=UTC),
    )
    await store.save_session(record)

    retrieved = await store.get_session("empty-session")
    assert retrieved is not None
    assert len(retrieved.stages) == 0
    assert len(retrieved.arbiter_reviews) == 0
    assert len(retrieved.routing_decisions) == 0

    await close_db(db)


# ── Orchestrator Integration Tests ────────────────────────────────


def _make_model_config(model: str = "model-a-v1", **overrides) -> ModelConfig:
    defaults = {
        "provider": "test",
        "model": model,
        "display_name": "Test Model",
        "api_key_env": "TEST_KEY",
        "context_window": 128000,
        "cost_input": 3.0,
        "cost_output": 15.0,
        "fitness": RoleFitness(
            architect=0.9, implementer=0.8,
            refactorer=0.7, verifier=0.85,
        ),
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_agent_message(stage: str = "architect"):
    from triad.schemas.messages import AgentMessage, MessageType, TokenUsage

    return AgentMessage(
        from_agent=stage,
        to_agent="implement",
        msg_type=MessageType.PROPOSAL,
        content=f"Output from {stage}",
        confidence=0.85,
        model="model-a-v1",
        token_usage=TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            cost=0.02,
        ),
    )


def _mock_provider(responses=None):
    """Build a mock LiteLLMProvider class that returns mock instances.

    The constructor returns a MagicMock whose .complete() is an AsyncMock
    that returns the given responses in sequence (or a default message).
    """
    from unittest.mock import MagicMock

    if responses is None:
        responses = [_make_agent_message()] * 4  # 4 stages

    mock_cls = MagicMock()
    mock_inst = MagicMock()
    mock_cls.return_value = mock_inst
    mock_inst.complete = AsyncMock(side_effect=responses)
    return mock_cls


@pytest.mark.asyncio
async def test_run_pipeline_generates_session_id():
    """run_pipeline assigns a UUID session_id to the result."""
    from triad.orchestrator import run_pipeline

    mock_cls = _mock_provider()

    mock_review = ArbiterReview(
        stage_reviewed="architect",
        reviewed_model="model-a-v1",
        arbiter_model="model-b-v1",
        verdict=Verdict.APPROVE,
        confidence=0.9,
        reasoning="LGTM",
        token_cost=0.01,
    )

    registry = {"model-a": _make_model_config()}
    task = _make_task()
    config = _make_config(persist_sessions=False)

    with (
        patch(_PROVIDER, mock_cls),
        patch(
            "triad.orchestrator.ArbiterEngine.review",
            new_callable=AsyncMock,
            return_value=mock_review,
        ),
    ):
        result = await run_pipeline(task, config, registry)

    assert result.session_id != ""
    # Should be a valid UUID format
    assert len(result.session_id) == 36


@pytest.mark.asyncio
async def test_run_pipeline_saves_session_when_enabled(tmp_path):
    """run_pipeline calls _save_session when persist_sessions is True."""
    from triad.orchestrator import run_pipeline

    mock_cls = _mock_provider()

    mock_review = ArbiterReview(
        stage_reviewed="architect",
        reviewed_model="model-a-v1",
        arbiter_model="model-b-v1",
        verdict=Verdict.APPROVE,
        confidence=0.9,
        reasoning="LGTM",
        token_cost=0.01,
    )

    db_path = str(tmp_path / "sessions.db")
    registry = {"model-a": _make_model_config()}
    task = _make_task()
    config = _make_config(
        persist_sessions=True,
        session_db_path=db_path,
    )

    with (
        patch(_PROVIDER, mock_cls),
        patch(
            "triad.orchestrator.ArbiterEngine.review",
            new_callable=AsyncMock,
            return_value=mock_review,
        ),
    ):
        result = await run_pipeline(task, config, registry)

    # Verify session was saved to the database
    db = await init_db(db_path)
    store = SessionStore(db)
    retrieved = await store.get_session(result.session_id)
    assert retrieved is not None
    assert retrieved.task.task == "Build a REST API"
    assert retrieved.success is True

    await close_db(db)


@pytest.mark.asyncio
async def test_run_pipeline_skips_save_when_disabled():
    """run_pipeline skips persistence when persist_sessions is False."""
    from triad.orchestrator import run_pipeline

    mock_cls = _mock_provider()

    mock_review = ArbiterReview(
        stage_reviewed="architect",
        reviewed_model="model-a-v1",
        arbiter_model="model-b-v1",
        verdict=Verdict.APPROVE,
        confidence=0.9,
        reasoning="LGTM",
        token_cost=0.01,
    )

    registry = {"model-a": _make_model_config()}
    task = _make_task()
    config = _make_config(persist_sessions=False)

    with (
        patch(_PROVIDER, mock_cls),
        patch(
            "triad.orchestrator.ArbiterEngine.review",
            new_callable=AsyncMock,
            return_value=mock_review,
        ),
        patch("triad.orchestrator._save_session", new_callable=AsyncMock) as mock_save,
    ):
        await run_pipeline(task, config, registry)

    mock_save.assert_not_called()


@pytest.mark.asyncio
async def test_save_session_failure_does_not_crash():
    """_save_session failure is logged but doesn't crash the pipeline."""
    from triad.orchestrator import _save_session

    result = PipelineResult(
        session_id="test-crash",
        task=_make_task(),
        config=_make_config(session_db_path="/nonexistent/path/db.sqlite"),
    )

    # Should not raise — failure is logged
    await _save_session(result, datetime.now(UTC))

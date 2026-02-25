from __future__ import annotations

import hashlib
from pathlib import Path
from uuid import uuid4

import pytest

from triad.memory.decision_log import DecisionLog
from triad.memory.memory import Memory
from triad.memory.patterns import PatternExtractor
from triad.memory.schema import Decision
from triad.memory.taxonomy import DecisionTaxonomy


@pytest.fixture()
def tmp_memory_dir(tmp_path: Path) -> Path:
    d = tmp_path / "memory"
    d.mkdir()
    return d


def _make_decision(
    content_type: str = "tweet",
    niche_id: str = "ai-orchestration",
    pillar_id: str | None = None,
    decision: str = "approve",
    decision_source: str = "human",
    arbiter_confidence: float | None = None,
    revision_notes: str | None = None,
    outcome_score: float | None = None,
) -> Decision:
    content = f"Test content {uuid4()}"
    return Decision(
        decision_id=str(uuid4()),
        timestamp="2025-06-15T12:00:00+00:00",
        content_type=content_type,
        content_hash=hashlib.sha256(content.encode()).hexdigest(),
        content_preview=content[:200],
        niche_id=niche_id,
        pillar_id=pillar_id,
        persona_id="crtx-ai",
        source_agent="content_agent",
        decision=decision,
        decision_source=decision_source,
        arbiter_confidence=arbiter_confidence,
    )


class TestDecisionRecordAndQuery:
    def test_record_and_query(self, tmp_memory_dir: Path) -> None:
        log = DecisionLog(tmp_memory_dir)

        d1 = _make_decision(content_type="tweet", niche_id="ai-orchestration", decision="approve")
        d2 = _make_decision(content_type="article", niche_id="ai-orchestration", decision="edit")
        d3 = _make_decision(content_type="tweet", niche_id="ou-sports", decision="skip")

        log.record(d1)
        log.record(d2)
        log.record(d3)

        # All
        all_decisions = log.query()
        assert len(all_decisions) == 3

        # Filter by content_type
        tweets = log.query(content_type="tweet")
        assert len(tweets) == 2

        # Filter by niche_id
        ai = log.query(niche_id="ai-orchestration")
        assert len(ai) == 2

        # Filter by decision
        edits = log.query(decision="edit")
        assert len(edits) == 1
        assert edits[0].content_type == "article"

        # Filter with limit
        limited = log.query(limit=1)
        assert len(limited) == 1


class TestTaxonomyStreak:
    def test_streak_to_auto_ship(self, tmp_memory_dir: Path) -> None:
        tax = DecisionTaxonomy(tmp_memory_dir)

        # 10 approvals should build streak to auto_ship
        for _ in range(10):
            tax.record_outcome("tweet", "ai-orchestration")

        result = tax.classify("tweet", "ai-orchestration", arbiter_confidence=0.90)
        assert result == "auto_ship"

    def test_edit_resets_to_flag(self, tmp_memory_dir: Path) -> None:
        tax = DecisionTaxonomy(tmp_memory_dir)

        # Build streak
        for _ in range(10):
            tax.record_outcome("tweet", "ai-orchestration")

        # Verify auto_ship
        assert tax.classify("tweet", "ai-orchestration", arbiter_confidence=0.90) == "auto_ship"

        # One edit resets
        tax.record_outcome("tweet", "ai-orchestration", human_decision="edit")

        assert tax.classify("tweet", "ai-orchestration", arbiter_confidence=0.90) == "flag"


class TestTaxonomyPauseOnLowConfidence:
    def test_low_confidence_pauses(self, tmp_memory_dir: Path) -> None:
        tax = DecisionTaxonomy(tmp_memory_dir)
        result = tax.classify("tweet", "ai-orchestration", arbiter_confidence=0.75)
        assert result == "pause"

    def test_high_confidence_no_pause(self, tmp_memory_dir: Path) -> None:
        tax = DecisionTaxonomy(tmp_memory_dir)
        result = tax.classify("tweet", "ai-orchestration", arbiter_confidence=0.85)
        # No rule exists yet, so "flag" (not "pause")
        assert result == "flag"


class TestPatternApproval:
    def test_approval_pattern_extracted(self, tmp_memory_dir: Path) -> None:
        log = DecisionLog(tmp_memory_dir)

        # 6 approvals + 1 edit = 6/7 = 85.7% approval (>= 85% threshold)
        for _ in range(6):
            log.record(_make_decision(decision="approve"))
        log.record(_make_decision(decision="edit"))

        extractor = PatternExtractor(log, tmp_memory_dir)
        patterns = extractor.extract()

        approval_patterns = [p for p in patterns if p.pattern_type == "always_approve"]
        assert len(approval_patterns) >= 1
        assert approval_patterns[0].confidence >= 0.85


class TestMemoryLearnCycle:
    def test_full_cycle(self, tmp_memory_dir: Path) -> None:
        mem = Memory(memory_dir=tmp_memory_dir)

        # Record enough decisions for patterns to emerge
        for _ in range(6):
            d = _make_decision(decision="approve", arbiter_confidence=0.95)
            mem.record(d)

        # Learn
        summary = mem.learn()
        assert summary["total_patterns"] >= 0
        assert "active" in summary
        assert "pattern_types" in summary

        # Get patterns
        patterns = mem.get_patterns()
        # Patterns should be a list (may be empty if thresholds not met)
        assert isinstance(patterns, list)

        # State should be updated
        assert mem.state.total_decisions == 6

    def test_generation_hints(self, tmp_memory_dir: Path) -> None:
        mem = Memory(memory_dir=tmp_memory_dir)

        for _ in range(6):
            d = _make_decision(decision="approve")
            mem.record(d)

        mem.learn()
        hints = mem.get_generation_hints("tweet", "ai-orchestration")
        assert "avoid" in hints
        assert "emphasize" in hints
        assert "common_edits" in hints
        assert "performance_hint" in hints


class TestIngestMissingFilesGraceful:
    def test_nonexistent_dir(self, tmp_memory_dir: Path) -> None:
        mem = Memory(memory_dir=tmp_memory_dir)
        count = mem.ingest_history(tmp_memory_dir / "nonexistent")
        assert count == 0

    def test_empty_dir(self, tmp_memory_dir: Path) -> None:
        empty_dir = tmp_memory_dir / "empty_project"
        empty_dir.mkdir()
        mem = Memory(memory_dir=tmp_memory_dir)
        count = mem.ingest_history(empty_dir)
        assert count == 0


class TestClassifyRevision:
    def test_tone_classification(self) -> None:
        assert Memory._classify_revision("The tone is too aggressive") == "tone"

    def test_accuracy_classification(self) -> None:
        assert Memory._classify_revision("This fact is wrong") == "accuracy"

    def test_length_classification(self) -> None:
        assert Memory._classify_revision("Too verbose, make it brief") == "length"

    def test_missing_data_classification(self) -> None:
        assert Memory._classify_revision("Missing key details") == "missing_data"

    def test_persona_drift_classification(self) -> None:
        assert Memory._classify_revision("Feels off-brand for persona") == "persona_drift"

    def test_unknown_classification(self) -> None:
        assert Memory._classify_revision("random unrelated feedback") == "other"


class TestTaxonomySkipDecrement:
    def test_skip_decrements_streak(self, tmp_memory_dir: Path) -> None:
        tax = DecisionTaxonomy(tmp_memory_dir)

        # Build streak of 5
        for _ in range(5):
            tax.record_outcome("tweet", "ai-orchestration")

        rule = tax._find_rule("tweet", "ai-orchestration")
        assert rule.consecutive_approvals == 5

        # Skip decrements by 2
        tax.record_outcome("tweet", "ai-orchestration", human_decision="skip")
        assert rule.consecutive_approvals == 3

    def test_skip_floor_at_zero(self, tmp_memory_dir: Path) -> None:
        tax = DecisionTaxonomy(tmp_memory_dir)
        tax.record_outcome("tweet", "ai-orchestration")  # streak = 1
        tax.record_outcome("tweet", "ai-orchestration", human_decision="skip")  # 1 - 2 = 0 (clamped)  # noqa: E501

        rule = tax._find_rule("tweet", "ai-orchestration")
        assert rule.consecutive_approvals == 0

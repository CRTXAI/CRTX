from __future__ import annotations

import json
from collections import defaultdict
from dataclasses import asdict
from datetime import datetime, timezone, timedelta
from pathlib import Path
from typing import Optional
from uuid import uuid4

from .decision_log import DecisionLog
from .schema import Pattern

MIN_SAMPLE_SIZE = 5
MIN_CONSISTENCY = 0.85

_WEAK_DAYS = 30
_RETIRED_DAYS = 60


class PatternExtractor:
    def __init__(self, log: DecisionLog, memory_dir: Path) -> None:
        self.log = log
        self.memory_dir = memory_dir
        self.patterns_file = memory_dir / "patterns.json"

    def extract(self) -> list[Pattern]:
        """Run all extractors, deduplicate, save, and return patterns."""
        new_patterns: list[Pattern] = []
        new_patterns.extend(self._extract_approval_patterns())
        new_patterns.extend(self._extract_revision_patterns())
        new_patterns.extend(self._extract_confidence_thresholds())
        new_patterns.extend(self._extract_performance_patterns())
        new_patterns.extend(self._extract_edit_trigger_patterns())

        existing = self._load_patterns()
        merged = self._merge_patterns(existing, new_patterns)
        self._save_patterns(merged)
        return merged

    def _extract_approval_patterns(self) -> list[Pattern]:
        """Groups by (content_type, niche_id, pillar_id), flags if approval_rate >= 85%."""
        decisions = self.log.query()
        groups: dict[tuple, list] = defaultdict(list)
        for d in decisions:
            key = (d.content_type, d.niche_id, d.pillar_id)
            groups[key].append(d)

        patterns: list[Pattern] = []
        now = datetime.now(timezone.utc).isoformat()
        for key, group in groups.items():
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            approvals = sum(1 for d in group if d.decision in ("approve", "auto_approve"))
            rate = approvals / len(group)
            if rate >= MIN_CONSISTENCY:
                patterns.append(Pattern(
                    pattern_id=str(uuid4()),
                    discovered_at=now,
                    last_confirmed=now,
                    pattern_type="always_approve",
                    description=f"Content ({key[0]}, {key[1]}, {key[2]}) approved {rate:.0%} of the time",
                    conditions={"content_type": key[0], "niche_id": key[1], "pillar_id": key[2]},
                    prediction="approve",
                    confidence=rate,
                    sample_size=len(group),
                    status="active",
                ))
        return patterns

    def _extract_revision_patterns(self) -> list[Pattern]:
        """Flags if edit_rate >= 50%."""
        decisions = self.log.query()
        groups: dict[tuple, list] = defaultdict(list)
        for d in decisions:
            key = (d.content_type, d.niche_id, d.pillar_id)
            groups[key].append(d)

        patterns: list[Pattern] = []
        now = datetime.now(timezone.utc).isoformat()
        for key, group in groups.items():
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            edits = sum(1 for d in group if d.decision == "edit")
            rate = edits / len(group)
            if rate >= 0.50:
                patterns.append(Pattern(
                    pattern_id=str(uuid4()),
                    discovered_at=now,
                    last_confirmed=now,
                    pattern_type="always_edit",
                    description=f"Content ({key[0]}, {key[1]}, {key[2]}) edited {rate:.0%} of the time",
                    conditions={"content_type": key[0], "niche_id": key[1], "pillar_id": key[2]},
                    prediction="edit",
                    confidence=rate,
                    sample_size=len(group),
                    status="active",
                ))
        return patterns

    def _extract_confidence_thresholds(self) -> list[Pattern]:
        """Find confidence level where approval >= 95% (scan 0.99 down to 0.80)."""
        decisions = self.log.query()
        scored = [d for d in decisions if d.arbiter_confidence is not None]
        if len(scored) < MIN_SAMPLE_SIZE:
            return []

        patterns: list[Pattern] = []
        now = datetime.now(timezone.utc).isoformat()

        for threshold_int in range(99, 79, -1):
            threshold = threshold_int / 100.0
            above = [d for d in scored if d.arbiter_confidence >= threshold]
            if len(above) < MIN_SAMPLE_SIZE:
                continue
            approvals = sum(1 for d in above if d.decision in ("approve", "auto_approve"))
            rate = approvals / len(above)
            if rate >= 0.95:
                patterns.append(Pattern(
                    pattern_id=str(uuid4()),
                    discovered_at=now,
                    last_confirmed=now,
                    pattern_type="confidence_threshold",
                    description=f"Confidence >= {threshold:.2f} yields {rate:.0%} approval rate",
                    conditions={"min_confidence": threshold},
                    prediction="approve",
                    confidence=rate,
                    sample_size=len(above),
                    status="active",
                ))
                break  # Take the lowest qualifying threshold
        return patterns

    def _extract_performance_patterns(self) -> list[Pattern]:
        """High performer if avg > 1.5x mean; low performer if avg < 0.5x mean."""
        decisions = self.log.query()
        scored = [d for d in decisions if d.outcome_score is not None]
        if len(scored) < MIN_SAMPLE_SIZE:
            return []

        overall_mean = sum(d.outcome_score for d in scored) / len(scored)
        if overall_mean == 0:
            return []

        groups: dict[tuple, list] = defaultdict(list)
        for d in scored:
            key = (d.content_type, d.niche_id, d.pillar_id)
            groups[key].append(d)

        patterns: list[Pattern] = []
        now = datetime.now(timezone.utc).isoformat()
        for key, group in groups.items():
            if len(group) < MIN_SAMPLE_SIZE:
                continue
            avg = sum(d.outcome_score for d in group) / len(group)
            if avg > 1.5 * overall_mean:
                patterns.append(Pattern(
                    pattern_id=str(uuid4()),
                    discovered_at=now,
                    last_confirmed=now,
                    pattern_type="high_performer",
                    description=f"Content ({key[0]}, {key[1]}, {key[2]}) avg score {avg:.2f} vs mean {overall_mean:.2f}",
                    conditions={"content_type": key[0], "niche_id": key[1], "pillar_id": key[2]},
                    prediction="high_performance",
                    confidence=avg / overall_mean,
                    sample_size=len(group),
                    status="active",
                ))
            elif avg < 0.5 * overall_mean:
                patterns.append(Pattern(
                    pattern_id=str(uuid4()),
                    discovered_at=now,
                    last_confirmed=now,
                    pattern_type="low_performer",
                    description=f"Content ({key[0]}, {key[1]}, {key[2]}) avg score {avg:.2f} vs mean {overall_mean:.2f}",
                    conditions={"content_type": key[0], "niche_id": key[1], "pillar_id": key[2]},
                    prediction="low_performance",
                    confidence=1 - (avg / overall_mean),
                    sample_size=len(group),
                    status="active",
                ))
        return patterns

    def _extract_edit_trigger_patterns(self) -> list[Pattern]:
        """Keyword-based categorization of revision_notes."""
        decisions = self.log.query()
        edited = [d for d in decisions if d.decision == "edit" and d.revision_notes]
        if len(edited) < MIN_SAMPLE_SIZE:
            return []

        keyword_categories = {
            "tone": ["tone", "voice", "formal", "casual", "aggressive", "soft"],
            "accuracy": ["wrong", "incorrect", "inaccurate", "fact", "error", "mistake"],
            "length": ["long", "short", "verbose", "brief", "wordy", "concise"],
            "missing_data": ["missing", "incomplete", "add", "include", "need"],
            "persona_drift": ["persona", "brand", "off-brand", "character", "voice"],
        }

        category_counts: dict[str, int] = defaultdict(int)
        for d in edited:
            notes_lower = d.revision_notes.lower()
            for category, keywords in keyword_categories.items():
                if any(kw in notes_lower for kw in keywords):
                    category_counts[category] += 1

        patterns: list[Pattern] = []
        now = datetime.now(timezone.utc).isoformat()
        for category, count in category_counts.items():
            if count >= 3:
                patterns.append(Pattern(
                    pattern_id=str(uuid4()),
                    discovered_at=now,
                    last_confirmed=now,
                    pattern_type="edit_trigger",
                    description=f"Edits frequently triggered by '{category}' issues ({count} times)",
                    conditions={"revision_category": category},
                    prediction="edit",
                    confidence=count / len(edited),
                    sample_size=count,
                    status="active",
                ))
        return patterns

    def _merge_patterns(
        self,
        existing: list[Pattern],
        new: list[Pattern],
    ) -> list[Pattern]:
        """Merge new patterns into existing. Update matches, add new, decay old."""
        now = datetime.now(timezone.utc)
        merged: dict[str, Pattern] = {}

        # Index existing by (pattern_type, conditions key)
        for p in existing:
            key = self._pattern_key(p)
            merged[key] = p

        # Merge new patterns
        for p in new:
            key = self._pattern_key(p)
            if key in merged:
                old = merged[key]
                old.last_confirmed = now.isoformat()
                old.confidence = p.confidence
                old.sample_size = p.sample_size
                old.description = p.description
                if old.status == "weak":
                    old.status = "active"
            else:
                merged[key] = p

        # Apply decay
        for p in merged.values():
            if p.status == "retired":
                continue
            try:
                last = datetime.fromisoformat(p.last_confirmed)
                if last.tzinfo is None:
                    last = last.replace(tzinfo=timezone.utc)
                age = now - last
                if age > timedelta(days=_RETIRED_DAYS):
                    p.status = "retired"
                elif age > timedelta(days=_WEAK_DAYS):
                    p.status = "weak"
            except (ValueError, TypeError):
                pass

        return list(merged.values())

    @staticmethod
    def _pattern_key(p: Pattern) -> str:
        """Unique key for dedup: pattern_type + sorted conditions."""
        cond_str = json.dumps(p.conditions, sort_keys=True)
        return f"{p.pattern_type}:{cond_str}"

    def _load_patterns(self) -> list[Pattern]:
        if not self.patterns_file.exists():
            return []
        try:
            with open(self.patterns_file, encoding="utf-8") as f:
                data = json.load(f)
            return [Pattern(**p) for p in data]
        except (json.JSONDecodeError, TypeError, OSError):
            return []

    def _save_patterns(self, patterns: list[Pattern]) -> None:
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self.patterns_file, "w", encoding="utf-8") as f:
            json.dump([asdict(p) for p in patterns], f, indent=2)

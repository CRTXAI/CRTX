from __future__ import annotations

import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional

from .decision_log import DecisionLog
from .patterns import PatternExtractor
from .schema import Decision, MemoryState, Pattern
from .taxonomy import DecisionTaxonomy


class Memory:
    def __init__(self, memory_dir: Path | None = None) -> None:
        self.memory_dir = memory_dir or Path.home() / ".crtx"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        self.state = self._load_state()
        self.log = DecisionLog(self.memory_dir)
        self.taxonomy = DecisionTaxonomy(self.memory_dir)
        self.extractor = PatternExtractor(self.log, self.memory_dir)

    def record(self, decision: Decision) -> None:
        """Classify revision, generate task_class, record to log, update taxonomy."""
        if decision.decision == "edit" and decision.revision_notes and not decision.revision_category:
            decision.revision_category = self._classify_revision(decision.revision_notes)

        if not decision.task_class:
            decision.task_class = f"{decision.content_type}:{decision.niche_id}"

        # Classify via taxonomy
        action = self.taxonomy.classify(
            decision.content_type,
            decision.niche_id,
            decision.pillar_id,
            decision.arbiter_confidence,
            decision.arbiter_model,
        )
        decision.taxonomy_action = action

        self.log.record(decision)

        # Update taxonomy streak if human decision
        if decision.decision_source == "human":
            self.taxonomy.record_outcome(
                decision.content_type,
                decision.niche_id,
                decision.pillar_id,
                decision.decision,
            )

        # Update state counters
        self.state.total_decisions += 1
        if action == "auto_ship":
            self.state.total_auto_ships += 1
        if decision.decision_source == "human" and decision.decision in ("edit", "skip"):
            self.state.total_human_overrides += 1
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self._save_state()

    def classify(
        self,
        content_type: str,
        niche_id: Optional[str] = None,
        pillar_id: Optional[str] = None,
        arbiter_confidence: Optional[float] = None,
        arbiter_model: Optional[str] = None,
    ) -> str:
        """Classify content via taxonomy. Returns 'auto_ship', 'flag', or 'pause'."""
        return self.taxonomy.classify(
            content_type, niche_id, pillar_id, arbiter_confidence, arbiter_model
        )

    def get_patterns(self, niche_id: Optional[str] = None) -> list[Pattern]:
        """Return active patterns, optionally filtered by niche."""
        patterns = self.extractor._load_patterns()
        active = [p for p in patterns if p.status == "active"]
        if niche_id:
            active = [
                p for p in active
                if p.conditions.get("niche_id") == niche_id or "niche_id" not in p.conditions
            ]
        return active

    def get_generation_hints(
        self,
        content_type: str,
        niche_id: Optional[str] = None,
        pillar_id: Optional[str] = None,
    ) -> dict:
        """Returns {avoid, emphasize, common_edits, performance_hint} from active patterns."""
        patterns = self.get_patterns(niche_id)
        avoid: list[str] = []
        emphasize: list[str] = []
        common_edits: list[str] = []
        performance_hint: Optional[str] = None

        for p in patterns:
            conds = p.conditions
            # Check relevance to this content
            if conds.get("content_type") and conds["content_type"] != content_type:
                continue
            if conds.get("pillar_id") and pillar_id and conds["pillar_id"] != pillar_id:
                continue

            if p.pattern_type == "edit_trigger":
                category = conds.get("revision_category", "unknown")
                common_edits.append(f"Frequently edited for: {category}")
                avoid.append(f"Avoid {category} issues")
            elif p.pattern_type == "always_approve":
                emphasize.append(f"High approval rate for this content type ({p.confidence:.0%})")
            elif p.pattern_type == "always_edit":
                avoid.append(f"High edit rate — review carefully ({p.confidence:.0%})")
            elif p.pattern_type == "high_performer":
                performance_hint = f"High performer: {p.description}"
                emphasize.append("This content type performs well — keep the pattern")
            elif p.pattern_type == "low_performer":
                performance_hint = f"Low performer: {p.description}"
                avoid.append("This content type underperforms — consider changes")

        return {
            "avoid": avoid,
            "emphasize": emphasize,
            "common_edits": common_edits,
            "performance_hint": performance_hint,
        }

    def learn(self) -> dict:
        """Extract patterns, update state, return summary."""
        patterns = self.extractor.extract()
        active = [p for p in patterns if p.status == "active"]
        weak = [p for p in patterns if p.status == "weak"]
        retired = [p for p in patterns if p.status == "retired"]

        self.state.patterns = [asdict(p) for p in patterns]
        self.state.last_updated = datetime.now(timezone.utc).isoformat()
        self._save_state()

        return {
            "total_patterns": len(patterns),
            "active": len(active),
            "weak": len(weak),
            "retired": len(retired),
            "pattern_types": list({p.pattern_type for p in active}),
        }

    def ingest_history(self, clawbucks_dir: Path) -> int:
        """Import existing data, then learn from it."""
        count = self.log.ingest_existing_data(clawbucks_dir)
        if count > 0:
            self.state.total_decisions += count
            self.state.last_updated = datetime.now(timezone.utc).isoformat()
            self._save_state()
            self.learn()
        return count

    @staticmethod
    def _classify_revision(notes: str) -> str:
        """Keyword-based revision category classification."""
        notes_lower = notes.lower()
        categories = {
            "tone": ["tone", "voice", "formal", "casual", "aggressive", "soft"],
            "accuracy": ["wrong", "incorrect", "inaccurate", "fact", "error", "mistake"],
            "length": ["long", "short", "verbose", "brief", "wordy", "concise"],
            "missing_data": ["missing", "incomplete", "add", "include", "need"],
            "persona_drift": ["persona", "brand", "off-brand", "character"],
        }
        for category, keywords in categories.items():
            if any(kw in notes_lower for kw in keywords):
                return category
        return "other"

    def _load_state(self) -> MemoryState:
        state_file = self.memory_dir / "memory.json"
        if not state_file.exists():
            state = MemoryState(
                created_at=datetime.now(timezone.utc).isoformat(),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )
            return state
        try:
            with open(state_file, encoding="utf-8") as f:
                data = json.load(f)
            return MemoryState(**data)
        except (json.JSONDecodeError, TypeError, OSError):
            return MemoryState(
                created_at=datetime.now(timezone.utc).isoformat(),
                last_updated=datetime.now(timezone.utc).isoformat(),
            )

    def _save_state(self) -> None:
        state_file = self.memory_dir / "memory.json"
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(state_file, "w", encoding="utf-8") as f:
            json.dump(asdict(self.state), f, indent=2)

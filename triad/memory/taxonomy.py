from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .schema import TaxonomyRule

DEFAULT_REQUIRED_STREAK = 10
DEFAULT_MAX_AUTO_SHIPS_PER_DAY = 20
DEFAULT_COOLDOWN_AFTER_EDIT = 5


class DecisionTaxonomy:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.taxonomy_file = memory_dir / "taxonomy.json"
        self.rules: list[TaxonomyRule] = self._load_rules()

    def classify(
        self,
        content_type: str,
        niche_id: str | None = None,
        pillar_id: str | None = None,
        arbiter_confidence: float | None = None,
        arbiter_model: str | None = None,
    ) -> str:
        """Returns 'auto_ship', 'flag', or 'pause'."""
        if arbiter_confidence is not None and arbiter_confidence < 0.80:
            return "pause"

        rule = self._find_rule(content_type, niche_id, pillar_id)
        if rule is None:
            return "flag"

        if rule.cooldown_after_edit > 0:
            return "flag"

        if rule.action != "auto_ship":
            return rule.action

        if rule.consecutive_approvals < rule.required_streak:
            return "flag"

        if arbiter_confidence is not None and arbiter_confidence < rule.min_arbiter_confidence:
            return "flag"

        if rule.requires_arbiter and arbiter_confidence is None:
            return "flag"

        if self._daily_auto_ship_count() >= rule.max_auto_ships_per_day:
            return "flag"

        return "auto_ship"

    def record_outcome(
        self,
        content_type: str,
        niche_id: str | None = None,
        pillar_id: str | None = None,
        human_decision: str = "approve",
    ) -> None:
        """Update streak based on human decision."""
        rule = self._find_or_create_rule(content_type, niche_id, pillar_id)

        if human_decision == "approve":
            rule.consecutive_approvals += 1
            if rule.cooldown_after_edit > 0:
                rule.cooldown_after_edit -= 1
        elif human_decision == "edit":
            rule.consecutive_approvals = 0
            rule.cooldown_after_edit = DEFAULT_COOLDOWN_AFTER_EDIT
            rule.last_human_override = datetime.now(UTC).isoformat()
            rule.action = "flag"
        elif human_decision == "skip":
            rule.consecutive_approvals = max(0, rule.consecutive_approvals - 2)

        # Promote to auto_ship if streak met
        if rule.consecutive_approvals >= rule.required_streak and rule.cooldown_after_edit == 0:
            rule.action = "auto_ship"

        self._save_rules()

    def get_status_report(self) -> dict:
        """Generate status report for daily briefing."""
        auto_ship_rules = [r for r in self.rules if r.action == "auto_ship"]
        flag_rules = [r for r in self.rules if r.action == "flag"]
        pause_rules = [r for r in self.rules if r.action == "pause"]

        return {
            "total_rules": len(self.rules),
            "auto_ship": len(auto_ship_rules),
            "flag": len(flag_rules),
            "pause": len(pause_rules),
            "daily_auto_ships": self._daily_auto_ship_count(),
            "rules": [
                {
                    "content_type": r.content_type,
                    "niche_id": r.niche_id,
                    "pillar_id": r.pillar_id,
                    "action": r.action,
                    "streak": r.consecutive_approvals,
                    "required": r.required_streak,
                    "cooldown": r.cooldown_after_edit,
                }
                for r in self.rules
            ],
        }

    def reset(self) -> None:
        """Reset all rules: clear streaks, set all actions to 'flag'."""
        for rule in self.rules:
            rule.consecutive_approvals = 0
            rule.action = "flag"
            rule.cooldown_after_edit = 0
        self._save_rules()

    def _find_rule(
        self,
        content_type: str,
        niche_id: str | None = None,
        pillar_id: str | None = None,
    ) -> TaxonomyRule | None:
        """Find most specific matching rule: pillar > niche > global."""
        # Pillar-level match (most specific)
        if pillar_id:
            for rule in self.rules:
                if (
                    rule.content_type == content_type
                    and rule.niche_id == niche_id
                    and rule.pillar_id == pillar_id
                ):
                    return rule

        # Niche-level match
        if niche_id:
            for rule in self.rules:
                if (
                    rule.content_type == content_type
                    and rule.niche_id == niche_id
                    and rule.pillar_id is None
                ):
                    return rule

        # Global match
        for rule in self.rules:
            if (
                rule.content_type == content_type
                and rule.niche_id is None
                and rule.pillar_id is None
            ):
                return rule

        return None

    def _find_or_create_rule(
        self,
        content_type: str,
        niche_id: str | None = None,
        pillar_id: str | None = None,
    ) -> TaxonomyRule:
        """Find or create a rule with defaults."""
        rule = self._find_rule(content_type, niche_id, pillar_id)
        if rule is not None:
            return rule

        rule = TaxonomyRule(
            rule_id=str(uuid4()),
            created_at=datetime.now(UTC).isoformat(),
            content_type=content_type,
            niche_id=niche_id,
            pillar_id=pillar_id,
            min_arbiter_confidence=0.85,
            action="flag",
            consecutive_approvals=0,
            required_streak=DEFAULT_REQUIRED_STREAK,
            max_auto_ships_per_day=DEFAULT_MAX_AUTO_SHIPS_PER_DAY,
        )
        self.rules.append(rule)
        self._save_rules()
        return rule

    def _daily_auto_ship_count(self) -> int:
        """Count today's auto_ships from the rules (approximation from taxonomy state)."""
        # This counts rules currently set to auto_ship as a proxy.
        # In production, this would query the decision log for today's auto_ship decisions.
        today = datetime.now(UTC).date().isoformat()
        decisions_file = self.memory_dir / "decisions.jsonl"
        if not decisions_file.exists():
            return 0

        count = 0
        try:
            import json as _json
            with open(decisions_file, encoding="utf-8") as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        data = _json.loads(line)
                    except (ValueError, KeyError):
                        continue
                    if (
                        data.get("taxonomy_action") == "auto_ship"
                        and data.get("timestamp", "")[:10] == today
                    ):
                        count += 1
        except OSError:
            pass
        return count

    def _load_rules(self) -> list[TaxonomyRule]:
        """Load taxonomy rules from file."""
        if not self.taxonomy_file.exists():
            return []
        try:
            with open(self.taxonomy_file, encoding="utf-8") as f:
                data = json.load(f)
            return [TaxonomyRule(**rule) for rule in data]
        except (json.JSONDecodeError, TypeError, OSError):
            return []

    def _save_rules(self) -> None:
        """Persist taxonomy rules to file."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self.taxonomy_file, "w", encoding="utf-8") as f:
            json.dump([asdict(r) for r in self.rules], f, indent=2)

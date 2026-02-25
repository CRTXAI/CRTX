from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from .schema import Decision


class DecisionLog:
    def __init__(self, memory_dir: Path) -> None:
        self.memory_dir = memory_dir
        self.decisions_file = memory_dir / "decisions.jsonl"

    def record(self, decision: Decision) -> None:
        """Append a decision to the append-only log."""
        self.memory_dir.mkdir(parents=True, exist_ok=True)
        with open(self.decisions_file, "a", encoding="utf-8") as f:
            f.write(json.dumps(asdict(decision)) + "\n")

    def query(
        self,
        content_type: str | None = None,
        niche_id: str | None = None,
        pillar_id: str | None = None,
        decision: str | None = None,
        since: str | None = None,
        limit: int | None = None,
    ) -> list[Decision]:
        """Filter and return decisions from the log."""
        results: list[Decision] = []
        if not self.decisions_file.exists():
            return results

        with open(self.decisions_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                data = json.loads(line)
                if content_type and data.get("content_type") != content_type:
                    continue
                if niche_id and data.get("niche_id") != niche_id:
                    continue
                if pillar_id and data.get("pillar_id") != pillar_id:
                    continue
                if decision and data.get("decision") != decision:
                    continue
                if since and data.get("timestamp", "") < since:
                    continue
                results.append(Decision(**data))

        if limit:
            results = results[-limit:]
        return results

    def ingest_existing_data(self, clawbucks_dir: Path) -> int:
        """One-time import of real human decisions from the draft queue.

        Only ingests entries where a human explicitly tapped Publish, Skip,
        or sent revision notes. Arbiter verdicts (runs.jsonl) and raw publish
        logs (x_posts.jsonl) are intentionally excluded — those are system
        events, not human decisions.

        Returns count of imported records.
        """
        return self._ingest_draft_queue(clawbucks_dir)

    def update_decision_reason(self, decision_id: str, reason: str) -> bool:
        """Update the decision_reason on an existing decision.

        Rewrites decisions.jsonl in-place. Safe at current scale (hundreds of
        decisions). Switch to SQLite if volume exceeds ~10K decisions.

        Returns True if the decision was found and updated.
        """
        if not self.decisions_file.exists():
            return False

        lines = self.decisions_file.read_text(encoding="utf-8").splitlines()
        updated = False
        new_lines: list[str] = []
        for line in lines:
            line = line.strip()
            if not line:
                continue
            data = json.loads(line)
            if data.get("decision_id") == decision_id:
                data["decision_reason"] = reason
                updated = True
            new_lines.append(json.dumps(data))

        if updated:
            self.decisions_file.write_text(
                "\n".join(new_lines) + "\n", encoding="utf-8"
            )
        return updated

    def backfill_engagement(self) -> None:
        # TODO: Needs the niche scorer to be built first
        pass

    def _ingest_draft_queue(self, clawbucks_dir: Path) -> int:
        """Import real human decisions from queue/*.json files.

        Only ingests drafts with an explicit human action:
          published          → decision="approve"  (human tapped Publish to X)
          skipped            → decision="skip"     (human tapped Skip)
          revision_requested → decision="edit"     (human sent revision notes)

        All other statuses (pending, failed, deleted, approved_for_medium) are
        ignored — they either haven't been acted on or aren't human approval decisions.
        """
        queue_dir = clawbucks_dir / "queue"
        if not queue_dir.exists():
            return 0

        # Status → decision mapping (only real human taps)
        status_map = {
            "published": "approve",
            "skipped": "skip",
            "revision_requested": "edit",
        }

        count = 0
        for draft_file in sorted(queue_dir.glob("*.json")):
            try:
                draft = json.loads(draft_file.read_text(encoding="utf-8"))
            except (json.JSONDecodeError, OSError):
                continue

            # Skip non-draft files (e.g. medium_queue.json is a list)
            if not isinstance(draft, dict):
                continue

            status = draft.get("status", "")
            if status not in status_map:
                continue  # Not a completed human decision

            decision_val = status_map[status]

            # Build content string for hashing/preview
            text = draft.get("text") or " ".join(draft.get("tweets", [])) or draft.get("title", "")
            content_hash = hashlib.sha256(text.encode()).hexdigest()

            decision = Decision(
                decision_id=str(uuid4()),
                timestamp=draft.get("updated_at", draft.get("created_at",
                                    datetime.now(UTC).isoformat())),
                content_type=draft.get("type", "tweet"),
                content_hash=content_hash,
                content_preview=text[:200],
                niche_id=draft.get("niche", "general"),
                pillar_id=draft.get("pillar"),
                persona_id=draft.get("persona"),
                source_agent=draft.get("source", "unknown"),
                decision=decision_val,
                decision_source="human",
                arbiter_confidence=draft.get("confidence"),
                arbiter_model=draft.get("arbiter_model"),
                arbiter_issues=None,
                revision_notes=draft.get("revision_notes"),
                revision_category=None,
                engagement_rate=None,
                impressions=None,
                outcome_score=None,
                task_class=None,
                taxonomy_action=None,
            )
            self.record(decision)
            count += 1

        return count

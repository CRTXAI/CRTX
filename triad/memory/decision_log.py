from __future__ import annotations

import hashlib
import json
from dataclasses import asdict
from datetime import datetime, timezone
from pathlib import Path
from typing import Optional
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
        content_type: Optional[str] = None,
        niche_id: Optional[str] = None,
        pillar_id: Optional[str] = None,
        decision: Optional[str] = None,
        since: Optional[str] = None,
        limit: Optional[int] = None,
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
        """One-time import from existing sources. Returns count of imported records."""
        count = 0
        count += self._ingest_runs(clawbucks_dir)
        count += self._ingest_x_posts(clawbucks_dir)
        count += self._ingest_pending_drafts(clawbucks_dir)
        return count

    def backfill_engagement(self) -> None:
        # TODO: Needs the niche scorer to be built first
        pass

    def _ingest_runs(self, clawbucks_dir: Path) -> int:
        """Import article Arbiter verdicts from logs/runs.jsonl."""
        runs_file = clawbucks_dir / "logs" / "runs.jsonl"
        if not runs_file.exists():
            return 0

        count = 0
        with open(runs_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                content = data.get("content", data.get("article", ""))
                content_str = content if isinstance(content, str) else json.dumps(content)

                decision = Decision(
                    decision_id=str(uuid4()),
                    timestamp=data.get("timestamp", datetime.now(timezone.utc).isoformat()),
                    content_type="article",
                    content_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                    content_preview=content_str[:200],
                    niche_id=data.get("niche_id", "unknown"),
                    pillar_id=data.get("pillar_id"),
                    persona_id=data.get("persona_id"),
                    source_agent="content_agent",
                    decision=data.get("verdict", data.get("decision", "approve")),
                    decision_source="arbiter",
                    arbiter_confidence=data.get("confidence"),
                    arbiter_model=data.get("model"),
                    arbiter_issues=data.get("issues"),
                )
                self.record(decision)
                count += 1
        return count

    def _ingest_x_posts(self, clawbucks_dir: Path) -> int:
        """Import published tweets from logs/x_posts.jsonl."""
        posts_file = clawbucks_dir / "logs" / "x_posts.jsonl"
        if not posts_file.exists():
            return 0

        seen_tweet_ids: set[str] = set()
        count = 0
        with open(posts_file, encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    data = json.loads(line)
                except json.JSONDecodeError:
                    continue

                tweet_id = data.get("tweet_id", data.get("id", ""))
                if tweet_id in seen_tweet_ids:
                    continue
                if tweet_id:
                    seen_tweet_ids.add(tweet_id)

                content = data.get("text", data.get("content", ""))
                content_str = content if isinstance(content, str) else json.dumps(content)

                decision = Decision(
                    decision_id=str(uuid4()),
                    timestamp=data.get("timestamp", data.get("created_at", datetime.now(timezone.utc).isoformat())),
                    content_type="tweet",
                    content_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                    content_preview=content_str[:200],
                    niche_id=data.get("niche_id", "unknown"),
                    pillar_id=data.get("pillar_id"),
                    persona_id=data.get("persona_id"),
                    source_agent="x_content_agent",
                    decision="approve",
                    decision_source="human",
                )
                self.record(decision)
                count += 1
        return count

    def _ingest_pending_drafts(self, clawbucks_dir: Path) -> int:
        """Import skipped/edited drafts from queue/pending_drafts.json."""
        drafts_file = clawbucks_dir / "queue" / "pending_drafts.json"
        if not drafts_file.exists():
            return 0

        try:
            with open(drafts_file, encoding="utf-8") as f:
                drafts = json.load(f)
        except (json.JSONDecodeError, ValueError):
            return 0

        if isinstance(drafts, dict):
            drafts = drafts.get("drafts", [drafts])

        count = 0
        for draft in drafts:
            content = draft.get("content", draft.get("text", ""))
            content_str = content if isinstance(content, str) else json.dumps(content)
            status = draft.get("status", "skip")
            decision_val = "edit" if status in ("edited", "edit") else "skip"

            decision = Decision(
                decision_id=str(uuid4()),
                timestamp=draft.get("timestamp", datetime.now(timezone.utc).isoformat()),
                content_type=draft.get("content_type", "tweet"),
                content_hash=hashlib.sha256(content_str.encode()).hexdigest(),
                content_preview=content_str[:200],
                niche_id=draft.get("niche_id", "unknown"),
                pillar_id=draft.get("pillar_id"),
                persona_id=draft.get("persona_id"),
                source_agent=draft.get("source_agent", "content_agent"),
                decision=decision_val,
                decision_source="human",
                revision_notes=draft.get("revision_notes"),
            )
            self.record(decision)
            count += 1
        return count

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass
class Decision:
    decision_id: str          # UUID
    timestamp: str            # ISO 8601
    content_type: str         # "tweet", "thread", "article", "reply", "scout_proposal"
    content_hash: str         # SHA256 of content for dedup
    content_preview: str      # First 200 chars
    niche_id: str             # "ai-orchestration", "ou-sports"
    pillar_id: str | None  # "verification-asymmetry", "build-in-public"
    persona_id: str | None  # "crtx-ai"
    source_agent: str         # "content_agent", "x_content_agent", "niche_scout"
    decision: str             # "approve", "edit", "skip", "auto_approve"
    decision_source: str      # "human", "arbiter", "taxonomy"
    arbiter_confidence: float | None = None
    arbiter_model: str | None = None
    arbiter_issues: int | None = None
    revision_notes: str | None = None
    revision_category: str | None = None  # "tone", "accuracy", "length", "missing_data", "persona_drift"  # noqa: E501
    engagement_rate: float | None = None
    impressions: int | None = None
    outcome_score: float | None = None
    task_class: str | None = None
    taxonomy_action: str | None = None  # "auto_ship", "flag", "pause"
    decision_reason: str | None = None  # Human-tagged reason: "on_voice", "strong_data", "too_generic", etc.  # noqa: E501


@dataclass
class Pattern:
    pattern_id: str
    discovered_at: str
    last_confirmed: str
    pattern_type: str         # "always_approve", "always_edit", "revision_predictor", "confidence_threshold", "high_performer", "low_performer", "edit_trigger"  # noqa: E501
    description: str
    conditions: dict
    prediction: str
    confidence: float
    sample_size: int
    status: str               # "active", "weak", "retired"
    false_positive_count: int = 0
    last_false_positive: str | None = None


@dataclass
class TaxonomyRule:
    rule_id: str
    created_at: str
    content_type: str
    niche_id: str | None
    pillar_id: str | None
    min_arbiter_confidence: float
    action: str               # "auto_ship", "flag", "pause"
    consecutive_approvals: int
    required_streak: int
    last_human_override: str | None = None
    max_auto_ships_per_day: int = 20
    requires_arbiter: bool = True
    cooldown_after_edit: int = 0


@dataclass
class MemoryState:
    version: str = "0.3.0"
    created_at: str = ""
    last_updated: str = ""
    total_decisions: int = 0
    total_auto_ships: int = 0
    total_human_overrides: int = 0
    patterns: list = field(default_factory=list)
    taxonomy_rules: list = field(default_factory=list)
    active_tasks: list = field(default_factory=list)

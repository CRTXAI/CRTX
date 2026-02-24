from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional


@dataclass
class Decision:
    decision_id: str          # UUID
    timestamp: str            # ISO 8601
    content_type: str         # "tweet", "thread", "article", "reply", "scout_proposal"
    content_hash: str         # SHA256 of content for dedup
    content_preview: str      # First 200 chars
    niche_id: str             # "ai-orchestration", "ou-sports"
    pillar_id: Optional[str]  # "verification-asymmetry", "build-in-public"
    persona_id: Optional[str]  # "crtx-ai"
    source_agent: str         # "content_agent", "x_content_agent", "niche_scout"
    decision: str             # "approve", "edit", "skip", "auto_approve"
    decision_source: str      # "human", "arbiter", "taxonomy"
    arbiter_confidence: Optional[float] = None
    arbiter_model: Optional[str] = None
    arbiter_issues: Optional[int] = None
    revision_notes: Optional[str] = None
    revision_category: Optional[str] = None  # "tone", "accuracy", "length", "missing_data", "persona_drift"
    engagement_rate: Optional[float] = None
    impressions: Optional[int] = None
    outcome_score: Optional[float] = None
    task_class: Optional[str] = None
    taxonomy_action: Optional[str] = None  # "auto_ship", "flag", "pause"


@dataclass
class Pattern:
    pattern_id: str
    discovered_at: str
    last_confirmed: str
    pattern_type: str         # "always_approve", "always_edit", "revision_predictor", "confidence_threshold", "high_performer", "low_performer", "edit_trigger"
    description: str
    conditions: dict
    prediction: str
    confidence: float
    sample_size: int
    status: str               # "active", "weak", "retired"
    false_positive_count: int = 0
    last_false_positive: Optional[str] = None


@dataclass
class TaxonomyRule:
    rule_id: str
    created_at: str
    content_type: str
    niche_id: Optional[str]
    pillar_id: Optional[str]
    min_arbiter_confidence: float
    action: str               # "auto_ship", "flag", "pause"
    consecutive_approvals: int
    required_streak: int
    last_human_override: Optional[str] = None
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

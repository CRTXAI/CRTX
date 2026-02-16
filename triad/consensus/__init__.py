"""Consensus protocol for the Triad Orchestrator.

Provides suggestion evaluation, escalation, vote tallying,
tiebreak resolution, and formal consensus across pipeline modes.
"""

from triad.consensus.protocol import ConsensusEngine
from triad.consensus.suggestions import (
    SuggestionEvaluator,
    format_suggestion_for_evaluation,
    parse_evaluation,
)
from triad.consensus.voting import (
    build_consensus_result,
    extract_winner,
    select_tiebreaker,
    tally_votes,
)

__all__ = [
    "ConsensusEngine",
    "SuggestionEvaluator",
    "build_consensus_result",
    "extract_winner",
    "format_suggestion_for_evaluation",
    "parse_evaluation",
    "select_tiebreaker",
    "tally_votes",
]

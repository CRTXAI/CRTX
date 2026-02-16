"""Vote collection, tallying, and tiebreak resolution.

Provides the core voting mechanics used by the consensus protocol
for both parallel mode winner selection and suggestion escalation.
"""

from __future__ import annotations

import logging
import re

from triad.schemas.consensus import ConsensusResult, VoteTally
from triad.schemas.pipeline import ModelConfig

logger = logging.getLogger(__name__)

# Regex for extracting WINNER: <key> from tiebreaker output
_WINNER_RE = re.compile(r"WINNER:\s*(\S+)")


def tally_votes(votes: dict[str, str]) -> VoteTally:
    """Count votes and identify winner or tie.

    Args:
        votes: Mapping of voter key → voted-for option.

    Returns:
        VoteTally with counts, winner (if clear), and tie info.
    """
    counts: dict[str, int] = {}
    for voted_for in votes.values():
        counts[voted_for] = counts.get(voted_for, 0) + 1

    if not counts:
        return VoteTally(counts={}, winner=None, is_tie=True, tied_options=[])

    max_count = max(counts.values())
    leaders = [k for k, c in counts.items() if c == max_count]

    if len(leaders) == 1:
        return VoteTally(
            counts=counts,
            winner=leaders[0],
            is_tie=False,
            tied_options=[],
        )

    return VoteTally(
        counts=counts,
        winner=None,
        is_tie=True,
        tied_options=sorted(leaders),
    )


def select_tiebreaker(registry: dict[str, ModelConfig]) -> str:
    """Select the tiebreaker model — highest verifier fitness.

    Args:
        registry: Model registry.

    Returns:
        Model key of the tiebreaker.

    Raises:
        RuntimeError: If the registry is empty.
    """
    if not registry:
        raise RuntimeError("No models available for tiebreaker")
    return max(registry, key=lambda k: registry[k].fitness.verifier)


def extract_winner(content: str) -> str | None:
    """Extract WINNER: <key> from tiebreaker model output."""
    match = _WINNER_RE.search(content)
    return match.group(1) if match else None


def build_consensus_result(
    votes: dict[str, str],
    tally: VoteTally,
    tiebreaker_model: str | None = None,
    tiebreaker_winner: str | None = None,
) -> ConsensusResult:
    """Build a ConsensusResult from votes and tally data.

    Args:
        votes: Voter → voted-for option.
        tally: The vote tally.
        tiebreaker_model: Model key of tiebreaker (if tie occurred).
        tiebreaker_winner: Winner chosen by tiebreaker (if tie occurred).

    Returns:
        ConsensusResult with method, winner, and rationale.
    """
    if not tally.is_tie and tally.winner:
        # Check for unanimity
        unique_choices = set(votes.values())
        method = "unanimous" if len(unique_choices) == 1 else "majority"
        return ConsensusResult(
            method=method,
            winner=tally.winner,
            votes=votes,
            tiebreaker_used=False,
            tiebreaker_model=None,
            rationale=(
                f"{method.capitalize()} consensus: "
                f"{tally.winner} with {tally.counts.get(tally.winner, 0)} votes"
            ),
        )

    # Tie — tiebreaker resolved it
    winner = tiebreaker_winner or (
        tally.tied_options[0] if tally.tied_options else ""
    )
    return ConsensusResult(
        method="tiebreak",
        winner=winner,
        votes=votes,
        tiebreaker_used=True,
        tiebreaker_model=tiebreaker_model,
        rationale=(
            f"Tie between {tally.tied_options}, "
            f"resolved by tiebreaker ({tiebreaker_model}): {winner}"
        ),
    )

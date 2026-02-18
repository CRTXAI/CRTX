"""Consensus protocol engine for CRTX.

Manages the full consensus flow: suggestion evaluation by primary
role-holder, escalation to group vote on high-confidence rejections,
and formal consensus resolution for parallel/debate voting.
"""

from __future__ import annotations

import logging

from triad.consensus.suggestions import SuggestionEvaluator
from triad.consensus.voting import (
    build_consensus_result,
    extract_winner,
    select_tiebreaker,
    tally_votes,
)
from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.consensus import (
    ConsensusResult,
    EscalationResult,
    SuggestionDecision,
)
from triad.schemas.messages import Suggestion
from triad.schemas.pipeline import ModelConfig, PipelineConfig

logger = logging.getLogger(__name__)

# Suggestions with confidence above this threshold can be escalated
_ESCALATION_THRESHOLD = 0.7


class ConsensusEngine:
    """Full consensus protocol engine.

    Coordinates suggestion evaluation, escalation, and consensus
    resolution across all pipeline modes.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
    ) -> None:
        self._config = config
        self._registry = registry
        self._evaluator = SuggestionEvaluator(
            registry, timeout=config.default_timeout,
        )

    async def evaluate_suggestion(
        self,
        suggestion: Suggestion,
        evaluator_key: str,
        task: str,
        current_approach: str = "",
    ) -> SuggestionDecision:
        """Evaluate a cross-domain suggestion via the primary role-holder.

        Args:
            suggestion: The suggestion to evaluate.
            evaluator_key: Registry key of the primary role-holder model.
            task: The task description.
            current_approach: Current approach in the target domain.

        Returns:
            SuggestionDecision with accept/reject and rationale.
        """
        return await self._evaluator.evaluate(
            suggestion=suggestion,
            evaluator_key=evaluator_key,
            task=task,
            current_approach=current_approach,
        )

    async def escalate_suggestion(
        self,
        suggestion: Suggestion,
        pipeline_models: dict[str, str],
        task: str,
        current_approach: str = "",
    ) -> EscalationResult:
        """Escalate a rejected suggestion to a group vote.

        All pipeline models vote accept/reject. Majority wins.

        Args:
            suggestion: The rejected suggestion to escalate.
            pipeline_models: Role → model_key mapping.
            task: The task description.
            current_approach: Current approach in the target domain.

        Returns:
            EscalationResult with votes and group decision.
        """
        return await self._evaluator.escalate(
            suggestion=suggestion,
            pipeline_models=pipeline_models,
            task=task,
            current_approach=current_approach,
        )

    def should_escalate(self, suggestion: Suggestion) -> bool:
        """Check if a rejected suggestion should be escalated.

        Suggestions with confidence above the escalation threshold
        (0.7) are eligible for group vote escalation.
        """
        return suggestion.confidence > _ESCALATION_THRESHOLD

    async def resolve_consensus(
        self,
        votes: dict[str, str],
        task: str,
        candidate_outputs: dict[str, str] | None = None,
    ) -> ConsensusResult:
        """Resolve a consensus from collected votes.

        Tallies votes, identifies majority winner or tie. On tie,
        invokes the tiebreaker model to decide.

        Args:
            votes: Voter model key → voted-for option key.
            task: The task description for tiebreaker context.
            candidate_outputs: Option key → output text (for tiebreaker
                               prompt context). Optional.

        Returns:
            ConsensusResult with method, winner, and rationale.
        """
        tally = tally_votes(votes)

        if not tally.is_tie and tally.winner:
            return build_consensus_result(votes, tally)

        # Tie — invoke tiebreaker
        tiebreaker_key = select_tiebreaker(self._registry)
        logger.info(
            "Vote tie between %s, invoking tiebreaker: %s",
            tally.tied_options, tiebreaker_key,
        )

        tiebreaker_winner = await self._run_tiebreaker(
            tiebreaker_key=tiebreaker_key,
            tied_options=tally.tied_options,
            task=task,
            candidate_outputs=candidate_outputs,
        )

        return build_consensus_result(
            votes, tally, tiebreaker_key, tiebreaker_winner,
        )

    async def _run_tiebreaker(
        self,
        tiebreaker_key: str,
        tied_options: list[str],
        task: str,
        candidate_outputs: dict[str, str] | None = None,
    ) -> str:
        """Run the tiebreaker model to resolve a tie.

        Returns the key of the winning option.
        """
        model_config = self._registry[tiebreaker_key]

        # Build tied_options data for template
        options_data = []
        for key in tied_options:
            output = (
                candidate_outputs.get(key, "No output available")
                if candidate_outputs
                else "No output available"
            )
            options_data.append({"key": key, "output": output})

        system = render_prompt(
            "tiebreak",
            task=task,
            tied_options=options_data,
        )

        provider = LiteLLMProvider(model_config)
        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Evaluate the tied options and select "
                    "a winner as described in your system instructions."
                ),
            }],
            system=system,
            timeout=self._config.default_timeout,
        )

        winner = extract_winner(msg.content)
        if winner and winner in tied_options:
            return winner

        # Fallback: pick first tied option
        logger.warning(
            "Tiebreaker output didn't contain valid WINNER, "
            "falling back to first tied option: %s",
            tied_options[0],
        )
        return tied_options[0]

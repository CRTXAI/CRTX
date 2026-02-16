"""Suggestion evaluation and escalation logic.

Handles the flow where cross-domain suggestions are evaluated by the
primary role-holder, and high-confidence rejected suggestions can be
escalated to a group vote.
"""

from __future__ import annotations

import logging
import re

from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.consensus import (
    EscalationResult,
    SuggestionDecision,
    SuggestionVerdict,
)
from triad.schemas.messages import Suggestion
from triad.schemas.pipeline import ModelConfig

logger = logging.getLogger(__name__)

# Regex for extracting DECISION: ACCEPT|REJECT from evaluation output
_DECISION_RE = re.compile(r"DECISION:\s*(ACCEPT|REJECT)", re.IGNORECASE)

# Regex for extracting RATIONALE: <text> from evaluation output
_RATIONALE_RE = re.compile(
    r"RATIONALE:\s*(.+)", re.IGNORECASE | re.DOTALL,
)


def format_suggestion_for_evaluation(suggestion: Suggestion) -> str:
    """Format a Suggestion into a readable prompt block.

    Args:
        suggestion: The cross-domain suggestion to format.

    Returns:
        Formatted string with all suggestion fields.
    """
    lines = [
        f"**Domain:** {suggestion.domain.value}",
        f"**Rationale:** {suggestion.rationale}",
        f"**Confidence:** {suggestion.confidence:.2f}",
    ]
    if suggestion.code_sketch:
        lines.append(f"**Code sketch:** {suggestion.code_sketch}")
    if suggestion.impact_assessment:
        lines.append(f"**Impact:** {suggestion.impact_assessment}")
    return "\n".join(lines)


def parse_evaluation(content: str) -> tuple[SuggestionVerdict, str]:
    """Parse an evaluation response into verdict and rationale.

    Returns:
        Tuple of (verdict, rationale). Defaults to REJECT if unparseable.
    """
    decision_match = _DECISION_RE.search(content)
    verdict = SuggestionVerdict.REJECT  # conservative default
    if decision_match:
        raw = decision_match.group(1).upper()
        if raw == "ACCEPT":
            verdict = SuggestionVerdict.ACCEPT

    rationale_match = _RATIONALE_RE.search(content)
    rationale = rationale_match.group(1).strip() if rationale_match else content

    return verdict, rationale


class SuggestionEvaluator:
    """Evaluates cross-domain suggestions via the primary role-holder.

    Renders the evaluate_suggestion prompt, calls the primary role-holder
    model, and parses the accept/reject decision.
    """

    def __init__(
        self,
        registry: dict[str, ModelConfig],
        timeout: int = 120,
    ) -> None:
        self._registry = registry
        self._timeout = timeout

    async def evaluate(
        self,
        suggestion: Suggestion,
        evaluator_key: str,
        task: str,
        current_approach: str = "",
    ) -> SuggestionDecision:
        """Evaluate a suggestion via the primary role-holder model.

        Args:
            suggestion: The cross-domain suggestion to evaluate.
            evaluator_key: Registry key of the primary role-holder model.
            task: The task description for context.
            current_approach: The current approach in the target domain.

        Returns:
            SuggestionDecision with accept/reject and rationale.
        """
        model_config = self._registry[evaluator_key]

        system = render_prompt(
            "evaluate_suggestion",
            task=task,
            domain=suggestion.domain.value,
            suggestion_rationale=suggestion.rationale,
            suggestion_confidence=f"{suggestion.confidence:.2f}",
            suggestion_code_sketch=suggestion.code_sketch,
            suggestion_impact=suggestion.impact_assessment,
            current_approach=current_approach,
        )

        provider = LiteLLMProvider(model_config)
        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Evaluate the suggestion described in your "
                    "system instructions."
                ),
            }],
            system=system,
            timeout=self._timeout,
        )

        verdict, rationale = parse_evaluation(msg.content)

        logger.info(
            "Suggestion evaluated by %s: %s",
            evaluator_key, verdict.value,
        )

        return SuggestionDecision(
            suggestion_id=f"{suggestion.domain.value}-{id(suggestion)}",
            evaluator_model=evaluator_key,
            decision=verdict,
            rationale=rationale,
        )

    async def escalate(
        self,
        suggestion: Suggestion,
        pipeline_models: dict[str, str],
        task: str,
        current_approach: str = "",
    ) -> EscalationResult:
        """Escalate a rejected suggestion to a group vote.

        All pipeline models vote accept/reject on the suggestion.
        Majority wins.

        Args:
            suggestion: The rejected suggestion to escalate.
            pipeline_models: Mapping of role → model_key for current pipeline.
            task: The task description.
            current_approach: Current approach in the target domain.

        Returns:
            EscalationResult with votes and final decision.
        """
        votes: dict[str, str] = {}
        seen_models: set[str] = set()

        for _role, model_key in pipeline_models.items():
            # Avoid duplicate votes from the same model
            if model_key in seen_models:
                continue
            seen_models.add(model_key)

            decision = await self.evaluate(
                suggestion=suggestion,
                evaluator_key=model_key,
                task=task,
                current_approach=current_approach,
            )
            votes[model_key] = decision.decision.value

        # Tally: majority wins
        accept_count = sum(
            1 for v in votes.values() if v == SuggestionVerdict.ACCEPT
        )
        reject_count = sum(
            1 for v in votes.values() if v == SuggestionVerdict.REJECT
        )

        if accept_count > reject_count:
            final = SuggestionVerdict.ACCEPT
            rationale = (
                f"Group voted {accept_count}-{reject_count} to accept"
            )
        else:
            # Tie or reject majority → reject (conservative)
            final = SuggestionVerdict.REJECT
            rationale = (
                f"Group voted {accept_count}-{reject_count} to reject"
            )

        suggestion_id = f"{suggestion.domain.value}-{id(suggestion)}"
        logger.info(
            "Escalation result for %s: %s (%s)",
            suggestion_id, final.value, rationale,
        )

        return EscalationResult(
            suggestion_id=suggestion_id,
            votes=votes,
            decision=final,
            rationale=rationale,
        )

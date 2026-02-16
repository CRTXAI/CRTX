"""Consensus schemas for the Triad Orchestrator.

Defines result types for multi-model pipeline modes (ParallelResult,
DebateResult), suggestion evaluation and escalation decisions
(SuggestionDecision, EscalationResult), vote tallying (VoteTally),
and consensus resolution (ConsensusResult).
"""

from __future__ import annotations

from enum import StrEnum

from pydantic import BaseModel, Field


class SuggestionVerdict(StrEnum):
    """Decision on a cross-domain suggestion."""

    ACCEPT = "accept"
    REJECT = "reject"


class ParallelResult(BaseModel):
    """Result from the parallel exploration pipeline mode.

    Captures each model's independent solution, cross-review scores,
    consensus votes, the winning model, and the synthesized final output.
    """

    individual_outputs: dict[str, str] = Field(
        default_factory=dict,
        description="Model key → independent solution output",
    )
    scores: dict[str, dict[str, dict[str, int]]] = Field(
        default_factory=dict,
        description=(
            "Cross-review scores: reviewer → reviewed → "
            "{architecture, implementation, quality} (1-10 each)"
        ),
    )
    votes: dict[str, str] = Field(
        default_factory=dict,
        description="Consensus votes: voter model key → voted-for model key",
    )
    winner: str = Field(
        default="",
        description="Model key of the winning approach",
    )
    synthesized_output: str = Field(
        default="",
        description="Final output after synthesis of winning + best elements",
    )


class DebateResult(BaseModel):
    """Result from the debate pipeline mode.

    Captures each model's position paper, structured rebuttals,
    updated final arguments, the judge's decision, and which model judged.
    """

    proposals: dict[str, str] = Field(
        default_factory=dict,
        description="Model key → initial position paper with tradeoff analysis",
    )
    rebuttals: dict[str, dict[str, str]] = Field(
        default_factory=dict,
        description=(
            "Structured rebuttals: rebutter model key → "
            "{target model key: rebuttal text}"
        ),
    )
    final_arguments: dict[str, str] = Field(
        default_factory=dict,
        description="Model key → updated proposal incorporating criticisms",
    )
    judgment: str = Field(
        default="",
        description="The judge's reasoned decision document",
    )
    judge_model: str = Field(
        default="",
        description="Model key of the judge that rendered the decision",
    )


class SuggestionDecision(BaseModel):
    """Result of a primary role-holder evaluating a cross-domain suggestion.

    When an agent suggests a change in another agent's domain, the primary
    role-holder for that domain evaluates it and issues an accept/reject.
    """

    suggestion_id: str = Field(
        description="Unique ID of the suggestion being evaluated",
    )
    evaluator_model: str = Field(
        description="Model key of the primary role-holder that evaluated",
    )
    decision: SuggestionVerdict = Field(
        description="Whether the suggestion was accepted or rejected",
    )
    rationale: str = Field(
        default="", description="Reasoning behind the decision",
    )


class EscalationResult(BaseModel):
    """Result of escalating a rejected suggestion to a group vote.

    When a high-confidence suggestion is rejected by the primary role-holder,
    the suggester can escalate to a group vote across all pipeline models.
    Majority wins.
    """

    suggestion_id: str = Field(
        description="Unique ID of the escalated suggestion",
    )
    votes: dict[str, str] = Field(
        default_factory=dict,
        description="Voter model key → 'accept' or 'reject'",
    )
    decision: SuggestionVerdict = Field(
        description="Final group decision on the suggestion",
    )
    rationale: str = Field(
        default="", description="Summary of the group's reasoning",
    )


class VoteTally(BaseModel):
    """Result of counting votes for a set of options.

    Used by both parallel mode voting and suggestion escalation.
    """

    counts: dict[str, int] = Field(
        default_factory=dict,
        description="Option → number of votes received",
    )
    winner: str | None = Field(
        default=None,
        description="The option with the most votes, or None if tied",
    )
    is_tie: bool = Field(
        default=False, description="Whether there is a tie for first place",
    )
    tied_options: list[str] = Field(
        default_factory=list,
        description="Options tied for first place (empty if no tie)",
    )


class ConsensusResult(BaseModel):
    """Result of a formal consensus resolution.

    Used when multiple models must agree on a winner — in parallel mode
    voting, debate judgment ties, or suggestion escalation.
    """

    method: str = Field(
        description="How consensus was reached: 'majority', 'tiebreak', or 'unanimous'",
    )
    winner: str = Field(
        description="The winning option (model key or suggestion decision)",
    )
    votes: dict[str, str] = Field(
        default_factory=dict,
        description="Voter → voted-for option",
    )
    tiebreaker_used: bool = Field(
        default=False,
        description="Whether a tiebreaker was needed to resolve",
    )
    tiebreaker_model: str | None = Field(
        default=None,
        description="Model key of the tiebreaker, if used",
    )
    rationale: str = Field(
        default="", description="Explanation of the consensus outcome",
    )

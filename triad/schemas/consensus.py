"""Consensus schemas for parallel exploration and debate pipeline modes.

Defines result types for the multi-model pipeline modes: ParallelResult
captures fan-out, cross-review, voting, and synthesis; DebateResult
captures proposals, rebuttals, final arguments, and judgment.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


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

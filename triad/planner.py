"""Task Planner — expands rough ideas into structured task specifications.

Quick mode: single LLM call to expand a description into a full spec.
Interactive mode: two-phase — asks clarifying questions, then expands
with user answers incorporated.

Uses the cheapest model above 0.70 architect fitness by default.
"""

from __future__ import annotations

import logging
import re

from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.messages import TokenUsage
from triad.schemas.pipeline import ModelConfig, TaskSpec
from triad.schemas.planner import PlannerResult

logger = logging.getLogger(__name__)

# Regex for extracting TECH_STACK: line from planner output
_TECH_STACK_RE = re.compile(r"TECH_STACK:\s*(.+)", re.IGNORECASE)


class TaskPlanner:
    """Expands rough task descriptions into structured specifications.

    Quick mode: one LLM call produces the full spec.
    Interactive mode: first call asks clarifying questions, second call
    produces the spec with user answers incorporated.
    """

    def __init__(self, registry: dict[str, ModelConfig]) -> None:
        self._registry = registry

    def select_model(self, model_override: str | None = None) -> str:
        """Select the planning model.

        Uses the cheapest model above 0.70 architect fitness. Falls back
        to the highest-fitness model if none meet the threshold. Can be
        overridden by passing a specific model key.

        Returns:
            The model registry key to use.

        Raises:
            RuntimeError: If the registry is empty or override not found.
        """
        if model_override:
            if model_override not in self._registry:
                raise RuntimeError(
                    f"Model '{model_override}' not found in registry. "
                    f"Available: {', '.join(sorted(self._registry))}"
                )
            return model_override

        if not self._registry:
            raise RuntimeError("No models available in registry")

        # Find cheapest model above 0.70 architect fitness
        eligible = {
            k: v for k, v in self._registry.items()
            if v.fitness.architect >= 0.70
        }

        if eligible:
            # Cheapest by average cost per token
            return min(
                eligible,
                key=lambda k: _avg_cost(eligible[k]),
            )

        # Fallback: highest architect fitness
        return max(
            self._registry,
            key=lambda k: self._registry[k].fitness.architect,
        )

    async def plan(
        self,
        description: str,
        interactive: bool = False,
        user_answers: str | None = None,
        model_override: str | None = None,
    ) -> PlannerResult:
        """Expand a rough description into a structured task specification.

        Args:
            description: The rough task description from the user.
            interactive: If True, returns clarifying questions on first
                call. Caller should collect answers and call again with
                user_answers set.
            user_answers: Answers to clarifying questions (interactive
                mode phase 2).
            model_override: Use a specific model instead of auto-select.

        Returns:
            PlannerResult with the expanded specification.
        """
        model_key = self.select_model(model_override)
        model_config = self._registry[model_key]
        provider = LiteLLMProvider(model_config)

        total_prompt_tokens = 0
        total_completion_tokens = 0
        total_cost = 0.0
        clarifying_questions: list[str] | None = None

        if interactive and user_answers is None:
            # Phase 1: Ask clarifying questions
            questions_text, usage = await self._ask_questions(
                description, provider,
            )
            clarifying_questions = _parse_questions(questions_text)

            return PlannerResult(
                original_description=description,
                expanded_spec=questions_text,
                task_spec=TaskSpec(task=description),
                tech_stack_inferred=[],
                model_used=model_key,
                token_usage=usage,
                cost=usage.cost,
                interactive=True,
                clarifying_questions=clarifying_questions,
                user_answers=None,
            )

        # Quick mode or interactive phase 2
        expanded, usage = await self._expand(
            description, provider, user_answers,
        )

        total_prompt_tokens += usage.prompt_tokens
        total_completion_tokens += usage.completion_tokens
        total_cost += usage.cost

        # Parse tech stack from output
        tech_stack = _extract_tech_stack(expanded)

        # Build a TaskSpec from the expanded output
        task_spec = TaskSpec(
            task=expanded,
            context="",
            domain_rules="",
        )

        total_usage = TokenUsage(
            prompt_tokens=total_prompt_tokens,
            completion_tokens=total_completion_tokens,
            cost=total_cost,
        )

        return PlannerResult(
            original_description=description,
            expanded_spec=expanded,
            task_spec=task_spec,
            tech_stack_inferred=tech_stack,
            model_used=model_key,
            token_usage=total_usage,
            cost=total_cost,
            interactive=interactive,
            clarifying_questions=None,
            user_answers=user_answers,
        )

    async def _ask_questions(
        self,
        description: str,
        provider: LiteLLMProvider,
    ) -> tuple[str, TokenUsage]:
        """Phase 1: Ask clarifying questions for interactive mode."""
        system = render_prompt("planner_questions", description=description)

        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Ask clarifying questions about this project idea."
                ),
            }],
            system=system,
            timeout=60,
        )

        usage = msg.token_usage or TokenUsage(
            prompt_tokens=0, completion_tokens=0, cost=0.0,
        )
        return msg.content, usage

    async def _expand(
        self,
        description: str,
        provider: LiteLLMProvider,
        user_answers: str | None = None,
    ) -> tuple[str, TokenUsage]:
        """Expand a description into a structured specification."""
        tpl_vars: dict = {"description": description}
        if user_answers:
            tpl_vars["user_answers"] = user_answers

        system = render_prompt("planner", **tpl_vars)

        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Expand this idea into a complete task specification."
                ),
            }],
            system=system,
            timeout=90,
        )

        usage = msg.token_usage or TokenUsage(
            prompt_tokens=0, completion_tokens=0, cost=0.0,
        )
        return msg.content, usage


def _avg_cost(config: ModelConfig) -> float:
    """Weighted average cost per token (4:1 input-to-output ratio)."""
    return (4 * config.cost_input + config.cost_output) / 5


def _extract_tech_stack(text: str) -> list[str]:
    """Extract the TECH_STACK line from planner output."""
    match = _TECH_STACK_RE.search(text)
    if match:
        raw = match.group(1)
        return [t.strip() for t in raw.split(",") if t.strip()]
    return []


def _parse_questions(text: str) -> list[str]:
    """Parse numbered questions from the planner output."""
    questions: list[str] = []
    for line in text.splitlines():
        stripped = line.strip()
        # Match lines starting with a number followed by . or )
        if re.match(r"^\d+[.)]\s+", stripped):
            questions.append(stripped)
    return questions

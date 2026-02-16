"""Sequential pipeline engine for the Triad Orchestrator.

Runs 4 stages in order: Architect -> Implement -> Refactor -> Verify.
Each stage loads its role prompt template, renders it with task context
and previous output, calls a LiteLLMProvider, and passes its output
to the next stage. Cross-domain suggestions are collected and injected
downstream. All AgentMessages are logged to a session audit trail.

The Arbiter layer reviews stage outputs (based on arbiter_mode) using
cross-model enforcement. REJECT triggers a retry with structured feedback,
HALT stops the pipeline. Optional ISR reconciliation runs after Verify.
"""

from __future__ import annotations

import logging
import re
import time

from triad.arbiter.arbiter import ArbiterEngine
from triad.arbiter.feedback import format_arbiter_feedback
from triad.arbiter.reconciler import ReconciliationEngine
from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.providers.registry import get_best_model_for_role
from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    Suggestion,
)
from triad.schemas.pipeline import (
    ArbiterMode,
    ModelConfig,
    PipelineConfig,
    PipelineResult,
    TaskSpec,
)

logger = logging.getLogger(__name__)

# Ordered pipeline stages
_STAGE_SEQUENCE: list[PipelineStage] = [
    PipelineStage.ARCHITECT,
    PipelineStage.IMPLEMENT,
    PipelineStage.REFACTOR,
    PipelineStage.VERIFY,
]

# Map pipeline stages to prompt template file names (without .md)
_STAGE_PROMPT: dict[PipelineStage, str] = {
    PipelineStage.ARCHITECT: "architect",
    PipelineStage.IMPLEMENT: "implementer",
    PipelineStage.REFACTOR: "refactorer",
    PipelineStage.VERIFY: "verifier",
}

# Map pipeline stages to the message type each produces
_STAGE_MSG_TYPE: dict[PipelineStage, MessageType] = {
    PipelineStage.ARCHITECT: MessageType.PROPOSAL,
    PipelineStage.IMPLEMENT: MessageType.IMPLEMENTATION,
    PipelineStage.REFACTOR: MessageType.IMPLEMENTATION,
    PipelineStage.VERIFY: MessageType.VERIFICATION,
}

# Map each stage to the next stage in the pipeline
_NEXT_STAGE: dict[PipelineStage, PipelineStage] = {
    PipelineStage.ARCHITECT: PipelineStage.IMPLEMENT,
    PipelineStage.IMPLEMENT: PipelineStage.REFACTOR,
    PipelineStage.REFACTOR: PipelineStage.VERIFY,
    PipelineStage.VERIFY: PipelineStage.VERIFY,
}

# Which stages the Arbiter reviews in each mode
_ARBITER_STAGES: dict[ArbiterMode, set[PipelineStage]] = {
    ArbiterMode.OFF: set(),
    ArbiterMode.FINAL_ONLY: {PipelineStage.VERIFY},
    ArbiterMode.BOOKEND: {PipelineStage.ARCHITECT, PipelineStage.VERIFY},
    ArbiterMode.FULL: {
        PipelineStage.ARCHITECT,
        PipelineStage.IMPLEMENT,
        PipelineStage.REFACTOR,
        PipelineStage.VERIFY,
    },
}

# Regex for extracting CONFIDENCE: <value> from model output
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([\d.]+)")


class PipelineOrchestrator:
    """Sequential pipeline engine with Arbiter integration.

    Coordinates the 4-stage pipeline (Architect -> Implement -> Refactor ->
    Verify), passing output between stages, collecting cross-domain
    suggestions, and tracking tokens/cost. The Arbiter layer reviews
    stage outputs and can trigger retries (REJECT), halt the pipeline
    (HALT), or inject flags (FLAG). Optional ISR reconciliation runs
    after the Verify stage.
    """

    def __init__(
        self,
        task: TaskSpec,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
    ) -> None:
        self._task = task
        self._config = config
        self._registry = registry
        self._session: list[AgentMessage] = []
        self._arbiter = ArbiterEngine(config, registry)
        self._reconciler = ReconciliationEngine(config, registry)

    @property
    def session(self) -> list[AgentMessage]:
        """The session audit trail — all AgentMessages from the run."""
        return list(self._session)

    async def run(self) -> PipelineResult:
        """Execute the full 4-stage pipeline and return a PipelineResult.

        The Arbiter reviews stages based on arbiter_mode. REJECT triggers
        a retry with feedback injection (up to max_retries). HALT stops
        the pipeline immediately. FLAG injects warnings into downstream
        context. After Verify, optional ISR reconciliation checks for
        spec drift.

        Raises:
            RuntimeError: If model selection fails or a stage errors out.
        """
        start = time.monotonic()

        stages: dict[PipelineStage, AgentMessage] = {}
        all_suggestions: list[Suggestion] = []
        arbiter_reviews: list[ArbiterReview] = []
        architect_output = ""
        previous_output = ""
        flagged_issues = ""
        halted = False
        halt_reason = ""

        reviewed_stages = _ARBITER_STAGES.get(
            self._config.arbiter_mode, set()
        )

        for stage in _STAGE_SEQUENCE:
            logger.info("Starting stage: %s", stage.value)

            template_vars = self._build_template_vars(
                stage=stage,
                previous_output=previous_output,
                architect_output=architect_output,
                suggestions=all_suggestions,
                flagged_issues=flagged_issues,
            )

            msg = await self._run_stage(stage, template_vars)

            stages[stage] = msg
            self._session.append(msg)
            all_suggestions.extend(msg.suggestions)

            # Chain output: current output becomes next stage's input
            previous_output = msg.content
            if stage == PipelineStage.ARCHITECT:
                architect_output = msg.content

            logger.info(
                "Completed stage %s (model=%s, confidence=%.2f)",
                stage.value,
                msg.model,
                msg.confidence,
            )

            # --- Arbiter review ---
            if stage in reviewed_stages:
                review = await self._arbiter.review(
                    stage=stage,
                    stage_model=msg.model,
                    stage_output=msg.content,
                    task=self._task,
                    architect_output=architect_output,
                )
                arbiter_reviews.append(review)

                if review.verdict == Verdict.HALT:
                    halted = True
                    halt_reason = review.reasoning
                    logger.warning(
                        "HALT verdict for %s — stopping pipeline", stage.value
                    )
                    break

                if review.verdict == Verdict.REJECT:
                    msg, retry_reviews = await self._retry_stage(
                        stage=stage,
                        review=review,
                        previous_output=previous_output,
                        architect_output=architect_output,
                        suggestions=all_suggestions,
                        flagged_issues=flagged_issues,
                    )
                    arbiter_reviews.extend(retry_reviews)

                    # Check if retries ended with HALT
                    if retry_reviews and retry_reviews[-1].verdict == Verdict.HALT:
                        halted = True
                        halt_reason = retry_reviews[-1].reasoning
                        break

                    # Update stage output with the successful retry
                    stages[stage] = msg
                    self._session.append(msg)
                    previous_output = msg.content
                    if stage == PipelineStage.ARCHITECT:
                        architect_output = msg.content

                if review.verdict == Verdict.FLAG:
                    flagged_issues = self._format_flags(review)

        # --- ISR Reconciliation ---
        recon_review = None
        if (
            not halted
            and self._config.reconciliation_enabled
            and PipelineStage.VERIFY in stages
        ):
            verify_msg = stages[PipelineStage.VERIFY]
            recon_review = await self._reconciler.reconcile(
                task=self._task,
                architect_output=architect_output,
                implementation_summary=verify_msg.content,
                verifier_model=verify_msg.model,
            )
            arbiter_reviews.append(recon_review)

            if recon_review.verdict == Verdict.HALT:
                halted = True
                halt_reason = recon_review.reasoning

        # --- Build result ---
        duration = time.monotonic() - start
        total_cost = sum(
            msg.token_usage.cost for msg in stages.values() if msg.token_usage
        )
        # Include arbiter costs
        total_cost += sum(r.token_cost for r in arbiter_reviews)

        total_tokens = sum(
            (msg.token_usage.prompt_tokens + msg.token_usage.completion_tokens)
            for msg in stages.values()
            if msg.token_usage
        )

        return PipelineResult(
            task=self._task,
            config=self._config,
            stages=stages,
            suggestions=all_suggestions,
            total_cost=total_cost,
            total_tokens=total_tokens,
            duration_seconds=duration,
            success=not halted,
            arbiter_reviews=arbiter_reviews,
            halted=halted,
            halt_reason=halt_reason,
        )

    async def _retry_stage(
        self,
        stage: PipelineStage,
        review: ArbiterReview,
        previous_output: str,
        architect_output: str,
        suggestions: list[Suggestion],
        flagged_issues: str,
    ) -> tuple[AgentMessage, list[ArbiterReview]]:
        """Retry a stage after REJECT, injecting Arbiter feedback.

        Returns the final AgentMessage and any additional ArbiterReviews
        from the retry attempts.
        """
        stage_config = self._config.stages.get(stage)
        max_retries = stage_config.max_retries if stage_config else self._config.max_retries
        retry_reviews: list[ArbiterReview] = []
        current_review = review

        for retry in range(1, max_retries + 1):
            logger.info(
                "Retrying stage %s (attempt %d/%d) after REJECT",
                stage.value, retry, max_retries,
            )

            feedback = format_arbiter_feedback(current_review, retry)
            template_vars = self._build_template_vars(
                stage=stage,
                previous_output=previous_output,
                architect_output=architect_output,
                suggestions=suggestions,
                flagged_issues=flagged_issues,
                arbiter_feedback=feedback,
                retry_number=retry,
            )

            msg = await self._run_stage(stage, template_vars)

            # Re-review the retried output
            reviewed_stages = _ARBITER_STAGES.get(
                self._config.arbiter_mode, set()
            )
            if stage in reviewed_stages:
                retry_review = await self._arbiter.review(
                    stage=stage,
                    stage_model=msg.model,
                    stage_output=msg.content,
                    task=self._task,
                    architect_output=architect_output,
                )
                retry_reviews.append(retry_review)

                if retry_review.verdict in (Verdict.APPROVE, Verdict.FLAG):
                    return msg, retry_reviews

                if retry_review.verdict == Verdict.HALT:
                    return msg, retry_reviews

                # Still REJECT — continue loop with new feedback
                current_review = retry_review
            else:
                # Not re-reviewed (shouldn't happen, but safe fallback)
                return msg, retry_reviews

        # Exhausted retries — return last attempt
        logger.warning(
            "Stage %s exhausted %d retries, proceeding with last output",
            stage.value, max_retries,
        )
        return msg, retry_reviews

    async def _run_stage(
        self,
        stage: PipelineStage,
        template_vars: dict,
    ) -> AgentMessage:
        """Execute a single pipeline stage.

        Resolves the model, renders the prompt template, calls the provider,
        and sets routing metadata on the returned AgentMessage.
        """
        model_key = self._resolve_model_key(stage)
        model_config = self._registry[model_key]
        provider = LiteLLMProvider(model_config)

        # Render the system prompt from the stage's template
        prompt_name = _STAGE_PROMPT[stage]
        system = render_prompt(prompt_name, **template_vars)

        timeout = self._get_timeout(stage)

        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": "Execute the task as described in your system instructions.",
            }],
            system=system,
            timeout=timeout,
        )

        # Set routing metadata
        msg.from_agent = stage
        msg.to_agent = _NEXT_STAGE[stage]
        msg.msg_type = _STAGE_MSG_TYPE[stage]

        # Extract confidence score from the model's output
        msg.confidence = _extract_confidence(msg.content)

        return msg

    def _resolve_model_key(self, stage: PipelineStage) -> str:
        """Determine which model to use for a pipeline stage.

        Checks for a stage-specific override first, then falls back to
        the best-fit model from the registry.

        Raises:
            RuntimeError: If the configured model isn't in the registry
                          or no models are available.
        """
        stage_config = self._config.stages.get(stage)
        if stage_config and stage_config.model:
            if stage_config.model in self._registry:
                return stage_config.model
            raise RuntimeError(
                f"Stage {stage.value} configured with model '{stage_config.model}' "
                f"but it is not in the model registry"
            )

        best = get_best_model_for_role(self._registry, stage)
        if best is None:
            raise RuntimeError(f"No models available for stage {stage.value}")
        return best

    def _get_timeout(self, stage: PipelineStage) -> int:
        """Get the timeout in seconds for a pipeline stage."""
        stage_config = self._config.stages.get(stage)
        if stage_config:
            return stage_config.timeout
        return self._config.default_timeout

    def _build_template_vars(
        self,
        stage: PipelineStage,
        previous_output: str,
        architect_output: str,
        suggestions: list[Suggestion],
        flagged_issues: str = "",
        arbiter_feedback: str = "",
        retry_number: int = 0,
    ) -> dict:
        """Build the Jinja2 template variables for a stage prompt."""
        tpl_vars: dict = {
            "task": self._task.task,
            "context": self._task.context,
            "domain_context": self._task.domain_rules,
        }

        if previous_output:
            tpl_vars["previous_output"] = previous_output

        # Verifier also receives the architect output for comparison
        if stage == PipelineStage.VERIFY and architect_output:
            tpl_vars["architect_output"] = architect_output

        # Format upstream suggestions targeted at this stage
        upstream = _format_suggestions(suggestions, stage)
        if upstream:
            tpl_vars["upstream_suggestions"] = upstream

        # Reconciliation flag for the verifier
        if stage == PipelineStage.VERIFY:
            tpl_vars["reconciliation_enabled"] = self._config.reconciliation_enabled

        # Arbiter FLAG warnings from a previous stage
        if flagged_issues:
            tpl_vars["flagged_issues"] = flagged_issues

        # Arbiter REJECT feedback for retry
        if arbiter_feedback:
            tpl_vars["arbiter_feedback"] = arbiter_feedback
            tpl_vars["retry_number"] = retry_number

        return tpl_vars

    def _format_flags(self, review: ArbiterReview) -> str:
        """Format a FLAG review's issues for downstream injection."""
        if not review.issues:
            return f"The Arbiter flagged concerns (confidence: {review.confidence:.2f}). " \
                   f"Review and address if appropriate."

        lines: list[str] = [
            "The Arbiter flagged the following concerns from a previous stage:"
        ]
        for issue in review.issues:
            lines.append(
                f"- [{issue.severity.value}] [{issue.category.value}] {issue.description}"
            )
        return "\n".join(lines)


def _extract_confidence(content: str) -> float:
    """Extract the CONFIDENCE: <value> score from model output.

    Returns 0.0 if no valid confidence score is found.
    """
    match = _CONFIDENCE_RE.search(content)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return 0.0


def _format_suggestions(
    suggestions: list[Suggestion], target_stage: PipelineStage
) -> str:
    """Format suggestions targeted at a specific stage into readable text.

    Filters suggestions by domain and returns a formatted string for
    injection into the downstream prompt template.

    Returns:
        Formatted string, or empty string if no relevant suggestions.
    """
    relevant = [s for s in suggestions if s.domain == target_stage]
    if not relevant:
        return ""

    lines: list[str] = []
    for s in relevant:
        line = f"- {s.rationale} (confidence: {s.confidence})"
        if s.code_sketch:
            line += f"\n  Code sketch: {s.code_sketch}"
        if s.impact_assessment:
            line += f"\n  Impact: {s.impact_assessment}"
        lines.append(line)

    return "\n".join(lines)

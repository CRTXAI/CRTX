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

import asyncio
import logging
import re
import time
import uuid
from datetime import UTC, datetime

from triad.arbiter.arbiter import ArbiterEngine
from triad.arbiter.feedback import format_arbiter_feedback
from triad.arbiter.reconciler import ReconciliationEngine
from triad.consensus.protocol import ConsensusEngine
from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.routing.engine import RoutingEngine
from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.consensus import (
    ConsensusResult,
    DebateResult,
    ParallelResult,
    SuggestionDecision,
    SuggestionVerdict,
)
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
    PipelineMode,
    PipelineResult,
    TaskSpec,
)
from triad.schemas.routing import RoutingDecision
from triad.schemas.session import SessionRecord, StageRecord

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
        event_emitter: object | None = None,
    ) -> None:
        self._task = task
        self._config = config
        self._registry = registry
        self._emitter = event_emitter
        self._session: list[AgentMessage] = []
        self._arbiter = ArbiterEngine(config, registry)
        self._reconciler = ReconciliationEngine(config, registry)
        self._router = RoutingEngine(config, registry)
        self._consensus = ConsensusEngine(config, registry)
        self._routing_decisions: list[RoutingDecision] = []
        self._suggestion_decisions: list[SuggestionDecision] = []
        self._stage_models: dict[PipelineStage, str] = {}

    async def _emit(self, event_type: str, **data: object) -> None:
        """Emit a dashboard event if an emitter is attached."""
        if self._emitter is not None:
            await self._emitter.emit(event_type, **data)

    def _display_name(self, model_key: str) -> str:
        """Resolve a registry key to its display_name for UI events."""
        cfg = self._registry.get(model_key)
        return cfg.display_name if cfg else model_key

    def _display_name_from_litellm_id(self, litellm_id: str) -> str:
        """Resolve a LiteLLM model ID to a display_name."""
        for cfg in self._registry.values():
            if cfg.model == litellm_id:
                return cfg.display_name
        return litellm_id

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

        await self._emit(
            "pipeline_started",
            mode=self._config.pipeline_mode.value,
            task_summary=self._task.task[:200],
            models=list(self._registry.keys()),
        )

        for stage in _STAGE_SEQUENCE:
            logger.info("Starting stage: %s", stage.value)

            stage_start = time.monotonic()

            template_vars = self._build_template_vars(
                stage=stage,
                previous_output=previous_output,
                architect_output=architect_output,
                suggestions=all_suggestions,
                flagged_issues=flagged_issues,
            )

            # _run_stage selects the model and emits stage_started
            msg = await self._run_stage(stage, template_vars)

            stages[stage] = msg
            self._session.append(msg)

            # Chain output: current output becomes next stage's input
            previous_output = msg.content
            if stage == PipelineStage.ARCHITECT:
                architect_output = msg.content

            stage_duration = time.monotonic() - stage_start
            stage_cost = msg.token_usage.cost if msg.token_usage else 0.0
            await self._emit(
                "stage_completed",
                stage=stage.value,
                model=self._display_name(self._stage_models[stage]),
                duration=stage_duration,
                cost=stage_cost,
                confidence=msg.confidence,
                content_preview=msg.content[:200],
            )

            logger.info(
                "Completed stage %s (model=%s, confidence=%.2f)",
                stage.value,
                msg.model,
                msg.confidence,
            )

            # --- Evaluate cross-domain suggestions ---
            accepted_suggestions = await self._evaluate_suggestions(
                new_suggestions=msg.suggestions,
                current_stage=stage,
                previous_output=previous_output,
            )
            all_suggestions.extend(accepted_suggestions)

            # --- Arbiter review ---
            if stage in reviewed_stages:
                await self._emit(
                    "arbiter_started",
                    stage=stage.value,
                    arbiter_model=self._display_name_from_litellm_id(
                        msg.model,
                    ),
                )
                review = await self._arbiter.review(
                    stage=stage,
                    stage_model=msg.model,
                    stage_output=msg.content,
                    task=self._task,
                    architect_output=architect_output,
                )
                arbiter_reviews.append(review)
                await self._emit(
                    "arbiter_verdict",
                    stage=stage.value,
                    verdict=review.verdict.value,
                    confidence=review.confidence,
                    issues_count=len(review.issues),
                    reasoning_preview=review.reasoning[:200],
                    arbiter_model=self._display_name_from_litellm_id(
                        review.arbiter_model,
                    ),
                )

                if review.verdict == Verdict.HALT:
                    halted = True
                    halt_reason = review.reasoning
                    logger.warning(
                        "HALT verdict for %s — stopping pipeline", stage.value
                    )
                    await self._emit(
                        "pipeline_halted",
                        stage=stage.value,
                        reason=halt_reason[:200],
                        arbiter_model=self._display_name_from_litellm_id(
                            review.arbiter_model,
                        ),
                    )
                    break

                if review.verdict == Verdict.REJECT:
                    await self._emit(
                        "retry_triggered",
                        stage=stage.value,
                        retry_number=1,
                        reason=review.reasoning[:200],
                    )
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

                    # Emit final stage_completed with cumulative totals
                    retry_msg_cost = msg.token_usage.cost if msg.token_usage else 0.0
                    retry_arbiter_cost = sum(r.token_cost for r in retry_reviews)
                    total_stage_cost = stage_cost + retry_msg_cost + retry_arbiter_cost
                    total_stage_duration = time.monotonic() - stage_start
                    await self._emit(
                        "stage_completed",
                        stage=stage.value,
                        model=self._display_name(self._stage_models[stage]),
                        duration=total_stage_duration,
                        cost=total_stage_cost,
                        confidence=msg.confidence,
                        content_preview=msg.content[:200],
                    )

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

        if not halted:
            await self._emit(
                "pipeline_completed",
                success=True,
                total_cost=total_cost,
                total_tokens=total_tokens,
                duration=duration,
                session_id="",
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
            routing_decisions=self._routing_decisions,
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

            # _run_stage emits stage_started with the selected model
            msg = await self._run_stage(stage, template_vars)

            # Re-review the retried output
            reviewed_stages = _ARBITER_STAGES.get(
                self._config.arbiter_mode, set()
            )
            if stage in reviewed_stages:
                await self._emit(
                    "arbiter_started",
                    stage=stage.value,
                    arbiter_model=self._display_name_from_litellm_id(
                        msg.model,
                    ),
                )
                retry_review = await self._arbiter.review(
                    stage=stage,
                    stage_model=msg.model,
                    stage_output=msg.content,
                    task=self._task,
                    architect_output=architect_output,
                )
                retry_reviews.append(retry_review)
                await self._emit(
                    "arbiter_verdict",
                    stage=stage.value,
                    verdict=retry_review.verdict.value,
                    confidence=retry_review.confidence,
                    issues_count=len(retry_review.issues),
                    reasoning_preview=retry_review.reasoning[:200],
                    arbiter_model=self._display_name_from_litellm_id(
                        retry_review.arbiter_model,
                    ),
                )

                if retry_review.verdict in (Verdict.APPROVE, Verdict.FLAG):
                    return msg, retry_reviews

                if retry_review.verdict == Verdict.HALT:
                    return msg, retry_reviews

                # Still REJECT — continue loop with new feedback
                await self._emit(
                    "retry_triggered",
                    stage=stage.value,
                    retry_number=retry + 1,
                    reason=retry_review.reasoning[:200],
                )
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

        Uses the RoutingEngine to select the model, renders the prompt
        template, calls the provider, and sets routing metadata on the
        returned AgentMessage. Emits stage_started with the selected model.
        """
        decision = self._router.select_model(stage)
        self._routing_decisions.append(decision)
        self._stage_models[stage] = decision.model_key
        model_config = self._registry[decision.model_key]

        await self._emit(
            "stage_started",
            stage=stage.value,
            model=model_config.display_name,
        )

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

    async def _evaluate_suggestions(
        self,
        new_suggestions: list[Suggestion],
        current_stage: PipelineStage,
        previous_output: str,
    ) -> list[Suggestion]:
        """Evaluate cross-domain suggestions from the current stage.

        Each suggestion is evaluated by the primary role-holder for its
        target domain. Rejected high-confidence suggestions (>0.7) are
        escalated to a group vote. Returns only accepted suggestions.
        """
        if not new_suggestions:
            return []

        accepted: list[Suggestion] = []
        pipeline_models = {
            stage.value: key
            for stage, key in self._stage_models.items()
        }

        for suggestion in new_suggestions:
            # Find the primary role-holder for this suggestion's domain
            target_model = self._stage_models.get(suggestion.domain)
            if not target_model:
                # Domain's model not yet assigned — accept by default
                accepted.append(suggestion)
                continue

            decision = await self._consensus.evaluate_suggestion(
                suggestion=suggestion,
                evaluator_key=target_model,
                task=self._task.task,
                current_approach=previous_output,
            )
            self._suggestion_decisions.append(decision)

            if decision.decision == SuggestionVerdict.ACCEPT:
                accepted.append(suggestion)
                continue

            # Rejected — check for escalation
            if self._consensus.should_escalate(suggestion):
                logger.info(
                    "Escalating high-confidence suggestion "
                    "(confidence=%.2f) to group vote",
                    suggestion.confidence,
                )
                escalation = await self._consensus.escalate_suggestion(
                    suggestion=suggestion,
                    pipeline_models=pipeline_models,
                    task=self._task.task,
                    current_approach=previous_output,
                )
                if escalation.decision == SuggestionVerdict.ACCEPT:
                    accepted.append(suggestion)

        return accepted


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


# ── Score extraction for parallel cross-review ────────────────────

_ARCH_SCORE_RE = re.compile(r"ARCHITECTURE:\s*(\d+)")
_IMPL_SCORE_RE = re.compile(r"IMPLEMENTATION:\s*(\d+)")
_QUALITY_SCORE_RE = re.compile(r"QUALITY:\s*(\d+)")


def _extract_scores(content: str) -> dict[str, int]:
    """Extract architecture/implementation/quality scores from review."""
    scores: dict[str, int] = {}
    for name, pattern in [
        ("architecture", _ARCH_SCORE_RE),
        ("implementation", _IMPL_SCORE_RE),
        ("quality", _QUALITY_SCORE_RE),
    ]:
        match = pattern.search(content)
        if match:
            scores[name] = max(1, min(10, int(match.group(1))))
        else:
            scores[name] = 5  # neutral default
    return scores


def _get_tiebreaker_key(registry: dict[str, ModelConfig]) -> str:
    """Select the tiebreaker model — highest-fitness verifier."""
    return max(registry, key=lambda k: registry[k].fitness.verifier)


# ── Parallel Exploration Orchestrator ─────────────────────────────


class ParallelOrchestrator:
    """Parallel exploration pipeline mode.

    All models in the registry produce independent solutions in parallel.
    Cross-review scoring determines a winner via consensus vote. The
    winner's approach is enhanced with the best elements from others.
    """

    def __init__(
        self,
        task: TaskSpec,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
        event_emitter: object | None = None,
    ) -> None:
        self._task = task
        self._config = config
        self._registry = registry
        self._emitter = event_emitter
        self._arbiter = ArbiterEngine(config, registry)
        self._consensus = ConsensusEngine(config, registry)

    async def _emit(self, event_type: str, **data: object) -> None:
        """Emit a dashboard event if an emitter is attached."""
        if self._emitter is not None:
            await self._emitter.emit(event_type, **data)

    async def run(self) -> PipelineResult:
        """Execute the parallel exploration pipeline."""
        start = time.monotonic()
        arbiter_reviews: list[ArbiterReview] = []
        all_messages: list[AgentMessage] = []
        timeout = self._config.default_timeout
        consensus_result: ConsensusResult | None = None

        # Phase 1: Fan-out — all models produce solutions in parallel
        logger.info("Parallel fan-out: %d models", len(self._registry))
        individual_outputs: dict[str, str] = {}
        fan_out_tasks = {}
        for key, cfg in self._registry.items():
            fan_out_tasks[key] = self._call_model(
                key, cfg, "architect", timeout,
            )

        results = await asyncio.gather(
            *fan_out_tasks.values(), return_exceptions=True,
        )
        for key, result in zip(fan_out_tasks, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Model %s failed in fan-out: %s", key, result)
                continue
            individual_outputs[key] = result.content
            all_messages.append(result)

        if len(individual_outputs) < 2:
            raise RuntimeError(
                f"Parallel mode requires at least 2 successful outputs, "
                f"got {len(individual_outputs)}"
            )

        # Phase 2: Cross-review — each model scores the others
        logger.info("Parallel cross-review")
        scores: dict[str, dict[str, dict[str, int]]] = {}
        review_tasks = []
        review_keys: list[tuple[str, str]] = []

        for reviewer_key in self._registry:
            if reviewer_key not in individual_outputs:
                continue
            reviewer_cfg = self._registry[reviewer_key]
            for reviewed_key, output in individual_outputs.items():
                if reviewed_key == reviewer_key:
                    continue
                review_tasks.append(
                    self._call_model(
                        reviewer_key,
                        reviewer_cfg,
                        "parallel_review",
                        timeout,
                        reviewed_model=reviewed_key,
                        solution=output,
                    )
                )
                review_keys.append((reviewer_key, reviewed_key))

        review_results = await asyncio.gather(
            *review_tasks, return_exceptions=True,
        )
        for (reviewer, reviewed), result in zip(
            review_keys, review_results, strict=True,
        ):
            if isinstance(result, Exception):
                logger.error(
                    "Review %s->%s failed: %s", reviewer, reviewed, result,
                )
                continue
            all_messages.append(result)
            if reviewer not in scores:
                scores[reviewer] = {}
            scores[reviewer][reviewed] = _extract_scores(result.content)

        # Phase 3: Vote — derive votes from scores
        votes: dict[str, str] = {}
        for voter, voter_scores in scores.items():
            if not voter_scores:
                continue
            # Vote for the model with highest total score
            votes[voter] = max(
                voter_scores,
                key=lambda k: sum(voter_scores[k].values()),
            )

        # Phase 4: Resolve consensus with formal protocol
        consensus_result = await self._consensus.resolve_consensus(
            votes=votes,
            task=self._task.task,
            candidate_outputs=individual_outputs,
        )
        winner = consensus_result.winner
        logger.info("Parallel winner: %s (method=%s)", winner, consensus_result.method)

        # Phase 5: Synthesis — winner enhances with best of others
        logger.info("Parallel synthesis by %s", winner)
        winner_cfg = self._registry[winner]
        other_outputs = {
            k: v for k, v in individual_outputs.items() if k != winner
        }
        synth_msg = await self._call_model(
            winner,
            winner_cfg,
            "parallel_synthesize",
            timeout,
            winner_model=winner,
            winning_output=individual_outputs[winner],
            other_outputs=other_outputs,
        )
        all_messages.append(synth_msg)
        synthesized_output = synth_msg.content

        # Phase 6: Arbiter review of synthesized output
        if self._config.arbiter_mode != ArbiterMode.OFF:
            review = await self._arbiter.review(
                stage=PipelineStage.VERIFY,
                stage_model=winner_cfg.model,
                stage_output=synthesized_output,
                task=self._task,
            )
            arbiter_reviews.append(review)

        # Build result
        duration = time.monotonic() - start
        total_cost = sum(
            m.token_usage.cost for m in all_messages if m.token_usage
        ) + sum(r.token_cost for r in arbiter_reviews)
        total_tokens = sum(
            (m.token_usage.prompt_tokens + m.token_usage.completion_tokens)
            for m in all_messages
            if m.token_usage
        )

        halted = any(r.verdict == Verdict.HALT for r in arbiter_reviews)

        return PipelineResult(
            task=self._task,
            config=self._config,
            total_cost=total_cost,
            total_tokens=total_tokens,
            duration_seconds=duration,
            success=not halted,
            arbiter_reviews=arbiter_reviews,
            halted=halted,
            halt_reason=(
                arbiter_reviews[-1].reasoning
                if halted and arbiter_reviews
                else ""
            ),
            parallel_result=ParallelResult(
                individual_outputs=individual_outputs,
                scores=scores,
                votes=votes,
                winner=winner,
                synthesized_output=synthesized_output,
            ),
        )

    async def _call_model(
        self,
        model_key: str,
        model_config: ModelConfig,
        template_name: str,
        timeout: int,
        **template_vars: object,
    ) -> AgentMessage:
        """Call a model with a rendered prompt template."""
        tpl_vars: dict = {
            "task": self._task.task,
            "context": self._task.context,
            "domain_context": self._task.domain_rules,
            **template_vars,
        }
        system = render_prompt(template_name, **tpl_vars)
        provider = LiteLLMProvider(model_config)

        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Execute the task as described "
                    "in your system instructions."
                ),
            }],
            system=system,
            timeout=timeout,
        )
        msg.from_agent = PipelineStage.ARCHITECT
        msg.to_agent = PipelineStage.ARCHITECT
        msg.msg_type = MessageType.PROPOSAL
        msg.confidence = _extract_confidence(msg.content)
        return msg


# ── Debate Orchestrator ───────────────────────────────────────────


class DebateOrchestrator:
    """Structured debate pipeline mode.

    Each model proposes an approach, writes rebuttals of other proposals,
    presents final arguments, and a judge model renders a decision.
    """

    def __init__(
        self,
        task: TaskSpec,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
        event_emitter: object | None = None,
    ) -> None:
        self._task = task
        self._config = config
        self._registry = registry
        self._emitter = event_emitter
        self._arbiter = ArbiterEngine(config, registry)
        self._consensus = ConsensusEngine(config, registry)

    async def _emit(self, event_type: str, **data: object) -> None:
        """Emit a dashboard event if an emitter is attached."""
        if self._emitter is not None:
            await self._emitter.emit(event_type, **data)

    async def run(self) -> PipelineResult:
        """Execute the debate pipeline."""
        start = time.monotonic()
        arbiter_reviews: list[ArbiterReview] = []
        all_messages: list[AgentMessage] = []
        timeout = self._config.default_timeout

        # Phase 1: Position papers
        logger.info(
            "Debate: position papers from %d models",
            len(self._registry),
        )
        proposals: dict[str, str] = {}
        prop_tasks = {}
        for key, cfg in self._registry.items():
            prop_tasks[key] = self._call_model(
                key, cfg, "debate_propose", timeout,
            )

        prop_results = await asyncio.gather(
            *prop_tasks.values(), return_exceptions=True,
        )
        for key, result in zip(prop_tasks, prop_results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Model %s failed in proposal: %s", key, result,
                )
                continue
            proposals[key] = result.content
            all_messages.append(result)

        if len(proposals) < 2:
            raise RuntimeError(
                f"Debate mode requires at least 2 proposals, "
                f"got {len(proposals)}"
            )

        # Phase 2: Rebuttals
        logger.info("Debate: rebuttals")
        rebuttals: dict[str, dict[str, str]] = {}
        rebuttal_tasks = {}
        for key, cfg in self._registry.items():
            if key not in proposals:
                continue
            other_proposals = {
                k: v for k, v in proposals.items() if k != key
            }
            rebuttal_tasks[key] = self._call_model(
                key,
                cfg,
                "debate_rebuttal",
                timeout,
                own_proposal=proposals[key],
                other_proposals=other_proposals,
            )

        reb_results = await asyncio.gather(
            *rebuttal_tasks.values(), return_exceptions=True,
        )
        for key, result in zip(rebuttal_tasks, reb_results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Model %s failed in rebuttal: %s", key, result,
                )
                continue
            all_messages.append(result)
            other_keys = [k for k in proposals if k != key]
            rebuttals[key] = self._parse_rebuttals(
                result.content, other_keys,
            )

        # Phase 3: Final arguments
        logger.info("Debate: final arguments")
        final_arguments: dict[str, str] = {}
        final_tasks = {}
        for key, cfg in self._registry.items():
            if key not in proposals:
                continue
            rebuttals_received = {}
            for rebutter, targets in rebuttals.items():
                if key in targets:
                    rebuttals_received[rebutter] = targets[key]

            final_tasks[key] = self._call_model(
                key,
                cfg,
                "debate_final",
                timeout,
                own_proposal=proposals[key],
                rebuttals_received=rebuttals_received,
            )

        final_results = await asyncio.gather(
            *final_tasks.values(), return_exceptions=True,
        )
        for key, result in zip(final_tasks, final_results, strict=True):
            if isinstance(result, Exception):
                logger.error(
                    "Model %s failed in final argument: %s", key, result,
                )
                continue
            final_arguments[key] = result.content
            all_messages.append(result)

        # Phase 4: Judgment
        judge_key = _get_tiebreaker_key(self._registry)
        judge_cfg = self._registry[judge_key]
        logger.info("Debate: judgment by %s", judge_key)

        judge_msg = await self._call_model(
            judge_key,
            judge_cfg,
            "debate_judge",
            timeout,
            proposals=proposals,
            all_rebuttals=rebuttals,
            final_arguments=final_arguments,
        )
        all_messages.append(judge_msg)
        judgment = judge_msg.content

        # Phase 5: Arbiter review of judgment
        if self._config.arbiter_mode != ArbiterMode.OFF:
            review = await self._arbiter.review(
                stage=PipelineStage.VERIFY,
                stage_model=judge_cfg.model,
                stage_output=judgment,
                task=self._task,
            )
            arbiter_reviews.append(review)

        # Build result
        duration = time.monotonic() - start
        total_cost = sum(
            m.token_usage.cost for m in all_messages if m.token_usage
        ) + sum(r.token_cost for r in arbiter_reviews)
        total_tokens = sum(
            (m.token_usage.prompt_tokens + m.token_usage.completion_tokens)
            for m in all_messages
            if m.token_usage
        )

        halted = any(r.verdict == Verdict.HALT for r in arbiter_reviews)

        return PipelineResult(
            task=self._task,
            config=self._config,
            total_cost=total_cost,
            total_tokens=total_tokens,
            duration_seconds=duration,
            success=not halted,
            arbiter_reviews=arbiter_reviews,
            halted=halted,
            halt_reason=(
                arbiter_reviews[-1].reasoning
                if halted and arbiter_reviews
                else ""
            ),
            debate_result=DebateResult(
                proposals=proposals,
                rebuttals=rebuttals,
                final_arguments=final_arguments,
                judgment=judgment,
                judge_model=judge_key,
            ),
        )

    def _parse_rebuttals(
        self,
        content: str,
        target_keys: list[str],
    ) -> dict[str, str]:
        """Parse a combined rebuttal output into per-target rebuttals.

        Looks for '## Rebuttal: <model_key>' headers. Falls back to
        assigning the entire content to all targets if parsing fails.
        """
        parsed: dict[str, str] = {}
        for key in target_keys:
            pattern = rf"##\s*Rebuttal:\s*{re.escape(key)}"
            match = re.search(pattern, content, re.IGNORECASE)
            if match:
                start_pos = match.end()
                next_match = re.search(
                    r"##\s*Rebuttal:",
                    content[start_pos:],
                    re.IGNORECASE,
                )
                end = (
                    start_pos + next_match.start()
                    if next_match
                    else len(content)
                )
                parsed[key] = content[start_pos:end].strip()
            else:
                parsed[key] = content

        return parsed

    async def _call_model(
        self,
        model_key: str,
        model_config: ModelConfig,
        template_name: str,
        timeout: int,
        **template_vars: object,
    ) -> AgentMessage:
        """Call a model with a rendered prompt template."""
        tpl_vars: dict = {
            "task": self._task.task,
            "context": self._task.context,
            "domain_context": self._task.domain_rules,
            **template_vars,
        }
        system = render_prompt(template_name, **tpl_vars)
        provider = LiteLLMProvider(model_config)

        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Execute the task as described "
                    "in your system instructions."
                ),
            }],
            system=system,
            timeout=timeout,
        )
        msg.from_agent = PipelineStage.ARCHITECT
        msg.to_agent = PipelineStage.ARCHITECT
        msg.msg_type = MessageType.PROPOSAL
        msg.confidence = _extract_confidence(msg.content)
        return msg


# ── Pipeline Dispatcher ──────────────────────────────────────────


async def run_pipeline(
    task: TaskSpec,
    config: PipelineConfig,
    registry: dict[str, ModelConfig],
    event_emitter: object | None = None,
) -> PipelineResult:
    """Dispatch to the appropriate pipeline mode and return a result.

    Routes to PipelineOrchestrator (sequential), ParallelOrchestrator,
    or DebateOrchestrator based on config.pipeline_mode. When context
    injection is enabled (context_dir set), scans the project and injects
    context into the task. When session persistence is enabled, saves the
    run to SQLite.

    Args:
        task: The task specification.
        config: Pipeline configuration.
        registry: Model registry.
        event_emitter: Optional dashboard event emitter for real-time updates.
    """
    session_id = str(uuid.uuid4())
    started_at = datetime.now(UTC)

    # Context injection
    if config.context_dir:
        task = _inject_context(task, config)

    if config.pipeline_mode == PipelineMode.PARALLEL:
        result = await ParallelOrchestrator(
            task, config, registry, event_emitter,
        ).run()
    elif config.pipeline_mode == PipelineMode.DEBATE:
        result = await DebateOrchestrator(
            task, config, registry, event_emitter,
        ).run()
    else:
        result = await PipelineOrchestrator(
            task, config, registry, event_emitter,
        ).run()

    result.session_id = session_id

    # Persist session if enabled
    if config.persist_sessions:
        await _save_session(result, started_at)

    return result


async def _save_session(result: PipelineResult, started_at: datetime) -> None:
    """Build a SessionRecord from the PipelineResult and persist it."""
    from triad.persistence.database import close_db, init_db
    from triad.persistence.session import SessionStore

    completed_at = datetime.now(UTC)

    # Convert stage AgentMessages to StageRecords
    stage_records: list[StageRecord] = []
    for stage, msg in result.stages.items():
        stage_records.append(StageRecord(
            stage=stage.value,
            model_key=msg.model,
            model_id=msg.model,
            content=msg.content,
            confidence=msg.confidence,
            cost=msg.token_usage.cost if msg.token_usage else 0.0,
            tokens=(
                (msg.token_usage.prompt_tokens + msg.token_usage.completion_tokens)
                if msg.token_usage
                else 0
            ),
            timestamp=msg.timestamp.isoformat(),
        ))

    record = SessionRecord(
        session_id=result.session_id,
        task=result.task,
        config=result.config,
        stages=stage_records,
        arbiter_reviews=result.arbiter_reviews,
        routing_decisions=result.routing_decisions,
        started_at=started_at,
        completed_at=completed_at,
        success=result.success,
        halted=result.halted,
        halt_reason=result.halt_reason,
        total_cost=result.total_cost,
        total_tokens=result.total_tokens,
        duration_seconds=result.duration_seconds,
        pipeline_mode=result.config.pipeline_mode.value,
    )

    try:
        db = await init_db(result.config.session_db_path)
        store = SessionStore(db)
        await store.save_session(record)
        await close_db(db)
        logger.info("Session %s persisted", result.session_id)
    except Exception:
        logger.exception("Failed to persist session %s", result.session_id)


def _inject_context(task: TaskSpec, config: PipelineConfig) -> TaskSpec:
    """Scan a project directory and inject context into the task spec.

    Returns a new TaskSpec with the scanned context prepended to
    the existing context field.
    """
    from triad.context.builder import ContextBuilder
    from triad.context.scanner import CodeScanner

    scanner = CodeScanner(
        root=config.context_dir,
        include=config.context_include,
        exclude=config.context_exclude,
    )
    files = scanner.scan()

    if not files:
        logger.info("Context injection: no files matched in %s", config.context_dir)
        return task

    builder = ContextBuilder(
        root_path=config.context_dir,
        token_budget=config.context_token_budget,
    )
    ctx = builder.build(files, task.task)

    logger.info(
        "Context injection: %d/%d files, ~%d tokens%s",
        ctx.files_included,
        ctx.files_scanned,
        ctx.token_estimate,
        " (truncated)" if ctx.truncated else "",
    )

    # Prepend project context to existing context
    new_context = ctx.context_text
    if task.context:
        new_context = f"{new_context}\n\n---\n\n{task.context}"

    return TaskSpec(
        task=task.task,
        context=new_context,
        domain_rules=task.domain_rules,
        output_dir=task.output_dir,
    )

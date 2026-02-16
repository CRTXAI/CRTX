"""Sequential pipeline engine for the Triad Orchestrator.

Runs 4 stages in order: Architect -> Implement -> Refactor -> Verify.
Each stage loads its role prompt template, renders it with task context
and previous output, calls a LiteLLMProvider, and passes its output
to the next stage. Cross-domain suggestions are collected and injected
downstream. All AgentMessages are logged to a session audit trail.
"""

from __future__ import annotations

import logging
import re
import time

from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.providers.registry import get_best_model_for_role
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    Suggestion,
)
from triad.schemas.pipeline import (
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

# Regex for extracting CONFIDENCE: <value> from model output
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([\d.]+)")


class PipelineOrchestrator:
    """Sequential pipeline engine.

    Coordinates the 4-stage pipeline (Architect -> Implement -> Refactor ->
    Verify), passing output between stages, collecting cross-domain
    suggestions, and tracking tokens/cost. Uses LiteLLMProvider exclusively
    for all model calls.
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

    @property
    def session(self) -> list[AgentMessage]:
        """The session audit trail — all AgentMessages from the run."""
        return list(self._session)

    async def run(self) -> PipelineResult:
        """Execute the full 4-stage pipeline and return a PipelineResult.

        Raises:
            RuntimeError: If model selection fails or a stage errors out.
        """
        start = time.monotonic()

        stages: dict[PipelineStage, AgentMessage] = {}
        all_suggestions: list[Suggestion] = []
        architect_output = ""
        previous_output = ""

        for stage in _STAGE_SEQUENCE:
            logger.info("Starting stage: %s", stage.value)

            template_vars = self._build_template_vars(
                stage=stage,
                previous_output=previous_output,
                architect_output=architect_output,
                suggestions=all_suggestions,
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

        duration = time.monotonic() - start
        total_cost = sum(
            msg.token_usage.cost for msg in stages.values() if msg.token_usage
        )
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
            success=True,
        )

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

        return tpl_vars


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

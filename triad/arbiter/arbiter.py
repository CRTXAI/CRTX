"""Arbiter review engine with cross-model enforcement.

Reviews pipeline stage outputs independently using a different model
than the one that generated the output. Parses the response into a
structured ArbiterReview with one of four verdicts.
"""

from __future__ import annotations

import logging
import re

from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ModelConfig, PipelineConfig, TaskSpec

logger = logging.getLogger(__name__)

# Minimum confidence for an APPROVE verdict to be accepted as-is.
# Below this threshold, APPROVE is downgraded to FLAG.
_MIN_APPROVE_CONFIDENCE = 0.50

# Regex for extracting VERDICT: <value> from arbiter output
_VERDICT_RE = re.compile(
    r"\*?\*?VERDICT:\s*(APPROVE|FLAG|REJECT|HALT)\*?\*?",
    re.IGNORECASE,
)

# Regex for extracting CONFIDENCE: <value> from arbiter output
_CONFIDENCE_RE = re.compile(r"CONFIDENCE:\s*([\d.]+)")


class ArbiterEngine:
    """Independent adversarial review engine.

    Reviews a pipeline stage's output using a model that is different
    from the one that produced the output (cross-model enforcement).
    Returns a structured ArbiterReview with a verdict that controls
    pipeline flow. Supports graceful degradation when arbiter models
    are unavailable.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
        health: object | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._health = health  # ProviderHealth instance (optional)

    async def review(
        self,
        stage: PipelineStage,
        stage_model: str,
        stage_output: str,
        task: TaskSpec,
        architect_output: str = "",
    ) -> ArbiterReview | None:
        """Review a stage's output and return a structured verdict.

        Tries multiple arbiter models if the primary is unavailable.
        Returns ``None`` when no arbiter model is available, signalling
        the caller to skip arbiter review for this stage.

        Args:
            stage: Which pipeline stage produced the output.
            stage_model: The model ID that generated the output.
            stage_output: The raw content from the stage's AgentMessage.
            task: The original task specification.
            architect_output: The Architect's output for reference.

        Returns:
            ArbiterReview, or None if all arbiter models are unavailable.
        """
        tried_keys: list[str] = []

        while True:
            try:
                arbiter_key = self._resolve_arbiter_model(
                    stage_model, exclude=tried_keys,
                )
            except RuntimeError:
                # No arbiter model available — degrade gracefully
                logger.warning(
                    "No arbiter models available for %s, skipping review",
                    stage.value,
                )
                return None

            arbiter_config = self._registry[arbiter_key]
            tried_keys.append(arbiter_key)

            try:
                provider = LiteLLMProvider(arbiter_config)

                system = render_prompt(
                    "arbiter",
                    stage_name=stage.value,
                    stage_model=stage_model,
                    task=task.task,
                    context=task.context,
                    domain_context=task.domain_rules,
                    stage_output=stage_output,
                    architect_output=(
                        architect_output if stage != PipelineStage.ARCHITECT else ""
                    ),
                )

                timeout = self._config.default_timeout
                msg = await provider.complete(
                    messages=[{
                        "role": "user",
                        "content": (
                            "Review the output as described in your "
                            "system instructions."
                        ),
                    }],
                    system=system,
                    timeout=timeout,
                )

                cost = msg.token_usage.cost if msg.token_usage else 0.0
                verdict = _extract_verdict(msg.content)
                confidence = _extract_confidence(msg.content)

                logger.info(
                    "Arbiter verdict for %s: %s (arbiter=%s, confidence=%.2f)",
                    stage.value,
                    verdict.value,
                    arbiter_config.model,
                    confidence,
                )

                # Downgrade low-confidence APPROVE to FLAG
                if (
                    verdict == Verdict.APPROVE
                    and confidence < _MIN_APPROVE_CONFIDENCE
                ):
                    logger.warning(
                        "Arbiter approved %s with low confidence (%.2f), "
                        "treating as FLAG",
                        stage.value, confidence,
                    )
                    verdict = Verdict.FLAG

                return ArbiterReview(
                    stage_reviewed=stage,
                    reviewed_model=stage_model,
                    arbiter_model=arbiter_config.model,
                    verdict=verdict,
                    confidence=confidence,
                    reasoning=msg.content,
                    token_cost=cost,
                )
            except (RuntimeError, TimeoutError) as e:
                logger.warning(
                    "Arbiter model %s failed for %s: %s — trying fallback",
                    arbiter_key, stage.value, e,
                )
                if self._health:
                    self._health.mark_unhealthy(arbiter_key)

    def _resolve_arbiter_model(
        self,
        stage_model: str,
        exclude: list[str] | None = None,
    ) -> str:
        """Select an arbiter model different from the stage model.

        Resolution order:
        1. config.arbiter_model (global override) — unless excluded/unhealthy
        2. Best available model excluding the stage model and excluded keys

        Raises:
            RuntimeError: If no valid arbiter model is available.
        """
        exclude = exclude or []

        # 1. Global override (if not excluded and healthy)
        if self._config.arbiter_model:
            key = self._config.arbiter_model
            if (
                key in self._registry
                and key not in exclude
                and self._registry[key].model != stage_model
                and (not self._health or self._health.is_healthy(key))
            ):
                return key

        # 2. Best available model excluding the stage model, excluded, unhealthy
        candidates = {
            k: v for k, v in self._registry.items()
            if v.model != stage_model
            and k not in exclude
            and (not self._health or self._health.is_healthy(k))
        }
        if not candidates:
            raise RuntimeError(
                f"No arbiter model available for stage model "
                f"'{stage_model}' (excluded={exclude})"
            )

        # Pick the candidate with the highest average fitness
        def _avg_fitness(cfg: ModelConfig) -> float:
            f = cfg.fitness
            return (f.architect + f.implementer + f.refactorer + f.verifier) / 4

        best_key = max(candidates, key=lambda k: _avg_fitness(candidates[k]))
        return best_key


def _extract_verdict(content: str) -> Verdict:
    """Extract the VERDICT: <value> from arbiter output.

    Returns Verdict.FLAG as a safe fallback if parsing fails — does not
    block the pipeline but does not silently approve either.
    """
    match = _VERDICT_RE.search(content)
    if match:
        try:
            return Verdict(match.group(1).lower())
        except ValueError:
            pass
    logger.warning("Could not parse arbiter verdict, defaulting to FLAG")
    return Verdict.FLAG


def _extract_confidence(content: str) -> float:
    """Extract the CONFIDENCE: <value> score from arbiter output."""
    match = _CONFIDENCE_RE.search(content)
    if match:
        try:
            val = float(match.group(1))
            return max(0.0, min(1.0, val))
        except ValueError:
            pass
    return 0.5

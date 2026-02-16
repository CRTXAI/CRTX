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
    pipeline flow.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
    ) -> None:
        self._config = config
        self._registry = registry

    async def review(
        self,
        stage: PipelineStage,
        stage_model: str,
        stage_output: str,
        task: TaskSpec,
        architect_output: str = "",
    ) -> ArbiterReview:
        """Review a stage's output and return a structured verdict.

        Args:
            stage: Which pipeline stage produced the output.
            stage_model: The model ID that generated the output.
            stage_output: The raw content from the stage's AgentMessage.
            task: The original task specification.
            architect_output: The Architect's output for reference.

        Returns:
            ArbiterReview with verdict, issues, alternatives, and reasoning.

        Raises:
            RuntimeError: If no valid arbiter model is available.
        """
        arbiter_key = self._resolve_arbiter_model(stage_model)
        arbiter_config = self._registry[arbiter_key]
        provider = LiteLLMProvider(arbiter_config)

        system = render_prompt(
            "arbiter",
            stage_name=stage.value,
            stage_model=stage_model,
            task=task.task,
            context=task.context,
            domain_context=task.domain_rules,
            stage_output=stage_output,
            architect_output=architect_output if stage != PipelineStage.ARCHITECT else "",
        )

        timeout = self._config.default_timeout
        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": "Review the output as described in your system instructions.",
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

        return ArbiterReview(
            stage_reviewed=stage,
            reviewed_model=stage_model,
            arbiter_model=arbiter_config.model,
            verdict=verdict,
            confidence=confidence,
            reasoning=msg.content,
            token_cost=cost,
        )

    def _resolve_arbiter_model(self, stage_model: str) -> str:
        """Select an arbiter model different from the stage model.

        Resolution order:
        1. config.arbiter_model (global override)
        2. Best available model excluding the stage model

        Raises:
            RuntimeError: If no valid arbiter model is available or the
                          configured model is the same as the stage model.
        """
        # 1. Global override
        if self._config.arbiter_model:
            key = self._config.arbiter_model
            if key not in self._registry:
                raise RuntimeError(
                    f"Configured arbiter model '{key}' is not in the model registry"
                )
            if self._registry[key].model == stage_model:
                raise RuntimeError(
                    f"Configured arbiter model '{key}' is the same model as the "
                    f"stage generator ({stage_model}). Cross-model enforcement "
                    f"requires arbiter != generator."
                )
            return key

        # 2. Best available model excluding the stage model
        candidates = {
            k: v for k, v in self._registry.items() if v.model != stage_model
        }
        if not candidates:
            raise RuntimeError(
                f"No arbiter model available that differs from stage model "
                f"'{stage_model}'. Register at least 2 models."
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

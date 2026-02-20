"""Implementation Summary Reconciliation (ISR) engine.

Compares the Verifier's ImplementationSummary against the original TaskSpec
and Architect scaffold using a cross-model Arbiter. Catches spec drift,
missing requirements, and silently dropped features.
"""

from __future__ import annotations

import logging

from triad.arbiter.arbiter import (
    _extract_confidence,
    _extract_verdict,
    _parse_alternatives,
    _parse_issues,
)
from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.arbiter import ArbiterReview
from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ModelConfig, PipelineConfig, TaskSpec

logger = logging.getLogger(__name__)


class ReconciliationEngine:
    """Implementation Summary Reconciliation Arbiter.

    Runs an independent cross-model review that compares the Verifier's
    ImplementationSummary against the original TaskSpec and Architect
    scaffold. The reconciliation model must differ from the Verifier model.
    """

    def __init__(
        self,
        config: PipelineConfig,
        registry: dict[str, ModelConfig],
    ) -> None:
        self._config = config
        self._registry = registry

    async def reconcile(
        self,
        task: TaskSpec,
        architect_output: str,
        implementation_summary: str,
        verifier_model: str,
    ) -> ArbiterReview:
        """Run the reconciliation pass and return a verdict.

        Args:
            task: The original task specification.
            architect_output: The Architect's scaffold output.
            implementation_summary: The Verifier's ImplementationSummary text.
            verifier_model: The model that produced the Verify stage output.

        Returns:
            ArbiterReview with reconciliation verdict.

        Raises:
            RuntimeError: If no valid reconciliation model is available.
        """
        recon_key = self._resolve_reconcile_model(verifier_model)
        recon_config = self._registry[recon_key]
        provider = LiteLLMProvider(recon_config)

        system = render_prompt(
            "reconciler",
            task=task.task,
            context=task.context,
            domain_context=task.domain_rules,
            architect_output=architect_output,
            implementation_summary=implementation_summary,
        )

        timeout = self._config.default_timeout
        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": (
                    "Perform the reconciliation review "
                    "as described in your system instructions."
                ),
            }],
            system=system,
            timeout=timeout,
        )

        cost = msg.token_usage.cost if msg.token_usage else 0.0
        verdict = _extract_verdict(msg.content)
        confidence = _extract_confidence(msg.content)
        issues = _parse_issues(msg.content)
        alternatives = _parse_alternatives(msg.content)

        logger.info(
            "Reconciliation verdict: %s (model=%s, confidence=%.2f, "
            "issues=%d, alternatives=%d)",
            verdict.value,
            recon_config.model,
            confidence,
            len(issues),
            len(alternatives),
        )

        return ArbiterReview(
            stage_reviewed=PipelineStage.VERIFY,
            reviewed_model=verifier_model,
            arbiter_model=recon_config.model,
            verdict=verdict,
            issues=issues,
            alternatives=alternatives,
            confidence=confidence,
            reasoning=msg.content,
            token_cost=cost,
        )

    def _resolve_reconcile_model(self, verifier_model: str) -> str:
        """Select a reconciliation model different from the verifier model.

        Resolution order:
        1. config.reconcile_model (explicit override)
        2. config.arbiter_model (global arbiter override)
        3. Best available model excluding the verifier model

        Raises:
            RuntimeError: If no valid reconciliation model is available.
        """
        # 1. Explicit reconciliation model override
        if self._config.reconcile_model:
            key = self._config.reconcile_model
            if key not in self._registry:
                raise RuntimeError(
                    f"Configured reconcile model '{key}' is not in the model registry"
                )
            if self._registry[key].model == verifier_model:
                raise RuntimeError(
                    f"Configured reconcile model '{key}' is the same model as the "
                    f"verifier ({verifier_model}). Cross-model enforcement requires "
                    f"reconciler != verifier."
                )
            return key

        # 2. Global arbiter override
        if self._config.arbiter_model:
            key = self._config.arbiter_model
            if key in self._registry and self._registry[key].model != verifier_model:
                return key

        # 3. Best available model excluding verifier
        candidates = {
            k: v for k, v in self._registry.items() if v.model != verifier_model
        }
        if not candidates:
            raise RuntimeError(
                f"No reconciliation model available that differs from verifier "
                f"model '{verifier_model}'. Register at least 2 models."
            )

        def _avg_fitness(cfg: ModelConfig) -> float:
            f = cfg.fitness
            return (f.architect + f.implementer + f.refactorer + f.verifier) / 4

        best_key = max(candidates, key=lambda k: _avg_fitness(candidates[k]))
        return best_key

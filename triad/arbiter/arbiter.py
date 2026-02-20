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
from triad.schemas.arbiter import (
    Alternative,
    ArbiterReview,
    Issue,
    IssueCategory,
    Severity,
    Verdict,
)
from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ModelConfig, PipelineConfig, TaskSpec

logger = logging.getLogger(__name__)

# Minimum confidence for an APPROVE verdict to be accepted as-is.
# Below this threshold, APPROVE is downgraded to FLAG.
_MIN_APPROVE_CONFIDENCE = 0.50

# Minimum response length to consider an arbiter response valid.
# Responses shorter than this trigger a fallback to the next model.
# The 0-issues fallback (below) catches longer but malformed responses.
_MIN_RESPONSE_LENGTH = 10

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
        stream_callback: object | None = None,
    ) -> None:
        self._config = config
        self._registry = registry
        self._health = health  # ProviderHealth instance (optional)
        self._stream_callback = stream_callback

    async def review(
        self,
        stage: PipelineStage,
        stage_model: str,
        stage_output: str,
        task: TaskSpec,
        architect_output: str = "",
        exclude_models: list[str] | None = None,
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
            exclude_models: Additional model IDs to exclude from arbiter
                selection (e.g. all fan-out participants in parallel mode).

        Returns:
            ArbiterReview, or None if all arbiter models are unavailable.
        """
        tried_keys: list[str] = []
        excluded_model_ids = {stage_model}
        if exclude_models:
            excluded_model_ids.update(exclude_models)

        # Last-resort review: if all fallback models also produce hollow
        # verdicts (FLAG/REJECT with 0 parsed issues), return the first
        # one rather than None.  Empty responses are never saved here.
        hollow_review: ArbiterReview | None = None

        while True:
            try:
                arbiter_key = self._resolve_arbiter_model(
                    excluded_model_ids, exclude=tried_keys,
                )
            except RuntimeError:
                # No more arbiter models — return the hollow review if
                # we captured one, otherwise degrade gracefully.
                if hollow_review is not None:
                    logger.warning(
                        "All arbiter models returned hollow verdicts for %s, "
                        "using first result",
                        stage.value,
                    )
                    return hollow_review
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

                logger.debug(
                    "Arbiter prompt for %s: %d chars, stage_output: %d chars",
                    stage.value, len(system), len(stage_output),
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

                logger.debug(
                    "Arbiter raw response for %s: %d chars",
                    stage.value, len(msg.content),
                )

                # Empty / too-short response — try fallback model.
                # Don't save as hollow_review; empty responses are useless.
                if len(msg.content.strip()) < _MIN_RESPONSE_LENGTH:
                    logger.warning(
                        "Arbiter model %s returned near-empty response for %s "
                        "(%d chars) — trying fallback",
                        arbiter_key, stage.value, len(msg.content),
                    )
                    if self._health:
                        self._health.mark_unhealthy(arbiter_key)
                    continue

                cost = msg.token_usage.cost if msg.token_usage else 0.0
                verdict = _extract_verdict(msg.content)
                confidence = _extract_confidence(msg.content)
                issues = _parse_issues(msg.content)
                alternatives = _parse_alternatives(msg.content)

                logger.info(
                    "Arbiter verdict for %s: %s (arbiter=%s, confidence=%.2f, "
                    "issues=%d, alternatives=%d)",
                    stage.value,
                    verdict.value,
                    arbiter_config.model,
                    confidence,
                    len(issues),
                    len(alternatives),
                )

                review = ArbiterReview(
                    stage_reviewed=stage,
                    reviewed_model=stage_model,
                    arbiter_model=arbiter_config.model,
                    verdict=verdict,
                    issues=issues,
                    alternatives=alternatives,
                    confidence=confidence,
                    reasoning=msg.content,
                    token_cost=cost,
                )

                # FLAG/REJECT with 0 parsed issues — response is likely
                # malformed or the model didn't follow the prompt format.
                # Try a fallback model, but save this as last resort.
                if not issues and verdict in (Verdict.FLAG, Verdict.REJECT):
                    logger.warning(
                        "Arbiter %s returned %s for %s but parsed 0 issues "
                        "(%d chars) — trying fallback",
                        arbiter_key, verdict.value, stage.value,
                        len(msg.content),
                    )
                    if hollow_review is None:
                        hollow_review = review
                    continue

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
                    review = review.model_copy(
                        update={"verdict": Verdict.FLAG},
                    )

                return review
            except (RuntimeError, TimeoutError) as e:
                logger.warning(
                    "Arbiter model %s failed for %s: %s — trying fallback",
                    arbiter_key, stage.value, e,
                )
                if self._health:
                    self._health.mark_unhealthy(arbiter_key)

    def _resolve_arbiter_model(
        self,
        excluded_model_ids: set[str],
        exclude: list[str] | None = None,
    ) -> str:
        """Select an arbiter model not in the excluded model IDs.

        Resolution order:
        1. config.arbiter_model (global override) — unless excluded/unhealthy
        2. Best available model excluding all listed model IDs and keys

        Args:
            excluded_model_ids: Set of model IDs (e.g. ``{"openai/o3",
                "anthropic/claude-3-opus"}``) that must not serve as arbiter.
                In parallel/debate modes this includes ALL participants.
            exclude: Registry keys already tried (for retry fallback).

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
                and self._registry[key].model not in excluded_model_ids
                and (not self._health or self._health.is_healthy(key))
            ):
                return key

        # 2. Best available model excluding all participant models, tried keys, unhealthy
        candidates = {
            k: v for k, v in self._registry.items()
            if v.model not in excluded_model_ids
            and k not in exclude
            and (not self._health or self._health.is_healthy(k))
        }
        if not candidates:
            raise RuntimeError(
                f"No arbiter model available (excluded_models="
                f"{excluded_model_ids}, excluded_keys={exclude})"
            )

        # Pick the candidate with the highest verifier fitness — the
        # arbiter's job is adversarial review, which maps to verifier.
        best_key = max(candidates, key=lambda k: candidates[k].fitness.verifier)
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


# ── Structured issue / alternative parsers ─────────────────────────

# Matches the first line of a numbered issue entry.
# Captures: (severity, category, description)
# Handles variations like:
#   1. **[critical]** [logic] — Missing error handling
#   2. **[WARNING]** **[security]** - API key exposed
_ISSUE_LINE_RE = re.compile(
    r"(?:\d+\.|[-*])\s*"               # numbered "1." or bullet "-"/"*"
    r"\*?\*?\[(\w+)\]\*?\*?\s*"         # [severity], optionally bold
    r"\*?\*?\[(\w+)\]\*?\*?\s*"         # [category], optionally bold
    r"[—–\-:]\s*"                       # separator (em-dash, en-dash, hyphen, colon)
    r"(.+)",                            # description (rest of line)
    re.IGNORECASE,
)

# Maps recognised severity strings to the enum.
_SEVERITY_MAP: dict[str, Severity] = {s.value: s for s in Severity}

# Maps recognised category strings to the enum.
_CATEGORY_MAP: dict[str, IssueCategory] = {c.value: c for c in IssueCategory}


def _parse_issues(content: str) -> list[Issue]:
    """Parse structured issues from the ``## Issues`` section.

    Robust to common LLM formatting variations (bold markers, mixed
    dashes, missing optional fields).  Returns an empty list when the
    section is absent or contains no parseable entries.
    """
    section = _extract_section(content, "Issues")
    if not section:
        return []

    items = _split_entries(section)
    parsed: list[Issue] = []

    for item in items:
        match = _ISSUE_LINE_RE.match(item.strip())
        if not match:
            continue

        severity_str = match.group(1).lower()
        category_str = match.group(2).lower()
        description = match.group(3).strip()

        severity = _SEVERITY_MAP.get(severity_str, Severity.WARNING)
        category = _CATEGORY_MAP.get(category_str, IssueCategory.LOGIC)

        location = _extract_field(item, "Location")
        evidence = _extract_field(item, "Evidence")
        suggestion = (
            _extract_field(item, "Suggestion")
            or _extract_field(item, "Fix")
        )

        parsed.append(Issue(
            severity=severity,
            category=category,
            location=location,
            description=description,
            suggestion=suggestion,
            evidence=evidence,
        ))

    return parsed


def _parse_alternatives(content: str) -> list[Alternative]:
    """Parse suggested alternatives from the ``## Alternatives`` section."""
    section = _extract_section(content, "Alternatives")
    if not section:
        return []

    items = _split_entries(section)
    parsed: list[Alternative] = []

    for item in items:
        # Match: 1. **Alternative**: description   or   - **Alternative**: desc
        desc_match = re.match(
            r"(?:\d+\.|[-*])\s*\*?\*?Alternative\*?\*?:\s*(.+?)$",
            item.strip(),
            re.IGNORECASE | re.MULTILINE,
        )
        if not desc_match:
            continue

        description = desc_match.group(1).strip()
        rationale = _extract_field(item, "Rationale")

        confidence_match = re.search(
            r"Confidence:\s*([\d.]+)", item, re.IGNORECASE,
        )
        confidence = 0.5
        if confidence_match:
            try:
                confidence = max(0.0, min(1.0, float(confidence_match.group(1))))
            except ValueError:
                pass

        code_match = re.search(r"```\w*\n(.*?)```", item, re.DOTALL)
        code_sketch = code_match.group(1).strip() if code_match else ""

        parsed.append(Alternative(
            description=description,
            rationale=rationale or "",
            code_sketch=code_sketch,
            confidence=confidence,
        ))

    return parsed


def _extract_section(content: str, heading: str) -> str:
    """Extract text between ``## <heading>`` and the next ``## `` or end."""
    pattern = re.compile(
        rf"##\s*{re.escape(heading)}\s*\n(.*?)(?=\n##\s|\Z)",
        re.DOTALL | re.IGNORECASE,
    )
    match = pattern.search(content)
    return match.group(1) if match else ""


def _split_entries(section: str) -> list[str]:
    """Split a section into individual numbered or bulleted entries."""
    parts = re.split(r"\n(?=\d+\.\s|[-*]\s\*?\*?\[)", section)
    return [p for p in parts if p.strip()]


def _extract_field(text: str, field_name: str) -> str:
    """Extract a ``Field: value`` line from an entry block.

    Handles optional bold markers (both ``**Field:** val`` and
    ``**Field**: val``) and backtick-wrapped values.
    """
    match = re.search(
        rf"^\s*\*{{0,2}}{field_name}\*{{0,2}}:\*{{0,2}}\s*`?(.+?)`?\s*$",
        text,
        re.IGNORECASE | re.MULTILINE,
    )
    return match.group(1).strip() if match else ""

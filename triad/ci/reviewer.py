"""Multi-model parallel code reviewer for CI/CD integration.

Runs each configured model independently on a git diff, parses their
findings, cross-validates through agreement mapping, and produces a
consolidated ReviewResult with a consensus recommendation.
"""

from __future__ import annotations

import asyncio
import logging
import re
import time

from triad.prompts import render_prompt
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.ci import (
    ModelAssessment,
    ReviewConfig,
    ReviewFinding,
    ReviewResult,
)
from triad.schemas.pipeline import ModelConfig

logger = logging.getLogger(__name__)

# Regex patterns for parsing model review output
_FINDING_RE = re.compile(
    r"FINDING:\s*\n"
    r"SEVERITY:\s*(?P<severity>\w+)\s*\n"
    r"FILE:\s*(?P<file>.+?)\s*\n"
    r"LINE:\s*(?P<line>.+?)\s*\n"
    r"DESCRIPTION:\s*(?P<description>.+?)\s*\n"
    r"SUGGESTION:\s*(?P<suggestion>.+?)(?=\n\n|\nFINDING:|\nASSESSMENT:|$)",
    re.DOTALL,
)

_ASSESSMENT_RE = re.compile(
    r"ASSESSMENT:\s*(?P<recommendation>\w+)",
    re.IGNORECASE,
)

_RATIONALE_RE = re.compile(
    r"RATIONALE:\s*(?P<rationale>.+?)(?=$|\n\n)",
    re.DOTALL,
)


class ReviewRunner:
    """Runs multi-model parallel review of a code diff.

    Each model independently reviews the diff. Findings are parsed,
    cross-validated by checking agreement across models, and consolidated
    into a single ReviewResult with a consensus recommendation.
    """

    def __init__(self, registry: dict[str, ModelConfig]) -> None:
        self._registry = registry

    async def review(
        self,
        diff: str,
        config: ReviewConfig,
    ) -> ReviewResult:
        """Run parallel review of a diff across configured models.

        Args:
            diff: Unified diff string.
            config: Review configuration.

        Returns:
            Consolidated ReviewResult.
        """
        start = time.monotonic()

        if not diff.strip():
            return ReviewResult(
                consensus_recommendation="approve",
                findings=[],
                model_assessments=[],
                total_findings=0,
                critical_count=0,
                models_used=[],
                total_cost=0.0,
                duration_seconds=0.0,
            )

        # Select models
        model_keys = config.models or list(self._registry.keys())
        model_keys = [k for k in model_keys if k in self._registry]

        if not model_keys:
            raise RuntimeError("No valid models available for review")

        # Run parallel reviews
        tasks = {
            key: self._review_with_model(key, diff, config)
            for key in model_keys
        }

        results = await asyncio.gather(
            *tasks.values(), return_exceptions=True,
        )

        assessments: list[ModelAssessment] = []
        for key, result in zip(tasks, results, strict=True):
            if isinstance(result, Exception):
                logger.error("Model %s failed review: %s", key, result)
                continue
            assessments.append(result)

        # Cross-validate findings
        all_findings = self._cross_validate(assessments, config.arbiter_enabled)

        # Determine consensus
        consensus = self._compute_consensus(assessments, all_findings)

        duration = time.monotonic() - start
        total_cost = sum(a.cost for a in assessments)

        critical_count = sum(
            1 for f in all_findings if f.severity == "critical"
        )

        return ReviewResult(
            consensus_recommendation=consensus,
            findings=all_findings,
            model_assessments=assessments,
            total_findings=len(all_findings),
            critical_count=critical_count,
            models_used=[a.model_key for a in assessments],
            total_cost=total_cost,
            duration_seconds=duration,
        )

    async def _review_with_model(
        self,
        model_key: str,
        diff: str,
        config: ReviewConfig,
    ) -> ModelAssessment:
        """Run a single model's review of the diff."""
        model_config = self._registry[model_key]
        provider = LiteLLMProvider(model_config)

        tpl_vars: dict = {"diff": diff}
        if config.focus_areas:
            tpl_vars["focus_areas"] = ", ".join(config.focus_areas)

        system = render_prompt("review_diff", **tpl_vars)

        msg = await provider.complete(
            messages=[{
                "role": "user",
                "content": "Review the code diff as described in your instructions.",
            }],
            system=system,
            timeout=90,
        )

        cost = msg.token_usage.cost if msg.token_usage else 0.0

        # Parse findings from output
        findings = _parse_findings(msg.content, model_key)

        # Parse assessment
        recommendation = "approve"
        rationale = ""

        assessment_match = _ASSESSMENT_RE.search(msg.content)
        if assessment_match:
            rec = assessment_match.group("recommendation").lower()
            if rec in ("request_changes", "request-changes"):
                recommendation = "request_changes"

        rationale_match = _RATIONALE_RE.search(msg.content)
        if rationale_match:
            rationale = rationale_match.group("rationale").strip()

        return ModelAssessment(
            model_key=model_key,
            recommendation=recommendation,
            findings=findings,
            rationale=rationale,
            cost=cost,
        )

    def _cross_validate(
        self,
        assessments: list[ModelAssessment],
        arbiter_enabled: bool,
    ) -> list[ReviewFinding]:
        """Cross-validate findings across models via agreement mapping.

        Findings reported by multiple models get confirmed=True and
        confidence="high". Unique findings remain needs_verification.
        """
        if not arbiter_enabled or len(assessments) <= 1:
            # No cross-validation — return all findings as-is
            all_findings: list[ReviewFinding] = []
            for a in assessments:
                all_findings.extend(a.findings)
            return all_findings

        # Group findings by (file, approximate_description)
        finding_groups: dict[str, list[tuple[str, ReviewFinding]]] = {}

        for a in assessments:
            for f in a.findings:
                # Key: file + first 50 chars of description (normalized)
                key = f"{f.file}:{f.severity}:{f.description[:50].lower().strip()}"
                if key not in finding_groups:
                    finding_groups[key] = []
                finding_groups[key].append((a.model_key, f))

        # Build deduplicated findings
        consolidated: list[ReviewFinding] = []
        for entries in finding_groups.values():
            reporters = list({model for model, _ in entries})
            base_finding = entries[0][1]
            confirmed = len(reporters) > 1
            confidence = "high" if confirmed else "needs_verification"

            consolidated.append(ReviewFinding(
                severity=base_finding.severity,
                file=base_finding.file,
                line=base_finding.line,
                description=base_finding.description,
                suggestion=base_finding.suggestion,
                reported_by=reporters,
                confirmed=confirmed,
                confidence=confidence,
            ))

        return consolidated

    def _compute_consensus(
        self,
        assessments: list[ModelAssessment],
        findings: list[ReviewFinding],
    ) -> str:
        """Determine consensus recommendation.

        Rules:
        - If any critical confirmed finding exists → request_changes
        - If majority of models say request_changes → request_changes
        - Otherwise → approve
        """
        if not assessments:
            return "approve"

        # Any confirmed critical finding forces request_changes
        if any(
            f.severity == "critical" and f.confirmed
            for f in findings
        ):
            return "request_changes"

        # Majority vote
        request_count = sum(
            1 for a in assessments if a.recommendation == "request_changes"
        )
        if request_count > len(assessments) / 2:
            return "request_changes"

        return "approve"


def _parse_findings(content: str, model_key: str) -> list[ReviewFinding]:
    """Parse FINDING blocks from model output."""
    findings: list[ReviewFinding] = []

    for match in _FINDING_RE.finditer(content):
        severity = match.group("severity").lower().strip()
        if severity not in ("critical", "warning", "suggestion"):
            severity = "suggestion"

        file_path = match.group("file").strip()
        line_str = match.group("line").strip()
        line: int | None = None
        if line_str.isdigit():
            line = int(line_str)

        findings.append(ReviewFinding(
            severity=severity,
            file=file_path,
            line=line,
            description=match.group("description").strip(),
            suggestion=match.group("suggestion").strip(),
            reported_by=[model_key],
        ))

    return findings

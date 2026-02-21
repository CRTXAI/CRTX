"""Arbiter reviewer — independent code review step for the Loop."""

from __future__ import annotations

import time
from dataclasses import dataclass, field

from triad.arbiter.arbiter import ArbiterEngine
from triad.arbiter.feedback import format_arbiter_feedback
from triad.loop.fixer import CodeFixer, FixResult
from triad.loop.router import RouteDecision
from triad.loop.test_runner import CheckResult, TestReport, TestRunner
from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ModelConfig, PipelineConfig, TaskSpec


@dataclass
class ReviewResult:
    """Result of an arbiter review step."""

    verdict: Verdict = Verdict.APPROVE
    review: ArbiterReview | None = None
    fix_result: FixResult | None = None
    final_test: TestReport | None = None
    cost: float = 0.0
    duration_seconds: float = 0.0


class ArbiterReviewer:
    """Thin wrapper around ArbiterEngine for Loop code review.

    After the test-fix cycle converges, a different model reviews the
    generated code. If the arbiter issues a REJECT (or HALT), one
    targeted fix cycle is triggered.
    """

    def __init__(self, registry: dict[str, ModelConfig]) -> None:
        config = PipelineConfig(default_timeout=300)
        self._arbiter = ArbiterEngine(config, registry)
        self._fixer = CodeFixer()
        self._tester = TestRunner()

    async def review(
        self,
        files: dict[str, str],
        prompt: str,
        route: RouteDecision,
        registry: dict[str, ModelConfig],
    ) -> ReviewResult:
        """Run arbiter review and optionally fix on REJECT.

        Args:
            files: Generated code files (path -> content).
            prompt: Original task description.
            route: Routing decision (contains generator model).
            registry: Model registry for fixer calls.

        Returns:
            ReviewResult with verdict, optional fix, and cost.
        """
        start = time.monotonic()

        # Format files for the arbiter
        formatted = self._format_files(files)
        task_spec = TaskSpec(task=prompt)

        arbiter_review = await self._arbiter.review(
            stage=PipelineStage.VERIFY,
            stage_model=route.model,
            stage_output=formatted,
            task=task_spec,
        )

        # No arbiter models available — degrade gracefully
        if arbiter_review is None:
            return ReviewResult(
                verdict=Verdict.APPROVE,
                cost=0.0,
                duration_seconds=time.monotonic() - start,
            )

        cost = arbiter_review.token_cost
        verdict = arbiter_review.verdict

        result = ReviewResult(
            verdict=verdict,
            review=arbiter_review,
            cost=cost,
            duration_seconds=time.monotonic() - start,
        )

        # Only REJECT and HALT trigger a fix cycle
        if verdict not in (Verdict.REJECT, Verdict.HALT):
            return result

        # Build a synthetic TestReport from arbiter feedback so
        # CodeFixer.fix() can use its existing broken-file logic.
        feedback = format_arbiter_feedback(arbiter_review, retry_number=1)
        synthetic_report = TestReport(
            tests=CheckResult(passed=False, details=feedback),
        )

        fix_result = await self._fixer.fix(
            files, synthetic_report, route, registry, iteration=0,
        )
        cost += fix_result.cost

        result.fix_result = fix_result

        if not fix_result.error:
            # Re-test the fixed files
            final_test = await self._tester.run_all(fix_result.files)
            result.final_test = final_test

        result.cost = cost
        result.duration_seconds = time.monotonic() - start
        return result

    @staticmethod
    def _format_files(files: dict[str, str]) -> str:
        """Format file dict into ``# file: path`` blocks for the arbiter."""
        parts: list[str] = []
        for path, content in sorted(files.items()):
            parts.append(f"# file: {path}\n{content}")
        return "\n\n".join(parts)

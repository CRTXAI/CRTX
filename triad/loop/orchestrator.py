"""Loop orchestrator — ties route → generate → test → fix together."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from triad.loop.fixer import CodeFixer, FixResult
from triad.loop.generator import (
    CodeGenerator,
    GenerationResult,
    extract_files_from_content,
)
from triad.loop.reviewer import ArbiterReviewer, ReviewResult
from triad.loop.router import RouteDecision, TaskRouter
from triad.loop.test_runner import TestReport, TestRunner
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.pipeline import ModelConfig

_TEST_FILE_RE = re.compile(r"(^|/)test_[^/]+\.py$|_test\.py$")

_TEST_GEN_SYSTEM = """\
You are an expert Python test engineer. Write comprehensive pytest tests.

Rules:
1. Output ONLY code using # file: headers.
2. Use direct imports (from module import Class), not relative imports.
3. Cover all public functions and edge cases.
4. Type hints on all test functions (-> None).
5. No explanations — just the test code.
"""

_GAP_DIAGNOSE_SYSTEM = """\
You are an expert Python debugger. A model wrote code and tried to fix it \
multiple times but these test failures remain. Do NOT write code. Analyze the \
failure and explain the root cause in 2-3 sentences. What assumption might be \
wrong? What specific mismatch exists between the source API and the test \
expectations?
"""

_GAP_FIX_SYSTEM = """\
You are an expert Python debugger. Fix the specific issues described in the \
diagnosis below.

Rules:
1. Output the complete fixed files using # file: path headers.
2. Include the FULL file content, not just changed lines.
3. Use direct imports (from module import X), not relative imports.
4. Fix either the source to pass the test, or the test if it's testing the \
wrong API. Never delete failing tests.
5. No explanations — just the fixed code.
"""

_GAP_SECOND_OPINION_SYSTEM = """\
You are an expert Python debugger brought in for a second opinion. A different \
model wrote this code and tried to fix it but couldn't resolve the remaining \
failures. Their diagnosis is included below but they couldn't fix it. Your \
job: either fix the source to pass the tests, or fix the tests if they're \
testing the wrong API.

Rules:
1. Output the complete fixed files using # file: path headers.
2. Include the FULL file content, not just changed lines.
3. Use direct imports (from module import X), not relative imports.
4. Never delete failing tests. Never hide failures.
5. No explanations — just the fixed code.
"""

# Preferred escalation model order per primary model family
_ESCALATION_PREFS: dict[str, list[str]] = {
    "anthropic": ["o3", "gemini-pro", "gemini-flash", "o3-mini"],
    "openai": ["claude-sonnet", "gemini-pro", "gemini-flash"],
    "google": ["claude-sonnet", "o3", "o3-mini"],
}


@dataclass
class GapResult:
    """Result of the close-the-gap escalation phase."""

    tier_reached: int = 0          # 0 = not attempted, 1-3 = highest tier tried
    resolved_at_tier: int = 0      # 0 = unresolved, 1-3 = which tier fixed it
    diagnosis: str = ""            # Tier 1 root-cause analysis
    escalation_model: str = ""     # Tier 3 second-opinion model (if used)
    cost: float = 0.0              # Total cost across all gap tiers
    duration_seconds: float = 0.0


@dataclass
class LoopStats:
    """Execution statistics for a Loop run."""

    iterations: int = 0
    generation_cost: float = 0.0
    fix_cost: float = 0.0
    total_cost: float = 0.0
    duration_seconds: float = 0.0
    gap_tier_reached: int = 0
    gap_diagnosis: str = ""
    gap_escalation_model: str = ""
    gap_resolved_at_tier: int = 0
    gap_cost: float = 0.0


@dataclass
class LoopResult:
    """Final result of a complete Loop execution."""

    files: dict[str, str] = field(default_factory=dict)
    test_report: TestReport = field(default_factory=TestReport)
    route: RouteDecision | None = None
    stats: LoopStats = field(default_factory=LoopStats)
    generation_result: GenerationResult | None = None
    fix_results: list[FixResult] = field(default_factory=list)
    review_result: ReviewResult | None = None
    gap_result: GapResult | None = None
    error: str = ""


class LoopOrchestrator:
    """Main Loop controller: route → generate → test → fix cycle."""

    def __init__(
        self,
        registry: dict[str, ModelConfig],
        *,
        arbiter: bool = True,
        on_route: object | None = None,
        on_generate: object | None = None,
        on_test: object | None = None,
        on_fix: object | None = None,
        on_review: object | None = None,
        on_escalation: object | None = None,
    ) -> None:
        self._registry = registry
        self._router = TaskRouter()
        self._generator = CodeGenerator()
        self._tester = TestRunner()
        self._fixer = CodeFixer()
        self._arbiter_enabled = arbiter

        # Callbacks for the presenter
        self._on_route = on_route
        self._on_generate = on_generate
        self._on_test = on_test
        self._on_fix = on_fix
        self._on_review = on_review
        self._on_escalation = on_escalation

    async def _generate_tests(
        self,
        prompt: str,
        files: dict[str, str],
        route: RouteDecision,
        gen_result: GenerationResult,
    ) -> tuple[dict[str, str], float]:
        """Generate test files when the initial generation produced none.

        Returns (test_files, cost).
        """
        model_config = self._generator._find_model_config(
            route.model, self._registry,
        )
        if model_config is None:
            return {}, 0.0

        source_context = "\n\n".join(
            f"# file: {fp}\n```python\n{code}\n```"
            for fp, code in files.items()
        )
        user_msg = (
            "The following code was generated but has no unit tests. "
            "Write comprehensive pytest tests for this code. "
            "Use direct imports, not relative imports. "
            "Output only the test file(s) using # file: headers.\n\n"
            f"Original task: {prompt}\n\n{source_context}"
        )

        provider = LiteLLMProvider(model_config)
        try:
            agent_msg = await provider.complete(
                [{"role": "user", "content": user_msg}],
                _TEST_GEN_SYSTEM,
                timeout=300,
            )
        except Exception:
            return {}, 0.0

        test_files = extract_files_from_content(agent_msg.content)
        for block in agent_msg.code_blocks:
            from triad.loop.generator import _sanitise_filepath
            fp = _sanitise_filepath(block.filepath)
            if fp not in test_files and block.content.strip():
                test_files[fp] = block.content

        cost = agent_msg.token_usage.cost if agent_msg.token_usage else 0.0
        return test_files, cost

    # ── Close-the-gap escalation tiers ─────────────────────────

    async def _close_the_gap(
        self,
        files: dict[str, str],
        test_report: TestReport,
        route: RouteDecision,
    ) -> tuple[dict[str, str], TestReport, GapResult]:
        """Three-tier escalation when the normal fix cycle can't resolve failures.

        Tier 1: Diagnose then fix (same model, two calls).
        Tier 2: Minimal context retry (same model, stripped to failing pair).
        Tier 3: Second opinion (different model with diagnosis context).
        """
        gap = GapResult()
        gap_start = time.monotonic()
        failure = test_report.failure_summary()

        # Identify the failing test file and its source dependency
        test_file, source_file = self._identify_failing_pair(files, test_report)

        # ── Tier 1: Diagnose then fix ─────────────────────────
        gap.tier_reached = 1
        model_config = self._generator._find_model_config(
            route.model, self._registry,
        )
        if model_config is None:
            gap.duration_seconds = time.monotonic() - gap_start
            return files, test_report, gap

        diagnosis, diag_cost = await self._gap_diagnose(
            model_config, files, test_file, source_file, failure,
        )
        gap.diagnosis = diagnosis
        gap.cost += diag_cost

        if diagnosis:
            fix_files, fix_cost = await self._gap_fix_from_diagnosis(
                model_config, files, test_file, source_file, failure, diagnosis,
            )
            gap.cost += fix_cost
            if fix_files:
                merged = {**files, **fix_files}
                report = await self._tester.run_all(merged)
                if report.all_pass:
                    gap.resolved_at_tier = 1
                    gap.duration_seconds = time.monotonic() - gap_start
                    if self._on_escalation:
                        self._on_escalation(gap, report)
                    return merged, report, gap
                # Keep the fix attempt even if it didn't fully resolve
                files = merged
                test_report = report

        # ── Tier 2: Minimal context retry ─────────────────────
        gap.tier_reached = 2
        if test_file and source_file:
            fix_files, fix_cost = await self._gap_minimal_fix(
                model_config, files, test_file, source_file, failure,
            )
            gap.cost += fix_cost
            if fix_files:
                merged = {**files, **fix_files}
                report = await self._tester.run_all(merged)
                if report.all_pass:
                    gap.resolved_at_tier = 2
                    gap.duration_seconds = time.monotonic() - gap_start
                    if self._on_escalation:
                        self._on_escalation(gap, report)
                    return merged, report, gap
                files = merged
                test_report = report

        # ── Tier 3: Second opinion (different model) ──────────
        gap.tier_reached = 3
        alt_config, alt_key = self._select_alt_model(route.model)
        if alt_config is not None:
            gap.escalation_model = alt_key
            fix_files, fix_cost = await self._gap_second_opinion(
                alt_config, files, test_file, source_file,
                failure, gap.diagnosis,
            )
            gap.cost += fix_cost
            if fix_files:
                merged = {**files, **fix_files}
                report = await self._tester.run_all(merged)
                if report.all_pass:
                    gap.resolved_at_tier = 3
                    gap.duration_seconds = time.monotonic() - gap_start
                    if self._on_escalation:
                        self._on_escalation(gap, report)
                    return merged, report, gap
                files = merged
                test_report = report

        # All three tiers failed
        gap.duration_seconds = time.monotonic() - gap_start
        if self._on_escalation:
            self._on_escalation(gap, test_report)
        return files, test_report, gap

    async def _gap_diagnose(
        self,
        model_config: ModelConfig,
        files: dict[str, str],
        test_file: str,
        source_file: str,
        failure: str,
    ) -> tuple[str, float]:
        """Tier 1 step 1: Ask the model to diagnose the root cause (no code)."""
        context_parts = []
        if test_file and test_file in files:
            context_parts.append(f"# file: {test_file}\n```python\n{files[test_file]}\n```")
        if source_file and source_file in files:
            context_parts.append(f"# file: {source_file}\n```python\n{files[source_file]}\n```")

        user_msg = (
            "This test fails after multiple fix attempts. Do NOT write code. "
            "Analyze the failure and explain the root cause in 2-3 sentences. "
            "What assumption might be wrong?\n\n"
            f"FAILURE OUTPUT:\n{failure[:2000]}\n\n"
            + "\n\n".join(context_parts)
        )

        provider = LiteLLMProvider(model_config)
        try:
            resp = await provider.complete(
                [{"role": "user", "content": user_msg}],
                _GAP_DIAGNOSE_SYSTEM,
                timeout=120,
            )
        except Exception:
            return "", 0.0

        cost = resp.token_usage.cost if resp.token_usage else 0.0
        return resp.content.strip(), cost

    async def _gap_fix_from_diagnosis(
        self,
        model_config: ModelConfig,
        files: dict[str, str],
        test_file: str,
        source_file: str,
        failure: str,
        diagnosis: str,
    ) -> tuple[dict[str, str], float]:
        """Tier 1 step 2: Fix informed by the diagnosis."""
        context_parts = []
        for fp in (test_file, source_file):
            if fp and fp in files:
                context_parts.append(f"# file: {fp}\n```python\n{files[fp]}\n```")

        user_msg = (
            f"DIAGNOSIS:\n{diagnosis}\n\n"
            f"FAILURE OUTPUT:\n{failure[:1500]}\n\n"
            "Based on this diagnosis, fix the code.\n\n"
            + "\n\n".join(context_parts)
        )
        return await self._call_for_fix(model_config, user_msg, _GAP_FIX_SYSTEM)

    async def _gap_minimal_fix(
        self,
        model_config: ModelConfig,
        files: dict[str, str],
        test_file: str,
        source_file: str,
        failure: str,
    ) -> tuple[dict[str, str], float]:
        """Tier 2: Strip context to only the failing test + its source file."""
        parts = []
        if test_file in files:
            parts.append(f"# file: {test_file}\n```python\n{files[test_file]}\n```")
        if source_file in files:
            parts.append(f"# file: {source_file}\n```python\n{files[source_file]}\n```")

        user_msg = (
            "Fix this code. Only these two files exist — nothing else.\n\n"
            f"FAILURE OUTPUT:\n{failure[:1500]}\n\n"
            + "\n\n".join(parts)
        )
        return await self._call_for_fix(model_config, user_msg, _GAP_FIX_SYSTEM)

    async def _gap_second_opinion(
        self,
        alt_config: ModelConfig,
        files: dict[str, str],
        test_file: str,
        source_file: str,
        failure: str,
        diagnosis: str,
    ) -> tuple[dict[str, str], float]:
        """Tier 3: Send to a different model with the primary's diagnosis."""
        context_parts = []
        for fp in (test_file, source_file):
            if fp and fp in files:
                context_parts.append(f"# file: {fp}\n```python\n{files[fp]}\n```")

        diag_section = ""
        if diagnosis:
            diag_section = (
                f"PRIMARY MODEL'S DIAGNOSIS (they couldn't fix it):\n"
                f"{diagnosis}\n\n"
            )

        user_msg = (
            f"{diag_section}"
            f"FAILURE OUTPUT:\n{failure[:1500]}\n\n"
            "What do you see? Fix the code.\n\n"
            + "\n\n".join(context_parts)
        )
        return await self._call_for_fix(alt_config, user_msg, _GAP_SECOND_OPINION_SYSTEM)

    async def _call_for_fix(
        self,
        model_config: ModelConfig,
        user_msg: str,
        system: str,
    ) -> tuple[dict[str, str], float]:
        """Shared helper: call a model and extract fix files."""
        provider = LiteLLMProvider(model_config)
        try:
            resp = await provider.complete(
                [{"role": "user", "content": user_msg}],
                system,
                timeout=300,
            )
        except Exception:
            return {}, 0.0

        fix_files = extract_files_from_content(resp.content)
        for block in resp.code_blocks:
            from triad.loop.generator import _sanitise_filepath
            fp = _sanitise_filepath(block.filepath)
            if fp not in fix_files and block.content.strip():
                fix_files[fp] = block.content

        cost = resp.token_usage.cost if resp.token_usage else 0.0
        return fix_files, cost

    @staticmethod
    def _identify_failing_pair(
        files: dict[str, str], report: TestReport,
    ) -> tuple[str, str]:
        """Find the primary failing test file and the source file it imports.

        Returns (test_file, source_file). Either may be empty if not identified.
        """
        import ast
        from pathlib import Path

        test_files = [fp for fp in files if "test" in fp.lower() and fp.endswith(".py")]
        source_files = [fp for fp in files if "test" not in fp.lower() and fp.endswith(".py")]

        # Try to find the specific failing test file from the report
        failing_test = ""
        details = report.tests.details + "\n" + report.runs.details
        for tf in test_files:
            if tf in details:
                failing_test = tf
                break
        if not failing_test and test_files:
            failing_test = test_files[0]

        # Find the source file the test imports
        target_source = ""
        if failing_test and failing_test in files:
            source_stems = {Path(fp).stem for fp in source_files}
            try:
                tree = ast.parse(files[failing_test])
                for node in ast.walk(tree):
                    if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                        stem = node.module.split(".")[0]
                        if stem in source_stems:
                            # Find the actual filepath for this stem
                            for fp in source_files:
                                if Path(fp).stem == stem:
                                    target_source = fp
                                    break
                            if target_source:
                                break
            except SyntaxError:
                pass

        if not target_source and source_files:
            target_source = source_files[0]

        return failing_test, target_source

    def _select_alt_model(
        self, primary_model: str,
    ) -> tuple[ModelConfig | None, str]:
        """Pick a different available model for Tier 3 escalation."""
        import os

        # Determine primary provider family
        primary_config = self._generator._find_model_config(
            primary_model, self._registry,
        )
        family = "anthropic"
        if primary_config:
            family = primary_config.provider

        prefs = _ESCALATION_PREFS.get(family, ["claude-sonnet", "o3", "gemini-pro"])

        for key in prefs:
            if key not in self._registry:
                continue
            cfg = self._registry[key]
            # Skip if it's the same model as the primary
            if cfg.model == primary_model:
                continue
            # Check API key is available
            if os.environ.get(cfg.api_key_env):
                return cfg, key

        return None, ""

    async def run(self, prompt: str) -> LoopResult:
        """Execute the full Loop cycle."""
        start = time.monotonic()

        # ── Route ────────────────────────────────────────────────
        route = self._router.classify(prompt)
        if self._on_route:
            self._on_route(route)

        # ── Generate ─────────────────────────────────────────────
        gen_result = await self._generator.generate(
            prompt, route, self._registry,
        )

        if self._on_generate:
            self._on_generate(gen_result)

        if gen_result.error:
            return LoopResult(
                route=route,
                generation_result=gen_result,
                error=gen_result.error,
                stats=LoopStats(
                    duration_seconds=time.monotonic() - start,
                    generation_cost=gen_result.cost,
                    total_cost=gen_result.cost,
                ),
            )

        files = gen_result.files
        if not files:
            return LoopResult(
                route=route,
                generation_result=gen_result,
                error="No files extracted from generation output",
                stats=LoopStats(
                    duration_seconds=time.monotonic() - start,
                    generation_cost=gen_result.cost,
                    total_cost=gen_result.cost,
                ),
            )

        # ── Test-generation fallback ──────────────────────────────
        # If generation produced zero test files, make a second call
        # so the fix cycle always has tests to work with.
        has_tests = any(_TEST_FILE_RE.search(fp) for fp in files)
        if not has_tests:
            test_files, test_cost = await self._generate_tests(
                prompt, files, route, gen_result,
            )
            files.update(test_files)
            gen_result = GenerationResult(
                files=files,
                raw_output=gen_result.raw_output,
                model=gen_result.model,
                cost=gen_result.cost + test_cost,
                prompt_tokens=gen_result.prompt_tokens,
                completion_tokens=gen_result.completion_tokens,
                duration_seconds=gen_result.duration_seconds,
            )

        # ── Test → Fix Loop ──────────────────────────────────────
        fix_cost = 0.0
        fix_results: list[FixResult] = []
        test_report = TestReport()
        iterations = 0

        for iteration in range(route.max_fix_iterations):
            iterations = iteration + 1

            test_report = await self._tester.run_all(files)
            if self._on_test:
                self._on_test(test_report, iteration)

            if test_report.all_pass:
                break

            # Don't fix on the last iteration — just report final state
            if iteration == route.max_fix_iterations - 1:
                break

            fix_result = await self._fixer.fix(
                files, test_report, route, self._registry, iteration,
            )
            fix_results.append(fix_result)
            fix_cost += fix_result.cost

            if self._on_fix:
                self._on_fix(fix_result, iteration)

            if fix_result.error:
                break

            files = fix_result.files

        # ── Final test if we just fixed ──────────────────────────
        if fix_results and not test_report.all_pass:
            test_report = await self._tester.run_all(files)
            if self._on_test:
                self._on_test(test_report, iterations)

        # ── Close the gap (3-tier escalation) ─────────────────
        gap_result: GapResult | None = None

        if not test_report.all_pass:
            files, test_report, gap_result = await self._close_the_gap(
                files, test_report, route,
            )
            fix_cost += gap_result.cost

        # ── Arbiter review ────────────────────────────────────────
        review_result: ReviewResult | None = None

        if self._arbiter_enabled:
            reviewer = ArbiterReviewer(self._registry)
            review_result = await reviewer.review(
                files, prompt, route, self._registry,
            )

            if self._on_review:
                self._on_review(review_result)

            # If REJECT fix was applied, update files and test report
            if review_result.fix_result and not review_result.fix_result.error:
                files = review_result.fix_result.files
                fix_cost += review_result.fix_result.cost
                if review_result.final_test is not None:
                    test_report = review_result.final_test

        total_cost = gen_result.cost + fix_cost
        if review_result:
            total_cost += review_result.cost
        elapsed = time.monotonic() - start

        # Build gap stats
        gap_stats_kw: dict[str, object] = {}
        if gap_result:
            gap_stats_kw = {
                "gap_tier_reached": gap_result.tier_reached,
                "gap_diagnosis": gap_result.diagnosis,
                "gap_escalation_model": gap_result.escalation_model,
                "gap_resolved_at_tier": gap_result.resolved_at_tier,
                "gap_cost": gap_result.cost,
            }

        return LoopResult(
            files=files,
            test_report=test_report,
            route=route,
            stats=LoopStats(
                iterations=iterations,
                generation_cost=gen_result.cost,
                fix_cost=fix_cost,
                total_cost=total_cost,
                duration_seconds=elapsed,
                **gap_stats_kw,
            ),
            generation_result=gen_result,
            fix_results=fix_results,
            review_result=review_result,
            gap_result=gap_result,
        )

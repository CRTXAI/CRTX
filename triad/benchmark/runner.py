"""Benchmark runner — orchestrates all conditions for a benchmark suite.

For each prompt, runs through the configured conditions (single-model and
CRTX pipeline variants), scores results, and saves artifacts.
"""

from __future__ import annotations

import json
import os
import time
from dataclasses import dataclass, field
from pathlib import Path

from rich.console import Console

from triad import __version__
from triad.benchmark.prompts import BenchmarkPrompt, get_quick_prompts
from triad.benchmark.scorer import BenchmarkScorer, ScoreBreakdown
from triad.benchmark.single_model import SingleModelResult, SingleModelRunner
from triad.loop.test_runner import TestReport, TestRunner
from triad.schemas.pipeline import ModelConfig


@dataclass
class VerificationData:
    """Results from running TestRunner on generated code."""

    parse_passed: bool = True
    imports_passed: bool = True
    tests_passed: bool = True
    runs_passed: bool = True
    test_passed: int = 0
    test_total: int = 0
    parse_errors: int = 0
    import_errors: int = 0
    syntax_errors: int = 0
    entry_failed: bool = False
    dev_time_minutes: float = 0.0


@dataclass
class ConditionResult:
    """Result for one (prompt, condition) pair."""

    prompt_id: str
    prompt_name: str
    condition: str
    score: ScoreBreakdown
    files: dict[str, str]
    raw_output: str
    duration_seconds: float
    cost: float
    error: str = ""
    verified: bool = False
    verification: VerificationData | None = None


@dataclass
class BenchmarkSuiteResult:
    """Full results of a benchmark run."""

    version: str
    timestamp: str
    conditions: list[str]
    results: list[ConditionResult] = field(default_factory=list)
    total_cost: float = 0.0
    total_duration: float = 0.0


# Condition definitions: (name, type, config)
# type: "single" = one-shot model call, "crtx" = pipeline run, "loop" = Loop cycle
QUICK_CONDITIONS = [
    ("single_sonnet", "single", "claude-sonnet"),
    ("single_o3", "single", "o3"),
    ("crtx_debate", "crtx", "debate"),
    ("crtx_loop", "loop", ""),
]

FULL_CONDITIONS = [
    ("single_sonnet", "single", "claude-sonnet"),
    ("single_o3", "single", "o3"),
    ("crtx_explore", "crtx", "explore"),
    ("crtx_debate", "crtx", "debate"),
    ("crtx_loop", "loop", ""),
]

# Tier-aware timeouts (seconds) for scoring
_PYTEST_TIMEOUT: dict[str, int] = {
    "simple": 60,
    "medium": 90,
    "complex": 120,
    "safety": 120,
}
_ENTRY_TIMEOUT: dict[str, int] = {
    "simple": 30,
    "medium": 45,
    "complex": 60,
    "safety": 60,
}


def _estimate_dev_time(vd: VerificationData, tier: str) -> float:
    """Estimate developer minutes to reach production quality.

    Formula: test failures × 3 + import errors × 2 + (20 if entry fails)
             + (5 per syntax error).  Cap at 45 for complex/safety tiers.
    """
    if vd.tests_passed and vd.imports_passed and vd.runs_passed and vd.parse_passed:
        return 0.0

    minutes = 0.0
    test_failures = vd.test_total - vd.test_passed
    minutes += test_failures * 3
    minutes += vd.import_errors * 2
    if vd.entry_failed:
        minutes += 20
    minutes += vd.syntax_errors * 5

    cap = 45.0 if tier in ("complex", "safety") else 30.0
    return min(minutes, cap)


def _build_verification(report: TestReport, tier: str) -> VerificationData:
    """Build VerificationData from a TestReport."""
    test_total = report.test_passed + report.test_failed + report.test_errors

    # Count syntax errors from parse details
    syntax_errors = 0
    if not report.parse.passed and report.parse.details:
        syntax_errors = len(report.parse.details.strip().splitlines())

    # Count import errors from import details
    import_errors = 0
    if not report.imports.passed and report.imports.details:
        lines = report.imports.details.strip().splitlines()
        import_errors = len([line for line in lines if "import" in line.lower()])

    vd = VerificationData(
        parse_passed=report.parse.passed,
        imports_passed=report.imports.passed,
        tests_passed=report.tests.passed,
        runs_passed=report.runs.passed,
        test_passed=report.test_passed,
        test_total=test_total,
        parse_errors=syntax_errors,
        import_errors=import_errors,
        syntax_errors=syntax_errors,
        entry_failed=not report.runs.passed,
    )
    vd.dev_time_minutes = _estimate_dev_time(vd, tier)
    return vd


class BenchmarkRunner:
    """Run benchmark prompts across multiple conditions."""

    def __init__(
        self,
        console: Console,
        registry: dict[str, ModelConfig],
        *,
        quick: bool = True,
    ) -> None:
        self._console = console
        self._registry = registry
        self._quick = quick
        self._conditions = QUICK_CONDITIONS if quick else FULL_CONDITIONS
        self._data_dir = Path.home() / ".crtx" / "benchmark-data" / __version__
        self._data_dir.mkdir(parents=True, exist_ok=True)

    async def run(self) -> BenchmarkSuiteResult:
        """Execute the full benchmark suite."""
        prompts = get_quick_prompts()
        condition_names = [c[0] for c in self._conditions]

        suite = BenchmarkSuiteResult(
            version=__version__,
            timestamp=time.strftime("%Y-%m-%dT%H:%M:%S"),
            conditions=condition_names,
        )

        suite_start = time.monotonic()

        for prompt in prompts:
            self._console.print(
                f"\n[bold cyan]Prompt {prompt.id}:[/bold cyan] {prompt.name} "
                f"[dim]({prompt.tier})[/dim]"
            )

            for cond_name, cond_type, cond_config in self._conditions:  # noqa: B007
                self._console.print(
                    f"  [dim]Running {cond_name}...[/dim]", end=""
                )

                try:
                    if cond_type == "single":
                        cr = await self._run_single(prompt, cond_name, cond_config)
                    elif cond_type == "loop":
                        cr = await self._run_loop(prompt, cond_name)
                    else:
                        cr = await self._run_crtx(prompt, cond_name, cond_config)
                except Exception as exc:
                    cr = ConditionResult(
                        prompt_id=prompt.id,
                        prompt_name=prompt.name,
                        condition=cond_name,
                        score=ScoreBreakdown(),
                        files={},
                        raw_output="",
                        duration_seconds=0.0,
                        cost=0.0,
                        error=str(exc),
                    )

                suite.results.append(cr)
                suite.total_cost += cr.cost

                # Print inline result
                if cr.error:
                    self._console.print(
                        f" [red]ERROR[/red] {cr.error[:60]}"
                    )
                else:
                    self._console.print(
                        f" [green]{cr.score.composite:.0%}[/green] "
                        f"({cr.duration_seconds:.1f}s, ${cr.cost:.4f})"
                    )

                # Save artifacts
                self._save_artifacts(cr)

        suite.total_duration = time.monotonic() - suite_start
        return suite

    async def _verify(
        self, files: dict[str, str], tier: str,
    ) -> VerificationData:
        """Run TestRunner on generated files and build verification data."""
        if not files:
            return VerificationData(
                parse_passed=False, dev_time_minutes=30.0,
            )
        pt = _PYTEST_TIMEOUT.get(tier, 60)
        et = _ENTRY_TIMEOUT.get(tier, 30)
        runner = TestRunner(pytest_timeout=pt, entry_timeout=et)
        report = await runner.run_all(files)
        return _build_verification(report, tier)

    async def _run_single(
        self,
        prompt: BenchmarkPrompt,
        cond_name: str,
        model_key: str,
    ) -> ConditionResult:
        """Run a single-model condition."""
        if model_key not in self._registry:
            return ConditionResult(
                prompt_id=prompt.id,
                prompt_name=prompt.name,
                condition=cond_name,
                score=ScoreBreakdown(),
                files={},
                raw_output="",
                duration_seconds=0.0,
                cost=0.0,
                error=f"Model '{model_key}' not in registry",
            )

        model_config = self._registry[model_key]

        # Check API key
        if not os.environ.get(model_config.api_key_env):
            return ConditionResult(
                prompt_id=prompt.id,
                prompt_name=prompt.name,
                condition=cond_name,
                score=ScoreBreakdown(),
                files={},
                raw_output="",
                duration_seconds=0.0,
                cost=0.0,
                error=f"No API key ({model_config.api_key_env})",
            )

        runner = SingleModelRunner(model_config)
        result: SingleModelResult = await runner.run(prompt)

        if result.error:
            return ConditionResult(
                prompt_id=prompt.id,
                prompt_name=prompt.name,
                condition=cond_name,
                score=ScoreBreakdown(),
                files=result.files,
                raw_output=result.raw_output,
                duration_seconds=result.duration_seconds,
                cost=result.cost,
                error=result.error,
            )

        # Score the output
        scorer = BenchmarkScorer(
            result.files,
            entry_point=prompt.entry_point,
            pytest_timeout=_PYTEST_TIMEOUT.get(prompt.tier, 60),
            entry_timeout=_ENTRY_TIMEOUT.get(prompt.tier, 30),
        )
        score = scorer.score_all()

        # Verification — run TestRunner on generated code
        verification = await self._verify(result.files, prompt.tier)

        return ConditionResult(
            prompt_id=prompt.id,
            prompt_name=prompt.name,
            condition=cond_name,
            score=score,
            files=result.files,
            raw_output=result.raw_output,
            duration_seconds=result.duration_seconds,
            cost=result.cost,
            verification=verification,
        )

    async def _run_crtx(
        self,
        prompt: BenchmarkPrompt,
        cond_name: str,
        preset_name: str,
    ) -> ConditionResult:
        """Run a CRTX pipeline condition using the specified preset."""
        from triad.dashboard.events import PipelineEventEmitter
        from triad.orchestrator import run_pipeline
        from triad.presets import resolve_preset
        from triad.providers.registry import load_pipeline_config
        from triad.schemas.pipeline import (
            ArbiterMode,
            PipelineConfig,
            PipelineMode,
            TaskSpec,
        )
        from triad.schemas.routing import RoutingStrategy

        mode, route, arbiter = resolve_preset(preset_name)
        base_config = load_pipeline_config()

        config = PipelineConfig(
            pipeline_mode=PipelineMode(mode),
            arbiter_mode=ArbiterMode(arbiter),
            routing_strategy=RoutingStrategy(route),
            reconciliation_enabled=base_config.reconciliation_enabled,
            default_timeout=300,
            max_retries=base_config.max_retries,
            reconciliation_retries=base_config.reconciliation_retries,
            stages=base_config.stages,
            arbiter_model=base_config.arbiter_model,
            reconcile_model=base_config.reconcile_model,
            min_fitness=base_config.min_fitness,
            persist_sessions=False,  # don't pollute session DB
        )

        task_spec = TaskSpec(
            task=prompt.prompt_text,
            output_dir=str(self._data_dir / prompt.id / cond_name),
        )

        emitter = PipelineEventEmitter()
        start = time.monotonic()

        try:
            pipeline_result = await run_pipeline(
                task_spec, config, self._registry, emitter,
            )
        except Exception as exc:
            return ConditionResult(
                prompt_id=prompt.id,
                prompt_name=prompt.name,
                condition=cond_name,
                score=ScoreBreakdown(),
                files={},
                raw_output="",
                duration_seconds=time.monotonic() - start,
                cost=0.0,
                error=str(exc),
            )

        elapsed = time.monotonic() - start

        # Extract files from the pipeline result using the same logic
        from triad.benchmark.single_model import _extract_files_from_content

        files: dict[str, str] = {}

        # Gather content from all stages
        for stage_msg in pipeline_result.stages.values():
            for block in stage_msg.code_blocks:
                fp = block.filepath
                if fp and not fp.startswith("untitled") and block.content.strip():
                    files[fp] = block.content

            # Also extract from raw content
            stage_files = _extract_files_from_content(stage_msg.content)
            for fp, code in stage_files.items():
                if fp not in files:
                    files[fp] = code

        # Check parallel/debate result content too
        extra_content = ""
        if pipeline_result.parallel_result:
            extra_content = pipeline_result.parallel_result.synthesized_output
        elif pipeline_result.debate_result:
            extra_content = pipeline_result.debate_result.judgment
        if extra_content:
            for fp, code in _extract_files_from_content(extra_content).items():
                if fp not in files:
                    files[fp] = code

        # Score
        scorer = BenchmarkScorer(
            files,
            entry_point=prompt.entry_point,
            pytest_timeout=_PYTEST_TIMEOUT.get(prompt.tier, 60),
            entry_timeout=_ENTRY_TIMEOUT.get(prompt.tier, 30),
        )
        score = scorer.score_all()

        # Verification — run TestRunner on generated code
        verification = await self._verify(files, prompt.tier)

        return ConditionResult(
            prompt_id=prompt.id,
            prompt_name=prompt.name,
            condition=cond_name,
            score=score,
            files=files,
            raw_output="",  # don't store full pipeline raw output
            duration_seconds=elapsed,
            cost=pipeline_result.total_cost,
            verification=verification,
        )

    async def _run_loop(
        self,
        prompt: BenchmarkPrompt,
        cond_name: str,
    ) -> ConditionResult:
        """Run a CRTX Loop condition (generate → test → fix cycle)."""
        from triad.loop.orchestrator import LoopOrchestrator

        orchestrator = LoopOrchestrator(self._registry)
        start = time.monotonic()

        try:
            loop_result = await orchestrator.run(prompt.prompt_text)
        except Exception as exc:
            return ConditionResult(
                prompt_id=prompt.id,
                prompt_name=prompt.name,
                condition=cond_name,
                score=ScoreBreakdown(),
                files={},
                raw_output="",
                duration_seconds=time.monotonic() - start,
                cost=0.0,
                error=str(exc),
            )

        elapsed = time.monotonic() - start

        if loop_result.error:
            return ConditionResult(
                prompt_id=prompt.id,
                prompt_name=prompt.name,
                condition=cond_name,
                score=ScoreBreakdown(),
                files=loop_result.files,
                raw_output="",
                duration_seconds=elapsed,
                cost=loop_result.stats.total_cost,
                error=loop_result.error,
            )

        scorer = BenchmarkScorer(
            loop_result.files,
            entry_point=prompt.entry_point,
            pytest_timeout=_PYTEST_TIMEOUT.get(prompt.tier, 60),
            entry_timeout=_ENTRY_TIMEOUT.get(prompt.tier, 30),
        )
        score = scorer.score_all()

        # Loop is inherently verified — it already ran test-fix cycles.
        # Build verification from the Loop's final test report.
        if loop_result.test_report is not None:
            verification = _build_verification(loop_result.test_report, prompt.tier)
        else:
            verification = await self._verify(loop_result.files, prompt.tier)

        return ConditionResult(
            prompt_id=prompt.id,
            prompt_name=prompt.name,
            condition=cond_name,
            score=score,
            files=loop_result.files,
            raw_output="",
            duration_seconds=elapsed,
            cost=loop_result.stats.total_cost,
            verified=True,
            verification=verification,
        )

    def _save_artifacts(self, cr: ConditionResult) -> None:
        """Save condition artifacts to the data directory."""
        artifact_dir = self._data_dir / cr.prompt_id / cr.condition
        artifact_dir.mkdir(parents=True, exist_ok=True)

        # Write generated code files
        for filepath, content in cr.files.items():
            dest = artifact_dir / "code" / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

        # Write score breakdown
        score_data: dict[str, object] = {
            "prompt_id": cr.prompt_id,
            "prompt_name": cr.prompt_name,
            "condition": cr.condition,
            "composite": cr.score.composite,
            "parse_rate": cr.score.parse_rate,
            "runs": cr.score.runs,
            "tests": cr.score.tests,
            "test_passed": cr.score.test_passed,
            "test_failed": cr.score.test_failed,
            "test_errors": cr.score.test_errors,
            "type_hints": cr.score.type_hints,
            "imports": cr.score.imports,
            "duration_seconds": cr.duration_seconds,
            "cost": cr.cost,
            "error": cr.error,
            "errors": cr.score.errors,
            "verified": cr.verified,
        }
        if cr.verification:
            vd = cr.verification
            score_data["verification"] = {
                "parse_passed": vd.parse_passed,
                "imports_passed": vd.imports_passed,
                "tests_passed": vd.tests_passed,
                "runs_passed": vd.runs_passed,
                "test_passed": vd.test_passed,
                "test_total": vd.test_total,
                "dev_time_minutes": vd.dev_time_minutes,
            }
        (artifact_dir / "score.json").write_text(
            json.dumps(score_data, indent=2), encoding="utf-8",
        )

    def estimate_cost(self) -> float:
        """Estimate total cost for the benchmark run.

        Uses conservative token projections:
        - Simple prompts: ~2K input, ~4K output tokens
        - Medium prompts: ~2K input, ~8K output tokens
        - Complex prompts: ~2K input, ~12K output tokens
        - CRTX pipeline: ~4x single model cost (4 stages)
        """
        prompts = get_quick_prompts()

        token_estimates = {
            "simple": (2000, 4000),
            "medium": (2000, 8000),
            "complex": (2000, 12000),
            "safety": (2000, 4000),
        }

        total = 0.0
        for prompt in prompts:
            input_tokens, output_tokens = token_estimates.get(
                prompt.tier, (2000, 8000)
            )
            for cond_name, cond_type, cond_config in self._conditions:  # noqa: B007
                if cond_type == "single":
                    if cond_config in self._registry:
                        cfg = self._registry[cond_config]
                        cost = (
                            (input_tokens / 1_000_000) * cfg.cost_input
                            + (output_tokens / 1_000_000) * cfg.cost_output
                        )
                        total += cost
                elif cond_type == "loop":
                    # Loop: ~1.5x single model (generation + potential fixes)
                    if "claude-sonnet" in self._registry:
                        cfg = self._registry["claude-sonnet"]
                        cost = (
                            (input_tokens / 1_000_000) * cfg.cost_input
                            + (output_tokens / 1_000_000) * cfg.cost_output
                        ) * 1.5
                        total += cost
                else:
                    # CRTX pipeline: estimate 4x a mid-range model call
                    # Use claude-sonnet as a proxy
                    if "claude-sonnet" in self._registry:
                        cfg = self._registry["claude-sonnet"]
                        cost = (
                            (input_tokens / 1_000_000) * cfg.cost_input
                            + (output_tokens / 1_000_000) * cfg.cost_output
                        ) * 4
                        total += cost

        return total


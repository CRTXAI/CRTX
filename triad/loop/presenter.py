"""Loop presenter — Rich terminal output for Loop execution."""

from __future__ import annotations

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triad.loop.fixer import FixResult
from triad.loop.generator import GenerationResult
from triad.loop.orchestrator import GapResult, LoopResult
from triad.loop.reviewer import ReviewResult
from triad.loop.router import RouteDecision
from triad.loop.test_runner import TestReport
from triad.schemas.arbiter import Verdict

# Brand colours matching cli_display.py
_MINT = "#00ffbb"
_GREEN = "#00ff88"
_DIM = "#6a8a6a"
_AMBER = "#ffaa00"
_RED = "#ff4444"


class LoopPresenter:
    """Render Loop progress and results to the terminal."""

    def __init__(self, console: Console) -> None:
        self._console = console

    # ── Phase callbacks (used by LoopOrchestrator) ───────────────

    def on_route(self, route: RouteDecision) -> None:
        tier_color = {
            "simple": _GREEN,
            "medium": _MINT,
            "complex": _AMBER,
            "safety": _RED,
        }.get(route.complexity.value, "white")

        debate = " + debate" if route.use_architecture_debate else ""
        self._console.print(
            Text.assemble(
                ("  Route: ", _DIM),
                (route.complexity.value, f"bold {tier_color}"),
                (f" \u2192 max {route.max_fix_iterations} iterations{debate}", _DIM),
            )
        )

    def on_generate(self, result: GenerationResult) -> None:
        if result.error:
            self._console.print(
                Text.assemble(
                    ("  Generate: ", _DIM),
                    (f"ERROR {result.error[:80]}", _RED),
                )
            )
            return

        file_count = len(result.files)
        line_count = sum(
            content.count("\n") + 1
            for content in result.files.values()
        )
        self._console.print(
            Text.assemble(
                ("  Generate: ", _DIM),
                (f"{file_count} files", _GREEN),
                (f", {line_count} lines ", "white"),
                (f"(${result.cost:.4f}, {result.duration_seconds:.1f}s)", _DIM),
            )
        )

    def on_test(self, report: TestReport, iteration: int) -> None:
        total = report.test_passed + report.test_failed + report.test_errors

        if report.all_pass:
            self._console.print(
                Text.assemble(
                    (f"  Test [{iteration + 1}]: ", _DIM),
                    (f"{report.test_passed}/{total} passing", f"bold {_GREEN}"),
                    (" \u2014 all checks pass", _GREEN),
                )
            )
        else:
            parts: list[tuple[str, str]] = []
            if report.test_passed:
                parts.append((f"{report.test_passed}P", _GREEN))
            if report.test_failed:
                if parts:
                    parts.append(("/", "white"))
                parts.append((f"{report.test_failed}F", _RED))
            if report.test_errors:
                if parts:
                    parts.append(("/", "white"))
                parts.append((f"{report.test_errors}E", _AMBER))
            if not parts:
                parts.append(("no tests", _DIM))

            failures: list[str] = []
            if not report.parse.passed:
                failures.append("parse")
            if not report.imports.passed:
                failures.append("imports")
            if not report.tests.passed:
                failures.append("tests")
            if not report.runs.passed:
                failures.append("entry")

            segments: list[tuple[str, str]] = [
                (f"  Test [{iteration + 1}]: ", _DIM),
            ]
            segments.extend(parts)
            segments.append((f" \u2014 failed: {', '.join(failures)}", _DIM))

            self._console.print(Text.assemble(*segments))

    def on_fix(self, result: FixResult, iteration: int) -> None:
        if result.error:
            self._console.print(
                Text.assemble(
                    (f"  Fix [{iteration + 1}]: ", _DIM),
                    (f"ERROR {result.error[:80]}", _RED),
                )
            )
            return

        self._console.print(
            Text.assemble(
                (f"  Fix [{iteration + 1}]: ", _DIM),
                (f"{len(result.files)} files updated", _MINT),
                (f" (${result.cost:.4f}, {result.duration_seconds:.1f}s)", _DIM),
            )
        )

    def on_review(self, result: ReviewResult) -> None:
        verdict = result.verdict
        verdict_color = {
            Verdict.APPROVE: _GREEN,
            Verdict.FLAG: _AMBER,
            Verdict.REJECT: _RED,
            Verdict.HALT: _RED,
        }.get(verdict, "white")

        segments: list[tuple[str, str]] = [
            ("  Review: ", _DIM),
            (verdict.value.upper(), f"bold {verdict_color}"),
        ]

        if result.review:
            segments.append(
                (f" (arbiter={result.review.arbiter_model}, "
                 f"${result.cost:.4f}, {result.duration_seconds:.1f}s)", _DIM),
            )

        self._console.print(Text.assemble(*segments))

        if result.fix_result and not result.fix_result.error:
            self._console.print(
                Text.assemble(
                    ("  Review fix: ", _DIM),
                    (f"{len(result.fix_result.files)} files updated", _MINT),
                    (f" (${result.fix_result.cost:.4f})", _DIM),
                )
            )

    def on_escalation(self, gap: GapResult, report: TestReport) -> None:
        tier_labels = {1: "diagnose+fix", 2: "minimal context", 3: "second opinion"}

        if gap.resolved_at_tier > 0:
            tier_name = tier_labels.get(gap.resolved_at_tier, f"tier {gap.resolved_at_tier}")
            model_note = ""
            if gap.resolved_at_tier == 3 and gap.escalation_model:
                model_note = f" via {gap.escalation_model}"
            self._console.print(
                Text.assemble(
                    ("  Gap: ", _DIM),
                    (f"resolved at tier {gap.resolved_at_tier}", f"bold {_GREEN}"),
                    (f" ({tier_name}{model_note}, ${gap.cost:.4f})", _DIM),
                )
            )
        else:
            tiers_tried = gap.tier_reached
            model_note = ""
            if gap.escalation_model:
                model_note = f", second opinion={gap.escalation_model}"
            self._console.print(
                Text.assemble(
                    ("  Gap: ", _DIM),
                    (f"unresolved after {tiers_tried} tiers", f"bold {_AMBER}"),
                    (f" (${gap.cost:.4f}{model_note})", _DIM),
                )
            )
            if gap.diagnosis:
                diag_display = gap.diagnosis[:120]
                if len(gap.diagnosis) > 120:
                    diag_display += "..."
                self._console.print(
                    Text.assemble(
                        ("    Diagnosis: ", _DIM),
                        (diag_display, _AMBER),
                    )
                )

    # ── Final results ────────────────────────────────────────────

    def show_result(self, result: LoopResult) -> None:
        """Display the final Loop result with summary panel."""
        self._console.print()

        if result.error:
            self._console.print(Panel(
                f"[bold {_RED}]Loop Failed[/bold {_RED}]\n\n{result.error}",
                border_style=_RED,
            ))
            return

        # Test summary
        tr = result.test_report
        total_tests = tr.test_passed + tr.test_failed + tr.test_errors
        if tr.all_pass:
            test_str = f"[bold {_GREEN}]{tr.test_passed}/{total_tests} passing[/bold {_GREEN}]"
        else:
            test_str = (
                f"[{_AMBER}]{tr.test_passed}/{total_tests} passing"
                f" ({tr.test_failed}F, {tr.test_errors}E)[/{_AMBER}]"
            )

        # Status
        if tr.all_pass:
            status = f"[bold {_GREEN}]Loop Complete[/bold {_GREEN}]"
        else:
            status = f"[bold {_AMBER}]Loop Complete (with failures)[/bold {_AMBER}]"

        # Arbiter verdict line
        arbiter_str = ""
        if result.review_result and result.review_result.review:
            rv = result.review_result
            v_color = {
                Verdict.APPROVE: _GREEN,
                Verdict.FLAG: _AMBER,
                Verdict.REJECT: _RED,
                Verdict.HALT: _RED,
            }.get(rv.verdict, "white")
            arbiter_str = (
                f"\n  Arbiter: [bold {v_color}]{rv.verdict.value.upper()}"
                f"[/bold {v_color}]"
            )

        # Gap escalation line
        gap_str = ""
        stats = result.stats
        if stats.gap_tier_reached > 0:
            tier_labels = {1: "diagnose+fix", 2: "minimal context", 3: "second opinion"}
            if stats.gap_resolved_at_tier > 0:
                tier_name = tier_labels.get(stats.gap_resolved_at_tier, "")
                model_note = ""
                if stats.gap_resolved_at_tier == 3 and stats.gap_escalation_model:
                    model_note = f" via {stats.gap_escalation_model}"
                gap_str = (
                    f"\n  Gap: [bold {_GREEN}]resolved at tier "
                    f"{stats.gap_resolved_at_tier}[/bold {_GREEN}]"
                    f" [{_DIM}]({tier_name}{model_note})[/{_DIM}]"
                )
            else:
                gap_str = (
                    f"\n  Gap: [{_AMBER}]unresolved after "
                    f"{stats.gap_tier_reached} tiers[/{_AMBER}]"
                )

        summary = (
            f"{status}\n\n"
            f"  Files: {len(result.files)}  |  "
            f"Iterations: {stats.iterations}\n"
            f"  Tests: {test_str}"
            f"{arbiter_str}"
            f"{gap_str}\n"
            f"  Cost: ${stats.total_cost:.4f}  |  "
            f"Time: {stats.duration_seconds:.0f}s"
        )

        self._console.print(Panel(summary, border_style=_MINT))

        # File list
        for name in sorted(result.files.keys()):
            is_test = "test" in name.lower()
            style = _DIM if is_test else "white"
            self._console.print(f"  [{style}]{name}[/{style}]")

        self._console.print()

    def show_header(self, task: str) -> None:
        """Display the task header before starting the Loop."""
        display = task if len(task) <= 120 else task[:117] + "..."
        self._console.print(Panel(
            f"[bold]{display}[/bold]",
            title=f"[bold {_MINT}]CRTX Loop[/bold {_MINT}]",
            border_style=_MINT,
        ))

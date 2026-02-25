"""Benchmark reporting — Rich terminal tables and JSON export."""

from __future__ import annotations

import json
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from triad.benchmark.runner import BenchmarkSuiteResult, ConditionResult, VerificationData


def display_results(suite: BenchmarkSuiteResult, console: Console) -> None:
    """Print per-prompt comparison tables and a final summary."""

    # Group results by prompt
    by_prompt: dict[str, list[ConditionResult]] = {}
    for cr in suite.results:
        by_prompt.setdefault(cr.prompt_id, []).append(cr)

    # Per-prompt tables
    for prompt_id, results in by_prompt.items():
        prompt_name = results[0].prompt_name if results else prompt_id

        table = Table(
            title=f"{prompt_id}: {prompt_name}",
            show_lines=True,
            title_style="bold cyan",
        )
        table.add_column("Condition", style="bold")
        table.add_column("Composite", justify="right")
        table.add_column("Parse", justify="right")
        table.add_column("Runs", justify="right")
        table.add_column("Tests", justify="right")
        table.add_column("Types", justify="right")
        table.add_column("Imports", justify="right")
        table.add_column("Verified", justify="center")
        table.add_column("Tests Run", justify="right")
        table.add_column("Dev Time", justify="right")
        table.add_column("Time", justify="right")
        table.add_column("Cost", justify="right")

        for cr in results:
            if cr.error:
                table.add_row(
                    cr.condition,
                    "[red]ERR[/red]",
                    "", "", "", "", "",
                    "", "", "",
                    f"{cr.duration_seconds:.1f}s",
                    f"${cr.cost:.4f}",
                )
            else:
                s = cr.score
                verified_cell, tests_run_cell, dev_time_cell = _verification_cells(
                    cr.verified, cr.verification,
                )
                table.add_row(
                    cr.condition,
                    _score_cell(s.composite),
                    _pct(s.parse_rate),
                    _pct(s.runs),
                    f"{s.test_passed}/{s.test_passed + s.test_failed + s.test_errors}",
                    _pct(s.type_hints),
                    _pct(s.imports),
                    verified_cell,
                    tests_run_cell,
                    dev_time_cell,
                    f"{cr.duration_seconds:.1f}s",
                    f"${cr.cost:.4f}",
                )

        console.print(table)
        console.print()

    # Summary table: average composite per condition
    _print_summary(suite, console)

    # Footer
    console.print(
        f"[dim]Total cost: ${suite.total_cost:.4f}  |  "
        f"Duration: {suite.total_duration:.0f}s  |  "
        f"Version: {suite.version}[/dim]"
    )


def _print_summary(suite: BenchmarkSuiteResult, console: Console) -> None:
    """Print condition-level summary with averages."""
    from collections import defaultdict

    scores: dict[str, list[float]] = defaultdict(list)
    costs: dict[str, float] = defaultdict(float)
    times: dict[str, float] = defaultdict(float)
    error_counts: dict[str, int] = defaultdict(int)
    dev_times: dict[str, list[float]] = defaultdict(list)

    for cr in suite.results:
        cond = cr.condition
        if cr.error:
            error_counts[cond] += 1
        else:
            scores[cond].append(cr.score.composite)
        costs[cond] += cr.cost
        times[cond] += cr.duration_seconds
        if cr.verification:
            dev_times[cond].append(cr.verification.dev_time_minutes)

    table = Table(title="Summary", show_lines=True, title_style="bold green")
    table.add_column("Condition", style="bold")
    table.add_column("Avg Score", justify="right")
    table.add_column("Min", justify="right")
    table.add_column("Max", justify="right")
    table.add_column("Errors", justify="right")
    table.add_column("Avg Dev Time", justify="right")
    table.add_column("Total Cost", justify="right")
    table.add_column("Total Time", justify="right")

    for cond in suite.conditions:
        cond_scores = scores.get(cond, [])
        if cond_scores:
            avg = sum(cond_scores) / len(cond_scores)
            mn = min(cond_scores)
            mx = max(cond_scores)
            avg_cell = _score_cell(avg)
            min_cell = f"{mn:.0%}"
            max_cell = f"{mx:.0%}"
        else:
            avg_cell = "[red]N/A[/red]"
            min_cell = ""
            max_cell = ""

        errs = error_counts.get(cond, 0)
        cond_dev = dev_times.get(cond, [])
        avg_dev = f"{sum(cond_dev) / len(cond_dev):.0f} min" if cond_dev else ""

        table.add_row(
            cond,
            avg_cell,
            min_cell,
            max_cell,
            str(errs) if errs else "[green]0[/green]",
            avg_dev,
            f"${costs.get(cond, 0):.4f}",
            f"{times.get(cond, 0):.0f}s",
        )

    console.print(table)
    console.print()


def write_results_json(suite: BenchmarkSuiteResult, output_path: Path) -> Path:
    """Write the full results to a JSON file."""
    data = {
        "version": suite.version,
        "timestamp": suite.timestamp,
        "conditions": suite.conditions,
        "total_cost": suite.total_cost,
        "total_duration": suite.total_duration,
        "results": [],
    }

    for cr in suite.results:
        entry: dict[str, object] = {
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
            "file_count": len(cr.files),
            "verified": cr.verified,
        }
        if cr.verification:
            vd = cr.verification
            entry["verification"] = {
                "parse_passed": vd.parse_passed,
                "imports_passed": vd.imports_passed,
                "tests_passed": vd.tests_passed,
                "runs_passed": vd.runs_passed,
                "test_passed": vd.test_passed,
                "test_total": vd.test_total,
                "dev_time_minutes": vd.dev_time_minutes,
            }
        data["results"].append(entry)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    output_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    return output_path


def display_last_results(console: Console) -> bool:
    """Load and display the most recent results.json. Returns False if none found."""
    from triad import __version__

    results_path = Path.home() / ".crtx" / "benchmark-data" / __version__ / "results.json"
    if not results_path.exists():
        # Try to find any version
        base = Path.home() / ".crtx" / "benchmark-data"
        if not base.exists():
            return False
        versions = sorted(base.iterdir(), reverse=True)
        for v in versions:
            candidate = v / "results.json"
            if candidate.exists():
                results_path = candidate
                break
        else:
            return False

    data = json.loads(results_path.read_text(encoding="utf-8"))

    # Reconstruct a minimal suite for display
    from triad.benchmark.scorer import ScoreBreakdown

    suite = BenchmarkSuiteResult(
        version=data["version"],
        timestamp=data["timestamp"],
        conditions=data["conditions"],
        total_cost=data["total_cost"],
        total_duration=data["total_duration"],
    )

    for r in data["results"]:
        verification = None
        vd_data = r.get("verification")
        if vd_data:
            verification = VerificationData(
                parse_passed=vd_data.get("parse_passed", True),
                imports_passed=vd_data.get("imports_passed", True),
                tests_passed=vd_data.get("tests_passed", True),
                runs_passed=vd_data.get("runs_passed", True),
                test_passed=vd_data.get("test_passed", 0),
                test_total=vd_data.get("test_total", 0),
                dev_time_minutes=vd_data.get("dev_time_minutes", 0.0),
            )

        suite.results.append(ConditionResult(
            prompt_id=r["prompt_id"],
            prompt_name=r["prompt_name"],
            condition=r["condition"],
            score=ScoreBreakdown(
                parse_rate=r["parse_rate"],
                runs=r["runs"],
                tests=r["tests"],
                test_passed=r["test_passed"],
                test_failed=r["test_failed"],
                test_errors=r["test_errors"],
                type_hints=r["type_hints"],
                imports=r["imports"],
                composite=r["composite"],
                errors=r.get("errors", []),
            ),
            files={},
            raw_output="",
            duration_seconds=r["duration_seconds"],
            cost=r["cost"],
            error=r.get("error", ""),
            verified=r.get("verified", False),
            verification=verification,
        ))

    console.print(Panel(
        f"[bold]Results from {data['timestamp']}[/bold]  (v{data['version']})",
        border_style="blue",
    ))
    display_results(suite, console)
    return True


def _score_cell(value: float) -> str:
    """Format a composite score with color coding."""
    pct = f"{value:.0%}"
    if value >= 0.8:
        return f"[bold green]{pct}[/bold green]"
    if value >= 0.5:
        return f"[yellow]{pct}[/yellow]"
    return f"[red]{pct}[/red]"


def _pct(value: float) -> str:
    """Format a 0–1 value as a percentage."""
    return f"{value:.0%}"


def _verification_cells(
    verified: bool, vd: VerificationData | None,
) -> tuple[str, str, str]:
    """Return (Verified, Tests Run, Dev Time) table cells."""
    if verified:
        verified_cell = "[bold green]Yes[/bold green]"
    else:
        verified_cell = "[dim]No[/dim]"

    if vd is None:
        return verified_cell, "", ""

    tests_run_cell = f"{vd.test_passed}/{vd.test_total}" if vd.test_total else "0/0"

    dt = vd.dev_time_minutes
    if dt == 0:
        dev_time_cell = "[green]0 min[/green]"
    elif dt <= 10:
        dev_time_cell = f"[yellow]{dt:.0f} min[/yellow]"
    else:
        dev_time_cell = f"[red]{dt:.0f} min[/red]"

    return verified_cell, tests_run_cell, dev_time_cell

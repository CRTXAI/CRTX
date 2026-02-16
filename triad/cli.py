"""Triad Orchestrator CLI — Typer + Rich terminal interface.

Commands: run, plan, estimate, models, config, sessions.
All output is Rich-powered with color-coded panels and tables.
"""

from __future__ import annotations

import asyncio
import os
from pathlib import Path

import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triad.providers.registry import load_models, load_pipeline_config
from triad.schemas.pipeline import (
    ArbiterMode,
    PipelineConfig,
    PipelineMode,
    TaskSpec,
)
from triad.schemas.routing import RoutingStrategy

console = Console()

# ── App and sub-apps ─────────────────────────────────────────────

app = typer.Typer(
    name="triad",
    help="Multi-model AI orchestration with adversarial Arbiter review.",
    no_args_is_help=True,
    rich_markup_mode="rich",
)

models_app = typer.Typer(
    name="models",
    help="Manage the model registry.",
    no_args_is_help=True,
)
app.add_typer(models_app, name="models")

config_app = typer.Typer(
    name="config",
    help="Show pipeline configuration.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

sessions_app = typer.Typer(
    name="sessions",
    help="Query session history.",
    no_args_is_help=True,
)
app.add_typer(sessions_app, name="sessions")


# ── Helpers ──────────────────────────────────────────────────────

def _load_registry():
    """Load the model registry, exit on error."""
    try:
        return load_models()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading models:[/red] {e}")
        raise typer.Exit(1) from None


def _load_config():
    """Load pipeline config, exit on error."""
    try:
        return load_pipeline_config()
    except (FileNotFoundError, ValueError) as e:
        console.print(f"[red]Error loading config:[/red] {e}")
        raise typer.Exit(1) from None


def _verdict_style(verdict: str) -> str:
    """Return a Rich style string for a verdict value."""
    return {
        "approve": "bold green",
        "flag": "bold yellow",
        "reject": "bold red",
        "halt": "bold bright_red",
    }.get(verdict.lower(), "white")


# ── triad run ────────────────────────────────────────────────────

@app.command()
def run(
    task: str = typer.Argument(..., help="Task description — what to build"),
    mode: str = typer.Option(
        "sequential", "--mode", "-m",
        help="Pipeline mode: sequential, parallel, or debate",
    ),
    route: str = typer.Option(
        "hybrid", "--route", "-r",
        help="Routing strategy: quality_first, cost_optimized, speed_first, hybrid",
    ),
    arbiter: str = typer.Option(
        "bookend", "--arbiter", "-a",
        help="Arbiter mode: off, final_only, bookend, full",
    ),
    reconcile: bool = typer.Option(
        False, "--reconcile",
        help="Enable Implementation Summary Reconciliation",
    ),
    context: str = typer.Option(
        "", "--context", "-c",
        help="Additional context for pipeline agents",
    ),
    domain_rules: str = typer.Option(
        "", "--domain-rules",
        help="Path to domain rules file (TOML)",
    ),
    timeout: int = typer.Option(
        120, "--timeout", "-t",
        help="Default per-stage timeout in seconds",
    ),
    max_retries: int = typer.Option(
        2, "--max-retries",
        help="Max Arbiter-triggered retries per stage",
    ),
    no_persist: bool = typer.Option(
        False, "--no-persist",
        help="Disable session persistence",
    ),
    output_dir: str = typer.Option(
        "triad-output", "--output-dir", "-o",
        help="Directory for pipeline output files",
    ),
) -> None:
    """Execute a pipeline run with the specified task and options."""
    # Validate enums
    try:
        pipeline_mode = PipelineMode(mode)
    except ValueError:
        console.print(
            f"[red]Invalid mode:[/red] '{mode}'. "
            f"Choose from: sequential, parallel, debate"
        )
        raise typer.Exit(1) from None

    try:
        routing_strategy = RoutingStrategy(route)
    except ValueError:
        console.print(
            f"[red]Invalid routing strategy:[/red] '{route}'. "
            f"Choose from: quality_first, cost_optimized, speed_first, hybrid"
        )
        raise typer.Exit(1) from None

    try:
        arbiter_mode = ArbiterMode(arbiter)
    except ValueError:
        console.print(
            f"[red]Invalid arbiter mode:[/red] '{arbiter}'. "
            f"Choose from: off, final_only, bookend, full"
        )
        raise typer.Exit(1) from None

    # Load domain rules from file if specified
    domain_context = ""
    if domain_rules:
        rules_path = Path(domain_rules)
        if not rules_path.exists():
            console.print(f"[red]Domain rules file not found:[/red] {domain_rules}")
            raise typer.Exit(1) from None
        domain_context = rules_path.read_text(encoding="utf-8")

    registry = _load_registry()
    base_config = _load_config()

    config = PipelineConfig(
        pipeline_mode=pipeline_mode,
        arbiter_mode=arbiter_mode,
        reconciliation_enabled=reconcile,
        default_timeout=timeout,
        max_retries=max_retries,
        reconciliation_retries=base_config.reconciliation_retries,
        stages=base_config.stages,
        arbiter_model=base_config.arbiter_model,
        reconcile_model=base_config.reconcile_model,
        routing_strategy=routing_strategy,
        min_fitness=base_config.min_fitness,
        persist_sessions=not no_persist,
        session_db_path=base_config.session_db_path,
    )

    task_spec = TaskSpec(
        task=task,
        context=context,
        domain_rules=domain_context,
        output_dir=output_dir,
    )

    # Show task summary
    _display_task_panel(task_spec, config)

    # Run pipeline
    from triad.orchestrator import run_pipeline

    console.print()
    with console.status("[bold blue]Running pipeline...", spinner="dots"):
        result = asyncio.run(run_pipeline(task_spec, config, registry))

    # Display results
    console.print()
    _display_result(result)

    # Write output files
    from triad.output.writer import write_pipeline_output

    write_pipeline_output(result, output_dir)
    console.print(f"\n[dim]Output written to:[/dim] {output_dir}/")


def _display_task_panel(task_spec: TaskSpec, config: PipelineConfig) -> None:
    """Show a summary panel for the task being run."""
    info = Text()
    info.append("Task: ", style="bold")
    info.append(task_spec.task)
    info.append("\n")
    info.append("Mode: ", style="bold")
    info.append(config.pipeline_mode.value)
    info.append("  Routing: ", style="bold")
    info.append(config.routing_strategy.value)
    info.append("  Arbiter: ", style="bold")
    info.append(config.arbiter_mode.value)
    if config.reconciliation_enabled:
        info.append("  +reconcile", style="dim")

    console.print(Panel(info, title="[bold blue]Triad Pipeline[/bold blue]", border_style="blue"))


def _display_result(result) -> None:
    """Display the pipeline result with colored verdicts and metrics."""

    # Status banner
    if result.halted:
        console.print(Panel(
            f"[bright_red]HALTED[/bright_red] — {result.halt_reason}",
            border_style="bright_red",
        ))
    elif result.success:
        console.print(Panel(
            "[bold green]Pipeline completed successfully[/bold green]",
            border_style="green",
        ))
    else:
        console.print(Panel(
            "[bold red]Pipeline failed[/bold red]",
            border_style="red",
        ))

    # Arbiter verdicts
    if result.arbiter_reviews:
        table = Table(title="Arbiter Verdicts", show_lines=True)
        table.add_column("Stage", style="cyan")
        table.add_column("Verdict")
        table.add_column("Arbiter Model", style="dim")
        table.add_column("Confidence", justify="right")
        table.add_column("Cost", justify="right")

        for review in result.arbiter_reviews:
            style = _verdict_style(review.verdict.value)
            table.add_row(
                review.stage_reviewed.value,
                Text(review.verdict.value.upper(), style=style),
                review.arbiter_model,
                f"{review.confidence:.2f}",
                f"${review.token_cost:.4f}",
            )
        console.print(table)
        console.print()

    # Routing decisions
    if result.routing_decisions:
        table = Table(title="Routing Decisions")
        table.add_column("Stage", style="cyan")
        table.add_column("Model", style="bold")
        table.add_column("Strategy", style="dim")
        table.add_column("Fitness", justify="right")
        table.add_column("Est. Cost", justify="right")

        for d in result.routing_decisions:
            table.add_row(
                d.role.value,
                d.model_key,
                d.strategy.value,
                f"{d.fitness_score:.2f}",
                f"${d.estimated_cost:.4f}",
            )
        console.print(table)
        console.print()

    # Summary metrics
    summary = Table.grid(padding=(0, 2))
    summary.add_row(
        "[bold]Total Cost:[/bold]", f"${result.total_cost:.4f}",
        "[bold]Total Tokens:[/bold]", f"{result.total_tokens:,}",
    )
    summary.add_row(
        "[bold]Duration:[/bold]", f"{result.duration_seconds:.1f}s",
        "[bold]Session:[/bold]", result.session_id or "n/a",
    )
    console.print(Panel(summary, title="[bold]Summary[/bold]"))


# ── triad plan ───────────────────────────────────────────────────

@app.command()
def plan() -> None:
    """Expand a rough idea into a structured task spec."""
    console.print(
        Panel(
            "[yellow]Coming in v0.1 — task planner[/yellow]\n\n"
            "Will expand a rough idea into a structured task spec\n"
            "with optional interactive clarifying questions.",
            title="[bold]triad plan[/bold]",
            border_style="yellow",
        )
    )


# ── triad estimate ───────────────────────────────────────────────

@app.command()
def estimate(
    task: str = typer.Argument(..., help="Task description for cost estimation"),
    mode: str = typer.Option(
        "sequential", "--mode", "-m",
        help="Pipeline mode: sequential, parallel, debate",
    ),
) -> None:
    """Show cost estimates across all routing strategies."""
    from triad.routing.engine import estimate_cost

    registry = _load_registry()
    base_config = _load_config()

    try:
        PipelineMode(mode)
    except ValueError:
        console.print(f"[red]Invalid mode:[/red] '{mode}'")
        raise typer.Exit(1) from None

    console.print(Panel(f"[bold]Cost Estimate:[/bold] {task}", border_style="blue"))
    console.print()

    strategies = [
        RoutingStrategy.QUALITY_FIRST,
        RoutingStrategy.COST_OPTIMIZED,
        RoutingStrategy.SPEED_FIRST,
        RoutingStrategy.HYBRID,
    ]

    # Summary comparison table
    summary_table = Table(title="Strategy Comparison")
    summary_table.add_column("Strategy", style="bold")
    summary_table.add_column("Architect", justify="right")
    summary_table.add_column("Implement", justify="right")
    summary_table.add_column("Refactor", justify="right")
    summary_table.add_column("Verify", justify="right")
    summary_table.add_column("Total", justify="right", style="bold green")

    for strategy in strategies:
        est = estimate_cost(base_config, registry, strategy)
        stage_costs = {d.role.value: d.estimated_cost for d in est.decisions}
        summary_table.add_row(
            strategy.value,
            f"${stage_costs.get('architect', 0):.4f}",
            f"${stage_costs.get('implement', 0):.4f}",
            f"${stage_costs.get('refactor', 0):.4f}",
            f"${stage_costs.get('verify', 0):.4f}",
            f"${est.total_estimated_cost:.4f}",
        )

    console.print(summary_table)
    console.print()

    # Model assignments per strategy
    detail_table = Table(title="Model Assignments")
    detail_table.add_column("Strategy", style="bold")
    detail_table.add_column("Architect")
    detail_table.add_column("Implement")
    detail_table.add_column("Refactor")
    detail_table.add_column("Verify")

    for strategy in strategies:
        est = estimate_cost(base_config, registry, strategy)
        models = {d.role.value: d.model_key for d in est.decisions}
        detail_table.add_row(
            strategy.value,
            models.get("architect", "?"),
            models.get("implement", "?"),
            models.get("refactor", "?"),
            models.get("verify", "?"),
        )

    console.print(detail_table)
    console.print()
    console.print("[dim]Estimates use conservative token projections. Actual costs may vary.[/dim]")


# ── triad models ─────────────────────────────────────────────────

@models_app.command("list")
def models_list() -> None:
    """Show all registered models as a table."""
    registry = _load_registry()

    table = Table(title="Registered Models", show_lines=True)
    table.add_column("Key", style="bold cyan")
    table.add_column("Display Name")
    table.add_column("Provider", style="dim")
    table.add_column("Context", justify="right")
    table.add_column("Input $/M", justify="right")
    table.add_column("Output $/M", justify="right")
    table.add_column("Arch", justify="right")
    table.add_column("Impl", justify="right")
    table.add_column("Refac", justify="right")
    table.add_column("Verif", justify="right")

    for key, cfg in sorted(registry.items()):
        table.add_row(
            key,
            cfg.display_name,
            cfg.provider,
            f"{cfg.context_window:,}",
            f"${cfg.cost_input:.2f}",
            f"${cfg.cost_output:.2f}",
            f"{cfg.fitness.architect:.2f}",
            f"{cfg.fitness.implementer:.2f}",
            f"{cfg.fitness.refactorer:.2f}",
            f"{cfg.fitness.verifier:.2f}",
        )

    console.print(table)
    console.print(f"\n[dim]{len(registry)} models registered[/dim]")


@models_app.command("show")
def models_show(
    key: str = typer.Argument(..., help="Model registry key"),
) -> None:
    """Show full details for one model."""
    registry = _load_registry()

    if key not in registry:
        console.print(f"[red]Model not found:[/red] '{key}'")
        console.print(f"[dim]Available: {', '.join(sorted(registry))}[/dim]")
        raise typer.Exit(1) from None

    cfg = registry[key]
    table = Table(title=f"Model: {key}", show_header=False, show_lines=True)
    table.add_column("Field", style="bold")
    table.add_column("Value")

    table.add_row("Display Name", cfg.display_name)
    table.add_row("Provider", cfg.provider)
    table.add_row("Model ID", cfg.model)
    table.add_row("API Key Env", cfg.api_key_env)
    if cfg.api_base:
        table.add_row("API Base", cfg.api_base)
    table.add_row("Context Window", f"{cfg.context_window:,}")
    table.add_row("Cost (Input)", f"${cfg.cost_input:.2f}/M tokens")
    table.add_row("Cost (Output)", f"${cfg.cost_output:.2f}/M tokens")

    caps = []
    if cfg.supports_tools:
        caps.append("tools")
    if cfg.supports_structured:
        caps.append("structured")
    if cfg.supports_vision:
        caps.append("vision")
    if cfg.supports_thinking:
        caps.append("thinking")
    table.add_row("Capabilities", ", ".join(caps) if caps else "none")

    table.add_row("Fitness: Architect", f"{cfg.fitness.architect:.2f}")
    table.add_row("Fitness: Implementer", f"{cfg.fitness.implementer:.2f}")
    table.add_row("Fitness: Refactorer", f"{cfg.fitness.refactorer:.2f}")
    table.add_row("Fitness: Verifier", f"{cfg.fitness.verifier:.2f}")

    # Check if API key is set
    api_key = os.environ.get(cfg.api_key_env, "")
    key_status = "[green]set[/green]" if api_key else "[red]not set[/red]"
    table.add_row("API Key Status", key_status)

    console.print(table)


@models_app.command("test")
def models_test(
    key: str = typer.Argument(..., help="Model registry key to test"),
) -> None:
    """Send a test prompt to verify a model is reachable."""
    registry = _load_registry()

    if key not in registry:
        console.print(f"[red]Model not found:[/red] '{key}'")
        raise typer.Exit(1) from None

    cfg = registry[key]
    api_key = os.environ.get(cfg.api_key_env, "")
    if not api_key:
        console.print(
            f"[red]API key not set:[/red] {cfg.api_key_env}\n"
            f"Set it with: export {cfg.api_key_env}=your-key"
        )
        raise typer.Exit(1) from None

    console.print(f"Testing [bold]{cfg.display_name}[/bold] ({cfg.model})...")

    from triad.providers.litellm_provider import LiteLLMProvider

    provider = LiteLLMProvider(cfg)

    async def _test():
        return await provider.complete(
            messages=[{"role": "user", "content": "Reply with: OK"}],
            system="You are a test probe. Reply with exactly 'OK'.",
            timeout=30,
        )

    try:
        with console.status("[bold blue]Sending test prompt...", spinner="dots"):
            msg = asyncio.run(_test())
        console.print(f"[green]Success![/green] Response: {msg.content[:100]}")
        if msg.token_usage:
            console.print(
                f"[dim]Tokens: {msg.token_usage.prompt_tokens}+"
                f"{msg.token_usage.completion_tokens} "
                f"Cost: ${msg.token_usage.cost:.4f}[/dim]"
            )
    except Exception as e:
        console.print(f"[red]Failed:[/red] {e}")
        raise typer.Exit(1) from None


# ── triad config ─────────────────────────────────────────────────

@config_app.command("show")
def config_show() -> None:
    """Show current pipeline configuration."""
    config = _load_config()

    table = Table(title="Pipeline Configuration", show_header=False, show_lines=True)
    table.add_column("Setting", style="bold")
    table.add_column("Value")

    table.add_row("Arbiter Mode", config.arbiter_mode.value)
    table.add_row("Reconciliation", str(config.reconciliation_enabled))
    table.add_row("Default Timeout", f"{config.default_timeout}s")
    table.add_row("Max Retries", str(config.max_retries))
    table.add_row("Reconciliation Retries", str(config.reconciliation_retries))
    table.add_row("Routing Strategy", config.routing_strategy.value)
    table.add_row("Min Fitness", f"{config.min_fitness:.2f}")
    table.add_row("Persist Sessions", str(config.persist_sessions))
    table.add_row("Session DB Path", config.session_db_path)
    if config.arbiter_model:
        table.add_row("Arbiter Model Override", config.arbiter_model)
    if config.reconcile_model:
        table.add_row("Reconcile Model Override", config.reconcile_model)

    console.print(table)

    if config.stages:
        stage_table = Table(title="Stage Overrides")
        stage_table.add_column("Stage", style="cyan")
        stage_table.add_column("Model")
        stage_table.add_column("Timeout")
        stage_table.add_column("Max Retries")

        for stage, scfg in config.stages.items():
            stage_table.add_row(
                stage.value,
                scfg.model or "(auto)",
                f"{scfg.timeout}s",
                str(scfg.max_retries),
            )
        console.print()
        console.print(stage_table)


@config_app.command("path")
def config_path() -> None:
    """Show configuration file locations."""
    config_dir = Path(__file__).parent / "config"
    files = [
        ("Models", config_dir / "models.toml"),
        ("Defaults", config_dir / "defaults.toml"),
        ("Routing", config_dir / "routing.toml"),
        ("Domain Rules", config_dir / "domain"),
    ]

    table = Table(title="Configuration Paths", show_header=False)
    table.add_column("Config", style="bold")
    table.add_column("Path")
    table.add_column("Status")

    for name, path in files:
        exists = path.exists()
        status = "[green]found[/green]" if exists else "[red]missing[/red]"
        table.add_row(name, str(path), status)

    console.print(table)


# ── triad sessions ───────────────────────────────────────────────

@sessions_app.command("list")
def sessions_list(
    limit: int = typer.Option(20, "--limit", "-n", help="Max sessions to show"),
    task_filter: str = typer.Option(None, "--task", help="Filter by task text"),
    model_filter: str = typer.Option(None, "--model", help="Filter by model key"),
    verdict_filter: str = typer.Option(None, "--verdict", help="Filter by verdict"),
    since: str = typer.Option(None, "--since", help="Filter sessions after date"),
) -> None:
    """Show recent sessions."""
    from triad.persistence.database import close_db, init_db
    from triad.persistence.session import SessionStore
    from triad.schemas.session import SessionQuery

    config = _load_config()

    query = SessionQuery(
        limit=limit,
        task_filter=task_filter,
        model_filter=model_filter,
        verdict_filter=verdict_filter,
        since=since,
    )

    async def _list():
        db = await init_db(config.session_db_path)
        store = SessionStore(db)
        summaries = await store.list_sessions(query)
        await close_db(db)
        return summaries

    summaries = asyncio.run(_list())

    if not summaries:
        console.print("[dim]No sessions found.[/dim]")
        return

    table = Table(title=f"Sessions ({len(summaries)} shown)")
    table.add_column("Session ID", style="cyan", max_width=12)
    table.add_column("Task", max_width=40)
    table.add_column("Mode", style="dim")
    table.add_column("Status")
    table.add_column("Cost", justify="right")
    table.add_column("Duration", justify="right")
    table.add_column("Verdicts", style="dim")

    for s in summaries:
        if s.halted:
            status = Text("HALTED", style="bright_red")
        elif s.success:
            status = Text("OK", style="green")
        else:
            status = Text("FAIL", style="red")

        table.add_row(
            s.session_id[:12] + "...",
            s.task_preview[:40],
            s.pipeline_mode,
            status,
            f"${s.total_cost:.4f}",
            f"{s.duration_seconds:.1f}s",
            s.arbiter_verdict_summary or "-",
        )

    console.print(table)


@sessions_app.command("show")
def sessions_show(
    session_id: str = typer.Argument(..., help="Session ID to show"),
) -> None:
    """Show full session details."""
    from triad.persistence.database import close_db, init_db
    from triad.persistence.session import SessionStore

    config = _load_config()

    async def _get():
        db = await init_db(config.session_db_path)
        store = SessionStore(db)
        record = await store.get_session(session_id)
        await close_db(db)
        return record

    record = asyncio.run(_get())

    if not record:
        console.print(f"[red]Session not found:[/red] {session_id}")
        raise typer.Exit(1) from None

    # Metadata
    meta = Table(title=f"Session: {record.session_id}", show_header=False, show_lines=True)
    meta.add_column("Field", style="bold")
    meta.add_column("Value")
    meta.add_row("Task", record.task.task)
    meta.add_row("Mode", record.pipeline_mode)
    meta.add_row("Started", record.started_at.isoformat())
    if record.completed_at:
        meta.add_row("Completed", record.completed_at.isoformat())
    meta.add_row("Duration", f"{record.duration_seconds:.1f}s")
    meta.add_row("Cost", f"${record.total_cost:.4f}")
    meta.add_row("Tokens", f"{record.total_tokens:,}")
    status = "HALTED" if record.halted else ("Success" if record.success else "Failed")
    meta.add_row("Status", status)
    if record.halt_reason:
        meta.add_row("Halt Reason", record.halt_reason)
    console.print(meta)

    # Stages
    if record.stages:
        console.print()
        stages_table = Table(title="Stages")
        stages_table.add_column("Stage", style="cyan")
        stages_table.add_column("Model")
        stages_table.add_column("Confidence", justify="right")
        stages_table.add_column("Cost", justify="right")
        stages_table.add_column("Tokens", justify="right")
        for s in record.stages:
            stages_table.add_row(
                s.stage,
                s.model_key,
                f"{s.confidence:.2f}",
                f"${s.cost:.4f}",
                f"{s.tokens:,}",
            )
        console.print(stages_table)

    # Arbiter reviews
    if record.arbiter_reviews:
        console.print()
        review_table = Table(title="Arbiter Reviews", show_lines=True)
        review_table.add_column("Stage", style="cyan")
        review_table.add_column("Verdict")
        review_table.add_column("Arbiter")
        review_table.add_column("Confidence", justify="right")
        review_table.add_column("Issues", justify="right")
        for r in record.arbiter_reviews:
            style = _verdict_style(r.verdict.value)
            review_table.add_row(
                r.stage_reviewed.value,
                Text(r.verdict.value.upper(), style=style),
                r.arbiter_model,
                f"{r.confidence:.2f}",
                str(len(r.issues)),
            )
        console.print(review_table)


@sessions_app.command("export")
def sessions_export(
    session_id: str = typer.Argument(..., help="Session ID to export"),
    fmt: str = typer.Option(
        "markdown", "--format", "-f",
        help="Export format: json or markdown",
    ),
) -> None:
    """Export a session as JSON or Markdown."""
    from triad.persistence.database import close_db, init_db
    from triad.persistence.export import export_json, export_markdown
    from triad.persistence.session import SessionStore

    config = _load_config()

    async def _get():
        db = await init_db(config.session_db_path)
        store = SessionStore(db)
        record = await store.get_session(session_id)
        await close_db(db)
        return record

    record = asyncio.run(_get())

    if not record:
        console.print(f"[red]Session not found:[/red] {session_id}")
        raise typer.Exit(1) from None

    if fmt == "json":
        console.print(export_json(record))
    elif fmt == "markdown":
        console.print(export_markdown(record))
    else:
        console.print(f"[red]Invalid format:[/red] '{fmt}'. Choose json or markdown.")
        raise typer.Exit(1) from None


@sessions_app.command("delete")
def sessions_delete(
    session_id: str = typer.Argument(..., help="Session ID to delete"),
    yes: bool = typer.Option(
        False, "--yes", "-y",
        help="Skip confirmation prompt",
    ),
) -> None:
    """Delete a session from history."""
    if not yes:
        confirm = typer.confirm(
            f"Delete session {session_id}? This cannot be undone."
        )
        if not confirm:
            console.print("[dim]Cancelled.[/dim]")
            return

    from triad.persistence.database import close_db, init_db
    from triad.persistence.session import SessionStore

    config = _load_config()

    async def _delete():
        db = await init_db(config.session_db_path)
        store = SessionStore(db)
        deleted = await store.delete_session(session_id)
        await close_db(db)
        return deleted

    deleted = asyncio.run(_delete())

    if deleted:
        console.print(f"[green]Session deleted:[/green] {session_id}")
    else:
        console.print(f"[red]Session not found:[/red] {session_id}")
        raise typer.Exit(1) from None


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    app()

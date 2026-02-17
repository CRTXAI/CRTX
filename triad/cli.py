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

from triad.keys import load_keys_env
from triad.providers.registry import load_models, load_pipeline_config
from triad.schemas.pipeline import (
    ArbiterMode,
    PipelineConfig,
    PipelineMode,
    TaskSpec,
)
from triad.schemas.routing import RoutingStrategy

# Load API keys from ~/.triad/keys.env and .env on startup
load_keys_env()

console = Console()

# ── App and sub-apps ─────────────────────────────────────────────

app = typer.Typer(
    name="triad",
    help="Multi-model AI orchestration with adversarial Arbiter review.",
    no_args_is_help=False,
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


# ── App Callback ────────────────────────────────────────────────


@app.callback(invoke_without_command=True)
def main(ctx: typer.Context) -> None:
    """Multi-model AI orchestration with adversarial Arbiter review."""
    if ctx.invoked_subcommand is None:
        from triad.cli_display import render_full_logo
        from triad.repl import TriadREPL

        render_full_logo(console)
        TriadREPL().run()


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


# ── triad setup ──────────────────────────────────────────────────


@app.command()
def setup(
    check: bool = typer.Option(
        False, "--check",
        help="Validate existing keys without prompting",
    ),
    reset: bool = typer.Option(
        False, "--reset",
        help="Clear saved keys and re-run setup",
    ),
) -> None:
    """Walk through interactive API key configuration.

    First-time setup for new users. Prompts for provider API keys,
    validates them, and saves to ~/.triad/keys.env.
    """
    from rich.prompt import Prompt

    from triad.keys import (
        KEYS_FILE,
        PROVIDER_NAMES,
        PROVIDERS,
        clear_keys,
        save_keys,
        validate_key,
    )

    # --check: just validate existing keys
    if check:
        _setup_check()
        return

    # --reset: clear and re-run
    if reset:
        removed = clear_keys()
        if removed:
            console.print(f"[dim]Cleared {KEYS_FILE}[/dim]\n")
        else:
            console.print("[dim]No saved keys to clear.[/dim]\n")
        # Clear env vars that came from the file so they're re-prompted
        for env_var, _, _, _ in PROVIDERS:
            if env_var in os.environ:
                del os.environ[env_var]

    # Banner
    console.print()
    banner = Text()
    banner.append("     ◆\n", style="#00ffbb")
    banner.append("    ╱", style="#00ff66")
    banner.append("◈", style="#D4A843")
    banner.append("╲", style="#00ff66")
    banner.append("    Triad Orchestrator v0.1.0\n", style="#00ff88")
    banner.append("   ◆", style="#00ffbb")
    banner.append("──", style="#00ff66")
    banner.append("◆", style="#00ffbb")
    banner.append("   First-time setup\n", style="#6a8a6a")
    console.print(banner)

    console.print(
        "  Triad needs API keys to call AI models. You only need [bold]ONE[/bold] provider\n"
        "  to get started, but more providers = better model selection.\n"
    )
    console.print("  [bold]── Provider Setup ────────────────────────────────────────────[/bold]\n")

    # Collect keys
    collected_keys: dict[str, str] = {}

    for i, (env_var, display_name, description, signup_url) in enumerate(PROVIDERS, 1):
        # Check if already set in environment
        existing = os.environ.get(env_var, "")

        console.print(f"  [bold][{i}] {display_name}[/bold]     — {description}")
        console.print(f"      Get a key: [link={signup_url}]{signup_url}[/link]")

        if existing:
            console.print(f"      {env_var}: [green]already set in environment[/green]\n")
            collected_keys[env_var] = existing
            continue

        key = Prompt.ask(
            f"      {env_var}",
            default="",
            show_default=False,
            password=True,
            console=console,
        )

        if key.strip():
            collected_keys[env_var] = key.strip()
            console.print("      [green]✓ Set[/green]\n")
        else:
            console.print("      [dim]⊘ Skipped[/dim]\n")

    # Check at least one key provided
    if not collected_keys:
        console.print("  [red]No API keys provided.[/red] You need at least one provider.")
        console.print("  Run [bold]triad setup[/bold] again when you have a key.\n")
        raise typer.Exit(1) from None

    # Validate keys
    console.print("  [bold]── Validating Keys ───────────────────────────────────────────[/bold]\n")

    validated_keys: dict[str, str] = {}
    active_count = 0

    async def _validate_all():
        results = {}
        for env_var in [e for e, _, _, _ in PROVIDERS]:
            key_val = collected_keys.get(env_var, "")
            if key_val:
                ok, detail = await validate_key(env_var, key_val)
                results[env_var] = (ok, detail)
            else:
                results[env_var] = (None, "Skipped")
        return results

    with console.status("[bold blue]Validating keys...", spinner="dots"):
        results = asyncio.run(_validate_all())

    for env_var, _, _, _ in PROVIDERS:
        name = PROVIDER_NAMES[env_var]
        ok, detail = results[env_var]
        if ok is True:
            console.print(f"  {name:<10} [green]✓ {detail}[/green]")
            validated_keys[env_var] = collected_keys[env_var]
            active_count += 1
        elif ok is False:
            console.print(f"  {name:<10} [red]✗ {detail}[/red]")
            # Still save the key — user might fix the issue later
            validated_keys[env_var] = collected_keys[env_var]
        else:
            console.print(f"  {name:<10} [dim]⊘ Skipped[/dim]")

    if active_count == 0:
        console.print(
            "\n  [yellow]Warning:[/yellow] No keys validated successfully. "
            "Keys are saved but may not work."
        )

    # Save keys
    saved_path = save_keys(validated_keys)

    # Also set them in the current environment
    for env_var, value in validated_keys.items():
        if value:
            os.environ[env_var] = value

    # Count available models
    try:
        registry = load_models()
        available_providers = {
            cfg.api_key_env for cfg in registry.values()
            if os.environ.get(cfg.api_key_env)
        }
        model_count = sum(
            1 for cfg in registry.values()
            if cfg.api_key_env in available_providers and os.environ.get(cfg.api_key_env)
        )
    except Exception:
        model_count = 0

    console.print()
    console.print("  [bold]── Configuration Saved ───────────────────────────────────────[/bold]\n")
    console.print(f"  Keys saved to: [bold]{saved_path}[/bold]")
    console.print(
        f"  {active_count} provider{'s' if active_count != 1 else ''} active"
        + (f", {model_count} models available" if model_count else "")
    )
    console.print()
    console.print(
        "  Run [bold]triad[/bold] to start, or "
        "[bold]triad run \"your task\"[/bold] for a quick test.\n"
    )


def _setup_check() -> None:
    """Validate existing keys without prompting (--check flag)."""
    from triad.keys import PROVIDER_NAMES, PROVIDERS, get_configured_keys, validate_key

    keys = get_configured_keys()

    has_any = any(keys.values())
    if not has_any:
        console.print(
            "[dim]No API keys configured.[/dim] "
            "Run [bold]triad setup[/bold] to get started."
        )
        raise typer.Exit(1) from None

    async def _validate_all():
        results = {}
        for env_var, _, _, _ in PROVIDERS:
            key_val = keys.get(env_var, "")
            if key_val:
                ok, detail = await validate_key(env_var, key_val)
                results[env_var] = (ok, detail)
            else:
                results[env_var] = (None, "Not configured")
        return results

    with console.status("[bold blue]Validating keys...", spinner="dots"):
        results = asyncio.run(_validate_all())

    any_invalid = False
    for env_var, _, _, _ in PROVIDERS:
        name = PROVIDER_NAMES[env_var]
        ok, detail = results[env_var]
        if ok is True:
            console.print(f"{name:<10} [green]✓ {detail}[/green]")
        elif ok is False:
            console.print(f"{name:<10} [red]✗ {detail}[/red]")
            any_invalid = True
        else:
            console.print(f"{name:<10} [dim]⊘ {detail}[/dim]")

    if any_invalid:
        raise typer.Exit(1) from None


# ── triad run ────────────────────────────────────────────────────

@app.command()
def run(
    task: str = typer.Argument(..., help="Task description — what to build"),
    mode: str = typer.Option(
        None, "--mode", "-m",
        help="Pipeline mode: sequential, parallel, or debate",
    ),
    route: str = typer.Option(
        None, "--route", "-r",
        help="Routing strategy: quality_first, cost_optimized, speed_first, hybrid",
    ),
    arbiter: str = typer.Option(
        None, "--arbiter", "-a",
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
        300, "--timeout", "-t",
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
    context_dir: str = typer.Option(
        None, "--context-dir",
        help="Project directory to scan for context injection",
    ),
    include: list[str] = typer.Option(
        None, "--include",
        help="Glob patterns for files to include (default: *.py)",
    ),
    exclude: list[str] = typer.Option(
        None, "--exclude",
        help="Glob patterns for files to exclude",
    ),
    context_budget: int = typer.Option(
        8000, "--context-budget",
        help="Max tokens for injected project context",
    ),
) -> None:
    """Execute a pipeline run with the specified task and options."""
    from triad.keys import has_any_key

    if not has_any_key():
        console.print(
            "[red]No API keys configured.[/red] "
            "Run [bold]triad setup[/bold] to get started."
        )
        raise typer.Exit(1) from None

    from triad.cli_display import (
        CompletionSummary,
        ConfigScreen,
        PipelineDisplay,
    )

    registry = _load_registry()
    base_config = _load_config()

    # Detect whether flags were explicitly set
    explicit_flags = mode is not None or route is not None or arbiter is not None

    # Load defaults from config for unset flags
    if mode is None:
        mode = base_config.pipeline_mode.value
    if route is None:
        route = base_config.routing_strategy.value
    if arbiter is None:
        arbiter = base_config.arbiter_mode.value

    # Interactive config screen when no explicit flags and running interactively
    from triad.cli_display import is_interactive as _is_interactive

    if not explicit_flags and _is_interactive():
        config_screen = ConfigScreen(task, base_config, registry)
        result = config_screen.show(console)
        if result is None:
            raise typer.Exit(0)
        mode, route, arbiter = result

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
        context_dir=context_dir,
        context_include=include if include else ["*.py"],
        context_exclude=exclude if exclude else [],
        context_token_budget=context_budget,
    )

    task_spec = TaskSpec(
        task=task,
        context=context,
        domain_rules=domain_context,
        output_dir=output_dir,
    )

    # Run pipeline
    from triad.cli_display import is_interactive
    from triad.dashboard.events import PipelineEventEmitter
    from triad.orchestrator import run_pipeline

    interactive = is_interactive()
    emitter = PipelineEventEmitter()

    if interactive:
        display = PipelineDisplay(console, mode, route, arbiter)
        emitter.add_listener(display.create_listener())
        with display:
            pipeline_result = asyncio.run(
                run_pipeline(task_spec, config, registry, emitter),
            )
    else:
        # Non-interactive: show static task panel + spinner
        _display_task_panel(task_spec, config)
        console.print()
        with console.status("[bold blue]Running pipeline...", spinner="dots"):
            pipeline_result = asyncio.run(
                run_pipeline(task_spec, config, registry, emitter),
            )

    # Display results
    console.print()
    _display_result(pipeline_result)

    # Write output files
    from triad.output.writer import write_pipeline_output

    write_pipeline_output(pipeline_result, output_dir)
    console.print(f"\n[dim]Output written to:[/dim] {output_dir}/")

    # Interactive completion summary (only in real terminals)
    if interactive:
        summary = CompletionSummary(console, pipeline_result, output_dir)
        action = summary.show()
        if action == "rerun":
            run(
                task=task, mode=mode, route=route, arbiter=arbiter,
                reconcile=reconcile, context=context, domain_rules=domain_rules,
                timeout=timeout, max_retries=max_retries, no_persist=no_persist,
                output_dir=output_dir, context_dir=context_dir, include=include,
                exclude=exclude, context_budget=context_budget,
            )


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
def plan(
    description: str = typer.Argument(..., help="Rough task description to expand"),
    interactive: bool = typer.Option(
        False, "--interactive", "-i",
        help="Ask clarifying questions before expanding",
    ),
    run_immediately: bool = typer.Option(
        False, "--run",
        help="Pipe expanded spec directly to pipeline (skip confirmation)",
    ),
    save: str = typer.Option(
        None, "--save", "-s",
        help="Save expanded spec to file",
    ),
    edit: bool = typer.Option(
        False, "--edit", "-e",
        help="Open expanded spec in $EDITOR before running",
    ),
    model: str = typer.Option(
        None, "--model",
        help="Override planner model selection",
    ),
    mode: str = typer.Option(
        "sequential", "--mode", "-m",
        help="Pipeline mode if --run is used",
    ),
    route: str = typer.Option(
        "hybrid", "--route", "-r",
        help="Routing strategy if --run is used",
    ),
) -> None:
    """Expand a rough idea into a structured task spec."""
    from triad.planner import TaskPlanner

    if not description.strip():
        console.print("[red]Error:[/red] Description cannot be empty.")
        raise typer.Exit(1) from None

    registry = _load_registry()

    planner = TaskPlanner(registry)

    try:
        selected_model = planner.select_model(model)
    except RuntimeError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1) from None

    model_display = registry[selected_model].display_name

    console.print(Panel(
        f"[bold]Description:[/bold] {description}\n"
        f"[bold]Model:[/bold] {model_display} ({selected_model})\n"
        f"[bold]Mode:[/bold] {'interactive' if interactive else 'quick'}",
        title="[bold blue]Task Planner[/bold blue]",
        border_style="blue",
    ))

    user_answers = None

    if interactive:
        # Phase 1: Ask clarifying questions
        console.print()
        with console.status("[bold blue]Generating questions...", spinner="dots"):
            phase1 = asyncio.run(
                planner.plan(description, interactive=True, model_override=model)
            )

        if phase1.clarifying_questions:
            console.print(Panel(
                "\n".join(phase1.clarifying_questions),
                title="[bold yellow]Clarifying Questions[/bold yellow]",
                border_style="yellow",
            ))
            console.print()
            console.print("[bold]Answer each question below (press Enter to skip):[/bold]")

            answers: list[str] = []
            for i, question in enumerate(phase1.clarifying_questions, 1):
                answer = typer.prompt(f"  Q{i}", default="", show_default=False)
                if answer:
                    answers.append(f"Q{i}: {question}\nA{i}: {answer}")

            user_answers = "\n\n".join(answers) if answers else "No answers provided."

            console.print(
                f"\n[dim]Phase 1 cost: ${phase1.cost:.4f} "
                f"({phase1.token_usage.prompt_tokens}+"
                f"{phase1.token_usage.completion_tokens} tokens)[/dim]"
            )
        else:
            user_answers = "No specific preferences."

    # Phase 2 (or quick mode): Expand the spec
    console.print()
    with console.status("[bold blue]Expanding task specification...", spinner="dots"):
        result = asyncio.run(
            planner.plan(
                description,
                interactive=interactive,
                user_answers=user_answers,
                model_override=model,
            )
        )

    # Display the expanded spec
    console.print()
    _display_plan_result(result)

    # Save to file
    if save:
        save_path = Path(save)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        save_path.write_text(result.expanded_spec, encoding="utf-8")
        console.print(f"\n[green]Spec saved to:[/green] {save}")

    # Edit in $EDITOR
    if edit:
        import tempfile

        with tempfile.NamedTemporaryFile(
            mode="w", suffix=".md", delete=False, encoding="utf-8",
        ) as tmp:
            tmp.write(result.expanded_spec)
            tmp_path = tmp.name

        editor = os.environ.get("EDITOR", os.environ.get("VISUAL", ""))
        if not editor:
            console.print(
                "[yellow]Warning:[/yellow] $EDITOR not set. "
                "Showing spec as-is."
            )
        else:
            import subprocess

            subprocess.run([editor, tmp_path], check=False)
            edited = Path(tmp_path).read_text(encoding="utf-8")
            result.expanded_spec = edited
            result.task_spec = TaskSpec(task=edited)

    # Run pipeline or prompt
    if run_immediately:
        _run_from_plan(result, registry, mode, route)
    elif not save:
        console.print()
        choice = typer.prompt(
            "[r]un pipeline  [s]ave to file  [q]uit",
            default="q",
            show_default=False,
        )
        if choice.lower() == "r":
            _run_from_plan(result, registry, mode, route)
        elif choice.lower() == "s":
            out_path = typer.prompt("Save path", default="task-spec.md")
            Path(out_path).write_text(
                result.expanded_spec, encoding="utf-8",
            )
            console.print(f"[green]Saved to:[/green] {out_path}")


def _display_plan_result(result) -> None:
    """Display the expanded spec with tech stack highlights."""
    from rich.markdown import Markdown

    console.print(Panel(
        Markdown(result.expanded_spec),
        title="[bold green]Generated Task Spec[/bold green]",
        border_style="green",
    ))

    if result.tech_stack_inferred:
        tech_text = "  ".join(
            f"[bold cyan]{t}[/bold cyan]" for t in result.tech_stack_inferred
        )
        console.print(f"\n[bold]Inferred Tech Stack:[/bold]  {tech_text}")

    console.print(
        f"\n[dim]Cost: ${result.cost:.4f} "
        f"({result.token_usage.prompt_tokens}+"
        f"{result.token_usage.completion_tokens} tokens) "
        f"Model: {result.model_used}[/dim]"
    )


def _run_from_plan(result, registry, mode_str: str, route_str: str) -> None:
    """Execute the pipeline using the planner's expanded spec."""
    try:
        pipeline_mode = PipelineMode(mode_str)
    except ValueError:
        console.print(f"[red]Invalid mode:[/red] '{mode_str}'")
        raise typer.Exit(1) from None

    try:
        routing_strategy = RoutingStrategy(route_str)
    except ValueError:
        console.print(f"[red]Invalid routing strategy:[/red] '{route_str}'")
        raise typer.Exit(1) from None

    base_config = _load_config()

    config = PipelineConfig(
        pipeline_mode=pipeline_mode,
        routing_strategy=routing_strategy,
        arbiter_mode=base_config.arbiter_mode,
        reconciliation_enabled=base_config.reconciliation_enabled,
        default_timeout=base_config.default_timeout,
        max_retries=base_config.max_retries,
        reconciliation_retries=base_config.reconciliation_retries,
        stages=base_config.stages,
        arbiter_model=base_config.arbiter_model,
        reconcile_model=base_config.reconcile_model,
        min_fitness=base_config.min_fitness,
        persist_sessions=base_config.persist_sessions,
        session_db_path=base_config.session_db_path,
    )

    task_spec = result.task_spec

    from triad.cli_display import PipelineDisplay, is_interactive
    from triad.dashboard.events import PipelineEventEmitter
    from triad.orchestrator import run_pipeline

    emitter = PipelineEventEmitter()

    if is_interactive():
        display = PipelineDisplay(
            console, mode_str, route_str, base_config.arbiter_mode.value,
        )
        emitter.add_listener(display.create_listener())
        with display:
            pipeline_result = asyncio.run(
                run_pipeline(task_spec, config, registry, emitter),
            )
    else:
        console.print()
        _display_task_panel(task_spec, config)
        console.print()
        with console.status("[bold blue]Running pipeline...", spinner="dots"):
            pipeline_result = asyncio.run(
                run_pipeline(task_spec, config, registry, emitter),
            )

    console.print()
    _display_result(pipeline_result)

    from triad.output.writer import write_pipeline_output

    output_dir = task_spec.output_dir or "triad-output"
    write_pipeline_output(pipeline_result, output_dir)
    console.print(f"\n[dim]Output written to:[/dim] {output_dir}/")


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


# ── triad review ─────────────────────────────────────────────────

@app.command()
def review(
    diff_file: str = typer.Option(
        None, "--diff", "-d",
        help="Path to a file containing a unified diff",
    ),
    ref: str = typer.Option(
        None, "--ref",
        help="Git ref range for diff (e.g. origin/main..HEAD)",
    ),
    stdin: bool = typer.Option(
        False, "--stdin",
        help="Read diff from stdin",
    ),
    focus: str = typer.Option(
        None, "--focus",
        help="Comma-separated focus areas (e.g. security,performance)",
    ),
    no_arbiter: bool = typer.Option(
        False, "--no-arbiter",
        help="Skip cross-validation of findings",
    ),
    fmt: str = typer.Option(
        "summary", "--format", "-f",
        help="Output format: summary, comments, or json",
    ),
    fail_on: str = typer.Option(
        "critical", "--fail-on",
        help="Exit with error on: critical, warning, or any",
    ),
) -> None:
    """Run multi-model parallel code review on a diff."""
    import sys

    from triad.ci.formatter import format_exit_code, format_summary
    from triad.ci.reviewer import ReviewRunner
    from triad.schemas.ci import ReviewConfig

    # Get the diff text
    diff_text = ""
    if diff_file:
        diff_path = Path(diff_file)
        if not diff_path.exists():
            console.print(f"[red]Diff file not found:[/red] {diff_file}")
            raise typer.Exit(1) from None
        diff_text = diff_path.read_text(encoding="utf-8")
    elif ref:
        import subprocess

        try:
            result = subprocess.run(
                ["git", "diff", ref],
                capture_output=True, text=True, check=True,
            )
            diff_text = result.stdout
        except (subprocess.CalledProcessError, FileNotFoundError) as e:
            console.print(f"[red]Failed to generate diff:[/red] {e}")
            raise typer.Exit(1) from None
    elif stdin:
        diff_text = sys.stdin.read()
    else:
        console.print(
            "[red]Error:[/red] Provide one of --diff, --ref, or --stdin"
        )
        raise typer.Exit(1) from None

    if not diff_text.strip():
        console.print("[dim]Empty diff — nothing to review.[/dim]")
        raise typer.Exit(0) from None

    registry = _load_registry()

    focus_areas = [a.strip() for a in focus.split(",")] if focus else []

    config = ReviewConfig(
        focus_areas=focus_areas,
        arbiter_enabled=not no_arbiter,
    )

    runner = ReviewRunner(registry)

    console.print(Panel(
        f"[bold]Models:[/bold] {len(registry)}\n"
        f"[bold]Focus:[/bold] {', '.join(focus_areas) if focus_areas else 'all'}\n"
        f"[bold]Arbiter:[/bold] {'enabled' if not no_arbiter else 'disabled'}",
        title="[bold blue]Triad Review[/bold blue]",
        border_style="blue",
    ))

    with console.status("[bold blue]Running parallel review...", spinner="dots"):
        review_result = asyncio.run(runner.review(diff_text, config))

    if fmt == "json":
        import json

        console.print(json.dumps(review_result.model_dump(), indent=2))
    elif fmt == "comments":
        from triad.ci.formatter import format_github_comments

        comments = format_github_comments(review_result)
        import json

        console.print(json.dumps(comments, indent=2))
    else:
        # summary (default)
        summary = format_summary(review_result)
        from rich.markdown import Markdown

        console.print(Markdown(summary))

        # Rich table for findings
        if review_result.findings:
            console.print()
            table = Table(title="Findings", show_lines=True)
            table.add_column("Severity")
            table.add_column("File", style="cyan")
            table.add_column("Line", justify="right")
            table.add_column("Description")
            table.add_column("Reporters", style="dim")

            severity_style = {
                "critical": "bold red",
                "warning": "bold yellow",
                "suggestion": "dim",
            }

            for f in review_result.findings:
                style = severity_style.get(f.severity, "white")
                confirmed = " ✓" if f.confirmed else ""
                table.add_row(
                    Text(f.severity.upper(), style=style),
                    f.file,
                    str(f.line) if f.line else "-",
                    f.description[:80],
                    ", ".join(f.reported_by) + confirmed,
                )
            console.print(table)

        # Summary metrics
        console.print()
        metrics = Table.grid(padding=(0, 2))
        metrics.add_row(
            "[bold]Consensus:[/bold]",
            review_result.consensus_recommendation.upper(),
            "[bold]Cost:[/bold]",
            f"${review_result.total_cost:.4f}",
        )
        metrics.add_row(
            "[bold]Findings:[/bold]",
            str(review_result.total_findings),
            "[bold]Duration:[/bold]",
            f"{review_result.duration_seconds:.1f}s",
        )
        console.print(Panel(metrics, title="[bold]Review Summary[/bold]"))

    # Exit code
    exit_code = format_exit_code(review_result, fail_on)
    if exit_code != 0:
        raise typer.Exit(exit_code) from None


# ── Dashboard ────────────────────────────────────────────────────


@app.command()
def dashboard(
    port: int = typer.Option(
        8420, "--port", "-p",
        help="Port to serve the dashboard on",
    ),
    no_browser: bool = typer.Option(
        False, "--no-browser",
        help="Don't auto-open browser",
    ),
) -> None:
    """Start the real-time pipeline dashboard server.

    Opens a browser with a live visualization of pipeline runs.
    Run `triad run --task "..."` in another terminal to see agents work.

    Requires: pip install triad-orchestrator[dashboard]
    """
    try:
        from triad.dashboard.server import create_app  # noqa: F811
    except ImportError:
        console.print(
            "[red]Dashboard requires extra dependencies.[/red]\n"
            "Install with: [bold]pip install triad-orchestrator\\[dashboard][/bold]"
        )
        raise typer.Exit(1) from None

    console.print(Panel(
        f"[bold]URL:[/bold] http://localhost:{port}\n"
        f"[bold]Auto-open:[/bold] {'no' if no_browser else 'yes'}",
        title="[bold blue]Triad Dashboard[/bold blue]",
        border_style="blue",
    ))
    console.print(
        "[dim]Run [bold]triad run --task \"...\"[/bold] in another terminal "
        "to see real-time pipeline events.[/dim]"
    )

    if not no_browser:
        import threading
        import webbrowser
        threading.Timer(1.5, lambda: webbrowser.open(f"http://localhost:{port}")).start()

    import uvicorn

    app_instance = create_app()
    uvicorn.run(app_instance, host="0.0.0.0", port=port, log_level="warning")


# ── Entry point ──────────────────────────────────────────────────

if __name__ == "__main__":
    app()

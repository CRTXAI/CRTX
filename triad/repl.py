"""Interactive REPL for the Triad Orchestrator.

Provides a persistent interactive session with command dispatch,
session state management, and task execution. Launch with `triad`
(no subcommand).
"""

from __future__ import annotations

import asyncio
import logging
import os
import shlex

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triad.cli_display import BRAND

logger = logging.getLogger(__name__)

console = Console()

# Commands that are dispatched to CLI subcommands
_CLI_COMMANDS = {
    "plan", "models", "estimate", "sessions", "config", "review", "dashboard",
    "setup",
}

# Session state commands
_STATE_COMMANDS = {"mode", "route", "arbiter"}

# All recognized first-words — anything matching these is NOT a task
_KNOWN_COMMANDS = (
    _CLI_COMMANDS | _STATE_COMMANDS
    | {"help", "status", "exit", "quit", "run", "show"}
)

# Valid values for session state
_VALID_MODES = {"sequential", "parallel", "debate"}
_VALID_ROUTES = {"hybrid", "quality_first", "cost_optimized", "speed_first"}
_VALID_ARBITERS = {"off", "final_only", "bookend", "full"}


class TriadREPL:
    """Interactive REPL loop for the Triad Orchestrator.

    Manages session state (mode, route, arbiter defaults) and
    dispatches commands or task descriptions to the appropriate
    handlers.
    """

    def __init__(self) -> None:
        self.mode = "sequential"
        self.route = "hybrid"
        self.arbiter = "bookend"
        # Last run state for the `show` command
        self._last_session_dir: str | None = None
        self._last_result: object | None = None
        # Cached provider health — populated once at startup
        # Maps env_var -> ("ok" | "degraded" | "error" | "none", detail)
        self._provider_health: dict[str, tuple[str, str]] = {}

    def run(self) -> None:
        """Main REPL loop."""
        self._check_providers()
        self._print_status_dashboard()
        self._print_quick_start()

        while True:
            try:
                prompt_text = Text()
                prompt_text.append("\ntriad", style=BRAND["green"])
                prompt_text.append(" ▸ ", style=BRAND["mint"])

                user_input = console.input(prompt_text).strip()

                if not user_input:
                    continue

                self._dispatch(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print(f"\n[{BRAND['dim']}]Goodbye.[/{BRAND['dim']}]")
                break

    def _check_providers(self) -> None:
        """Validate provider connectivity once and cache the results.

        Runs at REPL startup only. Makes a lightweight API call per
        configured provider to verify the key works. Errors are caught
        and recorded — they never crash the REPL.
        """
        from triad.keys import PROVIDERS, load_keys_env, validate_key

        load_keys_env()

        async def _validate_all() -> dict[str, tuple[str, str]]:
            import litellm

            results: dict[str, tuple[str, str]] = {}
            for env_var, _, _, _ in PROVIDERS:
                api_key = os.environ.get(env_var, "")
                if not api_key:
                    results[env_var] = ("none", "")
                    continue
                try:
                    ok, detail = await validate_key(env_var, api_key)
                    if ok:
                        results[env_var] = ("ok", detail)
                    else:
                        results[env_var] = ("error", detail)
                except (
                    litellm.RateLimitError,
                    litellm.ServiceUnavailableError,
                ) as e:
                    reason = "rate limited" if "429" in str(e) else "quota"
                    results[env_var] = ("degraded", reason)
                except Exception as e:
                    results[env_var] = ("error", str(e)[:60])
            return results

        try:
            with console.status(
                "[bold blue]Checking providers...", spinner="dots",
            ):
                self._provider_health = asyncio.run(_validate_all())
        except Exception:
            logger.debug("Provider health check failed, using key-only status")
            # Fallback: mark configured keys as ok, skip validation
            for env_var, _, _, _ in PROVIDERS:
                if os.environ.get(env_var):
                    self._provider_health[env_var] = ("ok", "key present")
                else:
                    self._provider_health[env_var] = ("none", "")

    def _print_status_dashboard(self) -> None:
        """Print the provider/model/defaults status panel.

        Uses cached health data from ``_check_providers()`` — never
        makes API calls itself.
        """
        from triad.keys import PROVIDER_NAMES, PROVIDERS

        # ── Provider connectivity (from cached health) ────────────
        provider_text = Text("  Providers:  ")
        for env_var, _, _, _ in PROVIDERS:
            name = PROVIDER_NAMES.get(env_var, env_var)
            status, detail = self._provider_health.get(
                env_var, ("none", ""),
            )
            if status == "ok":
                provider_text.append_text(
                    Text.from_markup(f"[green]✓[/green] {name}  ")
                )
            elif status == "degraded":
                suffix = f" ({detail})" if detail else ""
                provider_text.append_text(
                    Text.from_markup(
                        f"[yellow]⚠[/yellow] {name}{suffix}  "
                    )
                )
            elif status == "error":
                provider_text.append_text(
                    Text.from_markup(
                        f"[red]✗[/red] [dim]{name}[/dim]  "
                    )
                )
            else:
                provider_text.append_text(
                    Text.from_markup(
                        f"[dim red]✗[/dim red] [dim]{name}[/dim]  "
                    )
                )

        # ── Model count ───────────────────────────────────────────
        model_count = 0
        active_providers: set[str] = set()
        try:
            from triad.providers.registry import load_models

            registry = load_models()
            for cfg in registry.values():
                key_env = cfg.api_key_env
                if os.environ.get(key_env):
                    model_count += 1
                    active_providers.add(cfg.provider)
        except Exception:
            logger.debug("Could not load model registry for status dashboard")

        model_text = Text(f"  Models:     {model_count} available")
        if active_providers:
            model_text.append(f" across {len(active_providers)} provider")
            if len(active_providers) != 1:
                model_text.append("s")

        # ── Defaults ──────────────────────────────────────────────
        defaults_text = Text("  Defaults:   ")
        defaults_text.append(self.mode, style=BRAND["green"])
        defaults_text.append(" │ ", style="dim")
        defaults_text.append(self.arbiter, style=BRAND["green"])
        defaults_text.append(" │ ", style="dim")
        defaults_text.append(self.route, style=BRAND["green"])

        # ── Assemble panel ────────────────────────────────────────
        body = Text()
        body.append_text(provider_text)
        body.append("\n")
        body.append_text(model_text)
        body.append("\n")
        body.append_text(defaults_text)

        console.print(Panel(
            body,
            title="[bold]Status[/bold]",
            border_style="dim",
            expand=True,
            padding=(0, 1),
        ))

    def _print_quick_start(self) -> None:
        """Print compact quick-start command hints."""
        table = Table.grid(padding=(0, 4))
        table.add_column(style=BRAND["green"], width=24)
        table.add_column(style="dim")

        table.add_row("Just type your task", "Run a pipeline with current defaults")
        table.add_row("config", "View or change settings")
        table.add_row("models", "See available models + fitness scores")
        table.add_row("help", "All commands")

        console.print()
        console.print(Text("  Quick start:", style="bold"))
        console.print(table)
        console.print()

    def _dispatch(self, user_input: str) -> None:
        """Dispatch a user input line to the appropriate handler."""
        parts = user_input.split(None, 1)
        command = parts[0].lower()
        args = parts[1] if len(parts) > 1 else ""

        if command in ("exit", "quit"):
            raise EOFError

        if command == "help":
            self._show_help()
            return

        if command == "status":
            self._show_status()
            return

        if command in _STATE_COMMANDS:
            self._set_state(command, args)
            return

        # "run <task>" is an explicit task invocation
        if command == "run":
            if args:
                self._run_task(args)
            else:
                console.print(
                    f"  [{BRAND['dim']}]Usage: run <task description>[/{BRAND['dim']}]"
                )
            return

        if command == "show":
            self._handle_show(args)
            return

        if command in _CLI_COMMANDS:
            self._invoke_cli(user_input)
            return

        # Everything else is treated as a task description
        self._run_task(user_input)

    def _show_help(self) -> None:
        """Show available REPL commands."""
        help_text = Text()
        help_text.append("\n  Pipeline Commands\n", style="bold")
        help_text.append("    <task description>", style=BRAND["green"])
        help_text.append("  Run pipeline with current settings\n")
        help_text.append("    plan <desc>       ", style=BRAND["green"])
        help_text.append("  Expand a rough idea into a task spec\n")
        help_text.append("    estimate <desc>   ", style=BRAND["green"])
        help_text.append("  Show cost estimates\n")
        help_text.append("    show [view]       ", style=BRAND["green"])
        help_text.append("  View last run (summary/code/reviews/diffs)\n")

        help_text.append("\n  Session Settings\n", style="bold")
        help_text.append("    mode <value>      ", style=BRAND["green"])
        help_text.append("  Set pipeline mode (sequential/parallel/debate)\n")
        help_text.append("    route <value>     ", style=BRAND["green"])
        help_text.append("  Set routing strategy\n")
        help_text.append("    arbiter <value>   ", style=BRAND["green"])
        help_text.append("  Set arbiter mode (off/final_only/bookend/full)\n")
        help_text.append("    status            ", style=BRAND["green"])
        help_text.append("  Show current session settings\n")

        help_text.append("\n  Other Commands\n", style="bold")
        help_text.append("    models list       ", style=BRAND["green"])
        help_text.append("  Show registered models\n")
        help_text.append("    sessions list     ", style=BRAND["green"])
        help_text.append("  Show session history\n")
        help_text.append("    config show       ", style=BRAND["green"])
        help_text.append("  Show pipeline config\n")
        help_text.append("    help              ", style=BRAND["green"])
        help_text.append("  Show this help\n")
        help_text.append("    exit / quit       ", style=BRAND["green"])
        help_text.append("  Exit the REPL\n")

        console.print(help_text)

    def _show_status(self) -> None:
        """Show current session state."""
        status = Text()
        status.append("\n  Mode:    ", style="bold")
        status.append(self.mode, style=BRAND["green"])
        status.append("\n  Route:   ", style="bold")
        status.append(self.route, style=BRAND["green"])
        status.append("\n  Arbiter: ", style="bold")
        status.append(self.arbiter, style=BRAND["green"])
        status.append("\n")
        console.print(status)

    def _set_state(self, command: str, value: str) -> None:
        """Set a session state value."""
        value = value.strip().lower()
        g = BRAND["green"]
        d = BRAND["dim"]
        r = BRAND["red"]

        if command == "mode":
            if not value:
                console.print(f"  Current mode: [{g}]{self.mode}[/{g}]")
                opts = ", ".join(sorted(_VALID_MODES))
                console.print(f"  [{d}]Options: {opts}[/{d}]")
                return
            if value not in _VALID_MODES:
                console.print(
                    f"  [{r}]Invalid mode:[/{r}] '{value}'. "
                    f"Choose from: {', '.join(sorted(_VALID_MODES))}"
                )
                return
            self.mode = value
            console.print(f"  Mode set to [{g}]{value}[/{g}]")

        elif command == "route":
            if not value:
                console.print(f"  Current route: [{g}]{self.route}[/{g}]")
                opts = ", ".join(sorted(_VALID_ROUTES))
                console.print(f"  [{d}]Options: {opts}[/{d}]")
                return
            if value not in _VALID_ROUTES:
                console.print(
                    f"  [{r}]Invalid route:[/{r}] '{value}'. "
                    f"Choose from: {', '.join(sorted(_VALID_ROUTES))}"
                )
                return
            self.route = value
            console.print(f"  Route set to [{g}]{value}[/{g}]")

        elif command == "arbiter":
            if not value:
                console.print(
                    f"  Current arbiter: [{g}]{self.arbiter}[/{g}]"
                )
                opts = ", ".join(sorted(_VALID_ARBITERS))
                console.print(f"  [{d}]Options: {opts}[/{d}]")
                return
            if value not in _VALID_ARBITERS:
                console.print(
                    f"  [{BRAND['red']}]Invalid arbiter:[/{BRAND['red']}] '{value}'. "
                    f"Choose from: {', '.join(sorted(_VALID_ARBITERS))}"
                )
                return
            self.arbiter = value
            console.print(f"  Arbiter set to [{BRAND['green']}]{value}[/{BRAND['green']}]")

    def _invoke_cli(self, user_input: str) -> None:
        """Invoke a CLI subcommand via Typer runner."""
        try:
            from typer.testing import CliRunner

            from triad.cli import app

            args = shlex.split(user_input)
            cli_runner = CliRunner()
            result = cli_runner.invoke(app, args)
            if result.output:
                console.print(result.output, end="")
        except Exception as e:
            console.print(f"  [{BRAND['red']}]Error:[/{BRAND['red']}] {e}")

    def _run_task(self, task: str) -> None:
        """Run a pipeline task with current session settings.

        Executes the pipeline directly (not via CliRunner) so that
        the PipelineDisplay Live rendering works in the real terminal.
        Uses streaming display when conditions are met.
        """
        try:
            from triad.cli_display import PipelineDisplay
            from triad.dashboard.events import PipelineEventEmitter
            from triad.orchestrator import run_pipeline
            from triad.providers.registry import load_models, load_pipeline_config
            from triad.schemas.pipeline import (
                ArbiterMode,
                PipelineConfig,
                PipelineMode,
                TaskSpec,
            )
            from triad.schemas.routing import RoutingStrategy

            registry = load_models()
            base_config = load_pipeline_config()

            pipeline_mode = PipelineMode(self.mode)
            routing_strategy = RoutingStrategy(self.route)
            arbiter_mode = ArbiterMode(self.arbiter)

            config = PipelineConfig(
                pipeline_mode=pipeline_mode,
                arbiter_mode=arbiter_mode,
                default_timeout=300,
                max_retries=base_config.max_retries,
                reconciliation_retries=base_config.reconciliation_retries,
                stages=base_config.stages,
                arbiter_model=base_config.arbiter_model,
                reconcile_model=base_config.reconcile_model,
                routing_strategy=routing_strategy,
                min_fitness=base_config.min_fitness,
                persist_sessions=base_config.persist_sessions,
                session_db_path=base_config.session_db_path,
            )

            task_spec = TaskSpec(task=task)

            emitter = PipelineEventEmitter()
            stream_callback = None

            # Attach ProAgent for dashboard event forwarding (if configured)
            from triad.pro.agent import ProAgent

            _pro_agent = ProAgent.from_config()
            if _pro_agent:
                emitter.add_listener(_pro_agent.create_listener())

            # Try streaming display for sequential mode
            use_streaming = (
                pipeline_mode == PipelineMode.SEQUENTIAL
            )

            if use_streaming:
                try:
                    from triad.cli_streaming_display import ScrollingPipelineDisplay

                    streaming_display = ScrollingPipelineDisplay(
                        console, self.mode, self.route, self.arbiter,
                    )
                    emitter.add_listener(streaming_display.create_listener())
                    stream_callback = streaming_display.create_stream_callback()
                    with streaming_display:
                        pipeline_result = asyncio.run(
                            run_pipeline(
                                task_spec, config, registry, emitter,
                                stream_callback=stream_callback,
                            ),
                        )
                except ImportError:
                    use_streaming = False

            if not use_streaming:
                display = PipelineDisplay(console, self.mode, self.route, self.arbiter)
                emitter.add_listener(display.create_listener())
                with display:
                    pipeline_result = asyncio.run(
                        run_pipeline(task_spec, config, registry, emitter),
                    )

            from triad.output.writer import write_pipeline_output

            output_dir = "triad-output"
            actual_path = write_pipeline_output(pipeline_result, output_dir)

            # Store for the `show` command
            self._last_session_dir = actual_path
            self._last_result = pipeline_result

            # Display completion panel + interactive viewer
            from triad.cli import _display_completion

            _display_completion(pipeline_result, actual_path)

            from pathlib import Path

            from triad.post_run_viewer import PostRunViewer

            viewer = PostRunViewer(console, Path(actual_path), pipeline_result)
            viewer.run()

        except Exception as e:
            console.print(f"  [{BRAND['red']}]Error:[/{BRAND['red']}] {e}")

    def _handle_show(self, args: str) -> None:
        """Handle the 'show' command for viewing run outputs."""
        from pathlib import Path

        from triad.post_run_viewer import PostRunViewer

        parts = args.strip().split()
        subcommand = parts[0] if parts else ""

        if not self._last_session_dir or not self._last_result:
            console.print(
                f"  [{BRAND['dim']}]No previous run. "
                f"Use 'run' first or use 'triad show' from the CLI.[/{BRAND['dim']}]"
            )
            return

        viewer = PostRunViewer(
            console, Path(self._last_session_dir), self._last_result,
        )

        if not subcommand:
            viewer.run()
        elif subcommand == "summary":
            viewer._show_summary()
        elif subcommand == "code":
            if len(parts) > 1:
                try:
                    viewer._show_code_by_index(int(parts[1]))
                except ValueError:
                    viewer._show_code()
            else:
                viewer._show_code()
        elif subcommand == "reviews":
            viewer._show_reviews()
        elif subcommand == "diffs":
            viewer._show_diffs()
        else:
            console.print(
                f"  [{BRAND['dim']}]Usage: show [summary|code|reviews|diffs][/{BRAND['dim']}]"
            )

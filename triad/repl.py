"""Interactive REPL for the Triad Orchestrator.

Provides a persistent interactive session with command dispatch,
session state management, and task execution. Launch with `triad`
(no subcommand).
"""

from __future__ import annotations

import asyncio
import shlex

from rich.console import Console
from rich.text import Text

from triad.cli_display import BRAND

console = Console()

# Commands that are dispatched to CLI subcommands
_CLI_COMMANDS = {
    "plan", "models", "estimate", "sessions", "config", "review", "dashboard",
}

# Session state commands
_STATE_COMMANDS = {"mode", "route", "arbiter"}

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

    def run(self) -> None:
        """Main REPL loop."""
        self._show_help_hint()

        while True:
            try:
                prompt_text = Text()
                prompt_text.append("triad", style=BRAND["green"])
                prompt_text.append(" ▸ ", style=BRAND["mint"])

                console.print()
                user_input = console.input(prompt_text).strip()

                if not user_input:
                    continue

                self._dispatch(user_input)

            except (KeyboardInterrupt, EOFError):
                console.print(f"\n[{BRAND['dim']}]Goodbye.[/{BRAND['dim']}]")
                break

    def _show_help_hint(self) -> None:
        """Show a brief hint about available commands."""
        console.print(
            f"[{BRAND['dim']}]Type a task description to run the pipeline, "
            f"or 'help' for commands.[/{BRAND['dim']}]"
        )

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
                default_timeout=base_config.default_timeout,
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
            display = PipelineDisplay(console, self.mode, self.route, self.arbiter)
            emitter.add_listener(display.create_listener())

            with display:
                pipeline_result = asyncio.run(
                    run_pipeline(task_spec, config, registry, emitter),
                )

            # Display results inline
            from triad.cli import _display_result

            console.print()
            _display_result(pipeline_result)

            from triad.output.writer import write_pipeline_output

            output_dir = "triad-output"
            write_pipeline_output(pipeline_result, output_dir)
            console.print(f"\n[dim]Output written to:[/dim] {output_dir}/")

        except Exception as e:
            console.print(f"  [{BRAND['red']}]Error:[/{BRAND['red']}] {e}")

"""Streaming pipeline display with real-time token rendering.

Provides a multi-panel Rich Live display that shows streaming
code output, syntax highlighting, progress tracking, and an
activity log. Replaces PipelineDisplay in streaming mode.
"""

from __future__ import annotations

import re
import threading
import time
from collections import deque
from collections.abc import Callable
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.progress_bar import ProgressBar
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from triad.dashboard.events import EventListener, EventType, PipelineEvent
from triad.schemas.messages import PipelineStage
from triad.schemas.streaming import StreamChunk


# Stage symbols for progress display
_STAGE_SYMBOLS = {
    "architect": ("Architect", "cyan"),
    "implement": ("Implement", "green"),
    "refactor": ("Refactor", "yellow"),
    "verify": ("Verify", "magenta"),
}

_STAGE_ORDER = ["architect", "implement", "refactor", "verify"]


class StreamBuffer:
    """Accumulates streamed text and detects file boundaries.

    Watches for ``# file: path`` + `` ``` `` markers to detect
    when new files start and end. Used by the display layer to
    provide file-level awareness without burdening the orchestrator.
    """

    _FILE_HEADER_RE = re.compile(r"#\s*file:\s*(\S+)")
    _FENCE_OPEN_RE = re.compile(r"```(\w+)?\s*$")
    _FENCE_CLOSE_RE = re.compile(r"^```\s*$", re.MULTILINE)

    def __init__(self) -> None:
        self.accumulated: str = ""
        self.current_file: str = ""
        self.current_language: str = "text"
        self.current_code: str = ""
        self.in_code_block: bool = False
        self.files_completed: list[tuple[str, str, str]] = []  # (filepath, language, code)
        self.file_index: int = 0

    def feed(self, delta: str) -> list[dict[str, Any]]:
        """Process a text delta and return any file events.

        Returns:
            List of event dicts: {"type": "file_started"|"file_completed", ...}
        """
        events: list[dict[str, Any]] = []
        self.accumulated += delta

        # Process line by line from accumulated text
        lines = self.accumulated.split("\n")

        for line in lines:
            stripped = line.strip()

            # Check for file header
            fh_match = self._FILE_HEADER_RE.search(stripped)
            if fh_match and not self.in_code_block:
                self.current_file = fh_match.group(1)
                continue

            # Check for fence open
            fence_match = self._FENCE_OPEN_RE.match(stripped)
            if fence_match and not self.in_code_block:
                self.in_code_block = True
                self.current_language = fence_match.group(1) or "text"
                self.current_code = ""
                self.file_index += 1
                events.append({
                    "type": "file_started",
                    "filepath": self.current_file or f"file_{self.file_index}",
                    "language": self.current_language,
                    "file_index": self.file_index,
                })
                continue

            # Check for fence close
            if self._FENCE_CLOSE_RE.match(stripped) and self.in_code_block:
                self.in_code_block = False
                self.files_completed.append(
                    (self.current_file, self.current_language, self.current_code)
                )
                events.append({
                    "type": "file_completed",
                    "filepath": self.current_file or f"file_{self.file_index}",
                    "language": self.current_language,
                    "line_count": len(self.current_code.splitlines()),
                })
                self.current_file = ""
                continue

            # Accumulate code content
            if self.in_code_block:
                self.current_code += line + "\n"

        return events

    def get_display_code(self, max_lines: int = 500) -> tuple[str, str]:
        """Get the current code for syntax highlighting.

        Returns:
            Tuple of (code_text, language).
        """
        if self.in_code_block and self.current_code:
            lines = self.current_code.splitlines()
            if len(lines) > max_lines:
                lines = lines[-max_lines:]
            return "\n".join(lines), self.current_language
        return "", "text"


class StreamingPipelineDisplay:
    """Multi-panel streaming display for pipeline execution.

    Rich Layout structure:
    - progress (size=4): stage indicators + progress bar + elapsed + cost
    - main (ratio=1):
      - output (ratio=3): live syntax-highlighted code
      - activity (ratio=1, min=8): timestamped event log
    - cost (size=3): per-stage running cost
    """

    def __init__(
        self,
        console: Console,
        mode: str,
        route: str,
        arbiter: str,
    ) -> None:
        self._console = console
        self._mode = mode
        self._route = route
        self._arbiter = arbiter

        # State
        self._start_time = time.monotonic()
        self._current_stage: str = ""
        self._stage_status: dict[str, str] = {}  # stage -> "pending"|"running"|"done"
        self._stage_costs: dict[str, float] = {}
        self._stage_models: dict[str, str] = {}
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._activity_log: deque[tuple[float, str]] = deque(maxlen=50)
        self._stream_buffers: dict[str, StreamBuffer] = {}
        self._active_buffer: StreamBuffer | None = None
        self._paused: bool = False
        self._cancelled: bool = False
        self._lock = threading.Lock()

        # Initialize stage status
        for stage in _STAGE_ORDER:
            self._stage_status[stage] = "pending"

        # Rich Live
        self._live: Live | None = None

    def __enter__(self) -> StreamingPipelineDisplay:
        """Start the Rich Live display."""
        self._start_time = time.monotonic()
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.__enter__()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the Rich Live display."""
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def create_listener(self) -> EventListener:
        """Create an event listener for the pipeline event emitter."""

        def _handle(event: PipelineEvent) -> None:
            with self._lock:
                self._handle_event(event)
            self._refresh()

        return _handle

    def create_stream_callback(self) -> Callable:
        """Create a callback for streaming token delivery.

        Returns an async callable with signature (stage, chunk) -> None.
        """

        async def _on_stream(stage: PipelineStage, chunk: StreamChunk) -> None:
            with self._lock:
                stage_name = stage.value if hasattr(stage, "value") else str(stage)
                if stage_name not in self._stream_buffers:
                    self._stream_buffers[stage_name] = StreamBuffer()
                buf = self._stream_buffers[stage_name]
                self._active_buffer = buf

                # Feed the delta to the buffer
                file_events = buf.feed(chunk.delta)

                # Log file events
                for fe in file_events:
                    if fe["type"] == "file_started":
                        self._log(
                            f"[cyan]{stage_name}[/cyan] writing "
                            f"[bold]{fe['filepath']}[/bold] "
                            f"({fe['language']})"
                        )
                    elif fe["type"] == "file_completed":
                        self._log(
                            f"[cyan]{stage_name}[/cyan] completed "
                            f"[bold]{fe['filepath']}[/bold] "
                            f"({fe['line_count']} lines)"
                        )

                self._total_tokens = chunk.token_count

            self._refresh()

        return _on_stream

    def _handle_event(self, event: PipelineEvent) -> None:
        """Handle a pipeline event (called under lock)."""
        etype = event.type
        data = event.data

        if etype == EventType.PIPELINE_STARTED:
            self._log(
                f"Pipeline started — [bold]{self._mode}[/bold] mode, "
                f"[bold]{self._route}[/bold] routing"
            )

        elif etype == EventType.STAGE_STARTED:
            stage = data.get("stage", "?")
            model = data.get("model", "?")
            self._current_stage = stage
            self._stage_status[stage] = "running"
            self._stage_models[stage] = model
            self._log(
                f"[cyan]{stage}[/cyan] started — {model}"
            )

        elif etype == EventType.STAGE_COMPLETED:
            stage = data.get("stage", "?")
            cost = data.get("cost", 0.0)
            duration = data.get("duration", 0.0)
            self._stage_status[stage] = "done"
            self._stage_costs[stage] = cost
            self._total_cost += cost
            self._log(
                f"[green]{stage}[/green] completed "
                f"(${cost:.4f}, {duration:.1f}s)"
            )

        elif etype == EventType.ARBITER_STARTED:
            stage = data.get("stage", "?")
            self._log(f"[magenta]Arbiter[/magenta] reviewing {stage}")

        elif etype == EventType.ARBITER_VERDICT:
            stage = data.get("stage", "?")
            verdict = data.get("verdict", "?")
            confidence = data.get("confidence", 0.0)
            style = {
                "approve": "green", "flag": "yellow",
                "reject": "red", "halt": "bright_red",
            }.get(verdict, "white")
            self._log(
                f"[magenta]Arbiter[/magenta] {stage}: "
                f"[{style}]{verdict.upper()}[/{style}] "
                f"(conf {confidence:.2f})"
            )

        elif etype == EventType.RETRY_TRIGGERED:
            stage = data.get("stage", "?")
            retry_num = data.get("retry_number", "?")
            self._log(
                f"[yellow]Retry[/yellow] {stage} (attempt {retry_num})"
            )

        elif etype == EventType.MODEL_FALLBACK:
            original = data.get("original_model", "?")
            fallback = data.get("fallback_model", "?")
            reason = data.get("reason", "")
            self._log(
                f"[yellow]Fallback[/yellow] {original} -> {fallback} ({reason})"
            )

        elif etype == EventType.PIPELINE_COMPLETED:
            total_cost = data.get("total_cost", self._total_cost)
            self._total_cost = total_cost
            self._log("[green]Pipeline completed[/green]")

        elif etype == EventType.PIPELINE_HALTED:
            stage = data.get("stage", "?")
            self._log(f"[bright_red]HALTED[/bright_red] at {stage}")

        elif etype == EventType.ERROR:
            error = data.get("error", "Unknown error")
            self._log(f"[red]Error:[/red] {error}")

        elif etype == EventType.ARBITER_SKIPPED:
            stage = data.get("stage", "?")
            self._log(f"[dim]Arbiter skipped for {stage}[/dim]")

    def _log(self, message: str) -> None:
        """Add a timestamped entry to the activity log."""
        elapsed = time.monotonic() - self._start_time
        self._activity_log.append((elapsed, message))

    def _refresh(self) -> None:
        """Update the Live display."""
        if self._live and not self._paused:
            try:
                self._live.update(self._build_layout())
            except Exception:
                pass  # Swallow rendering errors during rapid updates

    def _build_layout(self) -> Layout:
        """Build the full Rich Layout."""
        layout = Layout()
        layout.split_column(
            Layout(name="progress", size=5),
            Layout(name="main", ratio=1),
            Layout(name="cost", size=3),
        )
        layout["main"].split_row(
            Layout(name="output", ratio=3),
            Layout(name="activity", ratio=1, minimum_size=30),
        )

        layout["progress"].update(self._build_progress_panel())
        layout["output"].update(self._build_output_panel())
        layout["activity"].update(self._build_activity_panel())
        layout["cost"].update(self._build_cost_panel())

        return layout

    def _build_progress_panel(self) -> Panel:
        """Build the stage progress panel."""
        table = Table.grid(padding=(0, 2))
        table.add_column(width=50)
        table.add_column(justify="right", width=20)

        # Stage indicators
        stages_text = Text()
        for stage in _STAGE_ORDER:
            name, color = _STAGE_SYMBOLS[stage]
            status = self._stage_status.get(stage, "pending")
            if status == "done":
                symbol = "[bold green]●[/bold green]"
            elif status == "running":
                symbol = "[bold yellow]◉[/bold yellow]"
            else:
                symbol = "[dim]○[/dim]"
            stages_text.append_text(Text.from_markup(f" {symbol} "))
            if status == "running":
                stages_text.append(name, style=f"bold {color}")
            elif status == "done":
                stages_text.append(name, style="dim")
            else:
                stages_text.append(name, style="dim")
            stages_text.append("  ")

        elapsed = time.monotonic() - self._start_time
        info_text = Text.from_markup(
            f"[dim]Elapsed:[/dim] {elapsed:.0f}s  "
            f"[dim]Cost:[/dim] ${self._total_cost:.4f}  "
            f"[dim]Tokens:[/dim] {self._total_tokens:,}"
        )

        table.add_row(stages_text, info_text)

        return Panel(
            table,
            title=f"[bold blue]Triad Pipeline[/bold blue] — {self._mode}",
            border_style="blue",
        )

    def _build_output_panel(self) -> Panel:
        """Build the code output panel with syntax highlighting."""
        if self._active_buffer:
            code, language = self._active_buffer.get_display_code(max_lines=500)
            if code.strip():
                try:
                    syntax = Syntax(
                        code,
                        language,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=False,
                    )
                    title = ""
                    if self._active_buffer.current_file:
                        title = f" {self._active_buffer.current_file}"
                    return Panel(
                        syntax,
                        title=f"[bold]Output[/bold]{title}",
                        border_style="green",
                    )
                except Exception:
                    pass

        # No code yet — show waiting message
        return Panel(
            Text("Waiting for output...", style="dim", justify="center"),
            title="[bold]Output[/bold]",
            border_style="dim",
        )

    def _build_activity_panel(self) -> Panel:
        """Build the activity log panel."""
        text = Text()
        entries = list(self._activity_log)
        # Show last N entries that fit
        for elapsed, message in entries[-20:]:
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            text.append(f"  {minutes:02d}:{seconds:02d}  ", style="dim")
            text.append_text(Text.from_markup(message))
            text.append("\n")

        if not entries:
            text.append("  Waiting for events...", style="dim")

        return Panel(
            text,
            title="[bold]Activity[/bold]",
            border_style="dim",
        )

    def _build_cost_panel(self) -> Panel:
        """Build the cost summary panel."""
        table = Table.grid(padding=(0, 3))
        for stage in _STAGE_ORDER:
            cost = self._stage_costs.get(stage, 0.0)
            model = self._stage_models.get(stage, "-")
            name, _ = _STAGE_SYMBOLS[stage]
            if cost > 0:
                table.add_row(
                    f"[dim]{name}:[/dim]",
                    f"${cost:.4f}",
                    f"[dim]{model}[/dim]",
                )

        return Panel(table, title="[bold]Costs[/bold]", border_style="dim")

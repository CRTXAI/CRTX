"""Streaming pipeline display with real-time token rendering.

Provides a multi-panel Rich Live display that shows streaming
code output, syntax highlighting, progress tracking, and an
activity log. Replaces PipelineDisplay in streaming mode.
"""

from __future__ import annotations

import platform
import re
import sys
import threading
import time
from collections import deque
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from rich.console import Console
from rich.layout import Layout
from rich.live import Live
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table
from rich.text import Text

from triad.dashboard.events import EventListener, EventType, PipelineEvent
from triad.schemas.messages import PipelineStage
from triad.schemas.streaming import StreamChunk


# ── Stage state tracking ──────────────────────────────────────────


class StageState(StrEnum):
    """Visual state for a pipeline stage."""

    PENDING = "pending"
    ACTIVE = "active"
    COMPLETE = "complete"
    FALLBACK = "fallback"
    FAILED = "failed"


_STAGE_SYMBOLS: dict[str, tuple[str, str]] = {
    "architect": ("Architect", "cyan"),
    "implement": ("Implement", "green"),
    "refactor": ("Refactor", "yellow"),
    "verify": ("Verify", "magenta"),
}

_STAGE_ORDER = ["architect", "implement", "refactor", "verify"]

_STATE_MARKUP: dict[StageState, str] = {
    StageState.PENDING: "[dim]○[/dim]",
    StageState.ACTIVE: "[bold cyan]◉[/bold cyan]",
    StageState.COMPLETE: "[bold green]●[/bold green]",
    StageState.FALLBACK: "[bold yellow]⚠[/bold yellow]",
    StageState.FAILED: "[bold red]✗[/bold red]",
}


# ── Keyboard handler ──────────────────────────────────────────────


def _is_interactive() -> bool:
    """Check if stdin is an interactive terminal."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _read_key() -> str:
    """Read a single keypress from the terminal.

    Returns the character, or special strings for arrow keys and
    control sequences. Uses msvcrt on Windows, tty/termios on Unix.
    """
    if not _is_interactive():
        return ""

    if platform.system() == "Windows":
        import msvcrt

        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":  # Ctrl+C
            return "ctrl+c"
        if ch == "\x1b":
            return "escape"
        if ch == "\t":
            return "tab"
        if ch == " ":
            return "space"
        # Arrow keys on Windows: first char is \x00 or \xe0
        if ch in ("\x00", "\xe0"):
            ch2 = msvcrt.getwch()
            if ch2 == "H":
                return "up"
            if ch2 == "P":
                return "down"
            return ""
        return ch
    else:
        import termios
        import tty

        fd = sys.stdin.fileno()
        old_settings = termios.tcgetattr(fd)
        try:
            tty.setraw(fd)
            ch = sys.stdin.read(1)
            if ch in ("\r", "\n"):
                return "enter"
            if ch == "\x03":
                return "ctrl+c"
            if ch == "\x1b":
                # Check for escape sequence (arrow keys)
                ch2 = sys.stdin.read(1)
                if ch2 == "[":
                    ch3 = sys.stdin.read(1)
                    if ch3 == "A":
                        return "up"
                    if ch3 == "B":
                        return "down"
                    return ""
                return "escape"
            if ch == "\t":
                return "tab"
            if ch == " ":
                return "space"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


class KeyboardHandler:
    """Daemon thread that reads keypresses and updates display state.

    Controls:
    - Tab: toggle focus between output and activity panels
    - j/k or Up/Down: scroll focused panel
    - f: toggle fullscreen on focused panel
    - c: toggle cost detail expansion
    - Space: pause/resume display updates
    - Ctrl+C: cancel pipeline
    """

    def __init__(self, display: StreamingPipelineDisplay) -> None:
        self._display = display
        self._thread: threading.Thread | None = None
        self._stop_event = threading.Event()

    def start(self) -> None:
        """Start the keyboard handler daemon thread."""
        if not _is_interactive():
            return
        self._thread = threading.Thread(
            target=self._run, daemon=True, name="keyboard-handler"
        )
        self._thread.start()

    def stop(self) -> None:
        """Signal the keyboard handler to stop."""
        self._stop_event.set()

    def _run(self) -> None:
        """Read keys in a loop until stopped."""
        while not self._stop_event.is_set():
            try:
                key = _read_key()
            except (EOFError, OSError):
                break

            if not key:
                continue

            with self._display._lock:
                self._dispatch(key)

            self._display._refresh()

    def _dispatch(self, key: str) -> None:
        """Handle a single keypress (called under lock)."""
        d = self._display

        if key == "tab":
            # Toggle focus between output and activity
            if d._focus_panel == "output":
                d._focus_panel = "activity"
            else:
                d._focus_panel = "output"

        elif key in ("j", "down"):
            d._scroll_offset = max(0, d._scroll_offset + 3)

        elif key in ("k", "up"):
            d._scroll_offset = max(0, d._scroll_offset - 3)

        elif key == "f":
            # Toggle fullscreen on focused panel
            if d._fullscreen_panel == d._focus_panel:
                d._fullscreen_panel = None
            else:
                d._fullscreen_panel = d._focus_panel

        elif key == "c":
            d._cost_expanded = not d._cost_expanded

        elif key == "space":
            d._paused = not d._paused

        elif key == "ctrl+c":
            d._cancelled = True
            d._cancel_event.set()


# ── Stream buffer ─────────────────────────────────────────────────


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


# ── Diff detection for refactor stage ─────────────────────────────

_DIFF_LINE_RE = re.compile(r"^[+\-@]{1,3}")


def _looks_like_diff(code: str) -> bool:
    """Heuristic: does this code look like a unified diff?

    Returns True if a significant portion of lines start with
    diff markers (+, -, @@).
    """
    lines = code.splitlines()
    if len(lines) < 3:
        return False
    marker_count = sum(1 for line in lines if _DIFF_LINE_RE.match(line))
    return marker_count / len(lines) > 0.3


def _render_diff_text(code: str) -> Text:
    """Render diff-formatted code with colored line prefixes."""
    text = Text()
    for line in code.splitlines():
        if line.startswith("+++") or line.startswith("---"):
            text.append(line + "\n", style="bold")
        elif line.startswith("+"):
            text.append(line + "\n", style="green")
        elif line.startswith("-"):
            text.append(line + "\n", style="red")
        elif line.startswith("@@"):
            text.append(line + "\n", style="cyan dim")
        else:
            text.append(line + "\n")
    return text


# ── Main display ──────────────────────────────────────────────────


class StreamingPipelineDisplay:
    """Multi-panel streaming display for pipeline execution.

    Rich Layout structure:
    - progress (size=5): stage indicators + progress bar + elapsed + cost
    - main (ratio=1):
      - output (ratio=3): live syntax-highlighted code / diff view
      - activity (ratio=1, min=30): timestamped event log
    - cost (size=3 or 8): per-model running cost (expandable)
    - help (size=1): keyboard shortcut help bar
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

        # Stage tracking
        self._start_time = time.monotonic()
        self._current_stage: str = ""
        self._stage_states: dict[str, StageState] = {}
        self._stage_costs: dict[str, float] = {}
        self._stage_models: dict[str, str] = {}
        self._stage_tokens: dict[str, int] = {}
        self._total_cost: float = 0.0
        self._total_tokens: int = 0
        self._active_model: str | None = None
        self._active_model_stage: str | None = None

        # Activity log
        self._activity_log: deque[tuple[float, str]] = deque(maxlen=50)

        # Stream buffers
        self._stream_buffers: dict[str, StreamBuffer] = {}
        self._active_buffer: StreamBuffer | None = None

        # Keyboard state
        self._focus_panel: str = "output"  # "output" | "activity"
        self._scroll_offset: int = 0
        self._fullscreen_panel: str | None = None
        self._cost_expanded: bool = False
        self._paused: bool = False
        self._cancelled: bool = False
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()

        # Initialize stage states
        for stage in _STAGE_ORDER:
            self._stage_states[stage] = StageState.PENDING

        # Backward-compat alias used by tests
        self._stage_status = self._stage_states

        # Rich Live + keyboard
        self._live: Live | None = None
        self._keyboard: KeyboardHandler | None = None

    @property
    def cancel_event(self) -> threading.Event:
        """Event that is set when the user presses Ctrl+C."""
        return self._cancel_event

    def __enter__(self) -> StreamingPipelineDisplay:
        """Start the Rich Live display and keyboard handler."""
        self._start_time = time.monotonic()
        self._live = Live(
            self._build_layout(),
            console=self._console,
            refresh_per_second=8,
            transient=True,
        )
        self._live.__enter__()
        self._keyboard = KeyboardHandler(self)
        self._keyboard.start()
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the Rich Live display and keyboard handler."""
        if self._keyboard:
            self._keyboard.stop()
            self._keyboard = None
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
                # Track per-stage tokens
                if stage_name in self._stage_models:
                    self._stage_tokens[stage_name] = chunk.token_count

            self._refresh()

        return _on_stream

    # ── Event handling ────────────────────────────────────────────

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
            self._stage_states[stage] = StageState.ACTIVE
            self._stage_models[stage] = model
            self._active_model = model
            self._active_model_stage = stage
            # Reset scroll on new stage
            self._scroll_offset = 0
            self._log(
                f"[cyan]{stage}[/cyan] started — {model}"
            )

        elif etype == EventType.STAGE_COMPLETED:
            stage = data.get("stage", "?")
            cost = data.get("cost", 0.0)
            duration = data.get("duration", 0.0)
            # Only mark complete if not already failed
            if self._stage_states.get(stage) != StageState.FAILED:
                self._stage_states[stage] = StageState.COMPLETE
            self._stage_costs[stage] = cost
            self._total_cost += cost
            # Clear active model if this was the active stage
            if self._active_model_stage == stage:
                self._active_model = None
                self._active_model_stage = None
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
            stage = data.get("stage", "")
            # Mark stage as fallback (still active but with warning symbol)
            if stage and self._stage_states.get(stage) == StageState.ACTIVE:
                self._stage_states[stage] = StageState.FALLBACK
            self._stage_models[stage] = fallback
            self._active_model = fallback
            self._log(
                f"[yellow]Fallback[/yellow] {original} -> {fallback} ({reason})"
            )

        elif etype == EventType.PIPELINE_COMPLETED:
            total_cost = data.get("total_cost", self._total_cost)
            self._total_cost = total_cost
            self._active_model = None
            self._active_model_stage = None
            self._log("[green]Pipeline completed[/green]")

        elif etype == EventType.PIPELINE_HALTED:
            stage = data.get("stage", "?")
            self._log(f"[bright_red]HALTED[/bright_red] at {stage}")

        elif etype == EventType.ERROR:
            error = data.get("error", "Unknown error")
            stage = data.get("stage", "")
            if stage and self._stage_states.get(stage) in (
                StageState.ACTIVE, StageState.FALLBACK,
            ):
                self._stage_states[stage] = StageState.FAILED
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

    # ── Layout builders ───────────────────────────────────────────

    def _build_layout(self) -> Layout:
        """Build the full Rich Layout."""
        layout = Layout()

        cost_size = 8 if self._cost_expanded else 3
        parts = [
            Layout(name="progress", size=5),
            Layout(name="main", ratio=1),
            Layout(name="cost", size=cost_size),
            Layout(name="help", size=1),
        ]
        layout.split_column(*parts)

        # Main panel: fullscreen or split
        if self._fullscreen_panel == "output":
            layout["main"].update(self._build_output_panel())
        elif self._fullscreen_panel == "activity":
            layout["main"].update(self._build_activity_panel())
        else:
            layout["main"].split_row(
                Layout(name="output", ratio=3),
                Layout(name="activity", ratio=1, minimum_size=30),
            )
            layout["main"]["output"].update(self._build_output_panel())
            layout["main"]["activity"].update(self._build_activity_panel())

        layout["progress"].update(self._build_progress_panel())
        layout["cost"].update(self._build_cost_panel())
        layout["help"].update(self._build_help_bar())

        return layout

    def _build_progress_panel(self) -> Panel:
        """Build the stage progress panel with distinct state symbols."""
        table = Table.grid(padding=(0, 2))
        table.add_column(width=60)
        table.add_column(justify="right", width=30)

        # Stage indicators with state-specific symbols
        stages_text = Text()
        for stage in _STAGE_ORDER:
            name, color = _STAGE_SYMBOLS[stage]
            state = self._stage_states.get(stage, StageState.PENDING)
            symbol = _STATE_MARKUP.get(state, "[dim]○[/dim]")
            model = self._stage_models.get(stage, "")

            stages_text.append_text(Text.from_markup(f" {symbol} "))

            if state == StageState.ACTIVE:
                label = f"{name} ({model})" if model else name
                stages_text.append(label, style=f"bold {color}")
            elif state == StageState.FALLBACK:
                label = f"{name} ({model})" if model else name
                stages_text.append(label, style="bold yellow")
            elif state == StageState.COMPLETE:
                label = f"{name} ({model})" if model else name
                stages_text.append(label, style="dim")
            elif state == StageState.FAILED:
                stages_text.append(name, style="red dim")
            else:
                stages_text.append(name, style="dim")
            stages_text.append("  ")

        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        info_text = Text.from_markup(
            f"[dim]Elapsed:[/dim] {minutes}:{seconds:02d}  "
            f"[dim]Cost:[/dim] ${self._total_cost:.4f}  "
            f"[dim]Tokens:[/dim] {self._total_tokens:,}"
        )

        table.add_row(stages_text, info_text)

        paused_tag = " [yellow]PAUSED[/yellow]" if self._paused else ""
        return Panel(
            table,
            title=f"[bold blue]Triad Pipeline[/bold blue] — {self._mode}{paused_tag}",
            border_style="blue",
        )

    def _build_output_panel(self) -> Panel:
        """Build the code output panel with syntax highlighting or diff coloring."""
        is_focused = self._focus_panel == "output"
        border = "green" if is_focused else "dim green"

        if self._active_buffer:
            code, language = self._active_buffer.get_display_code(max_lines=500)
            if code.strip():
                # Apply scroll offset
                lines = code.splitlines()
                if self._focus_panel == "output" and self._scroll_offset > 0:
                    offset = min(self._scroll_offset, max(0, len(lines) - 10))
                    lines = lines[offset:]
                    code = "\n".join(lines)

                # Check if this is a diff during refactor stage
                is_refactor = self._current_stage == "refactor"
                if is_refactor and _looks_like_diff(code):
                    content = _render_diff_text(code)
                    title_suffix = ""
                    if self._active_buffer.current_file:
                        title_suffix = f" {self._active_buffer.current_file}"
                    return Panel(
                        content,
                        title=f"[bold]Diff[/bold]{title_suffix}",
                        border_style=border,
                    )

                # Normal syntax-highlighted rendering
                try:
                    syntax = Syntax(
                        code,
                        language,
                        theme="monokai",
                        line_numbers=True,
                        word_wrap=False,
                    )
                    title_suffix = ""
                    if self._active_buffer.current_file:
                        title_suffix = f" {self._active_buffer.current_file}"
                    return Panel(
                        syntax,
                        title=f"[bold]Output[/bold]{title_suffix}",
                        border_style=border,
                    )
                except Exception:
                    pass

        return Panel(
            Text("Waiting for output...", style="dim", justify="center"),
            title="[bold]Output[/bold]",
            border_style="dim",
        )

    def _build_activity_panel(self) -> Panel:
        """Build the activity log panel."""
        is_focused = self._focus_panel == "activity"
        border = "bright_white" if is_focused else "dim"

        text = Text()
        entries = list(self._activity_log)

        # Apply scroll offset for activity panel
        display_entries = entries[-20:]
        if self._focus_panel == "activity" and self._scroll_offset > 0:
            end = max(0, len(entries) - self._scroll_offset)
            start = max(0, end - 20)
            display_entries = entries[start:end] if end > start else []

        for elapsed, message in display_entries:
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
            border_style=border,
        )

    def _build_cost_panel(self) -> Panel:
        """Build the cost summary panel.

        Compact mode (default): single-line summary with active model indicator.
        Expanded mode (press 'c'): per-model breakdown table.
        """
        if self._cost_expanded:
            return self._build_cost_expanded()
        return self._build_cost_compact()

    def _build_cost_compact(self) -> Panel:
        """One-line cost ticker with active model indicator."""
        text = Text()
        parts: list[str] = []

        for stage in _STAGE_ORDER:
            model = self._stage_models.get(stage, "")
            if not model:
                continue
            cost = self._stage_costs.get(stage, 0.0)
            tokens = self._stage_tokens.get(stage, 0)

            # Format token count
            if tokens >= 1000:
                tok_str = f"{tokens / 1000:.1f}K"
            else:
                tok_str = str(tokens)

            # Active model indicator
            is_active = (self._active_model_stage == stage)
            arrow = "↑" if is_active else ""

            parts.append(f"{model}: ${cost:.2f} ({tok_str}{arrow})")

        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)

        summary = "  ".join(parts)
        if summary:
            summary += f"  ║  Total: ${self._total_cost:.2f} · {minutes}:{seconds:02d}"
        else:
            summary = f"Total: ${self._total_cost:.4f} · {minutes}:{seconds:02d}"

        text.append_text(Text.from_markup(f"  {summary}"))

        return Panel(text, title="[bold]Costs[/bold]", border_style="dim")

    def _build_cost_expanded(self) -> Panel:
        """Multi-row cost breakdown table."""
        table = Table.grid(padding=(0, 3))
        table.add_column("Stage", width=12)
        table.add_column("Model", width=20)
        table.add_column("Cost", justify="right", width=10)
        table.add_column("Tokens", justify="right", width=10)
        table.add_column("", width=3)

        for stage in _STAGE_ORDER:
            name, _ = _STAGE_SYMBOLS[stage]
            model = self._stage_models.get(stage, "—")
            cost = self._stage_costs.get(stage, 0.0)
            tokens = self._stage_tokens.get(stage, 0)
            is_active = (self._active_model_stage == stage)

            if tokens >= 1000:
                tok_str = f"{tokens / 1000:.1f}K"
            else:
                tok_str = str(tokens) if tokens else "—"

            arrow = "[bold cyan]↑[/bold cyan]" if is_active else ""

            table.add_row(
                f"[dim]{name}[/dim]",
                model if model != "—" else "[dim]—[/dim]",
                f"${cost:.4f}" if cost > 0 else "[dim]—[/dim]",
                tok_str,
                arrow,
            )

        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        table.add_row(
            "[bold]Total[/bold]", "",
            f"[bold]${self._total_cost:.4f}[/bold]",
            f"{self._total_tokens:,}", f"{minutes}:{seconds:02d}",
        )

        return Panel(table, title="[bold]Costs[/bold] [dim](c to collapse)[/dim]", border_style="dim")

    def _build_help_bar(self) -> Text:
        """Build the keyboard shortcut help bar."""
        return Text.from_markup(
            " [dim][Tab] Focus  [↑↓] Scroll  [f] Fullscreen  "
            "[c] Costs  [Space] Pause[/dim]"
        )

"""Scrolling streaming display with a pinned status bar.

Prints pipeline output above a single-line Rich Live status bar.
Rich Live redraws only that one line at 4fps, so there's no flicker.
All content scrolls naturally in the terminal.
"""

from __future__ import annotations

import re
import threading
import time
from collections.abc import Callable
from enum import StrEnum
from typing import Any

from rich.console import Console
from rich.live import Live
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


# ── Scrolling display ─────────────────────────────────────────────


class ScrollingPipelineDisplay:
    """Scrolling streaming display with a pinned status bar.

    Prints pipeline output above a single-line Rich Live region.
    The Live renderable is just the status bar (1 line), so redraws
    at 4fps are imperceptible. All content scrolls naturally.
    """

    _FILE_HEADER_RE = re.compile(r"#\s*file:\s*(\S+)")
    _FENCE_OPEN_RE = re.compile(r"```(\w+)?\s*$")
    _FENCE_CLOSE_RE = re.compile(r"^```\s*$")

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

        # Line buffer for streaming text
        self._line_buffer: str = ""
        self._in_code_fence: bool = False
        self._current_file: str = ""
        self._current_language: str = ""

        # Cancel support
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()

        # Stream buffers (for file-level tracking)
        self._stream_buffers: dict[str, StreamBuffer] = {}

        # Rich Live
        self._live: Live | None = None

        # Initialize stage states
        for stage in _STAGE_ORDER:
            self._stage_states[stage] = StageState.PENDING

        # Backward-compat alias used by tests
        self._stage_status = self._stage_states

    @property
    def cancel_event(self) -> threading.Event:
        """Event that is set when the user cancels the pipeline."""
        return self._cancel_event

    def __enter__(self) -> ScrollingPipelineDisplay:
        """Start the Rich Live status bar."""
        self._start_time = time.monotonic()
        self._live = Live(
            self._build_status_bar(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        # Print pipeline header
        self._print(Text.from_markup(
            f"\n[bold blue]Pipeline started[/bold blue] — "
            f"{self._mode} | {self._route} | arbiter={self._arbiter}\n"
        ))
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the Rich Live status bar."""
        # Flush any remaining partial line
        self._flush_line_buffer()
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def __rich__(self) -> Text:
        """Called by Rich Live at 4fps — returns the 1-line status bar."""
        return self._build_status_bar()

    def create_listener(self) -> EventListener:
        """Create an event listener for the pipeline event emitter."""

        def _handle(event: PipelineEvent) -> None:
            with self._lock:
                self._handle_event(event)

        return _handle

    def create_stream_callback(self) -> Callable:
        """Create a callback for streaming token delivery."""

        async def _on_stream(stage: PipelineStage, chunk: StreamChunk) -> None:
            with self._lock:
                stage_name = stage.value if hasattr(stage, "value") else str(stage)
                if stage_name not in self._stream_buffers:
                    self._stream_buffers[stage_name] = StreamBuffer()
                buf = self._stream_buffers[stage_name]
                buf.feed(chunk.delta)

                self._total_tokens = chunk.token_count
                if stage_name in self._stage_models:
                    self._stage_tokens[stage_name] = chunk.token_count

                # Append delta to line buffer and flush complete lines
                self._line_buffer += chunk.delta
                self._flush_complete_lines()

        return _on_stream

    # ── Printing ───────────────────────────────────────────────────

    def _print(self, content: Text | str = "") -> None:
        """Print above the Live status bar.

        When Live is active, ``live.console.print()`` inserts content
        above the live region, so the status bar stays pinned at bottom.
        """
        if self._live:
            self._live.console.print(content, highlight=False)
        else:
            self._console.print(content, highlight=False)

    def _print_markup(self, markup: str) -> None:
        """Print Rich markup text above the status bar."""
        if self._live:
            self._live.console.print(markup, highlight=False)
        else:
            self._console.print(markup, highlight=False)

    # ── Status bar ─────────────────────────────────────────────────

    def _build_status_bar(self) -> Text:
        """Build the 1-line pinned status bar.

        Format: ● Architect  ◉ Implement (model)  ○ Refactor  ○ Verify   1:23  $0.12  4.2K tok
        """
        bar = Text()
        for stage in _STAGE_ORDER:
            name, color = _STAGE_SYMBOLS[stage]
            state = self._stage_states.get(stage, StageState.PENDING)
            symbol = _STATE_MARKUP.get(state, "[dim]○[/dim]")
            model = self._stage_models.get(stage, "")

            bar.append_text(Text.from_markup(f" {symbol} "))

            if state == StageState.ACTIVE:
                label = f"{name} ({model})" if model else name
                bar.append(label, style=f"bold {color}")
            elif state == StageState.FALLBACK:
                label = f"{name} ({model})" if model else name
                bar.append(label, style="bold yellow")
            elif state == StageState.COMPLETE:
                bar.append(name, style="dim")
            elif state == StageState.FAILED:
                bar.append(name, style="red dim")
            else:
                bar.append(name, style="dim")
            bar.append("  ")

        # Elapsed
        elapsed = time.monotonic() - self._start_time
        minutes = int(elapsed // 60)
        seconds = int(elapsed % 60)
        bar.append(f" {minutes}:{seconds:02d}", style="dim")

        # Cost
        bar.append(f"  ${self._total_cost:.2f}", style="dim")

        # Tokens
        if self._total_tokens >= 1000:
            tok_str = f"{self._total_tokens / 1000:.1f}K"
        else:
            tok_str = str(self._total_tokens)
        bar.append(f"  {tok_str} tok", style="dim")

        return bar

    # ── Line buffer processing ─────────────────────────────────────

    def _flush_complete_lines(self) -> None:
        """Split the line buffer on newlines and process complete lines."""
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._process_line(line)

    def _flush_line_buffer(self) -> None:
        """Print any remaining partial line in the buffer."""
        if self._line_buffer.strip():
            self._process_line(self._line_buffer)
            self._line_buffer = ""

    def _process_line(self, line: str) -> None:
        """Process a single line of streamed output.

        Handles file headers, code fence markers, diff coloring,
        and plain text.
        """
        stripped = line.strip()

        # File header: # file: path/to/file.py
        fh_match = self._FILE_HEADER_RE.search(stripped)
        if fh_match and not self._in_code_fence:
            filepath = fh_match.group(1)
            self._current_file = filepath
            # Detect language from extension
            ext = filepath.rsplit(".", 1)[-1] if "." in filepath else ""
            self._current_language = ext
            # Print file separator
            label = f" {filepath} ({ext}) " if ext else f" {filepath} "
            separator = f"──{label}" + "─" * max(0, 60 - len(label) - 2)
            self._print_markup(f"[dim]{separator}[/dim]")
            return

        # Code fence open
        fence_match = self._FENCE_OPEN_RE.match(stripped)
        if fence_match and not self._in_code_fence:
            self._in_code_fence = True
            lang = fence_match.group(1) or self._current_language or ""
            if lang:
                self._print_markup(f"[dim]```{lang}[/dim]")
            else:
                self._print_markup("[dim]```[/dim]")
            return

        # Code fence close
        if self._FENCE_CLOSE_RE.match(stripped) and self._in_code_fence:
            self._in_code_fence = False
            self._print_markup("[dim]```[/dim]")
            return

        # Inside code fence — check for diff lines during refactor
        if self._in_code_fence and self._current_stage == "refactor":
            if line.startswith("+++") or line.startswith("---"):
                self._print(Text(line, style="bold"))
                return
            if line.startswith("+"):
                self._print(Text(line, style="green"))
                return
            if line.startswith("-"):
                self._print(Text(line, style="red"))
                return
            if line.startswith("@@"):
                self._print(Text(line, style="cyan dim"))
                return

        # Default: print the line as-is
        self._print(Text(line))

    # ── Event handling ─────────────────────────────────────────────

    def _handle_event(self, event: PipelineEvent) -> None:
        """Handle a pipeline event (called under lock)."""
        etype = event.type
        data = event.data

        if etype == EventType.STAGE_STARTED:
            stage = data.get("stage", "?")
            model = data.get("model", "?")
            self._current_stage = stage
            self._stage_states[stage] = StageState.ACTIVE
            self._stage_models[stage] = model
            self._active_model = model
            self._active_model_stage = stage
            # Flush any buffered text from the previous stage
            self._flush_line_buffer()
            self._line_buffer = ""
            self._in_code_fence = False
            self._current_file = ""
            _, color = _STAGE_SYMBOLS.get(stage, (stage, "white"))
            self._print_markup(
                f"\n[bold {color}]◉ {stage.title()}[/bold {color}] ({model})"
            )

        elif etype == EventType.STAGE_COMPLETED:
            stage = data.get("stage", "?")
            cost = data.get("cost", 0.0)
            duration = data.get("duration", 0.0)
            # Flush remaining text
            self._flush_line_buffer()
            self._line_buffer = ""
            self._in_code_fence = False
            if self._stage_states.get(stage) != StageState.FAILED:
                self._stage_states[stage] = StageState.COMPLETE
            self._stage_costs[stage] = cost
            self._total_cost += cost
            if self._active_model_stage == stage:
                self._active_model = None
                self._active_model_stage = None
            self._print_markup(
                f"[green]● {stage.title()} completed[/green] "
                f"(${cost:.4f}, {duration:.1f}s)"
            )

        elif etype == EventType.ARBITER_STARTED:
            stage = data.get("stage", "?")
            self._print_markup(
                f"  [dim magenta]⚖ Arbiter reviewing {stage}...[/dim magenta]"
            )

        elif etype == EventType.ARBITER_VERDICT:
            stage = data.get("stage", "?")
            verdict = data.get("verdict", "?")
            confidence = data.get("confidence", 0.0)
            style = {
                "approve": "green", "flag": "yellow",
                "reject": "red", "halt": "bright_red",
            }.get(verdict, "white")
            self._print_markup(
                f"  [magenta]⚖ Arbiter {stage}:[/magenta] "
                f"[{style}]{verdict.upper()}[/{style}] "
                f"(conf {confidence:.2f})"
            )

        elif etype == EventType.RETRY_TRIGGERED:
            stage = data.get("stage", "?")
            retry_num = data.get("retry_number", "?")
            self._print_markup(
                f"  [yellow]↻ Retry {stage} (attempt {retry_num})[/yellow]"
            )

        elif etype == EventType.MODEL_FALLBACK:
            original = data.get("original_model", "?")
            fallback = data.get("fallback_model", "?")
            reason = data.get("reason", "")
            stage = data.get("stage", "")
            if stage and self._stage_states.get(stage) == StageState.ACTIVE:
                self._stage_states[stage] = StageState.FALLBACK
            self._stage_models[stage] = fallback
            self._active_model = fallback
            self._print_markup(
                f"  [yellow]⚠ Fallback:[/yellow] {original} → {fallback} ({reason})"
            )

        elif etype == EventType.PIPELINE_COMPLETED:
            total_cost = data.get("total_cost", self._total_cost)
            self._total_cost = total_cost
            self._active_model = None
            self._active_model_stage = None
            elapsed = time.monotonic() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self._print_markup(
                f"\n[bold green]Pipeline completed[/bold green] "
                f"— ${total_cost:.4f} in {minutes}:{seconds:02d}"
            )

        elif etype == EventType.PIPELINE_HALTED:
            stage = data.get("stage", "?")
            self._print_markup(
                f"\n[bold bright_red]HALTED[/bold bright_red] at {stage}"
            )

        elif etype == EventType.ERROR:
            error = data.get("error", "Unknown error")
            stage = data.get("stage", "")
            if stage and self._stage_states.get(stage) in (
                StageState.ACTIVE, StageState.FALLBACK,
            ):
                self._stage_states[stage] = StageState.FAILED
            self._print_markup(f"[red]Error:[/red] {error}")

        elif etype == EventType.ARBITER_SKIPPED:
            stage = data.get("stage", "?")
            self._print_markup(
                f"  [dim]Arbiter skipped for {stage}[/dim]"
            )

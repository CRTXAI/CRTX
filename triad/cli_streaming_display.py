"""Scrolling streaming display with a pinned status bar.

Prints file headers and stage lifecycle events above a single-line
Rich Live status bar. Code content is swallowed — only file names
scroll during streaming.
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


def _format_tokens(count: int) -> str:
    """Format a token count for display (e.g. 4200 -> '4.2K')."""
    if count >= 1000:
        return f"{count / 1000:.1f}K"
    return str(count)


# ── Scrolling streaming display ───────────────────────────────────


class ScrollingPipelineDisplay:
    """Scrolling streaming display with a pinned status bar.

    Prints file headers and stage lifecycle events above a single-line
    Rich Live status bar. Code content is swallowed — only file names
    scroll during streaming.
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

        # Cancel support
        self._cancel_event = threading.Event()
        self._lock = threading.Lock()

        # Stream buffers (for file-level tracking)
        self._stream_buffers: dict[str, StreamBuffer] = {}

        # Line buffer for stream processing
        self._line_buffer: str = ""
        self._in_code_fence: bool = False
        self._file_count: int = 0

        # Activity line — live preview of streaming content
        self._last_content_line: str = ""
        self._tok_samples: list[tuple[float, int]] = []

        # Rich Live — renders the activity line + status bar
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
            self._build_live_renderable(),
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.__enter__()
        # Print pipeline header
        self._print_markup(
            f"\n[bold blue]Pipeline started[/bold blue] — "
            f"{self._mode} | {self._route} | arbiter={self._arbiter}\n"
        )
        return self

    def __exit__(self, *args: object) -> None:
        """Stop the Rich Live status bar."""
        if self._live:
            self._live.__exit__(*args)
            self._live = None

    def __rich__(self) -> Text:
        """Called by Rich Live at 4fps — returns the live renderable."""
        return self._build_live_renderable()

    def create_listener(self) -> EventListener:
        """Create an event listener for the pipeline event emitter."""

        def _handle(event: PipelineEvent) -> None:
            with self._lock:
                self._handle_event(event)

        return _handle

    def create_stream_callback(self) -> Callable:
        """Create a callback for streaming token delivery.

        Appends deltas to a line buffer, flushes complete lines through
        ``_process_line()`` which prints file headers and swallows code.
        """

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

                # Tok/s rolling window
                self._tok_samples.append(
                    (time.monotonic(), chunk.token_count)
                )

                # Line buffer processing — detect file headers
                self._line_buffer += chunk.delta
                self._flush_complete_lines()

        return _on_stream

    # ── Printing ───────────────────────────────────────────────────

    def _print_markup(self, markup: str) -> None:
        """Print Rich markup text above the status bar."""
        if self._live:
            self._live.console.print(markup, highlight=False)
        else:
            self._console.print(markup, highlight=False)

    # ── Line buffer processing ────────────────────────────────────

    def _flush_complete_lines(self) -> None:
        """Split line buffer on newlines and process complete lines."""
        while "\n" in self._line_buffer:
            line, self._line_buffer = self._line_buffer.split("\n", 1)
            self._process_line(line)

    def _process_line(self, line: str) -> None:
        """Process a single line from the stream. Only print file headers."""
        stripped = line.strip()

        # File header → print (only outside code fences)
        if not self._in_code_fence:
            match = self._FILE_HEADER_RE.match(stripped)
            if match:
                filename = match.group(1)
                self._file_count += 1
                self._print_markup(f"  [dim]── {filename}[/]")
                self._last_content_line = ""
                return

        # If inside a code fence, check for close first
        if self._in_code_fence:
            if self._FENCE_CLOSE_RE.match(stripped):
                self._in_code_fence = False
            elif stripped and len(stripped) > 5:
                self._last_content_line = stripped
            return

        # Track fence open
        if self._FENCE_OPEN_RE.match(stripped):
            self._in_code_fence = True
            return

        # Outside code — update preview for prose/planning text
        if stripped and len(stripped) > 5:
            self._last_content_line = stripped

    # ── Live renderable ────────────────────────────────────────────

    def _calculate_tok_rate(self) -> float:
        """Calculate tokens per second over a 3-second rolling window."""
        now = time.monotonic()
        cutoff = now - 3.0
        self._tok_samples = [(t, c) for t, c in self._tok_samples if t >= cutoff]

        if len(self._tok_samples) < 2:
            return 0.0

        oldest_time, oldest_count = self._tok_samples[0]
        newest_time, newest_count = self._tok_samples[-1]
        elapsed = newest_time - oldest_time
        if elapsed <= 0:
            return 0.0

        return (newest_count - oldest_count) / elapsed

    def _build_live_renderable(self) -> Text:
        """Build the live renderable: activity preview + status bar."""
        lines = Text()

        # Activity preview line (only while actively streaming)
        if self._current_stage and self._last_content_line:
            console_width = getattr(self._console, "width", 80) or 80
            max_width = max(40, console_width - 20)
            preview = self._last_content_line[:max_width]
            tok_rate = self._calculate_tok_rate()
            rate_str = f"{tok_rate:,.0f} tok/s" if tok_rate > 0 else ""

            lines.append("  \u25b8 ", style="dim cyan")
            lines.append(preview, style="dim")
            padding = max(1, max_width - len(preview) - len(rate_str))
            lines.append(" " * padding)
            if rate_str:
                lines.append(rate_str, style="dim cyan")
            lines.append("\n")

        # Status bar (always)
        lines.append_text(self._build_status_bar())

        return lines

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
        bar.append(f"  {_format_tokens(self._total_tokens)} tok", style="dim")

        # Tok/s (only while actively streaming)
        if self._current_stage:
            tok_rate = self._calculate_tok_rate()
            if tok_rate > 0:
                bar.append(f"  {tok_rate:,.0f} tok/s", style="dim cyan")

        return bar

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
            self._last_content_line = ""
            self._tok_samples = []
            _, color = _STAGE_SYMBOLS.get(stage, (stage, "white"))
            self._print_markup(
                f"\n[bold {color}]◉ {stage.title()}[/bold {color}] ({model})"
            )

        elif etype == EventType.STAGE_COMPLETED:
            stage = data.get("stage", "?")
            cost = data.get("cost", 0.0)
            duration = data.get("duration", 0.0)
            if self._stage_states.get(stage) != StageState.FAILED:
                self._stage_states[stage] = StageState.COMPLETE
            self._stage_costs[stage] = cost
            self._total_cost += cost
            if self._active_model_stage == stage:
                self._active_model = None
                self._active_model_stage = None
            self._current_stage = ""
            self._last_content_line = ""
            # Flush any remaining partial line
            if self._line_buffer.strip():
                self._process_line(self._line_buffer)
                self._line_buffer = ""
            file_note = f", {self._file_count} files" if self._file_count > 0 else ""
            self._print_markup(
                f"[green]● {stage.title()} completed[/green] "
                f"[dim](${cost:.3f}, {duration:.1f}s{file_note})[/dim]"
            )
            self._file_count = 0
            self._in_code_fence = False

        elif etype == EventType.ARBITER_STARTED:
            stage = data.get("stage", "?")
            self._print_markup(
                f"  [dim magenta]⚖ Arbiter reviewing {stage}...[/dim magenta]"
            )

        elif etype == EventType.ARBITER_VERDICT:
            stage = data.get("stage", "?")
            verdict = data.get("verdict", "?")
            confidence = data.get("confidence", 0.0)
            reasoning = data.get("reasoning", "")
            style = {
                "approve": "green", "flag": "yellow",
                "reject": "red", "halt": "bright_red",
            }.get(verdict, "white")
            reason_brief = ""
            if reasoning:
                first_sentence = reasoning.split(". ")[0].split("\n")[0][:80]
                reason_brief = f" — {first_sentence}"
            self._print_markup(
                f"  [magenta]⚖ Arbiter {stage}:[/magenta] "
                f"[{style}]{verdict.upper()}[/{style}] "
                f"[dim](conf {confidence:.2f}){reason_brief}[/dim]"
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
            self._current_stage = ""
            elapsed = time.monotonic() - self._start_time
            minutes = int(elapsed // 60)
            seconds = int(elapsed % 60)
            self._print_markup(
                f"\n[bold green]Pipeline completed[/bold green] "
                f"— ${total_cost:.4f} in {minutes}:{seconds:02d}"
            )

        elif etype == EventType.PIPELINE_HALTED:
            stage = data.get("stage", "?")
            self._current_stage = ""
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

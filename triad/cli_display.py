"""Interactive CLI display components for CRTX.

Provides branded ASCII art, interactive configuration screen,
real-time pipeline status display, and post-completion summary.
All rendering uses Rich for premium terminal output.
"""

from __future__ import annotations

import platform
import sys
import threading
import time

from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

# ── Brand Colors ──────────────────────────────────────────────────

BRAND = {
    "mint": "#00ffbb",
    "emerald": "#00ff66",
    "lime": "#66ff88",
    "gold": "#D4A843",
    "green": "#00ff88",
    "dim": "#6a8a6a",
    "pending": "#354535",
    "amber": "#ffaa00",
    "red": "#ff4444",
}

VERSION = "0.1.0"

# ── ASCII Art Logos ───────────────────────────────────────────────

# Full logo — triangle + CRTX block letters
FULL_LOGO = r"""
        ◆                ▄▀▀▀▄ █▀▀▀▄ ▀▀▀█▀▀▀ █   █
       ╱ ╲               █     █   █    █    ▀▄ ▄▀
      ╱   ╲              █     █▀▀█▀    █      █
     ╱  ◈  ╲             ▀▄▄▄▀ █  ▀▄   █    ▄▀ ▀▄
    ╱       ╲
   ◆─────────◆
"""

COMPACT_LOGO_TEMPLATE = """\
 ◆
╱◈╲    CRTX v{version}
◆──◆   {mode} · {route} · {arbiter}"""


def render_full_logo(console: Console) -> None:
    """Print the full branded ASCII logo with per-segment Rich styling."""
    lines = FULL_LOGO.strip("\n").split("\n")

    for line in lines:
        text = Text()
        # Left side contains the triangle geometry
        # Right side contains the CRTX block letters
        # Split at column 22 (where block letters start)
        left = line[:22] if len(line) > 22 else line
        right = line[22:] if len(line) > 22 else ""

        # Color the triangle nodes
        for ch in left:
            if ch == "◆":
                text.append(ch, style=BRAND["mint"])
            elif ch == "◈":
                text.append(ch, style=BRAND["gold"])
            elif ch in ("╱", "╲", "─"):
                text.append(ch, style=BRAND["emerald"])
            else:
                text.append(ch)

        # Block letters in green
        if right:
            text.append(right, style=BRAND["green"])

        console.print(text)

    # Pronunciation + subtitle
    pronun = Text()
    pronun.append("    /kôr'teks/", style=BRAND["dim"])
    console.print(pronun)
    subtitle = Text()
    subtitle.append("    Every session smarter than the last.", style=BRAND["dim"])
    console.print(subtitle)
    console.print()


def render_compact_logo(
    console: Console, mode: str, route: str, arbiter: str,
) -> Text:
    """Build compact logo as a Rich Text for embedding in panels.

    Returns:
        Rich Text object with the compact logo.
    """
    logo_str = COMPACT_LOGO_TEMPLATE.format(
        version=VERSION,
        mode=mode.replace("_", " ").title(),
        route=route.replace("_", " ").title(),
        arbiter=arbiter.replace("_", " ").title(),
    )
    lines = logo_str.split("\n")
    text = Text()
    for i, line in enumerate(lines):
        if i > 0:
            text.append("\n")
        for ch in line:
            if ch == "◆":
                text.append(ch, style=BRAND["mint"])
            elif ch == "◈":
                text.append(ch, style=BRAND["gold"])
            elif ch in ("╱", "╲", "─"):
                text.append(ch, style=BRAND["emerald"])
            elif ch == "·":
                text.append(ch, style=BRAND["dim"])
            else:
                text.append(ch, style=BRAND["green"])
    return text


# ── Keypress Utilities ────────────────────────────────────────────


def is_interactive() -> bool:
    """Check if stdin is an interactive terminal (not a pipe or test runner)."""
    try:
        return sys.stdin.isatty()
    except (AttributeError, ValueError):
        return False


def _read_key() -> str:
    """Read a single keypress from the terminal.

    Returns the character, or special strings for Enter/escape.
    Uses msvcrt on Windows, tty/termios on Unix.
    Returns 'enter' immediately if stdin is not a real terminal.
    """
    if not is_interactive():
        return "enter"

    if platform.system() == "Windows":
        import msvcrt
        ch = msvcrt.getwch()
        if ch in ("\r", "\n"):
            return "enter"
        if ch == "\x03":  # Ctrl+C
            raise KeyboardInterrupt
        if ch == "\x1b":
            return "escape"
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
                raise KeyboardInterrupt
            if ch == "\x1b":
                return "escape"
            return ch
        finally:
            termios.tcsetattr(fd, termios.TCSADRAIN, old_settings)


# ── Config Screen ─────────────────────────────────────────────────


# Cycle lists for config screen options
_MODE_CYCLE = ["sequential", "parallel", "debate"]
_ROUTE_CYCLE = ["hybrid", "quality_first", "cost_optimized", "speed_first"]
_ARBITER_CYCLE = ["bookend", "off", "final_only", "full"]


class ConfigScreen:
    """Interactive pre-run configuration screen.

    Shows current pipeline settings and allows cycling through options
    with single keypresses before confirming the run.
    """

    def __init__(
        self,
        task: str,
        config: object,
        registry: dict,
    ) -> None:
        self._task = task
        self._config = config
        self._registry = registry
        self._mode_idx = 0
        self._route_idx = 0
        self._arbiter_idx = 0

        # Initialize indices from config defaults
        mode_val = getattr(config, "pipeline_mode", None)
        if mode_val:
            mode_str = mode_val.value if hasattr(mode_val, "value") else str(mode_val)
            if mode_str in _MODE_CYCLE:
                self._mode_idx = _MODE_CYCLE.index(mode_str)

        route_val = getattr(config, "routing_strategy", None)
        if route_val:
            route_str = route_val.value if hasattr(route_val, "value") else str(route_val)
            if route_str in _ROUTE_CYCLE:
                self._route_idx = _ROUTE_CYCLE.index(route_str)

        arbiter_val = getattr(config, "arbiter_mode", None)
        if arbiter_val:
            arbiter_str = arbiter_val.value if hasattr(arbiter_val, "value") else str(arbiter_val)
            if arbiter_str in _ARBITER_CYCLE:
                self._arbiter_idx = _ARBITER_CYCLE.index(arbiter_str)

    @property
    def mode(self) -> str:
        return _MODE_CYCLE[self._mode_idx]

    @property
    def route(self) -> str:
        return _ROUTE_CYCLE[self._route_idx]

    @property
    def arbiter(self) -> str:
        return _ARBITER_CYCLE[self._arbiter_idx]

    def _get_estimated_cost(self) -> str:
        """Get estimated cost string for current routing strategy."""
        try:
            from triad.routing.engine import estimate_cost
            from triad.schemas.routing import RoutingStrategy

            strategy = RoutingStrategy(self.route)
            est = estimate_cost(self._config, self._registry, strategy)
            return f"${est.total_estimated_cost:.4f}"
        except Exception:
            return "N/A"

    def _get_model_assignments(self) -> dict[str, str]:
        """Get model assignments for current routing strategy."""
        try:
            from triad.routing.engine import RoutingEngine
            from triad.schemas.routing import RoutingStrategy

            strategy = RoutingStrategy(self.route)
            engine = RoutingEngine(self._config, self._registry)
            decisions = engine.select_pipeline_models(strategy)
            return {d.role.value: d.model_key for d in decisions}
        except Exception:
            return {}

    def _build_panel(self, console: Console) -> Panel:
        """Build the config screen panel."""
        content = Text()

        # Task
        content.append("  Task: ", style="bold")
        task_preview = self._task[:70] + "..." if len(self._task) > 70 else self._task
        content.append(task_preview, style="white")
        content.append("\n\n")

        # Config options
        content.append("  [1] ", style=BRAND["dim"])
        content.append("Mode    ", style="bold")
        content.append(self.mode.replace("_", " ").title(), style=BRAND["green"])
        content.append("\n")

        content.append("  [2] ", style=BRAND["dim"])
        content.append("Route   ", style="bold")
        content.append(self.route.replace("_", " ").title(), style=BRAND["green"])
        content.append("\n")

        content.append("  [3] ", style=BRAND["dim"])
        content.append("Arbiter ", style="bold")
        content.append(self.arbiter.replace("_", " ").title(), style=BRAND["green"])
        content.append("\n\n")

        # Model assignments
        models = self._get_model_assignments()
        if models:
            content.append("  Models: ", style="bold")
            parts = []
            for role in ["architect", "implement", "refactor", "verify"]:
                model_key = models.get(role, "?")
                parts.append(f"{role[:4]}={model_key}")
            content.append("  ".join(parts), style=BRAND["dim"])
            content.append("\n")

        # Cost estimate
        cost = self._get_estimated_cost()
        content.append("  Est. Cost: ", style="bold")
        content.append(cost, style=BRAND["gold"])
        content.append("\n\n")

        # Instructions
        content.append("  [Enter] ", style=BRAND["mint"])
        content.append("Run", style="bold")
        content.append("  [q] ", style=BRAND["dim"])
        content.append("Quit", style=BRAND["dim"])

        return Panel(
            content,
            title=f"[{BRAND['green']}]◈ Pipeline Configuration[/{BRAND['green']}]",
            border_style=BRAND["emerald"],
            padding=(1, 2),
        )

    def show(self, console: Console) -> tuple[str, str, str] | None:
        """Show the interactive config screen.

        Returns:
            Tuple of (mode, route, arbiter) on confirmation, or None if quit.
        """
        try:
            while True:
                # Clear and render
                console.clear()
                console.print(self._build_panel(console))

                key = _read_key()

                if key == "1":
                    self._mode_idx = (self._mode_idx + 1) % len(_MODE_CYCLE)
                elif key == "2":
                    self._route_idx = (self._route_idx + 1) % len(_ROUTE_CYCLE)
                elif key == "3":
                    self._arbiter_idx = (self._arbiter_idx + 1) % len(_ARBITER_CYCLE)
                elif key == "enter":
                    console.clear()
                    return (self.mode, self.route, self.arbiter)
                elif key in ("q", "escape"):
                    return None
        except KeyboardInterrupt:
            return None


# ── Pipeline Display ──────────────────────────────────────────────

# Stage display order
_STAGE_ORDER = ["architect", "implement", "refactor", "verify"]

# Parallel-mode phase display
_PARALLEL_PHASE_ORDER = ["fan_out", "cross_review", "voting", "synthesis", "arbiter"]

_PARALLEL_PHASE_LABELS = {
    "fan_out": "Fan-Out",
    "cross_review": "Cross-Review",
    "voting": "Voting",
    "synthesis": "Synthesis",
    "arbiter": "Arbiter",
}

# Maps ParallelOrchestrator event stage names → display phase keys
_PARALLEL_EVENT_MAP = {
    "parallel_fan_out": "fan_out",
    "parallel_cross_review": "cross_review",
    "parallel_synthesis": "synthesis",
    "parallel_synthesis_retry": "synthesis",
}

# Status symbols
_STATUS_SYMBOLS = {
    "pending": "○",
    "running": "◉",
    "done": "●",
    "error": "✗",
}


class PipelineDisplay:
    """Real-time Rich Live pipeline status display.

    Shows a compact logo, stage progress table with STATUS/TIME/COST,
    and a scrolling activity log. Updated via event listener callbacks
    from the PipelineEventEmitter.
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
        self._live: Live | None = None
        self._lock = threading.Lock()

        # Mode-aware phase setup
        self._is_parallel = (mode == "parallel")
        self._phase_order = _PARALLEL_PHASE_ORDER if self._is_parallel else _STAGE_ORDER

        # Stage/phase state
        self._stages: dict[str, dict] = {}
        for stage in self._phase_order:
            self._stages[stage] = {
                "status": "pending",
                "model": "",
                "start_time": None,
                "duration": None,
                "cost": None,
                "confidence": None,
            }

        # Arbiter state
        self._arbiter_active: str | None = None
        self._arbiter_verdicts: list[dict] = []

        # Activity log
        self._log: list[tuple[str, str, str]] = []  # (time, style, message)

        # Pipeline totals
        self._total_cost: float | None = None
        self._total_tokens: int | None = None
        self._pipeline_start = time.monotonic()
        self._pipeline_done = False
        self._halted = False
        self._halt_reason = ""

    def create_listener(self):
        """Return a sync callback for PipelineEventEmitter.

        The callback updates internal state that Rich Live reads
        on each refresh cycle.
        """
        def listener(event) -> None:
            with self._lock:
                etype = event.type if hasattr(event.type, "value") else str(event.type)

                if etype == "stage_started":
                    stage = event.data.get("stage", "")
                    model = event.data.get("model", "")
                    # Resolve parallel event names to display phase keys
                    phase = _PARALLEL_EVENT_MAP.get(stage, stage) if self._is_parallel else stage
                    if phase in self._stages:
                        self._stages[phase]["status"] = "running"
                        self._stages[phase]["model"] = model
                        self._stages[phase]["start_time"] = time.monotonic()
                    self._add_log(
                        BRAND["green"],
                        f"▸ {stage.title()} started" + (f" ({model})" if model else ""),
                    )

                elif etype == "stage_completed":
                    stage = event.data.get("stage", "")
                    duration = event.data.get("duration", 0)
                    cost = event.data.get("cost", 0)
                    confidence = event.data.get("confidence", 0)
                    model = event.data.get("model", "")
                    # Resolve parallel event names to display phase keys
                    phase = _PARALLEL_EVENT_MAP.get(stage, stage) if self._is_parallel else stage
                    if phase in self._stages:
                        self._stages[phase]["status"] = "done"
                        self._stages[phase]["duration"] = duration
                        # Accumulate cost across retries
                        prev = self._stages[phase]["cost"] or 0
                        self._stages[phase]["cost"] = prev + cost
                        self._stages[phase]["confidence"] = confidence
                        if model:
                            self._stages[phase]["model"] = model
                    total_cost = (self._stages.get(phase, {}).get("cost") or 0)
                    self._add_log(
                        BRAND["mint"],
                        f"● {stage.title()} done ({duration:.1f}s, ${total_cost:.4f})",
                    )

                elif etype == "consensus_vote":
                    if self._is_parallel and "voting" in self._stages:
                        winner = event.data.get("winner", "")
                        method = event.data.get("method", "")
                        self._stages["voting"]["status"] = "done"
                        self._stages["voting"]["duration"] = 0
                        label = winner or method or "resolved"
                        self._add_log(
                            BRAND["mint"],
                            f"● Voting done ({label})",
                        )

                elif etype == "arbiter_started":
                    stage = event.data.get("stage", "")
                    arbiter_model = event.data.get("arbiter_model", "")
                    self._arbiter_active = stage
                    if self._is_parallel and "arbiter" in self._stages:
                        self._stages["arbiter"]["status"] = "running"
                        self._stages["arbiter"]["model"] = arbiter_model
                        self._stages["arbiter"]["start_time"] = time.monotonic()
                    model_note = f" ({arbiter_model})" if arbiter_model else ""
                    self._add_log(
                        BRAND["gold"],
                        f"⚖ Arbiter reviewing {stage}{model_note}",
                    )

                elif etype == "arbiter_verdict":
                    stage = event.data.get("stage", "")
                    verdict = event.data.get("verdict", "")
                    confidence = event.data.get("confidence", 0)
                    reasoning = event.data.get("reasoning_preview", "")
                    arbiter_model = event.data.get("arbiter_model", "")
                    self._arbiter_active = None
                    if self._is_parallel and "arbiter" in self._stages:
                        self._stages["arbiter"]["status"] = "done"
                        self._stages["arbiter"]["confidence"] = confidence
                        if self._stages["arbiter"]["start_time"] is not None:
                            self._stages["arbiter"]["duration"] = (
                                time.monotonic() - self._stages["arbiter"]["start_time"]
                            )
                    self._arbiter_verdicts.append({
                        "stage": stage,
                        "verdict": verdict,
                        "confidence": confidence,
                        "arbiter_model": arbiter_model,
                    })
                    style = _verdict_color(verdict)
                    # Main verdict line
                    self._add_log(
                        style,
                        f"⚖ {stage}: {verdict.upper()} (conf={confidence:.2f})",
                    )
                    # One-line reasoning summary for non-APPROVE verdicts
                    if reasoning and verdict.lower() != "approve":
                        # Extract first meaningful sentence
                        summary = reasoning.split("\n")[0].strip()[:80]
                        if summary:
                            self._add_log(BRAND["dim"], f"  └ {summary}")

                elif etype == "retry_triggered":
                    stage = event.data.get("stage", "")
                    retry_number = event.data.get("retry_number", 1)
                    if stage in self._stages:
                        self._stages[stage]["status"] = "running"
                    self._add_log(
                        BRAND["amber"],
                        f"↻ Retrying {stage} (attempt {retry_number})",
                    )

                elif etype == "pipeline_completed":
                    self._total_cost = event.data.get("total_cost", 0)
                    self._total_tokens = event.data.get("total_tokens", 0)
                    self._pipeline_done = True
                    self._add_log(
                        BRAND["green"],
                        f"✓ Pipeline completed (${self._total_cost:.4f})",
                    )

                elif etype == "pipeline_halted":
                    stage = event.data.get("stage", "")
                    reason = event.data.get("reason", "")
                    self._halted = True
                    self._halt_reason = reason
                    self._pipeline_done = True
                    self._add_log(BRAND["red"], f"✗ HALTED at {stage}: {reason[:60]}")

                elif etype == "model_fallback":
                    stage = event.data.get("stage", "")
                    original = event.data.get("original_model", "")
                    fallback = event.data.get("fallback_model", "")
                    reason = event.data.get("reason", "")
                    self._add_log(
                        BRAND["amber"],
                        f"⚠ {stage.title()}: {original} unavailable, using {fallback}",
                    )
                    if reason:
                        self._add_log(BRAND["dim"], f"  └ {reason[:80]}")

                elif etype == "arbiter_skipped":
                    stage = event.data.get("stage", "")
                    self._arbiter_active = None
                    self._add_log(
                        BRAND["dim"],
                        f"⚖ {stage}: arbiter skipped (no models available)",
                    )

                elif etype == "error":
                    error = event.data.get("error", "Unknown error")
                    self._add_log(BRAND["red"], f"✗ Error: {str(error)[:80]}")

        return listener

    def __rich__(self) -> Table:
        """Rich renderable protocol — called by Live on each refresh cycle."""
        return self._build_display()

    def __enter__(self):
        """Context manager support: start the Live display."""
        self.start()
        return self

    def __exit__(self, *_exc):
        """Context manager support: stop the Live display."""
        self.stop()

    def _add_log(self, style: str, message: str) -> None:
        """Add an entry to the activity log."""
        elapsed = time.monotonic() - self._pipeline_start
        ts = f"{elapsed:6.1f}s"
        self._log.append((ts, style, message))
        # Keep last 15 entries
        if len(self._log) > 15:
            self._log = self._log[-15:]

    def _build_display(self) -> Table:
        """Build the full display layout as a Rich Table (outer grid)."""
        outer = Table.grid(expand=True)
        outer.add_row(self._build_status_panel())
        outer.add_row(self._build_log_panel())
        return outer

    def _build_status_panel(self) -> Panel:
        """Build the pipeline status panel with compact logo + stage table."""
        content = Table.grid(padding=(0, 0))

        # Compact logo
        logo = render_compact_logo(self._console, self._mode, self._route, self._arbiter)
        content.add_row(logo)
        content.add_row(Text(""))

        # Stage table
        stage_table = Table(
            show_header=True,
            header_style="bold",
            box=None,
            padding=(0, 2),
            expand=True,
        )
        col_header = "Phase" if self._is_parallel else "Stage"
        stage_table.add_column(col_header, style="bold")
        stage_table.add_column("Status")
        stage_table.add_column("Model", style=BRAND["dim"])
        stage_table.add_column("Time", justify="right")
        stage_table.add_column("Cost", justify="right")

        with self._lock:
            for stage in self._phase_order:
                info = self._stages[stage]
                status = info["status"]

                # Status indicator
                symbol = _STATUS_SYMBOLS.get(status, "?")
                if status == "running":
                    status_text = Text(f"{symbol} Running", style=BRAND["amber"])
                elif status == "done":
                    status_text = Text(f"{symbol} Done", style=BRAND["green"])
                elif status == "error":
                    status_text = Text(f"{symbol} Error", style=BRAND["red"])
                else:
                    status_text = Text(f"{symbol} Pending", style=BRAND["pending"])

                # Model
                model = info.get("model", "") or ""

                # Time
                if info["duration"] is not None:
                    time_str = f"{info['duration']:.1f}s"
                elif info["start_time"] is not None:
                    elapsed = time.monotonic() - info["start_time"]
                    time_str = f"{elapsed:.1f}s..."
                else:
                    time_str = "-"

                # Cost
                cost_str = f"${info['cost']:.4f}" if info["cost"] is not None else "-"

                # Display name: use parallel labels or title-case
                if self._is_parallel:
                    display_name = _PARALLEL_PHASE_LABELS.get(stage, stage.title())
                else:
                    display_name = stage.title()

                stage_table.add_row(
                    display_name,
                    status_text,
                    model,
                    time_str,
                    cost_str,
                )

            # Arbiter row if active (sequential mode only — parallel has its own row)
            if self._arbiter_active and not self._is_parallel:
                stage_table.add_row(
                    "Arbiter",
                    Text(f"◉ Reviewing {self._arbiter_active}", style=BRAND["gold"]),
                    "",
                    "",
                    "",
                )

        content.add_row(stage_table)

        return Panel(
            content,
            title=f"[{BRAND['green']}]◈ Pipeline Status[/{BRAND['green']}]",
            border_style=BRAND["emerald"],
            padding=(1, 1),
        )

    def _build_log_panel(self) -> Panel:
        """Build the activity log panel."""
        log_text = Text()

        with self._lock:
            entries = list(self._log)

        if not entries:
            log_text.append("  Waiting for pipeline events...", style=BRAND["dim"])
        else:
            for i, (ts, style, msg) in enumerate(entries):
                if i > 0:
                    log_text.append("\n")
                log_text.append(f"  {ts}  ", style=BRAND["dim"])
                log_text.append(msg, style=style)

        return Panel(
            log_text,
            title=f"[{BRAND['dim']}]Activity Log[/{BRAND['dim']}]",
            border_style=BRAND["dim"],
            padding=(0, 1),
        )

    def start(self) -> Live:
        """Start the Rich Live display. Returns the Live context manager."""
        self._pipeline_start = time.monotonic()
        self._live = Live(
            self,
            console=self._console,
            refresh_per_second=4,
            transient=True,
        )
        self._live.start()
        return self._live

    def stop(self) -> None:
        """Stop the Rich Live display."""
        if self._live:
            self._live.stop()
            self._live = None

    def update(self) -> None:
        """Refresh the Live display with current state."""
        if self._live:
            self._live.update(self._build_display())


def _verdict_color(verdict: str) -> str:
    """Map a verdict string to a brand color."""
    return {
        "approve": BRAND["green"],
        "flag": BRAND["amber"],
        "reject": BRAND["red"],
        "halt": BRAND["red"],
    }.get(verdict.lower(), BRAND["dim"])


# ── Completion Summary ────────────────────────────────────────────


class CompletionSummary:
    """Post-run completion summary with interactive actions.

    Shows a summary panel with key metrics and offers interactive
    post-completion actions (view summary, export, rerun, quit).
    """

    def __init__(
        self,
        console: Console,
        result: object,
        output_dir: str,
    ) -> None:
        self._console = console
        self._result = result
        self._output_dir = output_dir

    def _build_panel(self) -> Panel:
        """Build the completion summary panel."""
        r = self._result
        content = Text()

        # Status banner
        if getattr(r, "halted", False):
            content.append("  ✗ PIPELINE HALTED", style=f"bold {BRAND['red']}")
            content.append("\n")
            reason = getattr(r, "halt_reason", "")
            if reason:
                content.append(f"  {reason[:100]}", style=BRAND["dim"])
                content.append("\n")
        elif getattr(r, "success", False):
            content.append("  ✓ PIPELINE COMPLETED SUCCESSFULLY", style=f"bold {BRAND['green']}")
            content.append("\n")
        else:
            content.append("  ✗ PIPELINE FAILED", style=f"bold {BRAND['red']}")
            content.append("\n")

        content.append("\n")

        # Metrics grid
        duration = getattr(r, "duration_seconds", 0)
        total_cost = getattr(r, "total_cost", 0)
        total_tokens = getattr(r, "total_tokens", 0)
        session_id = getattr(r, "session_id", "") or "n/a"
        stages_count = len(getattr(r, "stages", {}))
        reviews_count = len(getattr(r, "arbiter_reviews", []))

        content.append("  Duration:  ", style="bold")
        content.append(f"{duration:.1f}s", style="white")
        content.append("    Cost:  ", style="bold")
        content.append(f"${total_cost:.4f}", style=BRAND["gold"])
        content.append("    Tokens:  ", style="bold")
        content.append(f"{total_tokens:,}", style="white")
        content.append("\n")

        content.append("  Session:   ", style="bold")
        content.append(session_id[:20], style=BRAND["dim"])
        content.append("    Stages:  ", style="bold")
        content.append(str(stages_count), style="white")
        content.append("    Reviews:  ", style="bold")
        content.append(str(reviews_count), style="white")
        content.append("\n")

        content.append("  Output:    ", style="bold")
        content.append(f"{self._output_dir}/", style=BRAND["dim"])
        content.append("\n\n")

        # Arbiter verdicts
        reviews = getattr(r, "arbiter_reviews", [])
        if reviews:
            content.append("  Verdicts:  ", style="bold")
            for i, review in enumerate(reviews):
                if i > 0:
                    content.append("  ")
                v = review.verdict
                verdict_val = v.value if hasattr(v, "value") else str(v)
                color = _verdict_color(verdict_val)
                sr = review.stage_reviewed
                stage_val = sr.value if hasattr(sr, "value") else str(sr)
                content.append(f"{stage_val}=", style=BRAND["dim"])
                content.append(verdict_val.upper(), style=f"bold {color}")
            content.append("\n\n")

        # Actions
        content.append("  [v] ", style=BRAND["mint"])
        content.append("View summary", style="bold")
        content.append("  [e] ", style=BRAND["mint"])
        content.append("Export session", style="bold")
        content.append("  [r] ", style=BRAND["mint"])
        content.append("Rerun", style="bold")
        content.append("  [Enter/q] ", style=BRAND["dim"])
        content.append("Exit", style=BRAND["dim"])

        border_color = BRAND["green"] if getattr(r, "success", False) else BRAND["red"]

        return Panel(
            content,
            title=f"[{border_color}]◈ Pipeline Complete[/{border_color}]",
            border_style=border_color,
            padding=(1, 1),
        )

    def show(self) -> str | None:
        """Show the completion summary and handle interactive keys.

        Returns:
            'rerun' if user chose to rerun, None otherwise.
        """
        self._console.print()
        self._console.print(self._build_panel())

        try:
            while True:
                key = _read_key()

                if key == "v":
                    self._view_summary()
                elif key == "e":
                    self._export_session()
                    return None
                elif key == "r":
                    return "rerun"
                elif key in ("q", "enter", "escape"):
                    return None
        except (KeyboardInterrupt, EOFError):
            return None

    def _view_summary(self) -> None:
        """Render summary.md from the output directory."""
        from pathlib import Path

        from rich.markdown import Markdown

        summary_path = Path(self._output_dir) / "summary.md"
        if summary_path.exists():
            content = summary_path.read_text(encoding="utf-8")
            self._console.print()
            self._console.print(Panel(
                Markdown(content),
                title=f"[{BRAND['green']}]Summary[/{BRAND['green']}]",
                border_style=BRAND["dim"],
            ))
        else:
            dim = BRAND["dim"]
            self._console.print(
                f"\n  [{dim}]No summary.md found in"
                f" {self._output_dir}/[/{dim}]"
            )

        # Re-show the panel
        self._console.print()
        self._console.print(self._build_panel())

    def _export_session(self) -> None:
        """Export the session as JSON."""
        session_id = getattr(self._result, "session_id", "")
        if not session_id:
            dim = BRAND["dim"]
            self._console.print(
                f"\n  [{dim}]No session ID available"
                f" for export.[/{dim}]"
            )
            return

        from pathlib import Path

        output_path = Path(self._output_dir) / "session.json"
        if output_path.exists():
            green = BRAND["green"]
            self._console.print(
                f"\n  [{green}]Session exported"
                f" to:[/{green}] {output_path}"
            )
        else:
            dim = BRAND["dim"]
            self._console.print(
                f"\n  [{dim}]Session file not found."
                f" Run output may not have been"
                f" written yet.[/{dim}]"
            )

"""Tests for the interactive CLI display components.

Covers brand constants, logo rendering, ConfigScreen, PipelineDisplay,
and CompletionSummary classes.
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

from rich.console import Console
from rich.text import Text

from triad.cli_display import (
    BRAND,
    COMPACT_LOGO_TEMPLATE,
    FULL_LOGO,
    VERSION,
    CompletionSummary,
    ConfigScreen,
    PipelineDisplay,
    _verdict_color,
    render_compact_logo,
    render_full_logo,
)
from triad.dashboard.events import EventType, PipelineEvent

# ── Brand Constants ──────────────────────────────────────────────


class TestBrandConstants:
    def test_brand_has_required_colors(self):
        """BRAND dict contains all required color keys."""
        required = {"mint", "emerald", "lime", "gold", "green", "dim", "pending", "amber", "red"}
        assert required.issubset(BRAND.keys())

    def test_brand_values_are_hex(self):
        """All BRAND values are valid hex color strings."""
        for key, value in BRAND.items():
            assert value.startswith("#"), f"BRAND[{key}] = {value} is not a hex color"
            assert len(value) == 7, f"BRAND[{key}] = {value} is not 7 chars"

    def test_version_string(self):
        """VERSION is a valid semver-like string."""
        parts = VERSION.split(".")
        assert len(parts) == 3
        for part in parts:
            assert part.isdigit()


# ── Full Logo ────────────────────────────────────────────────────


class TestFullLogo:
    def test_full_logo_contains_triad(self):
        """FULL_LOGO ASCII art contains TRIAD block letters."""
        assert "TRIAD" in FULL_LOGO or "█" in FULL_LOGO or "▀" in FULL_LOGO

    def test_full_logo_contains_triangle_nodes(self):
        """FULL_LOGO contains the triangle node characters."""
        assert "◆" in FULL_LOGO
        assert "◈" in FULL_LOGO

    def test_render_full_logo(self):
        """render_full_logo prints without error."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        render_full_logo(console)
        # Just verify it completes without error


# ── Compact Logo ─────────────────────────────────────────────────


class TestCompactLogo:
    def test_compact_logo_template_has_placeholders(self):
        """COMPACT_LOGO_TEMPLATE has version/mode/route/arbiter placeholders."""
        assert "{version}" in COMPACT_LOGO_TEMPLATE
        assert "{mode}" in COMPACT_LOGO_TEMPLATE
        assert "{route}" in COMPACT_LOGO_TEMPLATE
        assert "{arbiter}" in COMPACT_LOGO_TEMPLATE

    def test_render_compact_logo_returns_text(self):
        """render_compact_logo returns a Rich Text object."""
        console = Console(force_terminal=True, width=120)
        result = render_compact_logo(console, "sequential", "hybrid", "bookend")
        assert isinstance(result, Text)

    def test_render_compact_logo_contains_values(self):
        """render_compact_logo output includes the formatted values."""
        console = Console(force_terminal=True, width=120)
        result = render_compact_logo(console, "sequential", "hybrid", "bookend")
        plain = result.plain
        assert "Sequential" in plain
        assert "Hybrid" in plain
        assert "Bookend" in plain
        assert VERSION in plain

    def test_render_compact_logo_underscore_handling(self):
        """render_compact_logo handles underscore-separated values."""
        console = Console(force_terminal=True, width=120)
        result = render_compact_logo(console, "parallel", "quality_first", "final_only")
        plain = result.plain
        assert "Quality First" in plain
        assert "Final Only" in plain


# ── Verdict Color ────────────────────────────────────────────────


class TestVerdictColor:
    def test_approve_is_green(self):
        assert _verdict_color("approve") == BRAND["green"]

    def test_flag_is_amber(self):
        assert _verdict_color("flag") == BRAND["amber"]

    def test_reject_is_red(self):
        assert _verdict_color("reject") == BRAND["red"]

    def test_halt_is_red(self):
        assert _verdict_color("halt") == BRAND["red"]

    def test_unknown_is_dim(self):
        assert _verdict_color("unknown") == BRAND["dim"]

    def test_case_insensitive(self):
        assert _verdict_color("APPROVE") == BRAND["green"]
        assert _verdict_color("Flag") == BRAND["amber"]


# ── ConfigScreen ─────────────────────────────────────────────────


class TestConfigScreen:
    def _make_config(self):
        """Create a mock config with required attributes."""
        config = MagicMock()
        config.pipeline_mode.value = "sequential"
        config.routing_strategy.value = "hybrid"
        config.arbiter_mode.value = "bookend"
        return config

    def test_initial_state(self):
        """ConfigScreen initializes with config defaults."""
        config = self._make_config()
        screen = ConfigScreen("Build a REST API", config, {})
        assert screen.mode == "sequential"
        assert screen.route == "hybrid"
        assert screen.arbiter == "bookend"

    def test_mode_property(self):
        """mode property returns the current mode value."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        assert screen.mode in ("sequential", "parallel", "debate")

    def test_route_property(self):
        """route property returns the current route value."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        assert screen.route in ("hybrid", "quality_first", "cost_optimized", "speed_first")

    def test_arbiter_property(self):
        """arbiter property returns the current arbiter value."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        assert screen.arbiter in ("bookend", "off", "final_only", "full")

    @patch("triad.cli_display._read_key", return_value="enter")
    def test_show_returns_tuple_on_enter(self, mock_key):
        """show() returns (mode, route, arbiter) on Enter."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = screen.show(console)
        assert result == ("sequential", "hybrid", "bookend")

    @patch("triad.cli_display._read_key", return_value="q")
    def test_show_returns_none_on_quit(self, mock_key):
        """show() returns None when user presses q."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = screen.show(console)
        assert result is None

    @patch("triad.cli_display._read_key", side_effect=["1", "enter"])
    def test_mode_cycling(self, mock_key):
        """Pressing 1 cycles the mode."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = screen.show(console)
        # sequential -> parallel
        assert result[0] == "parallel"

    @patch("triad.cli_display._read_key", side_effect=["2", "enter"])
    def test_route_cycling(self, mock_key):
        """Pressing 2 cycles the route."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = screen.show(console)
        # hybrid -> quality_first
        assert result[1] == "quality_first"

    @patch("triad.cli_display._read_key", side_effect=["3", "enter"])
    def test_arbiter_cycling(self, mock_key):
        """Pressing 3 cycles the arbiter."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = screen.show(console)
        # bookend -> off
        assert result[2] == "off"

    @patch("triad.cli_display._read_key", side_effect=KeyboardInterrupt)
    def test_show_returns_none_on_ctrl_c(self, mock_key):
        """show() returns None on Ctrl+C."""
        config = self._make_config()
        screen = ConfigScreen("test", config, {})
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = screen.show(console)
        assert result is None


# ── PipelineDisplay ──────────────────────────────────────────────


class TestPipelineDisplay:
    def test_create_listener_returns_callable(self):
        """create_listener returns a callable."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        assert callable(listener)

    def test_listener_handles_stage_started(self):
        """Listener updates stage status on STAGE_STARTED."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        event = PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "claude-opus"},
        )
        listener(event)

        assert display._stages["architect"]["status"] == "running"
        assert display._stages["architect"]["model"] == "claude-opus"

    def test_listener_handles_stage_completed(self):
        """Listener updates stage status on STAGE_COMPLETED."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        event = PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={
                "stage": "architect",
                "model": "claude-opus",
                "duration": 12.5,
                "cost": 0.05,
                "confidence": 0.85,
            },
        )
        listener(event)

        assert display._stages["architect"]["status"] == "done"
        assert display._stages["architect"]["duration"] == 12.5
        assert display._stages["architect"]["cost"] == 0.05

    def test_listener_handles_arbiter_verdict(self):
        """Listener records arbiter verdict."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        event = PipelineEvent(
            type=EventType.ARBITER_VERDICT,
            data={
                "stage": "architect",
                "verdict": "approve",
                "confidence": 0.9,
            },
        )
        listener(event)

        assert len(display._arbiter_verdicts) == 1
        assert display._arbiter_verdicts[0]["verdict"] == "approve"

    def test_listener_handles_pipeline_completed(self):
        """Listener records pipeline completion."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        event = PipelineEvent(
            type=EventType.PIPELINE_COMPLETED,
            data={"total_cost": 0.25, "total_tokens": 5000},
        )
        listener(event)

        assert display._pipeline_done is True
        assert display._total_cost == 0.25

    def test_listener_handles_pipeline_halted(self):
        """Listener records pipeline halt."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        event = PipelineEvent(
            type=EventType.PIPELINE_HALTED,
            data={"stage": "verify", "reason": "Critical bug found"},
        )
        listener(event)

        assert display._halted is True
        assert display._pipeline_done is True

    def test_listener_handles_retry(self):
        """Listener handles retry events."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        # First mark stage as done
        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "duration": 10, "cost": 0.05},
        ))

        # Then retry
        listener(PipelineEvent(
            type=EventType.RETRY_TRIGGERED,
            data={"stage": "architect", "retry_number": 1},
        ))

        assert display._stages["architect"]["status"] == "running"

    def test_activity_log_max_entries(self):
        """Activity log is capped at 15 entries."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        for i in range(20):
            listener(PipelineEvent(
                type=EventType.STAGE_STARTED,
                data={"stage": "architect", "model": f"model-{i}"},
            ))

        assert len(display._log) == 15

    def test_start_stop(self):
        """start/stop lifecycle works without error."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")

        display.start()
        assert display._live is not None
        display.stop()
        assert display._live is None

    def test_build_display_produces_table(self):
        """_build_display returns a renderable."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        result = display._build_display()
        assert result is not None


# ── CompletionSummary ────────────────────────────────────────────


class TestCompletionSummary:
    def _make_result(self, **overrides):
        """Create a mock pipeline result."""
        result = MagicMock()
        result.success = overrides.get("success", True)
        result.halted = overrides.get("halted", False)
        result.halt_reason = overrides.get("halt_reason", "")
        result.duration_seconds = overrides.get("duration_seconds", 45.0)
        result.total_cost = overrides.get("total_cost", 0.25)
        result.total_tokens = overrides.get("total_tokens", 6000)
        result.session_id = overrides.get("session_id", "test-session-id")
        result.stages = overrides.get("stages", {"arch": None, "impl": None})
        result.arbiter_reviews = overrides.get("arbiter_reviews", [])
        return result

    def test_build_panel_success(self):
        """_build_panel renders success state."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "triad-output")
        panel = summary._build_panel()
        assert panel is not None

    def test_build_panel_halted(self):
        """_build_panel renders halted state."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result(halted=True, success=False, halt_reason="Bug found")
        summary = CompletionSummary(console, result, "triad-output")
        panel = summary._build_panel()
        assert panel is not None

    def test_build_panel_failed(self):
        """_build_panel renders failure state."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result(success=False)
        summary = CompletionSummary(console, result, "triad-output")
        panel = summary._build_panel()
        assert panel is not None

    @patch("triad.cli_display._read_key", return_value="q")
    def test_show_returns_none_on_quit(self, mock_key):
        """show() returns None when user presses q."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "triad-output")
        action = summary.show()
        assert action is None

    @patch("triad.cli_display._read_key", return_value="enter")
    def test_show_returns_none_on_enter(self, mock_key):
        """show() returns None when user presses Enter."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "triad-output")
        action = summary.show()
        assert action is None

    @patch("triad.cli_display._read_key", return_value="r")
    def test_show_returns_rerun(self, mock_key):
        """show() returns 'rerun' when user presses r."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "triad-output")
        action = summary.show()
        assert action == "rerun"

    @patch("triad.cli_display._read_key", side_effect=KeyboardInterrupt)
    def test_show_returns_none_on_ctrl_c(self, mock_key):
        """show() returns None on KeyboardInterrupt."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "triad-output")
        action = summary.show()
        assert action is None

    def test_build_panel_with_arbiter_reviews(self):
        """_build_panel shows arbiter verdicts."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        review = MagicMock()
        review.verdict.value = "approve"
        review.stage_reviewed.value = "architect"
        result = self._make_result(arbiter_reviews=[review])
        summary = CompletionSummary(console, result, "triad-output")
        panel = summary._build_panel()
        assert panel is not None

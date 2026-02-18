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
    _DEBATE_PHASE_LABELS,
    _DEBATE_PHASE_ORDER,
    _PARALLEL_PHASE_LABELS,
    _PARALLEL_PHASE_ORDER,
    _STAGE_ORDER,
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


# ── Parallel PipelineDisplay ─────────────────────────────────────


class TestParallelPipelineDisplay:
    """Tests for PipelineDisplay in parallel mode."""

    def _make_display(self):
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        return PipelineDisplay(console, "parallel", "hybrid", "bookend")

    def test_parallel_initializes_phase_rows(self):
        """Parallel mode initializes _stages with all parallel phase keys."""
        display = self._make_display()
        assert set(display._stages.keys()) == set(_PARALLEL_PHASE_ORDER)
        for phase in _PARALLEL_PHASE_ORDER:
            assert display._stages[phase]["status"] == "pending"

    def test_parallel_fan_out_event_updates_row(self):
        """stage_started with parallel_fan_out sets fan_out to running."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "parallel_fan_out", "model": "gpt-4"},
        ))

        assert display._stages["fan_out"]["status"] == "running"
        assert display._stages["fan_out"]["model"] == "gpt-4"

    def test_parallel_fan_out_completed(self):
        """stage_completed with parallel_fan_out sets fan_out to done."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "parallel_fan_out", "duration": 5.0, "cost": 0.02},
        ))

        assert display._stages["fan_out"]["status"] == "done"
        assert display._stages["fan_out"]["duration"] == 5.0
        assert display._stages["fan_out"]["cost"] == 0.02

    def test_parallel_cross_review_events(self):
        """Cross-review start/complete updates the cross_review row."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "parallel_cross_review", "model": "claude-sonnet"},
        ))
        assert display._stages["cross_review"]["status"] == "running"

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "parallel_cross_review", "duration": 8.0, "cost": 0.03},
        ))
        assert display._stages["cross_review"]["status"] == "done"
        assert display._stages["cross_review"]["duration"] == 8.0

    def test_parallel_consensus_vote_updates_voting(self):
        """consensus_vote event sets voting row to done with winner info."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.CONSENSUS_VOTE,
            data={"winner": "agent_1", "method": "majority"},
        ))

        assert display._stages["voting"]["status"] == "done"

    def test_parallel_synthesis_events(self):
        """Synthesis start/complete updates the synthesis row."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "parallel_synthesis", "model": "claude-opus"},
        ))
        assert display._stages["synthesis"]["status"] == "running"

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "parallel_synthesis", "duration": 15.0, "cost": 0.08},
        ))
        assert display._stages["synthesis"]["status"] == "done"

    def test_parallel_arbiter_updates_row(self):
        """arbiter_started/verdict updates the arbiter row in parallel mode."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.ARBITER_STARTED,
            data={"stage": "verify", "arbiter_model": "claude-opus"},
        ))
        assert display._stages["arbiter"]["status"] == "running"
        assert display._stages["arbiter"]["model"] == "claude-opus"

        listener(PipelineEvent(
            type=EventType.ARBITER_VERDICT,
            data={"stage": "verify", "verdict": "approve", "confidence": 0.95},
        ))
        assert display._stages["arbiter"]["status"] == "done"
        assert display._stages["arbiter"]["confidence"] == 0.95

    def test_parallel_build_display_renders(self):
        """_build_display() produces output without errors in parallel mode."""
        display = self._make_display()
        result = display._build_display()
        assert result is not None

    def test_parallel_phase_labels_in_output(self):
        """Rendered table uses parallel phase labels like Fan-Out, Cross-Review."""
        display = self._make_display()
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        # Render to string
        with console.capture() as capture:
            console.print(display._build_status_panel())
        output = capture.get()

        for label in _PARALLEL_PHASE_LABELS.values():
            assert label in output, f"Expected '{label}' in rendered output"
        assert "Phase" in output

    def test_sequential_mode_unchanged(self):
        """Sequential mode still uses the 4 original stages."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert set(display._stages.keys()) == set(_STAGE_ORDER)
        assert display._is_parallel is False


# ── Debate PipelineDisplay ───────────────────────────────────────


class TestDebatePipelineDisplay:
    """Tests for PipelineDisplay in debate mode."""

    def _make_display(self):
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        return PipelineDisplay(console, "debate", "hybrid", "full")

    def test_debate_initializes_phase_rows(self):
        """Debate mode initializes _stages with all debate phase keys."""
        display = self._make_display()
        assert set(display._stages.keys()) == set(_DEBATE_PHASE_ORDER)
        for phase in _DEBATE_PHASE_ORDER:
            assert display._stages[phase]["status"] == "pending"

    def test_debate_proposals_event_updates_row(self):
        """stage_started with debate_proposals sets proposals to running."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "debate_proposals", "model": "3 debaters"},
        ))

        assert display._stages["proposals"]["status"] == "running"
        assert display._stages["proposals"]["model"] == "3 debaters"

    def test_debate_proposals_completed(self):
        """stage_completed with debate_proposals sets proposals to done."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "debate_proposals", "duration": 10.0, "cost": 0.05},
        ))

        assert display._stages["proposals"]["status"] == "done"
        assert display._stages["proposals"]["duration"] == 10.0

    def test_debate_rebuttals_events(self):
        """Rebuttal start/complete updates the rebuttals row."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "debate_rebuttals", "model": "3 rebuttals"},
        ))
        assert display._stages["rebuttals"]["status"] == "running"

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "debate_rebuttals", "duration": 12.0, "cost": 0.06},
        ))
        assert display._stages["rebuttals"]["status"] == "done"

    def test_debate_final_arguments_events(self):
        """Final arguments start/complete updates the final_args row."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "debate_final_arguments", "model": "3 models"},
        ))
        assert display._stages["final_args"]["status"] == "running"

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "debate_final_arguments", "duration": 8.0, "cost": 0.04},
        ))
        assert display._stages["final_args"]["status"] == "done"

    def test_debate_judgment_events(self):
        """Judgment start/complete updates the judgment row."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "debate_judgment", "model": "gpt-4o"},
        ))
        assert display._stages["judgment"]["status"] == "running"

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "debate_judgment", "model": "gpt-4o", "duration": 15.0, "cost": 0.08},
        ))
        assert display._stages["judgment"]["status"] == "done"

    def test_debate_arbiter_updates_row(self):
        """arbiter_started/verdict updates the arbiter row in debate mode."""
        display = self._make_display()
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.ARBITER_STARTED,
            data={"stage": "verify", "arbiter_model": "claude-opus"},
        ))
        assert display._stages["arbiter"]["status"] == "running"
        assert display._stages["arbiter"]["model"] == "claude-opus"

        listener(PipelineEvent(
            type=EventType.ARBITER_VERDICT,
            data={"stage": "verify", "verdict": "approve", "confidence": 0.9},
        ))
        assert display._stages["arbiter"]["status"] == "done"

    def test_debate_phase_labels_in_output(self):
        """Rendered table uses debate phase labels like Position Papers, Rebuttals."""
        display = self._make_display()
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        with console.capture() as capture:
            console.print(display._build_status_panel())
        output = capture.get()

        for label in _DEBATE_PHASE_LABELS.values():
            assert label in output, f"Expected '{label}' in rendered output"
        assert "Phase" in output

    def test_debate_build_display_renders(self):
        """_build_display() produces output without errors in debate mode."""
        display = self._make_display()
        result = display._build_display()
        assert result is not None

    def test_sequential_mode_still_unchanged(self):
        """Sequential mode is unaffected by debate additions."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        display = PipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert set(display._stages.keys()) == set(_STAGE_ORDER)
        assert display._is_parallel is False
        assert display._is_debate is False


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
        summary = CompletionSummary(console, result, "crtx-output")
        panel = summary._build_panel()
        assert panel is not None

    def test_build_panel_halted(self):
        """_build_panel renders halted state."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result(halted=True, success=False, halt_reason="Bug found")
        summary = CompletionSummary(console, result, "crtx-output")
        panel = summary._build_panel()
        assert panel is not None

    def test_build_panel_failed(self):
        """_build_panel renders failure state."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result(success=False)
        summary = CompletionSummary(console, result, "crtx-output")
        panel = summary._build_panel()
        assert panel is not None

    @patch("triad.cli_display._read_key", return_value="q")
    def test_show_returns_none_on_quit(self, mock_key):
        """show() returns None when user presses q."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "crtx-output")
        action = summary.show()
        assert action is None

    @patch("triad.cli_display._read_key", return_value="enter")
    def test_show_returns_none_on_enter(self, mock_key):
        """show() returns None when user presses Enter."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "crtx-output")
        action = summary.show()
        assert action is None

    @patch("triad.cli_display._read_key", return_value="r")
    def test_show_returns_rerun(self, mock_key):
        """show() returns 'rerun' when user presses r."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "crtx-output")
        action = summary.show()
        assert action == "rerun"

    @patch("triad.cli_display._read_key", side_effect=KeyboardInterrupt)
    def test_show_returns_none_on_ctrl_c(self, mock_key):
        """show() returns None on KeyboardInterrupt."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        result = self._make_result()
        summary = CompletionSummary(console, result, "crtx-output")
        action = summary.show()
        assert action is None

    def test_build_panel_with_arbiter_reviews(self):
        """_build_panel shows arbiter verdicts."""
        console = Console(file=MagicMock(), force_terminal=True, width=120)
        review = MagicMock()
        review.verdict.value = "approve"
        review.stage_reviewed.value = "architect"
        result = self._make_result(arbiter_reviews=[review])
        summary = CompletionSummary(console, result, "crtx-output")
        panel = summary._build_panel()
        assert panel is not None

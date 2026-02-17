"""Tests for the interactive REPL.

Covers command dispatch, session state management, and exit handling.
"""

from __future__ import annotations

from unittest.mock import patch

from triad.repl import _VALID_ARBITERS, _VALID_MODES, _VALID_ROUTES, TriadREPL

# ── Initial State ────────────────────────────────────────────────


class TestREPLState:
    def test_default_state(self):
        """REPL initializes with expected defaults."""
        repl = TriadREPL()
        assert repl.mode == "sequential"
        assert repl.route == "hybrid"
        assert repl.arbiter == "bookend"


# ── Session State Commands ───────────────────────────────────────


class TestStateCommands:
    def test_set_mode_valid(self):
        """Setting mode to a valid value updates the state."""
        repl = TriadREPL()
        repl._set_state("mode", "parallel")
        assert repl.mode == "parallel"

    def test_set_mode_invalid(self):
        """Setting mode to an invalid value keeps the old value."""
        repl = TriadREPL()
        repl._set_state("mode", "invalid")
        assert repl.mode == "sequential"  # unchanged

    def test_set_route_valid(self):
        """Setting route to a valid value updates the state."""
        repl = TriadREPL()
        repl._set_state("route", "quality_first")
        assert repl.route == "quality_first"

    def test_set_route_invalid(self):
        """Setting route to an invalid value keeps the old value."""
        repl = TriadREPL()
        repl._set_state("route", "invalid")
        assert repl.route == "hybrid"

    def test_set_arbiter_valid(self):
        """Setting arbiter to a valid value updates the state."""
        repl = TriadREPL()
        repl._set_state("arbiter", "full")
        assert repl.arbiter == "full"

    def test_set_arbiter_invalid(self):
        """Setting arbiter to an invalid value keeps the old value."""
        repl = TriadREPL()
        repl._set_state("arbiter", "invalid")
        assert repl.arbiter == "bookend"

    def test_set_mode_no_value_shows_current(self):
        """Setting mode with no value shows the current value."""
        repl = TriadREPL()
        # Should not error; just shows current value
        repl._set_state("mode", "")
        assert repl.mode == "sequential"

    def test_all_modes_accepted(self):
        """All valid modes are accepted."""
        repl = TriadREPL()
        for mode in _VALID_MODES:
            repl._set_state("mode", mode)
            assert repl.mode == mode

    def test_all_routes_accepted(self):
        """All valid routes are accepted."""
        repl = TriadREPL()
        for route in _VALID_ROUTES:
            repl._set_state("route", route)
            assert repl.route == route

    def test_all_arbiters_accepted(self):
        """All valid arbiter modes are accepted."""
        repl = TriadREPL()
        for arb in _VALID_ARBITERS:
            repl._set_state("arbiter", arb)
            assert repl.arbiter == arb

    def test_case_insensitive(self):
        """State commands are case-insensitive."""
        repl = TriadREPL()
        repl._set_state("mode", "PARALLEL")
        assert repl.mode == "parallel"

        repl._set_state("route", "Quality_First")
        assert repl.route == "quality_first"


# ── Command Dispatch ─────────────────────────────────────────────


class TestDispatch:
    def test_exit_raises_eof(self):
        """'exit' command raises EOFError."""
        repl = TriadREPL()
        try:
            repl._dispatch("exit")
            raise AssertionError("Should have raised EOFError")
        except EOFError:
            pass

    def test_quit_raises_eof(self):
        """'quit' command raises EOFError."""
        repl = TriadREPL()
        try:
            repl._dispatch("quit")
            raise AssertionError("Should have raised EOFError")
        except EOFError:
            pass

    def test_help_does_not_error(self):
        """'help' command runs without error."""
        repl = TriadREPL()
        repl._dispatch("help")  # Should not raise

    def test_status_does_not_error(self):
        """'status' command runs without error."""
        repl = TriadREPL()
        repl._dispatch("status")  # Should not raise

    def test_mode_dispatch(self):
        """'mode debate' updates state."""
        repl = TriadREPL()
        repl._dispatch("mode debate")
        assert repl.mode == "debate"

    def test_route_dispatch(self):
        """'route cost_optimized' updates state."""
        repl = TriadREPL()
        repl._dispatch("route cost_optimized")
        assert repl.route == "cost_optimized"

    def test_arbiter_dispatch(self):
        """'arbiter full' updates state."""
        repl = TriadREPL()
        repl._dispatch("arbiter full")
        assert repl.arbiter == "full"

    @patch.object(TriadREPL, "_invoke_cli")
    def test_cli_command_dispatched(self, mock_invoke):
        """Known CLI commands are dispatched to _invoke_cli."""
        repl = TriadREPL()
        repl._dispatch("models list")
        mock_invoke.assert_called_once_with("models list")

    @patch.object(TriadREPL, "_run_task")
    def test_unknown_input_is_task(self, mock_run):
        """Unknown input is treated as a task description."""
        repl = TriadREPL()
        repl._dispatch("Build a REST API for user management")
        mock_run.assert_called_once_with("Build a REST API for user management")


# ── REPL Loop ────────────────────────────────────────────────────


class TestREPLLoop:
    @patch("triad.repl.console")
    def test_ctrl_c_exits_cleanly(self, mock_console):
        """Ctrl+C (KeyboardInterrupt) exits the REPL."""
        mock_console.input.side_effect = KeyboardInterrupt
        repl = TriadREPL()
        repl.run()  # Should not raise

    @patch("triad.repl.console")
    def test_ctrl_d_exits_cleanly(self, mock_console):
        """Ctrl+D (EOFError) exits the REPL."""
        mock_console.input.side_effect = EOFError
        repl = TriadREPL()
        repl.run()  # Should not raise

    @patch("triad.repl.console")
    def test_empty_input_continues(self, mock_console):
        """Empty input is ignored."""
        mock_console.input.side_effect = ["", KeyboardInterrupt]
        repl = TriadREPL()
        repl.run()  # Should not raise

    @patch("triad.repl.console")
    def test_exit_command_exits(self, mock_console):
        """'exit' command exits the REPL cleanly."""
        mock_console.input.side_effect = ["exit"]
        repl = TriadREPL()
        repl.run()  # Should not raise (EOFError is caught)

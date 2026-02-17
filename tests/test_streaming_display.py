"""Tests for the streaming pipeline display."""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from triad.cli_streaming_display import (
    KeyboardHandler,
    StageState,
    StreamBuffer,
    StreamingPipelineDisplay,
    _looks_like_diff,
    _render_diff_text,
)
from triad.dashboard.events import EventType, PipelineEvent
from triad.schemas.messages import PipelineStage
from triad.schemas.streaming import StreamChunk


class TestStreamBuffer:
    def test_empty_feed(self):
        buf = StreamBuffer()
        events = buf.feed("")
        assert events == []

    def test_file_detection(self):
        buf = StreamBuffer()
        text = '# file: src/main.py\n```python\nprint("hello")\n```\n'
        events = buf.feed(text)
        # Should detect file start and complete
        started = [e for e in events if e["type"] == "file_started"]
        completed = [e for e in events if e["type"] == "file_completed"]
        assert len(started) >= 1
        assert len(completed) >= 1

    def test_accumulation(self):
        buf = StreamBuffer()
        buf.feed("hello ")
        buf.feed("world")
        assert buf.accumulated == "hello world"

    def test_code_tracking(self):
        buf = StreamBuffer()
        buf.feed("```python\n")
        assert buf.in_code_block is True
        buf.feed("print('hello')\n")
        buf.feed("```\n")
        assert buf.in_code_block is False
        assert len(buf.files_completed) == 1

    def test_get_display_code_empty(self):
        buf = StreamBuffer()
        code, lang = buf.get_display_code()
        assert code == ""
        assert lang == "text"

    def test_get_display_code_active(self):
        buf = StreamBuffer()
        buf.feed("```python\n")
        buf.feed("x = 1\n")
        code, lang = buf.get_display_code()
        assert "x = 1" in code
        assert lang == "python"

    def test_max_lines_cap(self):
        buf = StreamBuffer()
        buf.feed("```python\n")
        for i in range(600):
            buf.feed(f"line_{i}\n")
        code, lang = buf.get_display_code(max_lines=100)
        assert len(code.splitlines()) <= 100


class TestStageState:
    def test_stage_state_values(self):
        assert StageState.PENDING == "pending"
        assert StageState.ACTIVE == "active"
        assert StageState.COMPLETE == "complete"
        assert StageState.FALLBACK == "fallback"
        assert StageState.FAILED == "failed"


class TestDiffDetection:
    def test_looks_like_diff_true(self):
        diff_text = (
            "--- a/file.py\n"
            "+++ b/file.py\n"
            "@@ -1,3 +1,5 @@\n"
            " line1\n"
            "-old_line\n"
            "+new_line\n"
            "+added_line\n"
        )
        assert _looks_like_diff(diff_text) is True

    def test_looks_like_diff_false(self):
        code = "def hello():\n    print('hi')\n    return True\n"
        assert _looks_like_diff(code) is False

    def test_looks_like_diff_short(self):
        assert _looks_like_diff("+a\n") is False

    def test_render_diff_text_colors(self):
        diff = "+added\n-removed\n@@ hunk\nnormal\n"
        text = _render_diff_text(diff)
        plain = text.plain
        assert "+added" in plain
        assert "-removed" in plain
        assert "@@ hunk" in plain
        assert "normal" in plain

    def test_render_diff_text_headers(self):
        diff = "--- a/file.py\n+++ b/file.py\n"
        text = _render_diff_text(diff)
        assert "--- a/file.py" in text.plain


class TestStreamingPipelineDisplay:
    def test_create_listener(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        assert callable(listener)

    def test_create_stream_callback(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()
        assert callable(callback)

    def test_handle_stage_started(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        event = PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "Test Model"},
        )
        listener(event)
        assert display._stage_states["architect"] == StageState.ACTIVE
        assert display._stage_models["architect"] == "Test Model"
        assert display._active_model == "Test Model"

    def test_handle_stage_completed(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        event = PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.05, "duration": 10.0},
        )
        listener(event)
        assert display._stage_states["architect"] == StageState.COMPLETE
        assert display._stage_costs["architect"] == 0.05

    def test_handle_model_fallback(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        # First start the stage
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "implement", "model": "ModelA"},
        ))
        assert display._stage_states["implement"] == StageState.ACTIVE
        # Now trigger fallback
        listener(PipelineEvent(
            type=EventType.MODEL_FALLBACK,
            data={"stage": "implement", "original_model": "ModelA",
                  "fallback_model": "ModelB", "reason": "rate limit"},
        ))
        assert display._stage_states["implement"] == StageState.FALLBACK
        assert display._stage_models["implement"] == "ModelB"
        assert display._active_model == "ModelB"

    def test_handle_error_marks_failed(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        # Start stage first
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "verify", "model": "TestModel"},
        ))
        # Error with stage specified
        listener(PipelineEvent(
            type=EventType.ERROR,
            data={"stage": "verify", "error": "timeout"},
        ))
        assert display._stage_states["verify"] == StageState.FAILED

    def test_handle_pipeline_completed(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        event = PipelineEvent(
            type=EventType.PIPELINE_COMPLETED,
            data={"total_cost": 0.25},
        )
        listener(event)
        assert display._total_cost == 0.25
        assert display._active_model is None

    def test_activity_log(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._log("Test message")
        assert len(display._activity_log) == 1
        assert display._activity_log[0][1] == "Test message"

    def test_stream_callback_feeds_buffer(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        chunk = StreamChunk(delta="hello", accumulated="hello", token_count=1)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))

        assert "architect" in display._stream_buffers
        assert display._stream_buffers["architect"].accumulated == "hello"

    def test_cancel_event(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert not display.cancel_event.is_set()
        display._cancel_event.set()
        assert display.cancel_event.is_set()


class TestKeyboardState:
    """Test keyboard state changes via direct dispatch (no real terminal)."""

    def test_initial_state(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert display._focus_panel == "output"
        assert display._scroll_offset == 0
        assert display._fullscreen_panel is None
        assert display._cost_expanded is False
        assert display._paused is False

    def test_tab_toggles_focus(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        with display._lock:
            handler._dispatch("tab")
        assert display._focus_panel == "activity"
        with display._lock:
            handler._dispatch("tab")
        assert display._focus_panel == "output"

    def test_scroll_down(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        with display._lock:
            handler._dispatch("j")
        assert display._scroll_offset == 3
        with display._lock:
            handler._dispatch("down")
        assert display._scroll_offset == 6

    def test_scroll_up_clamped(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        # Already at 0, should stay at 0
        with display._lock:
            handler._dispatch("k")
        assert display._scroll_offset == 0
        # Go down then back up
        with display._lock:
            handler._dispatch("j")
            handler._dispatch("up")
        assert display._scroll_offset == 0

    def test_fullscreen_toggle(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        with display._lock:
            handler._dispatch("f")
        assert display._fullscreen_panel == "output"
        with display._lock:
            handler._dispatch("f")
        assert display._fullscreen_panel is None

    def test_cost_toggle(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        with display._lock:
            handler._dispatch("c")
        assert display._cost_expanded is True
        with display._lock:
            handler._dispatch("c")
        assert display._cost_expanded is False

    def test_space_pauses(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        with display._lock:
            handler._dispatch("space")
        assert display._paused is True
        with display._lock:
            handler._dispatch("space")
        assert display._paused is False

    def test_ctrl_c_cancels(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        handler = KeyboardHandler(display)
        with display._lock:
            handler._dispatch("ctrl+c")
        assert display._cancelled is True
        assert display._cancel_event.is_set()


class TestCostTicker:
    def test_active_model_tracking(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "o3"},
        ))
        assert display._active_model == "o3"
        assert display._active_model_stage == "architect"

    def test_active_model_cleared_on_complete(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "o3"},
        ))
        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.1, "duration": 5.0},
        ))
        assert display._active_model is None
        assert display._active_model_stage is None

"""Tests for the scrolling pipeline display."""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console
from rich.text import Text

from triad.cli_streaming_display import (
    ScrollingPipelineDisplay,
    StageState,
    StreamBuffer,
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


class TestScrollingPipelineDisplay:
    def test_create_listener(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        assert callable(listener)

    def test_create_stream_callback(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()
        assert callable(callback)

    def test_handle_stage_started(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
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
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
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
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
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
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
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
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        event = PipelineEvent(
            type=EventType.PIPELINE_COMPLETED,
            data={"total_cost": 0.25},
        )
        listener(event)
        assert display._total_cost == 0.25
        assert display._active_model is None

    def test_stream_callback_feeds_buffer(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        chunk = StreamChunk(delta="hello", accumulated="hello", token_count=1)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))

        assert "architect" in display._stream_buffers
        assert display._stream_buffers["architect"].accumulated == "hello"

    def test_cancel_event(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert not display.cancel_event.is_set()
        display._cancel_event.set()
        assert display.cancel_event.is_set()


class TestLineBuffer:
    """Tests for the line buffer and line processing."""

    def test_flush_complete_lines(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._line_buffer = "line one\nline two\npartial"
        display._flush_complete_lines()
        # "partial" should remain in the buffer
        assert display._line_buffer == "partial"

    def test_flush_line_buffer_prints_remainder(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._line_buffer = "trailing text"
        display._flush_line_buffer()
        assert display._line_buffer == ""

    def test_empty_line_buffer_noop(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._line_buffer = ""
        display._flush_line_buffer()
        assert display._line_buffer == ""

    def test_process_line_file_header(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._process_line("# file: src/app.py")
        assert display._current_file == "src/app.py"
        assert display._current_language == "py"

    def test_process_line_fence_open_close(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert display._in_code_fence is False
        display._process_line("```python")
        assert display._in_code_fence is True
        display._process_line("```")
        assert display._in_code_fence is False

    def test_process_line_diff_coloring_in_refactor(self):
        """Diff lines during refactor stage should be color-processed."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._current_stage = "refactor"
        display._in_code_fence = True
        # These should not raise — we're just verifying they go through
        # the diff coloring path without error
        display._process_line("+added line")
        display._process_line("-removed line")
        display._process_line("@@ -1,3 +1,5 @@")
        display._process_line("context line")

    def test_stream_callback_splits_lines(self):
        """Stream callback should process complete lines from chunks."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        # Send a chunk with a complete line and a partial
        chunk = StreamChunk(delta="hello\nwor", accumulated="hello\nwor", token_count=2)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        assert display._line_buffer == "wor"

        # Complete the second line
        chunk2 = StreamChunk(delta="ld\n", accumulated="hello\nworld\n", token_count=3)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk2))
        assert display._line_buffer == ""


class TestStatusBar:
    """Tests for the 1-line status bar."""

    def test_status_bar_returns_text(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        bar = display._build_status_bar()
        assert isinstance(bar, Text)

    def test_status_bar_shows_stage_names(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        bar = display._build_status_bar()
        plain = bar.plain
        assert "Architect" in plain
        assert "Implement" in plain
        assert "Refactor" in plain
        assert "Verify" in plain

    def test_status_bar_shows_active_model(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._stage_states["architect"] = StageState.ACTIVE
        display._stage_models["architect"] = "gemini-2.5-pro"
        bar = display._build_status_bar()
        assert "gemini-2.5-pro" in bar.plain

    def test_status_bar_shows_cost(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._total_cost = 0.12
        bar = display._build_status_bar()
        assert "$0.12" in bar.plain

    def test_status_bar_shows_tokens(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._total_tokens = 4200
        bar = display._build_status_bar()
        assert "4.2K tok" in bar.plain

    def test_status_bar_small_tokens(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._total_tokens = 50
        bar = display._build_status_bar()
        assert "50 tok" in bar.plain

    def test_rich_dunder_delegates_to_status_bar(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        result = display.__rich__()
        assert isinstance(result, Text)
        assert "Architect" in result.plain


class TestInlinePrinting:
    """Tests for inline event printing."""

    def test_stage_started_prints_header(self):
        """Stage start should update state and set current_stage."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "gpt-4o"},
        ))
        assert display._current_stage == "architect"
        assert display._stage_states["architect"] == StageState.ACTIVE

    def test_stage_completed_clears_active_model(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
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

    def test_arbiter_verdict_prints(self):
        """Arbiter events should not raise."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.ARBITER_VERDICT,
            data={"stage": "architect", "verdict": "approve", "confidence": 0.91},
        ))

    def test_retry_prints(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.RETRY_TRIGGERED,
            data={"stage": "implement", "retry_number": 2},
        ))

    def test_pipeline_halted_prints(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.PIPELINE_HALTED,
            data={"stage": "verify"},
        ))

    def test_stage_started_resets_line_buffer(self):
        """Starting a new stage should flush and reset the line buffer."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._line_buffer = "leftover text"
        display._in_code_fence = True
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "implement", "model": "claude"},
        ))
        assert display._line_buffer == ""
        assert display._in_code_fence is False

    def test_stage_completed_resets_line_buffer(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._line_buffer = "leftover"
        display._in_code_fence = True
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.05, "duration": 8.0},
        ))
        assert display._line_buffer == ""
        assert display._in_code_fence is False

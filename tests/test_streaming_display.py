"""Tests for the scrolling streaming pipeline display."""

from __future__ import annotations

import asyncio

from rich.console import Console
from rich.text import Text

from triad.cli_streaming_display import (
    ScrollingPipelineDisplay,
    StageState,
    StreamBuffer,
    _format_tokens,
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


class TestFormatTokens:
    def test_small_count(self):
        assert _format_tokens(50) == "50"

    def test_large_count(self):
        assert _format_tokens(4200) == "4.2K"

    def test_zero(self):
        assert _format_tokens(0) == "0"

    def test_exactly_1000(self):
        assert _format_tokens(1000) == "1.0K"


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
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "implement", "model": "ModelA"},
        ))
        assert display._stage_states["implement"] == StageState.ACTIVE
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
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "verify", "model": "TestModel"},
        ))
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


class TestTokenCounter:
    """Tests for the stream callback token counter."""

    def test_token_count_increments(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._stage_models["architect"] = "test-model"
        callback = display.create_stream_callback()

        chunk1 = StreamChunk(delta="hello ", accumulated="hello ", token_count=5)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk1))
        assert display._total_tokens == 5
        assert display._stage_tokens["architect"] == 5

        chunk2 = StreamChunk(delta="world", accumulated="hello world", token_count=10)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk2))
        assert display._total_tokens == 10
        assert display._stage_tokens["architect"] == 10

    def test_stream_callback_tracks_fences(self):
        """Stream callback should track fence state via line buffer."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        chunk = StreamChunk(
            delta="```python\nprint('hello')\n",
            accumulated="```python\nprint('hello')\n",
            token_count=5,
        )
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        assert display._in_code_fence is True

        chunk2 = StreamChunk(
            delta="```\n",
            accumulated="```python\nprint('hello')\n```\n",
            token_count=10,
        )
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk2))
        assert display._in_code_fence is False


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

    def test_rich_dunder_delegates(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        result = display.__rich__()
        assert isinstance(result, Text)
        assert "Architect" in result.plain


class TestFileFiltering:
    """Tests for file-header-only filtering in the stream callback."""

    def test_file_header_increments_count(self):
        """File header lines should increment _file_count."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        chunk = StreamChunk(
            delta="# file: src/app.py\n```python\nprint('hi')\n```\n",
            accumulated="# file: src/app.py\n```python\nprint('hi')\n```\n",
            token_count=10,
        )
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        assert display._file_count == 1

    def test_multiple_files_counted(self):
        """Multiple file headers should all be counted."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        text = (
            "# file: src/a.py\n```python\nx=1\n```\n"
            "# file: src/b.py\n```python\ny=2\n```\n"
            "# file: src/c.py\n```python\nz=3\n```\n"
        )
        chunk = StreamChunk(delta=text, accumulated=text, token_count=20)
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        assert display._file_count == 3

    def test_stage_complete_resets_file_count(self):
        """Stage completion should reset _file_count to 0."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._file_count = 5
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "test"},
        ))
        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.1, "duration": 5.0},
        ))
        assert display._file_count == 0

    def test_stage_complete_resets_fence_state(self):
        """Stage completion should reset _in_code_fence."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._in_code_fence = True
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "test"},
        ))
        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.1, "duration": 5.0},
        ))
        assert display._in_code_fence is False

    def test_live_renderable_status_bar_when_no_content(self):
        """Live renderable shows only status bar when no content line."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._current_stage = "architect"
        display._stage_states["architect"] = StageState.ACTIVE

        renderable = display._build_live_renderable()
        plain = renderable.plain
        # Should contain stage names (status bar) but no activity line
        assert "Architect" in plain
        assert "\u25b8" not in plain  # No ▸ marker

    def test_live_renderable_shows_activity_line(self):
        """Live renderable includes activity preview when content available."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._current_stage = "architect"
        display._stage_states["architect"] = StageState.ACTIVE
        display._last_content_line = "The token bucket strategy maintains a refill rate"

        renderable = display._build_live_renderable()
        plain = renderable.plain
        assert "\u25b8" in plain  # ▸ marker present
        assert "token bucket strategy" in plain
        # Status bar also present
        assert "Architect" in plain

    def test_activity_line_clears_on_stage_complete(self):
        """Activity line disappears after stage completes."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()

        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "architect", "model": "test"},
        ))
        display._last_content_line = "some preview text"

        listener(PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.1, "duration": 5.0},
        ))
        assert display._last_content_line == ""

        renderable = display._build_live_renderable()
        assert "\u25b8" not in renderable.plain


class TestInlinePrinting:
    """Tests for inline event printing."""

    def test_stage_started_updates_state(self):
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

    def test_arbiter_verdict_no_error(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.ARBITER_VERDICT,
            data={"stage": "architect", "verdict": "approve", "confidence": 0.91},
        ))

    def test_retry_no_error(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.RETRY_TRIGGERED,
            data={"stage": "implement", "retry_number": 2},
        ))

    def test_pipeline_halted_clears_stage(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._current_stage = "verify"
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.PIPELINE_HALTED,
            data={"stage": "verify"},
        ))
        assert display._current_stage == ""


class TestTokRate:
    """Tests for tok/s calculation."""

    def test_no_samples_returns_zero(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        assert display._calculate_tok_rate() == 0.0

    def test_single_sample_returns_zero(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        import time
        display._tok_samples = [(time.monotonic(), 100)]
        assert display._calculate_tok_rate() == 0.0

    def test_two_samples_calculates_rate(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        import time
        now = time.monotonic()
        display._tok_samples = [
            (now - 1.0, 100),
            (now, 200),
        ]
        rate = display._calculate_tok_rate()
        # 100 tokens in ~1 second
        assert 90 < rate < 110

    def test_old_samples_pruned(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        import time
        now = time.monotonic()
        display._tok_samples = [
            (now - 10.0, 50),   # older than 3s — should be pruned
            (now - 1.0, 100),
            (now, 200),
        ]
        rate = display._calculate_tok_rate()
        assert 90 < rate < 110
        # Old sample should have been removed
        assert len(display._tok_samples) == 2

    def test_stage_start_resets_samples(self):
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        import time
        display._tok_samples = [(time.monotonic(), 100), (time.monotonic(), 200)]
        listener = display.create_listener()
        listener(PipelineEvent(
            type=EventType.STAGE_STARTED,
            data={"stage": "implement", "model": "test"},
        ))
        assert display._tok_samples == []


class TestContentLineUpdate:
    """Tests for _last_content_line updates from streamed code."""

    def test_code_updates_content_line(self):
        """Code lines longer than 5 chars update the preview."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        callback = display.create_stream_callback()

        chunk = StreamChunk(
            delta="```python\ndef calculate_rate(tokens, elapsed):\n```\n",
            accumulated="```python\ndef calculate_rate(tokens, elapsed):\n```\n",
            token_count=10,
        )
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        assert "calculate_rate" in display._last_content_line

    def test_short_lines_ignored(self):
        """Lines 5 chars or shorter should not update preview."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._last_content_line = "previous content"
        callback = display.create_stream_callback()

        chunk = StreamChunk(
            delta="```python\nx=1\n```\n",
            accumulated="```python\nx=1\n```\n",
            token_count=5,
        )
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        # Short line "x=1" should not have overwritten
        assert display._last_content_line == "previous content"

    def test_file_header_clears_content_line(self):
        """File header should clear the preview line."""
        console = Console(quiet=True)
        display = ScrollingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        display._last_content_line = "old preview"
        callback = display.create_stream_callback()

        chunk = StreamChunk(
            delta="# file: src/app.py\n",
            accumulated="# file: src/app.py\n",
            token_count=5,
        )
        asyncio.run(callback(PipelineStage.ARCHITECT, chunk))
        assert display._last_content_line == ""

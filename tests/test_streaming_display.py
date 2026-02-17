"""Tests for the streaming pipeline display."""

from __future__ import annotations

import asyncio

import pytest
from rich.console import Console

from triad.cli_streaming_display import StreamBuffer, StreamingPipelineDisplay
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
        assert display._stage_status["architect"] == "running"
        assert display._stage_models["architect"] == "Test Model"

    def test_handle_stage_completed(self):
        console = Console(quiet=True)
        display = StreamingPipelineDisplay(console, "sequential", "hybrid", "bookend")
        listener = display.create_listener()
        event = PipelineEvent(
            type=EventType.STAGE_COMPLETED,
            data={"stage": "architect", "cost": 0.05, "duration": 10.0},
        )
        listener(event)
        assert display._stage_status["architect"] == "done"
        assert display._stage_costs["architect"] == 0.05

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

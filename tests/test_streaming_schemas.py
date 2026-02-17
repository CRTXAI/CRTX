"""Tests for streaming schemas."""

from __future__ import annotations

from triad.schemas.streaming import StreamChunk


class TestStreamChunk:
    def test_basic(self):
        chunk = StreamChunk(
            delta="hello",
            accumulated="hello",
            token_count=1,
        )
        assert chunk.delta == "hello"
        assert chunk.accumulated == "hello"
        assert chunk.token_count == 1
        assert chunk.is_complete is False

    def test_complete(self):
        chunk = StreamChunk(
            delta="",
            accumulated="full content",
            token_count=10,
            is_complete=True,
        )
        assert chunk.is_complete is True
        assert chunk.delta == ""

    def test_token_count_non_negative(self):
        chunk = StreamChunk(delta="x", accumulated="x", token_count=0)
        assert chunk.token_count == 0

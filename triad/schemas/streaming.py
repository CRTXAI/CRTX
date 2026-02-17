"""Streaming schemas for real-time token delivery.

Defines the StreamChunk model used by complete_streaming() to
deliver incremental output to the display layer.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class StreamChunk(BaseModel):
    """A single chunk of streaming output from a model."""

    delta: str = Field(description="New text in this chunk")
    accumulated: str = Field(description="Full text accumulated so far")
    token_count: int = Field(ge=0, description="Running output token count")
    is_complete: bool = Field(
        default=False, description="True on final chunk"
    )

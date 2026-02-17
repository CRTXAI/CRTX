"""Pydantic schemas for the Triad Pro ingestion API."""

from __future__ import annotations

from typing import Any

from pydantic import BaseModel, Field


class EventPayload(BaseModel):
    """A single pipeline event from the CLI ProAgent."""

    type: str = Field(description="Event type string (e.g. 'stage_started')")
    timestamp: float = Field(description="Unix timestamp from the CLI")
    data: dict[str, Any] = Field(default_factory=dict, description="Event payload")


class IngestRequest(BaseModel):
    """Batch of events sent by ProAgent."""

    session_id: str = Field(description="Pipeline session UUID")
    events: list[EventPayload] = Field(description="Batched pipeline events")


class IngestResponse(BaseModel):
    """Response to a successful ingestion."""

    status: str = "accepted"
    events_received: int = 0


class CheckoutRequest(BaseModel):
    """Request to create a Stripe Checkout session."""

    plan: str = Field(description="Plan to subscribe to: 'pro' or 'cloud'")
    success_url: str = Field(description="URL to redirect to after checkout")
    cancel_url: str = Field(description="URL to redirect to on cancel")

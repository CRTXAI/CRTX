"""Triad Pro Ingestion API.

FastAPI application that receives pipeline events from the CLI ProAgent
and writes them to Supabase Postgres. Supabase Realtime automatically
broadcasts INSERTs to connected dashboard clients.
"""

from __future__ import annotations

import hashlib
import json
import os
import secrets
from contextlib import asynccontextmanager
from typing import Any

import stripe
from fastapi import Depends, FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from supabase import Client, create_client

from auth import verify_api_key
from models import CheckoutRequest, EventPayload, IngestRequest, IngestResponse

# ---------------------------------------------------------------------------
# Supabase client (service role — bypasses RLS)
# ---------------------------------------------------------------------------

_supabase: Client | None = None


def _sb() -> Client:
    global _supabase
    if _supabase is None:
        _supabase = create_client(
            os.environ["SUPABASE_URL"],
            os.environ["SUPABASE_SERVICE_ROLE_KEY"],
        )
    return _supabase


# ---------------------------------------------------------------------------
# App
# ---------------------------------------------------------------------------


@asynccontextmanager
async def lifespan(app: FastAPI):
    stripe.api_key = os.environ.get("STRIPE_SECRET_KEY", "")
    yield


app = FastAPI(
    title="Triad Pro API",
    version="1.0.0",
    lifespan=lifespan,
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=[
        "https://triad-orchestrator.com",
        "https://app.triad-orchestrator.com",
        "http://localhost:3000",
    ],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ---------------------------------------------------------------------------
# Event ingestion
# ---------------------------------------------------------------------------


def _process_event(
    sb: Client,
    session_id: str,
    user_id: str,
    event: EventPayload,
) -> None:
    """Process a single pipeline event — update stages/verdicts/session."""
    data = event.data

    if event.type == "pipeline_started":
        # Upsert session row
        sb.table("sessions").upsert({
            "id": session_id,
            "user_id": user_id,
            "task": data.get("task", ""),
            "mode": data.get("mode", "sequential"),
            "route": data.get("route", "hybrid"),
            "arbiter_mode": data.get("arbiter_mode", "after_each"),
            "status": "running",
            "model_assignments": json.dumps(data.get("model_assignments", {})),
        }).execute()

    elif event.type == "stage_started":
        sb.table("stages").insert({
            "session_id": session_id,
            "stage_name": data.get("stage", ""),
            "model_id": data.get("model", ""),
            "status": "running",
        }).execute()

    elif event.type == "stage_completed":
        stage_name = data.get("stage", "")
        sb.table("stages").update({
            "status": "completed",
            "tokens_in": data.get("tokens_in", 0),
            "tokens_out": data.get("tokens_out", 0),
            "cost": data.get("cost", 0),
            "latency_ms": data.get("latency_ms"),
            "output_text": data.get("content", ""),
            "verdict": data.get("verdict"),
            "confidence": data.get("confidence"),
        }).eq("session_id", session_id).eq("stage_name", stage_name).eq("status", "running").execute()

    elif event.type == "arbiter_verdict":
        sb.table("verdicts").insert({
            "session_id": session_id,
            "stage_name": data.get("stage", ""),
            "verdict": data.get("verdict", ""),
            "confidence": data.get("confidence"),
            "issues": json.dumps(data.get("issues", [])),
            "reasoning": data.get("reasoning", ""),
            "model_id": data.get("model", ""),
            "token_cost": data.get("token_cost", 0),
        }).execute()

    elif event.type == "pipeline_completed":
        sb.table("sessions").update({
            "status": "completed",
            "cost_total": data.get("total_cost", 0),
            "tokens_total": data.get("total_tokens", 0),
            "duration_ms": data.get("duration_ms"),
            "confidence": data.get("confidence"),
            "completed_at": "now()",
        }).eq("id", session_id).execute()

    elif event.type == "pipeline_halted":
        sb.table("sessions").update({
            "status": "halted",
            "cost_total": data.get("total_cost", 0),
            "tokens_total": data.get("total_tokens", 0),
            "duration_ms": data.get("duration_ms"),
            "completed_at": "now()",
        }).eq("id", session_id).execute()


@app.post("/api/v1/events/ingest", response_model=IngestResponse, status_code=202)
async def ingest_events(
    body: IngestRequest,
    user_id: str = Depends(verify_api_key),
) -> IngestResponse:
    """Receive a batch of pipeline events from the CLI ProAgent."""
    sb = _sb()

    # Insert raw events (source of truth)
    raw_rows = [
        {
            "session_id": body.session_id,
            "event_type": e.type,
            "payload": json.dumps({"timestamp": e.timestamp, **e.data}),
        }
        for e in body.events
    ]
    if raw_rows:
        sb.table("events").insert(raw_rows).execute()

    # Process each event to update derived tables
    for event in body.events:
        try:
            _process_event(sb, body.session_id, user_id, event)
        except Exception:
            # Best-effort — don't fail the whole batch on one event
            pass

    return IngestResponse(status="accepted", events_received=len(body.events))


# ---------------------------------------------------------------------------
# API key management
# ---------------------------------------------------------------------------


@app.post("/api/v1/keys/create")
async def create_api_key(
    request: Request,
    user_id: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Create a new CLI API key for the authenticated user."""
    body = await request.json()
    label = body.get("label", "")

    # Generate key
    raw_key = f"sk_triad_{secrets.token_hex(16)}"
    key_hash = hashlib.sha256(raw_key.encode("utf-8")).hexdigest()
    key_prefix = raw_key[:12] + "..."

    _sb().table("api_keys").insert({
        "user_id": user_id,
        "key_hash": key_hash,
        "key_prefix": key_prefix,
        "label": label,
    }).execute()

    # Return the full key once — never stored in plaintext
    return {"key": raw_key, "prefix": key_prefix}


@app.post("/api/v1/keys/{key_id}/revoke")
async def revoke_api_key(
    key_id: str,
    user_id: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Revoke an API key."""
    _sb().table("api_keys").update(
        {"revoked_at": "now()"}
    ).eq("id", key_id).eq("user_id", user_id).execute()
    return {"status": "revoked"}


# ---------------------------------------------------------------------------
# Billing (Stripe)
# ---------------------------------------------------------------------------

_PRICE_IDS: dict[str, str] = {
    "pro": os.environ.get("STRIPE_PRO_PRICE_ID", ""),
    "cloud": os.environ.get("STRIPE_CLOUD_PRICE_ID", ""),
}


@app.post("/api/v1/billing/checkout")
async def create_checkout(
    body: CheckoutRequest,
    user_id: str = Depends(verify_api_key),
) -> dict[str, str]:
    """Create a Stripe Checkout session for subscription."""
    price_id = _PRICE_IDS.get(body.plan)
    if not price_id:
        raise HTTPException(status_code=400, detail="Invalid plan")

    session = stripe.checkout.Session.create(
        mode="subscription",
        line_items=[{"price": price_id, "quantity": 1}],
        success_url=body.success_url,
        cancel_url=body.cancel_url,
        metadata={"user_id": user_id},
    )
    return {"url": session.url}


@app.post("/api/v1/billing/webhook")
async def stripe_webhook(request: Request) -> dict[str, str]:
    """Handle Stripe webhook events."""
    payload = await request.body()
    sig = request.headers.get("stripe-signature", "")
    endpoint_secret = os.environ.get("STRIPE_WEBHOOK_SECRET", "")

    try:
        event = stripe.Webhook.construct_event(payload, sig, endpoint_secret)
    except (stripe.error.SignatureVerificationError, ValueError):
        raise HTTPException(status_code=400, detail="Invalid webhook signature")

    if event["type"] == "checkout.session.completed":
        session = event["data"]["object"]
        user_id = session.get("metadata", {}).get("user_id")
        if user_id:
            # Determine plan from price
            line_items = stripe.checkout.Session.list_line_items(session["id"])
            price_id = line_items["data"][0]["price"]["id"] if line_items["data"] else ""
            plan = "pro"
            for plan_name, pid in _PRICE_IDS.items():
                if pid == price_id:
                    plan = plan_name
                    break
            _sb().table("profiles").update({"plan": plan}).eq("id", user_id).execute()

    elif event["type"] in ("customer.subscription.updated", "customer.subscription.deleted"):
        sub = event["data"]["object"]
        # If subscription cancelled/expired, downgrade to free
        if sub.get("status") in ("canceled", "unpaid", "past_due"):
            customer_id = sub.get("customer")
            # Look up user by Stripe customer
            if customer_id:
                _sb().table("profiles").update({"plan": "free"}).eq(
                    "stripe_customer_id", customer_id
                ).execute()

    return {"status": "ok"}


# ---------------------------------------------------------------------------
# Analytics
# ---------------------------------------------------------------------------


@app.get("/api/v1/analytics/models")
async def model_performance(
    user_id: str = Depends(verify_api_key),
) -> list[dict[str, Any]]:
    """Return per-model performance metrics for the authenticated user."""
    result = _sb().table("model_performance").select("*").eq("user_id", user_id).execute()
    return result.data or []


@app.get("/api/v1/analytics/costs")
async def cost_rollup(
    user_id: str = Depends(verify_api_key),
    days: int = 30,
) -> list[dict[str, Any]]:
    """Return daily cost rollup for the authenticated user."""
    from datetime import datetime, timedelta
    cutoff = (datetime.utcnow() - timedelta(days=days)).isoformat()
    result = (
        _sb()
        .table("cost_rollup")
        .select("*")
        .eq("user_id", user_id)
        .gte("day", cutoff)
        .order("day", desc=False)
        .execute()
    )
    return result.data or []


@app.get("/api/v1/analytics/summary")
async def analytics_summary(
    user_id: str = Depends(verify_api_key),
) -> dict[str, Any]:
    """Return aggregate summary stats for the authenticated user."""
    sb = _sb()

    # Total sessions
    sessions_res = sb.table("sessions").select("id", count="exact").eq("user_id", user_id).execute()
    total_sessions = sessions_res.count or 0

    # Total cost and tokens
    cost_res = (
        sb.table("sessions")
        .select("cost_total, tokens_total")
        .eq("user_id", user_id)
        .execute()
    )
    total_cost = sum(row.get("cost_total", 0) or 0 for row in (cost_res.data or []))
    total_tokens = sum(row.get("tokens_total", 0) or 0 for row in (cost_res.data or []))

    # Verdict distribution
    verdicts_res = (
        sb.rpc("get_user_verdict_counts", {"p_user_id": user_id}).execute()
        if False  # RPC not yet created — fall back to client-side
        else sb.table("verdicts")
        .select("verdict")
        .in_("session_id", [r["id"] for r in (sessions_res.data or [])])
        .execute()
    )
    verdict_counts: dict[str, int] = {}
    for row in verdicts_res.data or []:
        v = row.get("verdict", "")
        verdict_counts[v] = verdict_counts.get(v, 0) + 1

    return {
        "total_sessions": total_sessions,
        "total_cost": float(total_cost),
        "total_tokens": total_tokens,
        "verdict_distribution": verdict_counts,
    }


# ---------------------------------------------------------------------------
# Health
# ---------------------------------------------------------------------------


@app.get("/health")
async def health() -> dict[str, str]:
    return {"status": "ok"}

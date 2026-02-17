"""API key authentication for the Triad Pro ingestion API."""

from __future__ import annotations

import hashlib
import os

from fastapi import HTTPException, Security
from fastapi.security import HTTPAuthorizationCredentials, HTTPBearer

from supabase import Client, create_client

_security = HTTPBearer()

_supabase: Client | None = None


def _get_supabase() -> Client:
    """Lazy-init Supabase service-role client."""
    global _supabase
    if _supabase is None:
        url = os.environ["SUPABASE_URL"]
        key = os.environ["SUPABASE_SERVICE_ROLE_KEY"]
        _supabase = create_client(url, key)
    return _supabase


def _hash_key(key: str) -> str:
    """SHA-256 hash of an API key for lookup."""
    return hashlib.sha256(key.encode("utf-8")).hexdigest()


async def verify_api_key(
    credentials: HTTPAuthorizationCredentials = Security(_security),
) -> str:
    """Validate the Bearer token against the api_keys table.

    Returns the user_id associated with the key.
    Raises 401 if the key is invalid or revoked.
    """
    token = credentials.credentials
    key_hash = _hash_key(token)

    sb = _get_supabase()
    result = (
        sb.table("api_keys")
        .select("user_id, revoked_at")
        .eq("key_hash", key_hash)
        .execute()
    )

    if not result.data:
        raise HTTPException(status_code=401, detail="Invalid API key")

    row = result.data[0]
    if row.get("revoked_at"):
        raise HTTPException(status_code=401, detail="API key has been revoked")

    user_id: str = row["user_id"]

    # Check plan
    profile = (
        sb.table("profiles")
        .select("plan")
        .eq("id", user_id)
        .single()
        .execute()
    )
    if profile.data and profile.data.get("plan") == "free":
        raise HTTPException(status_code=403, detail="Pro subscription required")

    # Update last_used_at (fire-and-forget)
    sb.table("api_keys").update({"last_used_at": "now()"}).eq("key_hash", key_hash).execute()

    return user_id

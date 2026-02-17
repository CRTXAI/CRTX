"""Provider health tracking for automatic model fallback.

Tracks which models are healthy or unhealthy during a pipeline session.
When a model fails with a transient error (429, 529, timeout), it is
marked unhealthy for a cooldown period. The routing engine checks health
before selecting a model, and the orchestrator uses fallback models
when the primary selection is unavailable.

Design: simple in-memory dict, per-session, no persistence.
"""

from __future__ import annotations

import time


class ProviderHealth:
    """Per-session health tracker for model providers.

    After a model fails with a transient error, it is marked unhealthy
    for ``cooldown`` seconds. Subsequent routing decisions skip unhealthy
    models, preventing wasted retries on providers known to be down.
    """

    def __init__(self, cooldown: float = 300.0) -> None:
        self._unhealthy: dict[str, float] = {}  # model_key -> monotonic time
        self._cooldown = cooldown

    def mark_unhealthy(self, model_key: str) -> None:
        """Mark a model as unhealthy (just failed with a transient error)."""
        self._unhealthy[model_key] = time.monotonic()

    def is_healthy(self, model_key: str) -> bool:
        """Check whether a model is healthy (or its cooldown has expired)."""
        marked_at = self._unhealthy.get(model_key)
        if marked_at is None:
            return True
        if time.monotonic() - marked_at > self._cooldown:
            del self._unhealthy[model_key]
            return True
        return False

    def unhealthy_models(self) -> set[str]:
        """Return the set of currently-unhealthy model keys.

        Also cleans up any entries whose cooldown has expired.
        """
        now = time.monotonic()
        expired = [k for k, t in self._unhealthy.items() if now - t > self._cooldown]
        for k in expired:
            del self._unhealthy[k]
        return set(self._unhealthy)

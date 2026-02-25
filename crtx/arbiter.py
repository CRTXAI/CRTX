"""CRTX Arbiter — public SDK API + standalone review."""
from __future__ import annotations

import asyncio
import json
import os
import re
import time

# Public API (pipeline context)
from triad.arbiter.arbiter import ArbiterEngine as Arbiter  # noqa: F401
# Note: triad uses ArbiterReview (not ArbiterResult) -- aliased for SDK consistency
from triad.schemas.arbiter import ArbiterReview as ArbiterResult  # noqa: F401

# ── Standalone review ─────────────────────────────────────────────────────────

_DEFAULT_REVIEW_MODEL = "xai/grok-4-0709"
_FALLBACK_REVIEW_MODEL = "anthropic/claude-sonnet-4-5-20250929"

_REVIEW_SYSTEM = """\
You are an independent content and code quality reviewer.
Evaluate the content provided by the user and return a structured verdict.

Verdicts:
  APPROVE — high quality, ready to ship, no significant issues
  FLAG    — minor issues present; human review recommended
  REJECT  — significant issues found; revisions required

Respond ONLY with valid JSON in exactly this schema (no markdown fences, no extra text):
{
  "verdict": "APPROVE" | "FLAG" | "REJECT",
  "confidence": <float 0.0-1.0>,
  "checks": {
    "quality": <bool>,
    "clarity": <bool>,
    "completeness": <bool>,
    "accuracy": <bool>
  },
  "notes": ["<issue or observation>", ...]
}
"""

_JSON_SUFFIX = (
    "\n\nIMPORTANT: Your entire response must be a single valid JSON object. "
    "No markdown code fences. No explanation before or after the JSON."
)


def _build_user_message(content: str, content_type: str) -> str:
    return f"Content type: {content_type}\n\nContent to review:\n\n{content}"


def _strip_fences(text: str) -> str:
    """Remove markdown code fences and leading/trailing whitespace."""
    text = text.strip()
    text = re.sub(r"^```[a-zA-Z]*\n?", "", text)
    text = re.sub(r"\n?```$", "", text)
    return text.strip()


def _fallback_verdict(reason: str) -> dict:
    return {
        "verdict": "FLAG",
        "confidence": 0.0,
        "checks": {
            "quality": False,
            "clarity": False,
            "completeness": False,
            "accuracy": False,
        },
        "notes": [f"Review failed: {reason}"],
    }


async def standalone_review(
    content: str,
    content_type: str = "general",
    model: str | None = None,
    *,
    timeout: int = 60,
) -> dict:
    """One-shot Arbiter review without pipeline context.

    Makes a direct LLM call and returns a structured verdict dict.
    Falls back to FLAG on any error — never raises.

    Args:
        content:      The text/code to review.
        content_type: Hint for the reviewer ("article", "tweet", "code", etc.).
        model:        Override the default model (xai/grok-4-0709).
        timeout:      API call timeout in seconds.

    Returns:
        Dict with keys: verdict, confidence, checks, notes.
    """
    try:
        import litellm  # type: ignore
    except ImportError:
        return _fallback_verdict("litellm not installed")

    # Load keys the same way the triad pipeline does
    try:
        from triad.keys import load_keys_env
        load_keys_env()
    except Exception:
        pass

    review_model = model or _DEFAULT_REVIEW_MODEL
    system_prompt = _REVIEW_SYSTEM
    user_message = _build_user_message(content, content_type) + _JSON_SUFFIX

    async def _call(m: str) -> dict:
        resp = await litellm.acompletion(
            model=m,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_message},
            ],
            timeout=timeout,
            temperature=0.0,
        )
        raw = resp.choices[0].message.content or ""
        cleaned = _strip_fences(raw)
        return json.loads(cleaned)

    # Try primary model, then fallback
    for attempt_model in (review_model, _FALLBACK_REVIEW_MODEL):
        if attempt_model == review_model and attempt_model == _FALLBACK_REVIEW_MODEL:
            # custom model specified — only one attempt
            try:
                return await _call(attempt_model)
            except Exception as exc:
                return _fallback_verdict(str(exc))
        try:
            return await _call(attempt_model)
        except Exception:
            continue

    return _fallback_verdict("all review models failed")

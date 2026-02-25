"""Tests for crtx check command and standalone_review().

All LLM calls are mocked — no real API calls made in this suite.
"""
from __future__ import annotations

import asyncio
import json
from pathlib import Path
from typing import Any
from unittest.mock import AsyncMock, MagicMock, patch

import pytest
from typer.testing import CliRunner

from crtx.arbiter import _fallback_verdict, _strip_fences, standalone_review
from triad.cli import app

runner = CliRunner()

# ── helpers ───────────────────────────────────────────────────────────────────

def _mock_litellm_response(verdict: str = "APPROVE", confidence: float = 0.95) -> Any:
    """Build a fake litellm response object."""
    payload = json.dumps(
        {
            "verdict": verdict,
            "confidence": confidence,
            "checks": {
                "quality": True,
                "clarity": True,
                "completeness": True,
                "accuracy": True,
            },
            "notes": ["Looks good."],
        }
    )
    msg = MagicMock()
    msg.content = payload
    choice = MagicMock()
    choice.message = msg
    resp = MagicMock()
    resp.choices = [choice]
    return resp


# ── _strip_fences ─────────────────────────────────────────────────────────────

def test_strip_fences_removes_json_fence():
    raw = "```json\n{\"a\": 1}\n```"
    assert _strip_fences(raw) == '{"a": 1}'


def test_strip_fences_plain_text_unchanged():
    raw = '{"verdict": "APPROVE"}'
    assert _strip_fences(raw) == raw


def test_strip_fences_strips_whitespace():
    raw = "  \n{\"x\": 1}\n  "
    assert _strip_fences(raw) == '{"x": 1}'


# ── _fallback_verdict ─────────────────────────────────────────────────────────

def test_fallback_verdict_structure():
    result = _fallback_verdict("network error")
    assert result["verdict"] == "FLAG"
    assert result["confidence"] == 0.0
    assert "network error" in result["notes"][0]
    assert "quality" in result["checks"]


# ── standalone_review ─────────────────────────────────────────────────────────

@pytest.mark.asyncio
async def test_standalone_review_approve():
    mock_resp = _mock_litellm_response("APPROVE", 0.97)
    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        result = await standalone_review("Is 2+2=4?", content_type="general")

    assert result["verdict"] == "APPROVE"
    assert result["confidence"] == pytest.approx(0.97)
    assert "quality" in result["checks"]
    assert isinstance(result["notes"], list)


@pytest.mark.asyncio
async def test_standalone_review_flag():
    mock_resp = _mock_litellm_response("FLAG", 0.72)
    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        result = await standalone_review("Some questionable content", content_type="tweet")

    assert result["verdict"] == "FLAG"


@pytest.mark.asyncio
async def test_standalone_review_custom_model():
    mock_resp = _mock_litellm_response("APPROVE", 0.99)
    mock_call = AsyncMock(return_value=mock_resp)
    with patch("litellm.acompletion", new=mock_call):
        result = await standalone_review(
            "Test content", model="anthropic/claude-sonnet-4-5-20250929"
        )

    assert result["verdict"] == "APPROVE"
    called_model = mock_call.call_args[1].get("model") or mock_call.call_args[0][0]
    assert called_model == "anthropic/claude-sonnet-4-5-20250929"


@pytest.mark.asyncio
async def test_standalone_review_falls_back_on_parse_error():
    """If primary model returns garbage, falls back to FLAG verdict."""
    bad_msg = MagicMock()
    bad_msg.content = "this is not json at all!!!"
    bad_choice = MagicMock()
    bad_choice.message = bad_msg
    bad_resp = MagicMock()
    bad_resp.choices = [bad_choice]

    good_resp = _mock_litellm_response("APPROVE", 0.90)

    call_count = 0

    async def side_effect(**kwargs):
        nonlocal call_count
        call_count += 1
        if call_count == 1:
            return bad_resp
        return good_resp

    with patch("litellm.acompletion", new=side_effect):
        result = await standalone_review("some content")

    # Should have tried fallback model and returned APPROVE
    assert result["verdict"] == "APPROVE"
    assert call_count == 2


@pytest.mark.asyncio
async def test_standalone_review_all_models_fail():
    """If all models raise exceptions, returns FLAG with error note."""
    with patch("litellm.acompletion", new=AsyncMock(side_effect=RuntimeError("API down"))):
        result = await standalone_review("some content")

    assert result["verdict"] == "FLAG"
    assert result["confidence"] == 0.0
    assert any("API down" in n or "failed" in n for n in result["notes"])


# -- CLI check command -------------------------------------------------------------

def test_cli_check_json_output():
    mock_resp = _mock_litellm_response("APPROVE", 0.95)
    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        result = runner.invoke(
            app, ["check", "--prompt", "Is 2+2=4?", "--format", "json"]
        )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["verdict"] == "APPROVE"
    assert "confidence" in parsed
    assert "checks" in parsed
    assert "notes" in parsed


def test_cli_check_text_output():
    mock_resp = _mock_litellm_response("APPROVE", 0.95)
    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        result = runner.invoke(app, ["check", "--prompt", "Good content"])

    assert result.exit_code == 0
    assert "APPROVE" in result.output


def test_cli_check_reject_output():
    mock_resp = _mock_litellm_response("REJECT", 0.88)
    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        result = runner.invoke(
            app, ["check", "--prompt", "Bad content", "--format", "json"]
        )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["verdict"] == "REJECT"


def test_cli_check_file_flag(tmp_path: Path):
    content_file = tmp_path / "content.txt"
    content_file.write_text("Article content goes here.")
    mock_resp = _mock_litellm_response("FLAG", 0.65)
    with patch("litellm.acompletion", new=AsyncMock(return_value=mock_resp)):
        result = runner.invoke(
            app,
            ["check", "--file", str(content_file), "--type", "article", "--format", "json"],
        )

    assert result.exit_code == 0
    parsed = json.loads(result.output)
    assert parsed["verdict"] == "FLAG"


def test_cli_check_missing_file():
    result = runner.invoke(app, ["check", "--file", "nonexistent.txt"])
    assert result.exit_code != 0
    assert "not found" in result.output.lower()


def test_cli_check_no_input():
    result = runner.invoke(app, ["check"])
    assert result.exit_code != 0

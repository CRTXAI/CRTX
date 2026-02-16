"""Tests for triad.providers.litellm_provider — LiteLLM adapter."""

from __future__ import annotations

import json
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

import pytest
from pydantic import BaseModel

from triad.providers.litellm_provider import (
    LiteLLMProvider,
    _language_extension,
    extract_code_blocks,
)
from triad.schemas.pipeline import ModelConfig, RoleFitness

# Shorthand for the mock target
_ACOMP = "triad.providers.litellm_provider.litellm.acompletion"
_SLEEP = "triad.providers.litellm_provider.asyncio.sleep"


# ── Helpers ───────────────────────────────────────────────────


def _make_config(**overrides) -> ModelConfig:
    """Create a ModelConfig with sensible defaults."""
    defaults = {
        "provider": "anthropic",
        "model": "claude-sonnet-4-5-20250929",
        "display_name": "Claude Sonnet 4.5",
        "api_key_env": "ANTHROPIC_API_KEY",
        "context_window": 200000,
        "supports_tools": True,
        "supports_structured": True,
        "supports_vision": True,
        "supports_thinking": True,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "fitness": RoleFitness(
            architect=0.8, implementer=0.85,
            refactorer=0.9, verifier=0.85,
        ),
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


def _make_response(
    content: str = "Hello",
    prompt_tokens: int = 100,
    completion_tokens: int = 50,
    model: str | None = None,
) -> SimpleNamespace:
    """Build a mock LiteLLM ModelResponse-like object."""
    message = SimpleNamespace(content=content, tool_calls=None)
    choice = SimpleNamespace(
        message=message, finish_reason="stop", index=0,
    )
    usage = SimpleNamespace(
        prompt_tokens=prompt_tokens,
        completion_tokens=completion_tokens,
        total_tokens=prompt_tokens + completion_tokens,
    )
    return SimpleNamespace(
        choices=[choice],
        usage=usage,
        model=model or "claude-sonnet-4-5-20250929",
    )


# ── Code Block Extraction ────────────────────────────────────


class TestExtractCodeBlocks:
    def test_single_python_block(self):
        content = (
            "Here is the code:\n\n"
            "```python\n"
            "def hello():\n"
            '    print("world")\n'
            "```\n"
        )
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].language == "python"
        assert "def hello():" in blocks[0].content

    def test_multiple_blocks(self):
        content = (
            "First:\n\n```python\ndef foo():\n    pass\n```\n\n"
            "Second:\n\n```javascript\nfunction bar() {}\n```\n"
        )
        blocks = extract_code_blocks(content)
        assert len(blocks) == 2
        assert blocks[0].language == "python"
        assert blocks[1].language == "javascript"

    def test_no_language_specified(self):
        content = "```\nsome code\n```\n"
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].language == "text"

    def test_filepath_hint_hash_comment(self):
        content = "# file: src/main.py\n```python\nprint('hi')\n```\n"
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].filepath == "src/main.py"

    def test_filepath_hint_slash_comment(self):
        content = (
            "// filepath: src/app.ts\n"
            "```typescript\nconst x = 1;\n```\n"
        )
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].filepath == "src/app.ts"

    def test_no_filepath_generates_default(self):
        content = "```python\nx = 1\n```\n"
        blocks = extract_code_blocks(content)
        assert len(blocks) == 1
        assert blocks[0].filepath == "untitled.py"

    def test_empty_content(self):
        assert extract_code_blocks("") == []
        assert extract_code_blocks("No code blocks here") == []

    def test_preserves_indentation(self):
        content = (
            "```python\ndef foo():\n"
            "    if True:\n        return 42\n```\n"
        )
        blocks = extract_code_blocks(content)
        assert "    if True:" in blocks[0].content
        assert "        return 42" in blocks[0].content


class TestLanguageExtension:
    def test_known_languages(self):
        assert _language_extension("python") == "py"
        assert _language_extension("javascript") == "js"
        assert _language_extension("typescript") == "ts"
        assert _language_extension("rust") == "rs"
        assert _language_extension("go") == "go"
        assert _language_extension("bash") == "sh"

    def test_case_insensitive(self):
        assert _language_extension("Python") == "py"
        assert _language_extension("JAVASCRIPT") == "js"

    def test_unknown_language_returns_self(self):
        assert _language_extension("zig") == "zig"
        assert _language_extension("haskell") == "haskell"


# ── LiteLLMProvider Init ─────────────────────────────────────


class TestLiteLLMProviderInit:
    def test_init_from_config(self):
        config = _make_config()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-123"}):
            provider = LiteLLMProvider(config)
        assert provider.model_id == "claude-sonnet-4-5-20250929"
        assert provider.provider_id == "anthropic"
        assert provider.display_name == "Claude Sonnet 4.5"

    def test_missing_api_key_is_empty_string(self):
        config = _make_config(api_key_env="NONEXISTENT_KEY_12345")
        with patch.dict("os.environ", {}, clear=False):
            provider = LiteLLMProvider(config)
        assert provider._api_key == ""


# ── LiteLLMProvider.complete() ────────────────────────────────


class TestLiteLLMProviderComplete:
    @pytest.fixture
    def provider(self):
        config = _make_config()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_basic_completion(self, provider):
        resp = _make_response(
            content="Here is the scaffold...",
            prompt_tokens=200, completion_tokens=100,
        )
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "Build an API"}],
                system="You are an architect.",
            )

        assert result.content == "Here is the scaffold..."
        assert result.model == "claude-sonnet-4-5-20250929"
        assert result.token_usage is not None
        assert result.token_usage.prompt_tokens == 200
        assert result.token_usage.completion_tokens == 100

    @pytest.mark.asyncio
    async def test_system_prompt_prepended(self, provider):
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response()
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="You are an architect.",
            )

        msgs = mock.call_args[1]["messages"]
        assert msgs[0] == {
            "role": "system", "content": "You are an architect.",
        }
        assert msgs[1] == {"role": "user", "content": "test"}

    @pytest.mark.asyncio
    async def test_api_key_passed(self, provider):
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response()
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert mock.call_args[1]["api_key"] == "sk-test"

    @pytest.mark.asyncio
    async def test_api_base_passed_when_set(self):
        config = _make_config(
            provider="xai", model="xai/grok-4-0709",
            api_key_env="XAI_API_KEY", api_base="https://api.x.ai/v1",
        )
        with patch.dict("os.environ", {"XAI_API_KEY": "xai-test"}):
            provider = LiteLLMProvider(config)

        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response()
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert mock.call_args[1]["api_base"] == "https://api.x.ai/v1"

    @pytest.mark.asyncio
    async def test_api_base_not_passed_when_empty(self, provider):
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response()
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert "api_base" not in mock.call_args[1]

    @pytest.mark.asyncio
    async def test_timeout_passed(self, provider):
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response()
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
                timeout=300,
            )

        assert mock.call_args[1]["timeout"] == 300.0

    @pytest.mark.asyncio
    async def test_code_blocks_extracted(self, provider):
        content = (
            "Here is the code:\n\n"
            "# file: src/main.py\n"
            "```python\n"
            "def main():\n"
            '    print("hello")\n'
            "```\n"
        )
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content=content)
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert len(result.code_blocks) == 1
        assert result.code_blocks[0].language == "python"
        assert result.code_blocks[0].filepath == "src/main.py"
        assert "def main():" in result.code_blocks[0].content


class TestLiteLLMProviderStructuredOutput:
    @pytest.fixture
    def provider(self):
        config = _make_config(supports_structured=True)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_response_format_set(self, provider):
        class Schema(BaseModel):
            name: str
            value: int

        resp = _make_response(
            content=json.dumps({"name": "test", "value": 42}),
        )
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
                output_schema=Schema,
            )

        assert mock.call_args[1]["response_format"] is Schema

    @pytest.mark.asyncio
    async def test_not_set_when_unsupported(self):
        config = _make_config(supports_structured=False)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            provider = LiteLLMProvider(config)

        class Schema(BaseModel):
            name: str

        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content="just text")
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
                output_schema=Schema,
            )

        assert "response_format" not in mock.call_args[1]

    @pytest.mark.asyncio
    async def test_fallback_to_text_on_invalid_json(self, provider):
        class Schema(BaseModel):
            name: str
            value: int

        resp = _make_response(content="This is plain text, not JSON")
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
                output_schema=Schema,
            )

        assert result.content == "This is plain text, not JSON"

    @pytest.mark.asyncio
    async def test_json_in_code_block_fallback(self, provider):
        class Schema(BaseModel):
            name: str
            value: int

        content = (
            "Here is the result:\n\n"
            '```json\n{"name": "test", "value": 42}\n```\n'
        )
        resp = _make_response(content=content)
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
                output_schema=Schema,
            )

        assert "test" in result.content


class TestLiteLLMProviderCostCalculation:
    @pytest.fixture
    def provider(self):
        config = _make_config(cost_input=3.00, cost_output=15.00)
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_cost_from_config_rates(self, provider):
        resp = _make_response(
            prompt_tokens=10000, completion_tokens=5000,
        )
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        # 10000/1M * 3.00 = 0.03, 5000/1M * 15.00 = 0.075
        assert result.token_usage is not None
        expected = 0.03 + 0.075
        assert abs(result.token_usage.cost - expected) < 1e-10

    @pytest.mark.asyncio
    async def test_zero_usage(self, provider):
        resp = _make_response(prompt_tokens=0, completion_tokens=0)
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.token_usage.cost == 0.0


class TestLiteLLMProviderRetry:
    @pytest.fixture
    def provider(self):
        config = _make_config()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_retries_on_rate_limit(self, provider):
        import litellm as litellm_mod

        mock_acomp = AsyncMock(side_effect=[
            litellm_mod.RateLimitError(
                message="rate limited", model="test",
                llm_provider="test",
            ),
            _make_response(),
        ])
        with (
            patch(_ACOMP, mock_acomp),
            patch(_SLEEP, new_callable=AsyncMock),
        ):
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.content == "Hello"
        assert mock_acomp.call_count == 2

    @pytest.mark.asyncio
    async def test_retries_on_server_error(self, provider):
        import litellm as litellm_mod

        mock_acomp = AsyncMock(side_effect=[
            litellm_mod.InternalServerError(
                message="server error", model="test",
                llm_provider="test",
            ),
            litellm_mod.ServiceUnavailableError(
                message="unavailable", model="test",
                llm_provider="test",
            ),
            _make_response(),
        ])
        with (
            patch(_ACOMP, mock_acomp),
            patch(_SLEEP, new_callable=AsyncMock),
        ):
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.content == "Hello"
        assert mock_acomp.call_count == 3

    @pytest.mark.asyncio
    async def test_auth_error_not_retried(self, provider):
        import litellm as litellm_mod

        mock_acomp = AsyncMock(
            side_effect=litellm_mod.AuthenticationError(
                message="bad key", model="test",
                llm_provider="test",
            ),
        )
        with patch(_ACOMP, mock_acomp), pytest.raises(
            RuntimeError, match="Authentication failed",
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert mock_acomp.call_count == 1

    @pytest.mark.asyncio
    async def test_bad_request_not_retried(self, provider):
        import litellm as litellm_mod

        mock_acomp = AsyncMock(
            side_effect=litellm_mod.BadRequestError(
                message="invalid params", model="test",
                llm_provider="test",
            ),
        )
        with patch(_ACOMP, mock_acomp), pytest.raises(
            RuntimeError, match="Bad request",
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert mock_acomp.call_count == 1

    @pytest.mark.asyncio
    async def test_all_retries_exhausted(self, provider):
        import litellm as litellm_mod

        mock_acomp = AsyncMock(
            side_effect=litellm_mod.RateLimitError(
                message="rate limited", model="test",
                llm_provider="test",
            ),
        )
        with (
            patch(_ACOMP, mock_acomp),
            patch(_SLEEP, new_callable=AsyncMock),
            pytest.raises(RuntimeError, match="failed after 3"),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert mock_acomp.call_count == 3

    @pytest.mark.asyncio
    async def test_timeout_raises_timeout_error(self, provider):
        mock_acomp = AsyncMock(side_effect=TimeoutError())
        with (
            patch(_ACOMP, mock_acomp),
            patch(_SLEEP, new_callable=AsyncMock),
            pytest.raises(TimeoutError, match="timed out"),
        ):
            await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
                timeout=30,
            )


class TestLiteLLMProviderEdgeCases:
    @pytest.fixture
    def provider(self):
        config = _make_config()
        with patch.dict("os.environ", {"ANTHROPIC_API_KEY": "sk-test"}):
            return LiteLLMProvider(config)

    @pytest.mark.asyncio
    async def test_empty_response_content(self, provider):
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = _make_response(content="")
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.content == ""
        assert result.code_blocks == []

    @pytest.mark.asyncio
    async def test_none_content_handled(self, provider):
        msg = SimpleNamespace(content=None, tool_calls=None)
        choice = SimpleNamespace(
            message=msg, finish_reason="stop", index=0,
        )
        usage = SimpleNamespace(
            prompt_tokens=10, completion_tokens=5, total_tokens=15,
        )
        resp = SimpleNamespace(
            choices=[choice], usage=usage, model="test",
        )
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.content == ""

    @pytest.mark.asyncio
    async def test_no_choices_handled(self, provider):
        usage = SimpleNamespace(
            prompt_tokens=0, completion_tokens=0, total_tokens=0,
        )
        resp = SimpleNamespace(
            choices=[], usage=usage, model="test",
        )
        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.content == ""

    @pytest.mark.asyncio
    async def test_missing_usage_handled(self, provider):
        msg = SimpleNamespace(content="hello", tool_calls=None)
        choice = SimpleNamespace(
            message=msg, finish_reason="stop", index=0,
        )
        resp = SimpleNamespace(choices=[choice], model="test")

        with patch(_ACOMP, new_callable=AsyncMock) as mock:
            mock.return_value = resp
            result = await provider.complete(
                messages=[{"role": "user", "content": "test"}],
                system="test",
            )

        assert result.token_usage.prompt_tokens == 0
        assert result.token_usage.completion_tokens == 0
        assert result.token_usage.cost == 0.0

"""Tests for the streaming provider implementation."""

from __future__ import annotations

import asyncio
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.providers.base import ModelProvider
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.messages import AgentMessage
from triad.schemas.pipeline import ModelConfig, RoleFitness
from triad.schemas.streaming import StreamChunk


def _make_config() -> ModelConfig:
    return ModelConfig(
        provider="test",
        model="test-model-v1",
        display_name="Test Model",
        api_key_env="TEST_API_KEY",
        context_window=100000,
        cost_input=1.0,
        cost_output=2.0,
        fitness=RoleFitness(architect=0.9, implementer=0.8, refactorer=0.7, verifier=0.6),
    )


class TestBaseProviderStreaming:
    """Test that the base class default falls back to complete()."""

    def test_complete_streaming_exists(self):
        """Base class has complete_streaming method."""
        assert hasattr(ModelProvider, "complete_streaming")

    @pytest.mark.asyncio
    async def test_default_falls_back_to_complete(self):
        """Default complete_streaming falls back to complete()."""
        config = _make_config()
        provider = LiteLLMProvider(config)

        # Mock complete to avoid actual API calls
        mock_msg = AgentMessage(
            from_agent="architect",
            to_agent="implement",
            msg_type="implementation",
            content="test",
            confidence=0.9,
            model="test-model",
        )

        with patch.object(provider, "complete", new_callable=AsyncMock, return_value=mock_msg):
            result = await ModelProvider.complete_streaming(
                provider,
                messages=[{"role": "user", "content": "test"}],
                system="test system",
            )
            assert result.content == "test"


class TestLiteLLMProviderStreaming:
    @pytest.mark.asyncio
    async def test_complete_streaming_method_exists(self):
        config = _make_config()
        provider = LiteLLMProvider(config)
        assert hasattr(provider, "complete_streaming")

    @pytest.mark.asyncio
    async def test_streaming_calls_on_chunk(self):
        """Streaming invokes on_chunk callback for each delta."""
        config = _make_config()
        provider = LiteLLMProvider(config)

        # Create mock streaming response
        chunks_received: list[StreamChunk] = []

        def on_chunk(chunk):
            chunks_received.append(chunk)

        # Mock the streaming response
        mock_chunk1 = MagicMock()
        mock_chunk1.choices = [MagicMock()]
        mock_chunk1.choices[0].delta = MagicMock()
        mock_chunk1.choices[0].delta.content = "hello "

        mock_chunk2 = MagicMock()
        mock_chunk2.choices = [MagicMock()]
        mock_chunk2.choices[0].delta = MagicMock()
        mock_chunk2.choices[0].delta.content = "world"
        mock_chunk2.usage = None

        async def mock_aiter():
            yield mock_chunk1
            yield mock_chunk2

        mock_response = mock_aiter()

        with patch.object(
            provider, "_call_streaming_with_retry",
            new_callable=AsyncMock,
            return_value=mock_response,
        ):
            result = await provider.complete_streaming(
                messages=[{"role": "user", "content": "test"}],
                system="system prompt",
                on_chunk=on_chunk,
            )

        assert result.content == "hello world"
        # on_chunk called for each delta + final
        assert len(chunks_received) >= 2
        # Last chunk should be the complete marker
        assert chunks_received[-1].is_complete is True

    @pytest.mark.asyncio
    async def test_streaming_returns_agent_message(self):
        """Streaming returns a proper AgentMessage."""
        config = _make_config()
        provider = LiteLLMProvider(config)

        mock_chunk = MagicMock()
        mock_chunk.choices = [MagicMock()]
        mock_chunk.choices[0].delta = MagicMock()
        mock_chunk.choices[0].delta.content = "code output"
        mock_chunk.usage = None

        async def mock_aiter():
            yield mock_chunk

        with patch.object(
            provider, "_call_streaming_with_retry",
            new_callable=AsyncMock,
            return_value=mock_aiter(),
        ):
            result = await provider.complete_streaming(
                messages=[{"role": "user", "content": "test"}],
                system="system prompt",
            )

        assert isinstance(result, AgentMessage)
        assert result.content == "code output"
        assert result.token_usage is not None

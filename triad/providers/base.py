"""Abstract base class for all model providers.

Defines the ModelProvider interface that every LLM adapter must implement.
The orchestrator interacts exclusively through this interface — it never
calls provider SDKs directly.
"""

from __future__ import annotations

from abc import ABC, abstractmethod

from pydantic import BaseModel

from triad.schemas.messages import AgentMessage
from triad.schemas.pipeline import ModelConfig, RoleFitness


class ModelProvider(ABC):
    """Abstract interface for any LLM that can participate in the pipeline.

    Initialized from a ModelConfig loaded from the TOML registry. Exposes
    identity, capabilities, cost info, and a single async complete() method
    that all providers must implement.
    """

    def __init__(self, config: ModelConfig) -> None:
        self._config = config

    # ── Identity ──────────────────────────────────────────────

    @property
    def provider_id(self) -> str:
        """Provider identifier (e.g. 'anthropic', 'openai', 'xai')."""
        return self._config.provider

    @property
    def model_id(self) -> str:
        """LiteLLM model identifier used for routing."""
        return self._config.model

    @property
    def display_name(self) -> str:
        """Human-friendly model name for CLI output."""
        return self._config.display_name

    # ── Capabilities ──────────────────────────────────────────

    @property
    def context_window(self) -> int:
        """Maximum context window size in tokens."""
        return self._config.context_window

    @property
    def supports_tools(self) -> bool:
        """Whether the model supports tool/function calling."""
        return self._config.supports_tools

    @property
    def supports_structured(self) -> bool:
        """Whether the model supports structured JSON output."""
        return self._config.supports_structured

    @property
    def supports_vision(self) -> bool:
        """Whether the model supports image inputs."""
        return self._config.supports_vision

    @property
    def supports_thinking(self) -> bool:
        """Whether the model supports extended thinking."""
        return self._config.supports_thinking

    # ── Cost ──────────────────────────────────────────────────

    @property
    def cost_per_1m_input(self) -> float:
        """Cost per 1M input tokens in USD."""
        return self._config.cost_input

    @property
    def cost_per_1m_output(self) -> float:
        """Cost per 1M output tokens in USD."""
        return self._config.cost_output

    # ── Fitness ───────────────────────────────────────────────

    @property
    def fitness(self) -> RoleFitness:
        """Role fitness scores from benchmarks."""
        return self._config.fitness

    @property
    def config(self) -> ModelConfig:
        """The full ModelConfig backing this provider."""
        return self._config

    # ── Core interface ────────────────────────────────────────

    @abstractmethod
    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str,
        *,
        output_schema: type[BaseModel] | None = None,
        timeout: int = 120,
    ) -> AgentMessage:
        """Send a completion request and return a structured AgentMessage.

        This is the only method a provider must implement. The orchestrator
        calls this for every pipeline stage and Arbiter review.

        Args:
            messages: Conversation messages in OpenAI format
                      (list of {"role": ..., "content": ...} dicts).
            system: System prompt for this call.
            output_schema: Optional Pydantic model for structured output.
                           When provided and the model supports it, the
                           provider should request JSON mode. When the model
                           doesn't support structured output or parsing fails,
                           fall back to text extraction.
            timeout: Timeout in seconds for the model call.

        Returns:
            An AgentMessage with content, code_blocks, token_usage, and model
            fields populated from the provider response.

        Raises:
            TimeoutError: If the model call exceeds the timeout.
            RuntimeError: If the model call fails after all retries.
        """

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str,
        *,
        timeout: int = 120,
        on_chunk: object | None = None,
    ) -> AgentMessage:
        """Send a streaming completion request.

        Default implementation falls back to non-streaming complete().
        Providers that support streaming should override this method.

        Args:
            messages: Conversation messages in OpenAI format.
            system: System prompt for this call.
            timeout: Timeout in seconds.
            on_chunk: Optional callback invoked with each StreamChunk.

        Returns:
            An AgentMessage with the full accumulated response.
        """
        return await self.complete(messages, system, timeout=timeout)

    def calculate_cost(self, prompt_tokens: int, completion_tokens: int) -> float:
        """Calculate the USD cost for a given token count.

        Args:
            prompt_tokens: Number of input tokens.
            completion_tokens: Number of output tokens.

        Returns:
            Estimated cost in USD.
        """
        input_cost = (prompt_tokens / 1_000_000) * self.cost_per_1m_input
        output_cost = (completion_tokens / 1_000_000) * self.cost_per_1m_output
        return input_cost + output_cost

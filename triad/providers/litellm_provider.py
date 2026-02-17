"""Universal LiteLLM adapter implementing the ModelProvider interface.

Routes completion requests to any LLM provider via LiteLLM's unified API.
Handles structured output parsing, code block extraction, token tracking,
cost calculation, timeouts, and retry with exponential backoff.
"""

from __future__ import annotations

import asyncio
import json
import logging
import os
import re

import litellm
from pydantic import BaseModel, ValidationError

# Suppress LiteLLM's "Give Feedback / Get Help" and "Provider List" banners
litellm.suppress_debug_info = True

from triad.providers.base import ModelProvider
from triad.schemas.messages import (
    AgentMessage,
    CodeBlock,
    MessageType,
    PipelineStage,
    TokenUsage,
)
from triad.schemas.pipeline import ModelConfig

logger = logging.getLogger(__name__)

# Regex for extracting fenced code blocks from markdown
_CODE_BLOCK_RE = re.compile(
    r"```(\w+)?\s*\n"       # opening fence with optional language
    r"(.*?)"                 # code content (non-greedy)
    r"\n```",                # closing fence
    re.DOTALL,
)

# Regex for extracting filepath hints from comments before code blocks
# Matches patterns like: # file: path/to/file.py  or  // filepath: src/main.ts
_FILEPATH_RE = re.compile(
    r"(?:#|//)\s*(?:file(?:path)?|File(?:path)?)\s*:\s*(\S+)"
)

# Max retries for transient failures
_MAX_RETRIES = 3
_BASE_BACKOFF = 1.0  # seconds


class LiteLLMProvider(ModelProvider):
    """Universal LLM adapter powered by LiteLLM.

    Routes calls to any provider (Anthropic, OpenAI, Google, xAI, etc.)
    through litellm.acompletion(). This is the ONLY way models are called
    in the Triad pipeline — no direct SDK imports anywhere else.
    """

    def __init__(self, config: ModelConfig) -> None:
        super().__init__(config)
        # Resolve API key from environment
        self._api_key = os.environ.get(config.api_key_env, "")

    async def complete(
        self,
        messages: list[dict[str, str]],
        system: str,
        *,
        output_schema: type[BaseModel] | None = None,
        timeout: int = 120,
    ) -> AgentMessage:
        """Send a completion request via LiteLLM and return an AgentMessage.

        Attempts structured output (JSON mode) when output_schema is provided
        and the model supports it. Falls back to text extraction on failure.

        Args:
            messages: Conversation messages in OpenAI format.
            system: System prompt for this call.
            output_schema: Optional Pydantic model for structured output.
            timeout: Timeout in seconds for the model call.

        Returns:
            AgentMessage populated with content, code_blocks, token_usage.

        Raises:
            TimeoutError: If the call exceeds timeout after all retries.
            RuntimeError: If the call fails after all retries.
        """
        # Build the full messages list with system prompt prepended
        full_messages = [{"role": "system", "content": system}, *messages]

        # Build kwargs for litellm.acompletion
        kwargs = self._build_completion_kwargs(full_messages, output_schema, timeout)

        # Call with retry
        response = await self._call_with_retry(kwargs)

        # Extract content from response
        content = self._extract_content(response)

        # Try structured output parsing (validates but uses raw content)
        if output_schema and content:
            self._try_parse_structured(content, output_schema)

        # Extract code blocks from markdown content
        code_blocks = extract_code_blocks(content)

        # Build token usage
        token_usage = self._build_token_usage(response)

        return AgentMessage(
            from_agent=PipelineStage.ARCHITECT,  # overridden by orchestrator
            to_agent=PipelineStage.IMPLEMENT,     # overridden by orchestrator
            msg_type=MessageType.IMPLEMENTATION,  # overridden by orchestrator
            content=content,
            code_blocks=code_blocks,
            confidence=0.0,  # set by the agent/orchestrator from response content
            token_usage=token_usage,
            model=self._config.model,
        )

    def _build_completion_kwargs(
        self,
        messages: list[dict[str, str]],
        output_schema: type[BaseModel] | None,
        timeout: int,
    ) -> dict:
        """Build the kwargs dict for litellm.acompletion."""
        kwargs: dict = {
            "model": self._config.model,
            "messages": messages,
            "timeout": float(timeout),
        }

        # Set API key if available
        if self._api_key:
            kwargs["api_key"] = self._api_key

        # Set custom API base if configured
        if self._config.api_base:
            kwargs["api_base"] = self._config.api_base

        # Request structured output when model supports it
        if output_schema and self._config.supports_structured:
            kwargs["response_format"] = output_schema

        return kwargs

    async def _call_with_retry(self, kwargs: dict) -> litellm.ModelResponse:
        """Call litellm.acompletion with exponential backoff retry.

        Retries on transient errors (rate limits, server errors, timeouts).
        Non-retryable errors (auth, invalid request) are raised immediately.

        Raises:
            TimeoutError: If all retries time out.
            RuntimeError: If all retries fail with non-timeout errors.
        """
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.acompletion(**kwargs)
                return response
            except TimeoutError:
                last_error = TimeoutError(
                    f"Model call timed out after {kwargs.get('timeout')}s "
                    f"(attempt {attempt + 1}/{_MAX_RETRIES})"
                )
            except litellm.AuthenticationError:
                raise RuntimeError(
                    f"Authentication failed for {self._config.model}. "
                    f"Check that {self._config.api_key_env} is set correctly."
                ) from None
            except litellm.BadRequestError as e:
                raise RuntimeError(
                    f"Bad request to {self._config.model}: {e}"
                ) from e
            except (
                litellm.RateLimitError,
                litellm.ServiceUnavailableError,
                litellm.InternalServerError,
                litellm.APIConnectionError,
            ) as e:
                last_error = e

            # Exponential backoff with jitter avoidance
            if attempt < _MAX_RETRIES - 1:
                backoff = _BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "Retry %d/%d for %s after error: %s (backoff: %.1fs)",
                    attempt + 1,
                    _MAX_RETRIES,
                    self._config.model,
                    last_error,
                    backoff,
                )
                await asyncio.sleep(backoff)

        # All retries exhausted
        if isinstance(last_error, TimeoutError):
            raise last_error
        raise RuntimeError(
            f"Model call to {self._config.model} failed after {_MAX_RETRIES} "
            f"retries: {last_error}"
        ) from last_error

    def _extract_content(self, response: litellm.ModelResponse) -> str:
        """Extract text content from a LiteLLM response."""
        if not response.choices:
            return ""
        message = response.choices[0].message
        return message.content or "" if message else ""

    def _try_parse_structured(
        self, content: str, schema: type[BaseModel]
    ) -> BaseModel | None:
        """Attempt to parse content as structured JSON matching the schema.

        Returns the parsed model instance, or None if parsing fails.
        """
        try:
            # Try direct JSON parse
            data = json.loads(content)
            return schema.model_validate(data)
        except (json.JSONDecodeError, ValidationError):
            pass

        # Try extracting JSON from markdown code block
        json_match = re.search(r"```(?:json)?\s*\n(.*?)\n```", content, re.DOTALL)
        if json_match:
            try:
                data = json.loads(json_match.group(1))
                return schema.model_validate(data)
            except (json.JSONDecodeError, ValidationError):
                pass

        logger.debug("Structured output parsing failed for %s, using raw text", self._config.model)
        return None

    def _build_token_usage(self, response: litellm.ModelResponse) -> TokenUsage:
        """Build TokenUsage from the LiteLLM response usage data."""
        usage = getattr(response, "usage", None)
        prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
        completion_tokens = getattr(usage, "completion_tokens", 0) or 0
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        return TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )


def extract_code_blocks(content: str) -> list[CodeBlock]:
    """Extract fenced code blocks from markdown-formatted content.

    Looks for ```language ... ``` patterns and optionally extracts filepath
    hints from comments immediately preceding the code block.

    Args:
        content: The raw markdown content to parse.

    Returns:
        List of CodeBlock instances extracted from the content.
    """
    if not content:
        return []

    blocks: list[CodeBlock] = []

    for match in _CODE_BLOCK_RE.finditer(content):
        language = match.group(1) or "text"
        code_content = match.group(2)

        # Look for a filepath hint in the line(s) before this code block
        block_start = content[:match.start()].rstrip()
        preceding_lines = block_start.split("\n")
        filepath = ""

        if preceding_lines:
            last_line = preceding_lines[-1].strip()
            fp_match = _FILEPATH_RE.search(last_line)
            if fp_match:
                filepath = fp_match.group(1)

        # If no filepath hint found, generate one from language
        if not filepath:
            filepath = f"untitled.{_language_extension(language)}"

        blocks.append(
            CodeBlock(
                language=language,
                filepath=filepath,
                content=code_content.strip(),
            )
        )

    return blocks


def _language_extension(language: str) -> str:
    """Map a language identifier to a file extension."""
    extensions = {
        "python": "py",
        "javascript": "js",
        "typescript": "ts",
        "rust": "rs",
        "ruby": "rb",
        "go": "go",
        "java": "java",
        "c": "c",
        "cpp": "cpp",
        "csharp": "cs",
        "html": "html",
        "css": "css",
        "sql": "sql",
        "bash": "sh",
        "shell": "sh",
        "yaml": "yaml",
        "yml": "yaml",
        "toml": "toml",
        "json": "json",
        "markdown": "md",
        "text": "txt",
    }
    return extensions.get(language.lower(), language.lower())

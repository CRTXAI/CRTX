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
from triad.schemas.streaming import StreamChunk

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


def _short_error_reason(error: Exception) -> str:
    """Extract a short, user-friendly reason from a LiteLLM error.

    Maps error types and status codes to concise descriptions instead
    of dumping full JSON error payloads.
    """
    error_str = str(error).lower()
    if "rate" in error_str or "429" in error_str:
        return "rate limit"
    if "overloaded" in error_str or "529" in error_str:
        return "overloaded"
    if "timeout" in error_str or isinstance(error, TimeoutError):
        return "timeout"
    if "503" in error_str or "unavailable" in error_str:
        return "service unavailable"
    if "500" in error_str or "internal" in error_str:
        return "server error"
    if "connection" in error_str:
        return "connection error"
    # Fallback: first 80 chars of the error
    return str(error)[:80]


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

    async def complete_streaming(
        self,
        messages: list[dict[str, str]],
        system: str,
        *,
        timeout: int = 120,
        on_chunk: object | None = None,
    ) -> AgentMessage:
        """Send a streaming completion request via LiteLLM.

        Streams token-by-token, invoking on_chunk for each delta.
        Returns the same AgentMessage contract as complete().

        Args:
            messages: Conversation messages in OpenAI format.
            system: System prompt for this call.
            timeout: Timeout in seconds.
            on_chunk: Optional callback invoked with each StreamChunk.

        Returns:
            AgentMessage with the full accumulated response.
        """
        full_messages = [{"role": "system", "content": system}, *messages]
        kwargs = self._build_completion_kwargs(full_messages, None, timeout)
        kwargs["stream"] = True

        accumulated = ""
        token_count = 0

        response = await self._call_streaming_with_retry(kwargs)

        async for chunk in response:
            delta = ""
            if chunk.choices and chunk.choices[0].delta:
                delta = chunk.choices[0].delta.content or ""

            if delta:
                accumulated += delta
                token_count += 1  # approximate, 1 chunk ~= 1 token

                if on_chunk is not None:
                    stream_chunk = StreamChunk(
                        delta=delta,
                        accumulated=accumulated,
                        token_count=token_count,
                        is_complete=False,
                    )
                    result = on_chunk(stream_chunk)
                    if asyncio.iscoroutine(result):
                        await result

        # Final chunk
        if on_chunk is not None:
            final_chunk = StreamChunk(
                delta="",
                accumulated=accumulated,
                token_count=token_count,
                is_complete=True,
            )
            result = on_chunk(final_chunk)
            if asyncio.iscoroutine(result):
                await result

        # Extract code blocks from accumulated content
        code_blocks = extract_code_blocks(accumulated)

        # Approximate token usage (streaming doesn't always give exact counts)
        # Try to get usage from the last chunk if available
        prompt_tokens = 0
        completion_tokens = token_count
        usage = getattr(chunk, "usage", None) if chunk else None
        if usage:
            prompt_tokens = getattr(usage, "prompt_tokens", 0) or 0
            completion_tokens = getattr(usage, "completion_tokens", 0) or token_count
        cost = self.calculate_cost(prompt_tokens, completion_tokens)

        token_usage = TokenUsage(
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cost=cost,
        )

        return AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.IMPLEMENTATION,
            content=accumulated,
            code_blocks=code_blocks,
            confidence=0.0,
            token_usage=token_usage,
            model=self._config.model,
        )

    async def _call_streaming_with_retry(self, kwargs: dict):
        """Call litellm.acompletion with stream=True and retry on failure.

        Same 3-attempt backoff as _call_with_retry, but restarts the
        entire stream on failure.
        """
        last_error: Exception | None = None

        for attempt in range(_MAX_RETRIES):
            try:
                response = await litellm.acompletion(**kwargs)
                return response
            except TimeoutError:
                last_error = TimeoutError(
                    f"Streaming call timed out after {kwargs.get('timeout')}s "
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

            if attempt < _MAX_RETRIES - 1:
                backoff = _BASE_BACKOFF * (2**attempt)
                logger.warning(
                    "Streaming retry %d/%d for %s (%s, backoff: %.1fs)",
                    attempt + 1, _MAX_RETRIES, self._config.display_name,
                    _short_error_reason(last_error), backoff,
                )
                await asyncio.sleep(backoff)

        if isinstance(last_error, TimeoutError):
            raise last_error
        raise RuntimeError(
            f"Streaming call to {self._config.model} failed after "
            f"{_MAX_RETRIES} retries: {last_error}"
        ) from last_error

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
                reason = _short_error_reason(last_error)
                logger.warning(
                    "Retry %d/%d for %s (%s, backoff: %.1fs)",
                    attempt + 1,
                    _MAX_RETRIES,
                    self._config.display_name,
                    reason,
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
    hints from comments immediately preceding the code block or on the
    first line inside it.

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

        # Check the first line inside the code block for a filepath hint
        if not filepath and code_content.strip():
            first_line = code_content.strip().split("\n", 1)[0]
            fp_match = _FILEPATH_RE.search(first_line)
            if fp_match:
                filepath = fp_match.group(1)
                # Strip the filepath hint line from the code content
                lines = code_content.strip().split("\n", 1)
                code_content = lines[1] if len(lines) > 1 else ""

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

    merged = _merge_filepath_blocks(blocks)
    return _filter_untitled_fragments(merged)


def _merge_filepath_blocks(blocks: list[CodeBlock]) -> list[CodeBlock]:
    """Merge standalone filepath-only blocks with the following code block.

    Handles the pattern where an LLM produces a block containing only a
    ``# file: path`` hint followed by a separate block with the actual code.
    """
    if len(blocks) < 2:
        return blocks

    merged: list[CodeBlock] = []
    skip_next = False

    for i, block in enumerate(blocks):
        if skip_next:
            skip_next = False
            continue

        # A block is "filepath-only" when it has no meaningful code content
        # and its filepath is not a generated default (untitled.*)
        is_hint_only = (
            not block.content.strip()
            and not block.filepath.startswith("untitled.")
        )

        if is_hint_only and i + 1 < len(blocks):
            next_block = blocks[i + 1]
            merged.append(
                CodeBlock(
                    language=next_block.language,
                    filepath=block.filepath,
                    content=next_block.content,
                )
            )
            skip_next = True
        else:
            merged.append(block)

    return merged


def _filter_untitled_fragments(blocks: list[CodeBlock]) -> list[CodeBlock]:
    """Filter out untitled code blocks that are clearly not real files.

    Discards blocks where the content is ONLY a filepath comment with no
    actual code (e.g., ``# file: src/email_validator.py`` and nothing else).
    These arise when ``_merge_filepath_blocks`` can't merge a standalone
    hint block.
    """
    # Regex for blocks that are only a filepath comment
    _FILEPATH_ONLY = re.compile(
        r"^\s*(?:#|//)\s*(?:file(?:path)?|File(?:path)?)\s*:\s*\S+\s*$"
    )

    filtered: list[CodeBlock] = []
    for block in blocks:
        content = block.content.strip()

        # Discard blocks that are only a filepath comment line
        if content and _FILEPATH_ONLY.match(content) and "\n" not in content:
            continue

        # Discard any block with no code content
        if not content:
            continue

        filtered.append(block)

    return filtered


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

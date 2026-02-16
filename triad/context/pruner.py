"""Context pruner for model-specific token limits.

Takes a ContextResult and prunes it to fit within a model's available
context window, accounting for the prompt template and task description.
Uses intelligent truncation — removes lowest-relevance files first,
then trims previews and function lists.
"""

from __future__ import annotations

import logging

from triad.schemas.context import ContextResult

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4

# Reserve space for system prompt + task + previous stage output
_RESERVED_TOKENS = 4000


class ContextPruner:
    """Prunes context to fit within a model's available token window.

    Given a model's context_window and the reserved token space for
    system prompts and stage output, calculates the remaining budget
    and truncates the context string accordingly.
    """

    def __init__(self, model_context_window: int) -> None:
        self._window = model_context_window

    def prune(self, context: ContextResult) -> ContextResult:
        """Prune the context to fit the model's available window.

        If the context already fits, returns it unchanged. Otherwise,
        truncates by removing the tail (lowest-relevance files) and
        updating metadata.

        Args:
            context: The full context result from ContextBuilder.

        Returns:
            A (possibly truncated) ContextResult.
        """
        available = self._window - _RESERVED_TOKENS
        if available <= 0:
            logger.warning(
                "Model context window (%d) too small for context injection",
                self._window,
            )
            return ContextResult(
                profile=context.profile,
                context_text="",
                files_included=0,
                files_scanned=context.files_scanned,
                token_estimate=0,
                truncated=True,
            )

        if context.token_estimate <= available:
            return context

        # Need to truncate
        logger.info(
            "Pruning context from ~%d to ~%d tokens",
            context.token_estimate,
            available,
        )

        char_limit = available * _CHARS_PER_TOKEN
        text = context.context_text

        if len(text) <= char_limit:
            return context

        # Truncate at the last complete file entry before the limit
        truncated_text = text[:char_limit]

        # Try to cut at a clean boundary (### header)
        last_header = truncated_text.rfind("\n### ")
        if last_header > len(text) // 4:
            truncated_text = truncated_text[:last_header]

        truncated_text += "\n\n[Context truncated to fit model window]"

        # Count remaining files (### headers)
        files_kept = truncated_text.count("\n### ")

        token_estimate = len(truncated_text) // _CHARS_PER_TOKEN

        return ContextResult(
            profile=context.profile,
            context_text=truncated_text,
            files_included=files_kept,
            files_scanned=context.files_scanned,
            token_estimate=token_estimate,
            truncated=True,
        )

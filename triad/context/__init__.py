"""Context injection module for codebase-aware pipeline runs.

Scans a project directory, builds a ranked context string from the most
relevant files, and prunes it to fit within a model's token budget.
"""

from triad.context.builder import ContextBuilder
from triad.context.pruner import ContextPruner
from triad.context.scanner import CodeScanner

__all__ = [
    "CodeScanner",
    "ContextBuilder",
    "ContextPruner",
]

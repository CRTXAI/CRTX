"""CI/CD integration module for multi-model code review.

Provides ReviewRunner for parallel diff review across models, and
formatters for GitHub comments, Markdown summary, and CI exit codes.
"""

from triad.ci.formatter import format_exit_code, format_github_comments, format_summary
from triad.ci.reviewer import ReviewRunner

__all__ = [
    "ReviewRunner",
    "format_exit_code",
    "format_github_comments",
    "format_summary",
]

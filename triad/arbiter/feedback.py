"""Structured feedback injection for Arbiter REJECT retries.

Formats an ArbiterReview into a readable feedback block for injection into
the system prompt when a stage is retried after a REJECT verdict.
"""

from __future__ import annotations

from triad.schemas.arbiter import ArbiterReview, Severity


def format_arbiter_feedback(review: ArbiterReview, retry_number: int) -> str:
    """Format an ArbiterReview into structured feedback for prompt injection.

    Produces a Markdown block with critical issues first, then warnings,
    then alternatives. Included in the retry prompt so the re-generating
    model knows exactly what the Arbiter objected to.

    Args:
        review: The ArbiterReview that triggered the REJECT.
        retry_number: Which retry attempt this is (1-based).

    Returns:
        Formatted feedback string ready for template injection.
    """
    max_retries = 2  # pipeline default, matches PipelineConfig.max_retries
    lines: list[str] = [
        f"## ARBITER FEEDBACK — REJECTED (Retry {retry_number} of {max_retries})",
        "",
        f"Your previous output was reviewed by an independent Arbiter "
        f"({review.arbiter_model}) and **REJECTED**.",
        "",
    ]

    # Split issues by severity
    critical = [i for i in review.issues if i.severity == Severity.CRITICAL]
    warnings = [i for i in review.issues if i.severity == Severity.WARNING]

    if critical:
        lines.append("### CRITICAL ISSUES (must fix)")
        lines.append("")
        for idx, issue in enumerate(critical, 1):
            lines.append(f"{idx}. **[{issue.category.value}]** {issue.description}")
            if issue.location:
                lines.append(f"   Location: {issue.location}")
            if issue.suggestion:
                lines.append(f"   Fix: {issue.suggestion}")
            if issue.evidence:
                lines.append(f"   Evidence: {issue.evidence}")
            lines.append("")

    if warnings:
        lines.append("### WARNINGS (should fix)")
        lines.append("")
        for idx, issue in enumerate(warnings, 1):
            lines.append(f"{idx}. **[{issue.category.value}]** {issue.description}")
            if issue.location:
                lines.append(f"   Location: {issue.location}")
            if issue.suggestion:
                lines.append(f"   Fix: {issue.suggestion}")
            lines.append("")

    if review.alternatives:
        lines.append("### ALTERNATIVES TO CONSIDER")
        lines.append("")
        for alt in review.alternatives:
            lines.append(f"- **{alt.description}** (confidence: {alt.confidence:.2f})")
            lines.append(f"  Rationale: {alt.rationale}")
            if alt.code_sketch:
                lines.append(f"  ```\n  {alt.code_sketch}\n  ```")
            lines.append("")

    lines.append("Address the issues above in your revised output.")

    return "\n".join(lines)

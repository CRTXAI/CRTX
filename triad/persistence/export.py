"""Session export formatters.

Provides JSON and Markdown export functions for session records.
"""

from __future__ import annotations

from triad.schemas.session import SessionRecord


def export_json(record: SessionRecord) -> str:
    """Export a session record as a formatted JSON string.

    Returns:
        Pretty-printed JSON string of the full session record.
    """
    return record.model_dump_json(indent=2)


def export_markdown(record: SessionRecord) -> str:
    """Export a session record as a human-readable Markdown report.

    Generates a structured Markdown document with sections for
    session metadata, stage outputs, arbiter reviews, routing
    decisions, and aggregate metrics.

    Returns:
        Markdown-formatted string.
    """
    lines: list[str] = []

    # Header
    lines.append(f"# Session Report: {record.session_id}")
    lines.append("")

    # Metadata
    lines.append("## Metadata")
    lines.append("")
    lines.append(f"- **Task:** {record.task.task}")
    if record.task.context:
        lines.append(f"- **Context:** {record.task.context}")
    lines.append(f"- **Pipeline Mode:** {record.pipeline_mode}")
    lines.append(f"- **Started:** {record.started_at.isoformat()}")
    if record.completed_at:
        lines.append(f"- **Completed:** {record.completed_at.isoformat()}")
    lines.append(f"- **Duration:** {record.duration_seconds:.1f}s")
    lines.append(f"- **Total Cost:** ${record.total_cost:.4f}")
    lines.append(f"- **Total Tokens:** {record.total_tokens:,}")
    status = "Halted" if record.halted else ("Success" if record.success else "Failed")
    lines.append(f"- **Status:** {status}")
    if record.halted and record.halt_reason:
        lines.append(f"- **Halt Reason:** {record.halt_reason}")
    lines.append("")

    # Stages
    if record.stages:
        lines.append("## Pipeline Stages")
        lines.append("")
        for stage in record.stages:
            lines.append(f"### {stage.stage.title()}")
            lines.append("")
            lines.append(f"- **Model:** {stage.model_key} (`{stage.model_id}`)")
            lines.append(f"- **Confidence:** {stage.confidence:.2f}")
            lines.append(f"- **Cost:** ${stage.cost:.4f}")
            lines.append(f"- **Tokens:** {stage.tokens:,}")
            if stage.timestamp:
                lines.append(f"- **Timestamp:** {stage.timestamp}")
            lines.append("")
            if stage.content:
                lines.append("<details>")
                lines.append(f"<summary>Stage output ({len(stage.content):,} chars)</summary>")
                lines.append("")
                lines.append(stage.content)
                lines.append("")
                lines.append("</details>")
                lines.append("")

    # Arbiter Reviews
    if record.arbiter_reviews:
        lines.append("## Arbiter Reviews")
        lines.append("")
        for review in record.arbiter_reviews:
            verdict_emoji = {
                "approve": "APPROVE",
                "flag": "FLAG",
                "reject": "REJECT",
                "halt": "HALT",
            }.get(review.verdict.value, review.verdict.value.upper())
            lines.append(
                f"### {review.stage_reviewed.value.title()} — {verdict_emoji}"
            )
            lines.append("")
            lines.append(f"- **Reviewed Model:** {review.reviewed_model}")
            lines.append(f"- **Arbiter Model:** {review.arbiter_model}")
            lines.append(f"- **Confidence:** {review.confidence:.2f}")
            lines.append(f"- **Cost:** ${review.token_cost:.4f}")
            lines.append("")
            if review.reasoning:
                lines.append(f"**Reasoning:** {review.reasoning}")
                lines.append("")
            if review.issues:
                lines.append("**Issues:**")
                lines.append("")
                for issue in review.issues:
                    loc = f" at `{issue.location}`" if issue.location else ""
                    lines.append(
                        f"- [{issue.severity.value.upper()}] "
                        f"[{issue.category.value}]{loc}: {issue.description}"
                    )
                    if issue.suggestion:
                        lines.append(f"  - Fix: {issue.suggestion}")
                lines.append("")

    # Routing Decisions
    if record.routing_decisions:
        lines.append("## Routing Decisions")
        lines.append("")
        lines.append("| Role | Model | Strategy | Fitness | Est. Cost |")
        lines.append("|------|-------|----------|---------|-----------|")
        for decision in record.routing_decisions:
            lines.append(
                f"| {decision.role.value} | {decision.model_key} | "
                f"{decision.strategy.value} | {decision.fitness_score:.2f} | "
                f"${decision.estimated_cost:.4f} |"
            )
        lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by CRTX*")
    lines.append("")

    return "\n".join(lines)

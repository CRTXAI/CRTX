"""Markdown summary generator for pipeline results.

Produces a human-readable summary.md covering task, pipeline mode,
models used, stage summaries, arbiter verdicts, routing decisions,
cost breakdown, and session ID.
"""

from __future__ import annotations

from triad.schemas.pipeline import PipelineResult


def _display_name_from_litellm_id(litellm_id: str) -> str:
    """Resolve a LiteLLM model ID to a human-friendly display name."""
    try:
        from triad.providers.registry import load_models
        registry = load_models()
        for cfg in registry.values():
            if cfg.model == litellm_id:
                return cfg.display_name
    except Exception:
        pass
    return litellm_id


def render_summary(result: PipelineResult) -> str:
    """Produce a Markdown summary report from a PipelineResult.

    Returns:
        Markdown-formatted string suitable for writing to summary.md.
    """
    lines: list[str] = []

    # Title
    lines.append("# Triad Pipeline Summary")
    lines.append("")

    # Session
    if result.session_id:
        lines.append(f"**Session:** `{result.session_id}`")
        lines.append("")

    # Task
    lines.append("## Task")
    lines.append("")
    lines.append(result.task.task)
    if result.task.context:
        lines.append("")
        lines.append(f"**Context:** {result.task.context}")
    lines.append("")

    # Pipeline Mode
    lines.append("## Pipeline Configuration")
    lines.append("")
    lines.append(f"- **Mode:** {result.config.pipeline_mode.value}")
    lines.append(f"- **Arbiter:** {result.config.arbiter_mode.value}")
    lines.append(
        f"- **Reconciliation:** "
        f"{'enabled' if result.config.reconciliation_enabled else 'disabled'}"
    )
    lines.append(f"- **Routing:** {result.config.routing_strategy.value}")
    lines.append("")

    # Status
    status = "HALTED" if result.halted else ("SUCCESS" if result.success else "FAILED")
    lines.append(f"## Result: {status}")
    lines.append("")
    if result.halted and result.halt_reason:
        lines.append(f"**Halt Reason:** {result.halt_reason}")
        lines.append("")

    # Models Used
    if result.routing_decisions:
        lines.append("## Models Used")
        lines.append("")
        lines.append("| Stage | Model | Strategy | Fitness | Est. Cost |")
        lines.append("|-------|-------|----------|---------|-----------|")
        for d in result.routing_decisions:
            lines.append(
                f"| {d.role.value} | {d.model_key} | "
                f"{d.strategy.value} | {d.fitness_score:.2f} | "
                f"${d.estimated_cost:.4f} |"
            )
        lines.append("")

    # Stage Summaries
    if result.stages:
        lines.append("## Stage Summaries")
        lines.append("")
        for stage, msg in result.stages.items():
            lines.append(f"### {stage.value.title()}")
            lines.append("")
            lines.append(f"- **Model:** {_display_name_from_litellm_id(msg.model)}")
            lines.append(f"- **Confidence:** {msg.confidence:.2f}")
            if msg.token_usage:
                total_tokens = msg.token_usage.prompt_tokens + msg.token_usage.completion_tokens
                lines.append(f"- **Tokens:** {total_tokens:,}")
                lines.append(f"- **Cost:** ${msg.token_usage.cost:.4f}")
            # Show first 500 chars of output
            preview = msg.content[:500]
            if len(msg.content) > 500:
                preview += "..."
            lines.append("")
            lines.append(f"> {preview}")
            lines.append("")

    # Parallel Mode Results
    if result.parallel_result:
        pr = result.parallel_result
        lines.append("## Parallel Exploration Results")
        lines.append("")
        lines.append(f"- **Winner:** {pr.winner}")
        lines.append(f"- **Models:** {len(pr.individual_outputs)}")
        lines.append("")

        # Voting summary
        if pr.votes:
            lines.append("### Votes")
            lines.append("")
            lines.append("| Voter | Voted For |")
            lines.append("|-------|-----------|")
            for voter, voted_for in pr.votes.items():
                lines.append(f"| {voter} | {voted_for} |")
            lines.append("")

        # Cross-review scores
        if pr.scores:
            lines.append("### Cross-Review Scores")
            lines.append("")
            lines.append("| Reviewer | Reviewed | Arch | Impl | Quality |")
            lines.append("|----------|----------|------|------|---------|")
            for reviewer, targets in pr.scores.items():
                for reviewed, scores in targets.items():
                    lines.append(
                        f"| {reviewer} | {reviewed} | "
                        f"{scores.get('architecture', '-')} | "
                        f"{scores.get('implementation', '-')} | "
                        f"{scores.get('quality', '-')} |"
                    )
            lines.append("")

        # Synthesized output preview
        if pr.synthesized_output:
            preview = pr.synthesized_output[:500]
            if len(pr.synthesized_output) > 500:
                preview += "..."
            lines.append("### Synthesized Output")
            lines.append("")
            lines.append(f"> {preview}")
            lines.append("")

    # Debate Mode Results
    if result.debate_result:
        dr = result.debate_result
        lines.append("## Debate Results")
        lines.append("")
        lines.append(f"- **Judge:** {dr.judge_model}")
        lines.append(f"- **Debaters:** {len(dr.proposals)}")
        lines.append("")

        # Proposals
        if dr.proposals:
            lines.append("### Position Papers")
            lines.append("")
            for model_key, proposal in dr.proposals.items():
                lines.append(f"#### {model_key}")
                lines.append("")
                preview = proposal[:300]
                if len(proposal) > 300:
                    preview += "..."
                lines.append(f"> {preview}")
                lines.append("")

        # Judgment preview
        if dr.judgment:
            preview = dr.judgment[:500]
            if len(dr.judgment) > 500:
                preview += "..."
            lines.append("### Judgment")
            lines.append("")
            lines.append(f"> {preview}")
            lines.append("")

    # Arbiter Verdicts
    if result.arbiter_reviews:
        lines.append("## Arbiter Verdicts")
        lines.append("")
        lines.append(
            "| Stage | Verdict | Arbiter Model | Confidence | Cost |"
        )
        lines.append(
            "|-------|---------|---------------|------------|------|"
        )
        for r in result.arbiter_reviews:
            lines.append(
                f"| {r.stage_reviewed.value} | "
                f"{r.verdict.value.upper()} | "
                f"{_display_name_from_litellm_id(r.arbiter_model)} | "
                f"{r.confidence:.2f} | "
                f"${r.token_cost:.4f} |"
            )
        lines.append("")

    # Cost Breakdown
    lines.append("## Cost Summary")
    lines.append("")
    lines.append(f"- **Total Cost:** ${result.total_cost:.4f}")
    lines.append(f"- **Total Tokens:** {result.total_tokens:,}")
    lines.append(f"- **Duration:** {result.duration_seconds:.1f}s")
    lines.append("")

    # Footer
    lines.append("---")
    lines.append("*Generated by Triad Orchestrator*")
    lines.append("")

    return "\n".join(lines)

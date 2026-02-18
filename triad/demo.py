"""Guided 60-second demo showcasing cross-model generation and review.

Generates code with one provider and reviews it with an arbiter from a
different provider, annotating every step with educational context.
"""

from __future__ import annotations

import os
import time
import uuid

import typer
from rich.console import Console
from rich.panel import Panel

from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.messages import (
    MessageType,
    PipelineStage,
)
from triad.schemas.pipeline import (
    ArbiterMode,
    ModelConfig,
    PipelineConfig,
    PipelineResult,
    TaskSpec,
)

# ── Constants ────────────────────────────────────────────────────

DEMO_TASK = (
    "Write a Python function that validates email addresses using regex, "
    "handles edge cases like plus-addressing and international domains, "
    "and includes comprehensive test cases."
)

DEMO_SYSTEM_PROMPT = (
    "You are an expert Python developer. Write clean, production-quality code.\n\n"
    "Task:\n{task}\n\n"
    "Return ONLY the code — no explanation, no markdown fences around the "
    "entire response. Use ```python fences for each code block with a "
    "# file: <path> comment on the first line inside the fence."
)

DEMO_ANNOTATIONS = {
    "routing": (
        "Why different providers? Models from the same company share training "
        "data and blind spots. Cross-provider review finds more bugs."
    ),
    "generating": (
        "The generator writes code like any AI coding tool. The difference "
        "is what happens next."
    ),
    "reviewing": (
        "This is the key innovation: an independent model reviewing for bugs, "
        "hallucinations, and security issues the generator can't see in its "
        "own output."
    ),
    "complete": (
        "This is what CRTX does at scale — multiple models generating, "
        "reviewing, debating, and improving code."
    ),
}

# ── Model selection ──────────────────────────────────────────────

# Preference lists — first match wins
_GEN_PREFERENCE = ["gpt-4o-mini", "gemini-flash", "claude-haiku"]
_ARB_PREFERENCE = ["claude-sonnet", "claude-haiku", "gemini-flash", "gpt-4o-mini"]


def select_demo_models(
    registry: dict[str, ModelConfig],
) -> tuple[tuple[str, ModelConfig], tuple[str, ModelConfig]]:
    """Pick a generator and arbiter from *different* providers.

    Args:
        registry: Full model registry from ``load_models()``.

    Returns:
        ``((gen_key, gen_cfg), (arb_key, arb_cfg))``

    Raises:
        RuntimeError: If fewer than two providers have reachable models.
    """
    # Filter to reachable models (API key present in env)
    reachable = {
        k: v for k, v in registry.items()
        if os.environ.get(v.api_key_env)
    }

    # Group by provider
    providers: dict[str, list[tuple[str, ModelConfig]]] = {}
    for k, v in reachable.items():
        providers.setdefault(v.provider, []).append((k, v))

    if len(providers) < 2:
        n = len(providers)
        raise RuntimeError(
            f"Demo requires models from at least 2 providers, but only "
            f"{n} provider{'s' if n != 1 else ''} {'are' if n != 1 else 'is'} "
            f"reachable. Run [bold]crtx setup[/bold] to configure more API keys."
        )

    # --- Generator ---
    gen_key, gen_cfg = _pick_preferred(reachable, _GEN_PREFERENCE)

    # --- Arbiter (must be different provider) ---
    arb_candidates = {
        k: v for k, v in reachable.items()
        if v.provider != gen_cfg.provider
    }
    arb_key, arb_cfg = _pick_preferred(arb_candidates, _ARB_PREFERENCE)

    return (gen_key, gen_cfg), (arb_key, arb_cfg)


def _pick_preferred(
    candidates: dict[str, ModelConfig],
    preference: list[str],
) -> tuple[str, ModelConfig]:
    """Return the first preferred key present, else cheapest."""
    for key in preference:
        if key in candidates:
            return key, candidates[key]
    # Fallback: cheapest by total cost
    cheapest_key = min(
        candidates,
        key=lambda k: candidates[k].cost_input + candidates[k].cost_output,
    )
    return cheapest_key, candidates[cheapest_key]


# ── Demo runner ──────────────────────────────────────────────────


async def run_demo(
    console: Console,
    *,
    skip_confirm: bool = False,
) -> PipelineResult:
    """Run the guided 60-second demo flow.

    Steps:
        0. Pre-flight — load keys, select models, confirm
        1. Routing annotation — show model choices
        2. Generate code with cheap model
        3. Arbiter review from different provider
        4. Summary + next-steps
        5. Write output files

    Returns:
        The ``PipelineResult`` built from the demo run.
    """
    from triad.keys import load_keys_env
    from triad.providers.registry import load_models

    # ── Step 0: Pre-flight ────────────────────────────────────
    load_keys_env()
    registry = load_models()

    try:
        (gen_key, gen_cfg), (arb_key, arb_cfg) = select_demo_models(registry)
    except RuntimeError as exc:
        console.print(f"\n[red]Demo unavailable:[/red] {exc}\n")
        raise typer.Exit(1) from None

    console.print()
    console.print(Panel(
        (
            "This demo runs a real multi-model pipeline to show\n"
            "how CRTX catches bugs that single-model tools miss.\n"
            "\n"
            "[dim]Estimated cost:[/dim]  [bold]~$0.15[/bold]\n"
            "[dim]Estimated time:[/dim]  [bold]~60 seconds[/bold]\n"
            "[dim]Models:[/dim]          [bold]2[/bold] "
            "(cheapest available from your providers)"
        ),
        title="[bold]CRTX Demo[/bold]",
        border_style="bright_blue",
    ))

    if not skip_confirm:
        if not typer.confirm("\nContinue?", default=True):
            raise typer.Exit(0) from None

    wall_start = time.time()
    total_cost = 0.0
    total_tokens = 0

    # ── Step 1: Routing annotation ────────────────────────────
    console.print(
        f"\n[bold cyan]ROUTING[/bold cyan]\n"
        f"  Generator: [bold]{gen_cfg.display_name}[/bold] "
        f"({gen_cfg.provider}) — fast, cheap, good enough to build\n"
        f"  Arbiter:   [bold]{arb_cfg.display_name}[/bold] "
        f"({arb_cfg.provider}) — different provider, catches what "
        f"{gen_cfg.display_name.split()[0]} misses\n\n"
        f"  [dim]{DEMO_ANNOTATIONS['routing']}[/dim]"
    )

    # ── Step 2: Generate code ─────────────────────────────────
    console.print(
        f"\n[bold yellow]GENERATING[/bold yellow] — "
        f"{gen_cfg.display_name} is writing the code...\n"
        f"  [dim]{DEMO_ANNOTATIONS['generating']}[/dim]"
    )

    from triad.providers.litellm_provider import LiteLLMProvider

    gen_provider = LiteLLMProvider(gen_cfg)
    system = DEMO_SYSTEM_PROMPT.format(task=DEMO_TASK)

    gen_start = time.time()
    with console.status(
        f"[bold cyan]Generating with {gen_cfg.display_name}...[/bold cyan]",
    ):
        msg = await gen_provider.complete(
            messages=[{"role": "user", "content": DEMO_TASK}],
            system=system,
            timeout=120,
        )
    gen_elapsed = time.time() - gen_start

    gen_cost = msg.token_usage.cost if msg.token_usage else 0.0
    gen_tokens = (
        (msg.token_usage.prompt_tokens + msg.token_usage.completion_tokens)
        if msg.token_usage else 0
    )
    total_cost += gen_cost
    total_tokens += gen_tokens

    # Count lines and infer structure
    content_lines = msg.content.strip().splitlines()
    line_count = len(content_lines)
    test_count = sum(
        1 for ln in content_lines if ln.strip().startswith("def test_")
    )

    console.print(
        f"\n  [green]Generated:[/green] validate_email() + "
        f"{test_count} test cases ({line_count} lines)\n"
        f"  [dim]Cost so far: ${gen_cost:.4f} · "
        f"{gen_tokens:,} tokens · {gen_elapsed:.1f}s[/dim]"
    )

    # ── Step 3: Arbiter review ────────────────────────────────
    console.print(
        f"\n[bold magenta]ARBITER REVIEW[/bold magenta] — "
        f"{arb_cfg.display_name} is checking "
        f"{gen_cfg.display_name}'s work...\n"
        f"  [dim]({DEMO_ANNOTATIONS['reviewing']})[/dim]"
    )

    from triad.arbiter.arbiter import ArbiterEngine

    arb_config = PipelineConfig(
        arbiter_model=arb_key,
        arbiter_mode=ArbiterMode.FULL,
        default_timeout=120,
    )
    arb_registry = {gen_key: gen_cfg, arb_key: arb_cfg}
    engine = ArbiterEngine(arb_config, arb_registry)

    task_spec = TaskSpec(task=DEMO_TASK, output_dir="crtx-output")

    arb_start = time.time()
    with console.status(
        f"[bold cyan]Reviewing with {arb_cfg.display_name}...[/bold cyan]",
    ):
        review: ArbiterReview | None = await engine.review(
            stage=PipelineStage.IMPLEMENT,
            stage_model=gen_cfg.model,
            stage_output=msg.content,
            task=task_spec,
        )
    arb_elapsed = time.time() - arb_start

    if review is not None:
        total_cost += review.token_cost
        # Estimate arbiter tokens from cost
        if arb_cfg.cost_output > 0:
            arb_tokens = int(
                review.token_cost / (arb_cfg.cost_output / 1_000_000)
            )
        else:
            arb_tokens = 0
        total_tokens += arb_tokens
        _display_verdict(
            console, review, gen_cfg, arb_cfg, arb_elapsed, total_cost,
        )
    else:
        console.print(Panel(
            "[yellow]Arbiter review returned no result.[/yellow]\n"
            "This can happen if the arbiter model is temporarily unavailable.",
            title="[bold yellow]Review Skipped[/bold yellow]",
            border_style="yellow",
        ))

    # ── Step 4: Summary ───────────────────────────────────────
    wall_elapsed = time.time() - wall_start

    # Determine what-happened summary based on verdict
    issue_count = len(review.issues) if review else 0
    if review and review.verdict != Verdict.APPROVE:
        what_happened = (
            f"  1. {gen_cfg.display_name} generated an email validator\n"
            f"  2. {arb_cfg.display_name} independently reviewed it\n"
            f"  3. The review caught {issue_count} issue"
            f"{'s' if issue_count != 1 else ''} the generator missed"
        )
    else:
        what_happened = (
            f"  1. {gen_cfg.display_name} generated an email validator\n"
            f"  2. {arb_cfg.display_name} independently reviewed it\n"
            f"  3. The review found no critical issues this time"
        )

    console.print()
    console.print(Panel(
        (
            f"[dim]Total cost:[/dim]  [bold]${total_cost:.2f}[/bold]\n"
            f"[dim]Total time:[/dim]  [bold]{wall_elapsed:.0f} seconds[/bold]\n"
            f"[dim]Tokens used:[/dim] [bold]{total_tokens:,}[/bold]\n"
            "\n"
            "[bold]What just happened:[/bold]\n"
            f"{what_happened}\n"
            "\n"
            f"  {DEMO_ANNOTATIONS['complete']}\n"
            "\n"
            "[bold]Next steps:[/bold]\n"
            '  [cyan]crtx run "your task"[/cyan]              '
            "— run a full pipeline\n"
            '  [cyan]crtx run "task" --preset explore[/cyan]  '
            "— 3 models in parallel\n"
            '  [cyan]crtx run "task" --preset debate[/cyan]   '
            "— adversarial debate\n"
            "  [cyan]crtx repl[/cyan]                         "
            "— interactive session"
        ),
        title="[bold]Demo Complete[/bold]",
        border_style="bright_blue",
    ))

    # ── Step 5: Write output ──────────────────────────────────
    session_id = str(uuid.uuid4())

    # Build AgentMessage with from_agent=IMPLEMENT
    impl_msg = msg.model_copy(
        update={
            "from_agent": PipelineStage.IMPLEMENT,
            "to_agent": PipelineStage.VERIFY,
            "msg_type": MessageType.IMPLEMENTATION,
        },
    )

    result = PipelineResult(
        session_id=session_id,
        task=task_spec,
        config=arb_config,
        stages={PipelineStage.IMPLEMENT: impl_msg},
        arbiter_reviews=[review] if review else [],
        total_cost=total_cost,
        total_tokens=total_tokens,
        duration_seconds=wall_elapsed,
        success=True,
    )

    from triad.output.writer import write_pipeline_output

    output_path = write_pipeline_output(result, task_spec.output_dir)
    console.print(
        f"\n[dim]Output saved to[/dim] [bold]{output_path}/[/bold]\n"
    )

    return result


def _display_verdict(
    console: Console,
    review: ArbiterReview,
    gen_cfg: ModelConfig,
    arb_cfg: ModelConfig,
    elapsed: float,
    total_cost: float,
) -> None:
    """Display the arbiter verdict with appropriate styling."""
    verdict = review.verdict
    confidence = review.confidence

    if verdict == Verdict.APPROVE:
        # ── Approve path ──────────────────────────────────────
        console.print(Panel(
            (
                f"{arb_cfg.display_name} reviewed "
                f"{gen_cfg.display_name}'s code and found\n"
                "no critical issues. This happens sometimes — the\n"
                "generator got it right. But when it doesn't, the\n"
                "Arbiter catches it. Run [bold]crtx run[/bold] on a "
                "complex task\nto see the full pipeline in action."
            ),
            title="[bold green]ARBITER APPROVED[/bold green]",
            border_style="green",
        ))
        return

    # ── Flag / Reject / Halt path ─────────────────────────────
    if verdict == Verdict.FLAG:
        style = "yellow"
        title = "ARBITER FOUND ISSUES"
    elif verdict == Verdict.REJECT:
        style = "red"
        title = "ARBITER FOUND ISSUES"
    else:
        style = "bold red"
        title = "ARBITER HALTED"

    body_parts: list[str] = [
        f"[bold]Verdict:[/bold] [{style}]{verdict.value.upper()}"
        f"[/{style}] · Confidence: {confidence:.0%}",
        "",
        f"{arb_cfg.display_name} found {len(review.issues)} issue"
        f"{'s' if len(review.issues) != 1 else ''} in "
        f"{gen_cfg.display_name}'s code:",
        "",
    ]

    for i, issue in enumerate(review.issues, 1):
        sev_style = {
            "critical": "red",
            "warning": "yellow",
            "suggestion": "dim",
        }.get(issue.severity, "dim")
        body_parts.append(
            f"  {i}. [{sev_style}][{issue.severity}][/{sev_style}] "
            f"{issue.description}"
        )
        if issue.suggestion:
            body_parts.append(f"     [dim]-> {issue.suggestion}[/dim]")

    body_parts.append("")
    body_parts.append(
        "[dim]A single model missed these. The cross-model review\n"
        f"caught them in {elapsed:.0f} seconds for "
        f"${review.token_cost:.4f}.[/dim]"
    )

    console.print(Panel(
        "\n".join(body_parts),
        title=f"[bold {style}]{title}[/bold {style}]",
        border_style=style,
    ))

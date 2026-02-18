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
from rich.table import Table

from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    TokenUsage,
)
from triad.schemas.pipeline import (
    ArbiterMode,
    ModelConfig,
    PipelineConfig,
    PipelineMode,
    PipelineResult,
    TaskSpec,
)

# ── Constants ────────────────────────────────────────────────────

DEMO_TASK = (
    "Write a Python module `email_validator.py` that validates email addresses. "
    "Include a function `is_valid_email(email: str) -> bool` that checks format "
    "using a regex, rejects disposable domains (mailinator.com, tempmail.com, "
    "throwaway.email), and normalises gmail dot-tricks. Add clear docstrings "
    "and at least five unit tests."
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
        "[bold]How CRTX picks models:[/bold] The registry lists every model "
        "you have API keys for. CRTX chose the cheapest reachable model to "
        "[italic]generate[/italic] the code, then picked an arbiter from a "
        "[italic]different provider[/italic] to review it — cross-model "
        "enforcement catches blind spots a single model would miss."
    ),
    "generation": (
        "[bold]Generation:[/bold] The generator writes code from scratch. "
        "In a full pipeline this would be the IMPLEMENT stage, fed by an "
        "ARCHITECT's design."
    ),
    "arbiter": (
        "[bold]Arbiter review:[/bold] An independent model from a different "
        "provider acts as an adversarial reviewer. It scores correctness, "
        "security, and edge-case coverage, then issues a verdict: "
        "[green]APPROVE[/green], [yellow]FLAG[/yellow], "
        "[red]REJECT[/red], or [bold red]HALT[/bold red]."
    ),
    "verdict": (
        "[bold]What just happened:[/bold]\n"
        "  1. CRTX selected two models from different providers\n"
        "  2. Model A generated code for the task\n"
        "  3. Model B reviewed it independently (cross-model enforcement)\n"
        "  4. You got working code [italic]and[/italic] a quality verdict\n\n"
        "A full [bold]crtx run[/bold] adds Architect → Implement → Refactor "
        "→ Verify stages with arbiter checkpoints between each."
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
) -> None:
    """Run the guided 60-second demo flow.

    Steps:
        0. Pre-flight — load keys, select models, confirm
        1. Routing annotation — show model table
        2. Generate code with cheap model
        3. Arbiter review from different provider
        4. Summary + next-steps
        5. Write output files
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

    est_cost = (
        (gen_cfg.cost_input + gen_cfg.cost_output) * 2 / 1_000_000
        + (arb_cfg.cost_input + arb_cfg.cost_output) * 2 / 1_000_000
    )

    console.print()
    console.print(
        Panel(
            f"[bold]CRTX Demo[/bold] — generate code + cross-provider review\n\n"
            f"  Generator : [cyan]{gen_cfg.display_name}[/cyan] ({gen_cfg.provider})\n"
            f"  Arbiter   : [cyan]{arb_cfg.display_name}[/cyan] ({arb_cfg.provider})\n"
            f"  Est. cost : [green]~${est_cost:.4f}[/green]",
            title="[bold]60-Second Demo[/bold]",
            border_style="bright_blue",
        )
    )

    if not skip_confirm:
        if not typer.confirm("\nProceed?", default=True):
            raise typer.Exit(0) from None

    wall_start = time.time()
    total_cost = 0.0
    total_tokens = 0

    # ── Step 1: Routing annotation ────────────────────────────
    console.print(f"\n{DEMO_ANNOTATIONS['routing']}\n")

    table = Table(title="Selected Models", show_header=True, header_style="bold")
    table.add_column("Role", style="bold")
    table.add_column("Model")
    table.add_column("Provider")
    table.add_column("Cost (per 1M tokens)", justify="right")
    table.add_row(
        "Generator",
        gen_cfg.display_name,
        gen_cfg.provider,
        f"${gen_cfg.cost_input:.2f} in / ${gen_cfg.cost_output:.2f} out",
    )
    table.add_row(
        "Arbiter",
        arb_cfg.display_name,
        arb_cfg.provider,
        f"${arb_cfg.cost_input:.2f} in / ${arb_cfg.cost_output:.2f} out",
    )
    console.print(table)

    # ── Step 2: Generate code ─────────────────────────────────
    console.print(f"\n{DEMO_ANNOTATIONS['generation']}\n")

    from triad.providers.litellm_provider import LiteLLMProvider

    gen_provider = LiteLLMProvider(gen_cfg)
    system = DEMO_SYSTEM_PROMPT.format(task=DEMO_TASK)

    gen_start = time.time()
    with console.status(
        f"[bold cyan]Generating with {gen_cfg.display_name}...[/bold cyan]"
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

    # Show generation summary
    preview_lines = msg.content.strip().splitlines()[:15]
    preview = "\n".join(preview_lines)
    if len(msg.content.strip().splitlines()) > 15:
        preview += "\n[dim]... (truncated)[/dim]"

    console.print(
        Panel(
            f"[bold]Model:[/bold] {gen_cfg.display_name}\n"
            f"[bold]Tokens:[/bold] {gen_tokens:,}  "
            f"[bold]Cost:[/bold] ${gen_cost:.4f}  "
            f"[bold]Time:[/bold] {gen_elapsed:.1f}s\n\n"
            f"{preview}",
            title="[bold green]Generation Complete[/bold green]",
            border_style="green",
        )
    )

    # ── Step 3: Arbiter review ────────────────────────────────
    console.print(f"\n{DEMO_ANNOTATIONS['arbiter']}\n")

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
        f"[bold cyan]Reviewing with {arb_cfg.display_name}...[/bold cyan]"
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
        arb_tokens = int(review.token_cost / (arb_cfg.cost_output / 1_000_000))
        total_tokens += arb_tokens
        _display_verdict(console, review, arb_cfg, arb_elapsed)
    else:
        console.print(
            Panel(
                "[yellow]Arbiter review returned no result.[/yellow]\n"
                "This can happen if the arbiter model is temporarily unavailable.",
                title="[bold yellow]Review Skipped[/bold yellow]",
                border_style="yellow",
            )
        )

    # ── Step 4: Summary ───────────────────────────────────────
    wall_elapsed = time.time() - wall_start

    console.print(f"\n{DEMO_ANNOTATIONS['verdict']}\n")

    summary_table = Table(title="Demo Summary", show_header=False)
    summary_table.add_column("Metric", style="bold")
    summary_table.add_column("Value")
    summary_table.add_row("Total cost", f"${total_cost:.4f}")
    summary_table.add_row("Total tokens", f"{total_tokens:,}")
    summary_table.add_row("Wall time", f"{wall_elapsed:.1f}s")
    summary_table.add_row("Generator", f"{gen_cfg.display_name} ({gen_cfg.provider})")
    summary_table.add_row("Arbiter", f"{arb_cfg.display_name} ({arb_cfg.provider})")
    console.print(summary_table)

    console.print(
        Panel(
            "[bold]Next steps:[/bold]\n\n"
            "  [cyan]crtx run[/cyan] \"Build a REST API with FastAPI\"   "
            "— full 4-stage pipeline\n"
            "  [cyan]crtx run -p fast[/cyan] \"Create a CLI tool\"        "
            "— speed-optimized preset\n"
            "  [cyan]crtx run -p thorough[/cyan] \"Payment processor\"    "
            "— maximum quality\n"
            "  [cyan]crtx repl[/cyan]                                   "
            "— interactive session",
            title="[bold]What's Next?[/bold]",
            border_style="bright_blue",
        )
    )

    # ── Step 5: Write output ──────────────────────────────────
    session_id = str(uuid.uuid4())

    # Build AgentMessage with from_agent=IMPLEMENT
    impl_msg = msg.model_copy(
        update={
            "from_agent": PipelineStage.IMPLEMENT,
            "to_agent": PipelineStage.VERIFY,
            "msg_type": MessageType.IMPLEMENTATION,
        }
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
    console.print(f"\n[green]Output written to:[/green] [bold]{output_path}[/bold]\n")


def _display_verdict(
    console: Console,
    review: ArbiterReview,
    arb_cfg: ModelConfig,
    elapsed: float,
) -> None:
    """Display the arbiter verdict with appropriate styling."""
    verdict = review.verdict
    confidence = review.confidence

    if verdict == Verdict.APPROVE:
        style = "green"
        icon = "APPROVED"
    elif verdict == Verdict.FLAG:
        style = "yellow"
        icon = "FLAGGED"
    elif verdict == Verdict.REJECT:
        style = "red"
        icon = "REJECTED"
    else:
        style = "bold red"
        icon = "HALTED"

    body_parts = [
        f"[bold]Verdict:[/bold] [{style}]{icon}[/{style}]  "
        f"(confidence: {confidence:.0%})",
        f"[bold]Arbiter:[/bold] {arb_cfg.display_name}  "
        f"[bold]Cost:[/bold] ${review.token_cost:.4f}  "
        f"[bold]Time:[/bold] {elapsed:.1f}s",
    ]

    if review.issues:
        body_parts.append("")
        body_parts.append("[bold]Issues:[/bold]")
        for issue in review.issues:
            sev_style = {
                "critical": "red",
                "warning": "yellow",
                "suggestion": "dim",
            }.get(issue.severity, "dim")
            body_parts.append(
                f"  [{sev_style}]{issue.severity.upper()}[/{sev_style}]: "
                f"{issue.description}"
            )

    # Show a snippet of reasoning
    reasoning_lines = review.reasoning.strip().splitlines()[:5]
    if reasoning_lines:
        body_parts.append("")
        body_parts.append("[bold]Reasoning (excerpt):[/bold]")
        for line in reasoning_lines:
            body_parts.append(f"  [dim]{line}[/dim]")

    console.print(
        Panel(
            "\n".join(body_parts),
            title=f"[bold {style}]Arbiter Verdict: {icon}[/bold {style}]",
            border_style=style,
        )
    )

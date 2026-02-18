"""Pipeline presets — named combinations of mode, route, and arbiter.

Each preset is a tuple of (pipeline_mode, routing_strategy, arbiter_mode).
Users pick a preset as a starting point and can override individual values.
"""

from __future__ import annotations

PRESETS: dict[str, tuple[str, str, str]] = {
    "fast": ("sequential", "speed_first", "off"),
    "balanced": ("sequential", "hybrid", "bookend"),
    "thorough": ("sequential", "quality_first", "full"),
    "explore": ("parallel", "hybrid", "bookend"),
    "debate": ("debate", "quality_first", "bookend"),
    "cheap": ("sequential", "cost_optimized", "off"),
}

DEFAULT_PRESET = "balanced"

# Short labels for prompt display
_MODE_SHORT = {
    "sequential": "seq",
    "parallel": "par",
    "debate": "deb",
    "review": "rev",
    "improve": "imp",
}

_ROUTE_SHORT = {
    "hybrid": "hyb",
    "quality_first": "qual",
    "cost_optimized": "cost",
    "speed_first": "speed",
}

_ARBITER_SHORT = {
    "off": "off",
    "final_only": "final",
    "bookend": "book",
    "full": "full",
}


def resolve_preset(
    preset: str | None = None,
    *,
    mode: str | None = None,
    route: str | None = None,
    arbiter: str | None = None,
) -> tuple[str, str, str]:
    """Resolve a preset with optional per-field overrides.

    Args:
        preset: Preset name (defaults to DEFAULT_PRESET).
        mode: Override pipeline mode.
        route: Override routing strategy.
        arbiter: Override arbiter mode.

    Returns:
        Tuple of (mode, route, arbiter).

    Raises:
        ValueError: If the preset name is unknown.
    """
    name = preset or DEFAULT_PRESET
    if name not in PRESETS:
        valid = ", ".join(PRESETS)
        msg = f"Unknown preset: '{name}'. Choose from: {valid}"
        raise ValueError(msg)

    preset_mode, preset_route, preset_arbiter = PRESETS[name]
    return (
        mode or preset_mode,
        route or preset_route,
        arbiter or preset_arbiter,
    )


def matching_preset(mode: str, route: str, arbiter: str) -> str | None:
    """Return the preset name if (mode, route, arbiter) matches one exactly."""
    combo = (mode, route, arbiter)
    for name, values in PRESETS.items():
        if values == combo:
            return name
    return None


def format_prompt_tag(mode: str, route: str, arbiter: str) -> str:
    """Format a short tag for the REPL prompt.

    Returns the preset name if settings match one exactly,
    otherwise returns abbreviated settings like ``seq|hyb|off``.
    """
    name = matching_preset(mode, route, arbiter)
    if name:
        return name
    m = _MODE_SHORT.get(mode, mode[:3])
    r = _ROUTE_SHORT.get(route, route[:4])
    a = _ARBITER_SHORT.get(arbiter, arbiter[:4])
    return f"{m}|{r}|{a}"

# CRTX SDK — public API surface
from crtx.version import __version__

# Engine core
from crtx.loop import Loop, LoopResult, LoopStats
from crtx.arbiter import Arbiter, ArbiterResult, check, standalone_review
from crtx.router import Router, RouteDecision, TaskComplexity
from crtx.providers import ModelProvider, LiteLLMProvider

# Memory (v0.3.0)
from crtx.memory import Memory, Decision, Pattern, TaxonomyRule

__all__ = [
    # Loop
    "Loop", "LoopResult", "LoopStats",
    # Arbiter — pipeline API
    "Arbiter", "ArbiterResult",
    # Arbiter — standalone (no pipeline context needed)
    "check",            # sync: check(content, content_type) -> dict
    "standalone_review", # async: await standalone_review(content, content_type) -> dict
    # Router
    "Router", "RouteDecision", "TaskComplexity",
    # Providers
    "ModelProvider", "LiteLLMProvider",
    # Memory (v0.3.0)
    "Memory", "Decision", "Pattern", "TaxonomyRule",
]

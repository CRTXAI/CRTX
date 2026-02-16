"""Smart Routing Engine for the Triad Orchestrator.

Provides model-to-role assignment using configurable strategies:
quality-first, cost-optimized, speed-first, and hybrid.
"""

from triad.routing.engine import RoutingEngine, estimate_cost
from triad.routing.strategies import (
    cost_optimized,
    hybrid,
    quality_first,
    speed_first,
)

__all__ = [
    "RoutingEngine",
    "cost_optimized",
    "estimate_cost",
    "hybrid",
    "quality_first",
    "speed_first",
]

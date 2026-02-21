"""Task router — classifies prompt complexity and picks execution strategy."""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum


class TaskComplexity(StrEnum):
    SIMPLE = "simple"
    MEDIUM = "medium"
    COMPLEX = "complex"
    SAFETY = "safety"


@dataclass
class RouteDecision:
    """Routing decision for a prompt."""

    complexity: TaskComplexity
    model: str
    use_architecture_debate: bool
    max_fix_iterations: int
    arbiter_required: bool


# Default model for all tiers (benchmark winner)
_DEFAULT_MODEL = "anthropic/claude-sonnet-4-5-20250929"

_SAFETY_KEYWORDS = [
    "safety", "medical", "financial", "autonomous", "real-time",
    "critical", "interlock", "audit", "compliance", "hipaa",
]

_COMPLEX_KEYWORDS = [
    "architecture", "system design", "coordinate", "concurrent",
    "distributed", "state machine", "event sourc", "websocket",
    "formation", "orchestrat", "microservice", "pipeline",
]

_MEDIUM_KEYWORDS = [
    "api", "fastapi", "flask", "django", "database", "crud",
    "authentication", "pagination", "search", "import", "export",
    "migration", "rest", "graphql", "oauth", "jwt",
]


class TaskRouter:
    """Classify prompt complexity and select execution strategy."""

    def classify(self, prompt: str) -> RouteDecision:
        """Rule-based complexity classifier.

        Returns a RouteDecision with model selection and iteration limits.
        """
        lower = prompt.lower()
        word_count = len(prompt.split())

        # Safety tier
        if any(kw in lower for kw in _SAFETY_KEYWORDS):
            return RouteDecision(
                complexity=TaskComplexity.SAFETY,
                model=_DEFAULT_MODEL,
                use_architecture_debate=True,
                max_fix_iterations=5,
                arbiter_required=True,
            )

        # Complex tier
        if any(kw in lower for kw in _COMPLEX_KEYWORDS) or word_count > 200:
            return RouteDecision(
                complexity=TaskComplexity.COMPLEX,
                model=_DEFAULT_MODEL,
                use_architecture_debate=True,
                max_fix_iterations=4,
                arbiter_required=True,
            )

        # Medium tier
        if any(kw in lower for kw in _MEDIUM_KEYWORDS) or word_count > 100:
            return RouteDecision(
                complexity=TaskComplexity.MEDIUM,
                model=_DEFAULT_MODEL,
                use_architecture_debate=False,
                max_fix_iterations=3,
                arbiter_required=True,
            )

        # Simple tier
        return RouteDecision(
            complexity=TaskComplexity.SIMPLE,
            model=_DEFAULT_MODEL,
            use_architecture_debate=False,
            max_fix_iterations=2,
            arbiter_required=False,
        )

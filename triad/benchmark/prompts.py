"""Benchmark prompt definitions.

Each prompt is a self-contained coding task with requirements that can be
mechanically checked (parses, runs, passes tests, has type hints, etc.).
"""

from __future__ import annotations

from dataclasses import dataclass, field


@dataclass(frozen=True)
class BenchmarkPrompt:
    """A single benchmark coding prompt."""

    id: str
    name: str
    tier: str  # simple | medium | complex | safety
    prompt_text: str
    requirements: list[str] = field(default_factory=list)
    entry_point: str = ""  # filename to execute for run check


BENCHMARK_PROMPTS: list[BenchmarkPrompt] = [
    # ── 1. Password Generator (simple) ───────────────────────────
    BenchmarkPrompt(
        id="p01",
        name="Password Generator",
        tier="simple",
        prompt_text=(
            "Build a Python CLI password generator.\n\n"
            "Requirements:\n"
            "- Accept --length (default 16), --uppercase, --lowercase, "
            "--digits, --special flags\n"
            "- Default: all character classes enabled\n"
            "- Use secrets module for cryptographic randomness\n"
            "- Guarantee at least one character from each enabled class\n"
            "- Add --count flag to generate multiple passwords\n"
            "- Include a strength estimator that prints entropy bits\n"
            "- Write unit tests covering all flags and edge cases\n"
            "- All functions must have type annotations\n"
            "- Provide a # file: header for every file"
        ),
        requirements=[
            "Uses secrets module (not random)",
            "Respects --length flag",
            "Guarantees at least one char per enabled class",
            "--count generates multiple passwords",
            "Prints entropy/strength estimate",
            "Has unit tests",
            "All functions have type hints",
        ],
        entry_point="password_generator.py",
    ),

    # ── 5. Bookmark API (medium) ─────────────────────────────────
    BenchmarkPrompt(
        id="p05",
        name="Bookmark API",
        tier="medium",
        prompt_text=(
            "Build a Python bookmark manager REST API using FastAPI.\n\n"
            "Requirements:\n"
            "- CRUD endpoints: POST /bookmarks, GET /bookmarks, "
            "GET /bookmarks/{id}, PUT /bookmarks/{id}, DELETE /bookmarks/{id}\n"
            "- Bookmark model: id (UUID), url, title, description, tags (list), "
            "created_at, updated_at\n"
            "- In-memory storage (dict-based, no database)\n"
            "- Input validation with Pydantic models\n"
            "- GET /bookmarks supports ?tag= filter and ?search= full-text search\n"
            "- Proper HTTP status codes (201, 404, 422)\n"
            "- Write pytest tests for every endpoint\n"
            "- All functions must have type annotations\n"
            "- Provide a # file: header for every file"
        ),
        requirements=[
            "All 5 CRUD endpoints implemented",
            "UUID-based bookmark IDs",
            "Pydantic request/response models",
            "Tag filtering works",
            "Search filtering works",
            "Correct HTTP status codes",
            "Has pytest tests for all endpoints",
            "All functions have type hints",
        ],
        entry_point="app.py",
    ),

    # ── 10. Multi-Agent Coordination (complex) ───────────────────
    BenchmarkPrompt(
        id="p10",
        name="Multi-Agent Coordination",
        tier="complex",
        prompt_text=(
            "Build a Python multi-agent task coordination system.\n\n"
            "Requirements:\n"
            "- Agent base class with: name, capabilities (list[str]), "
            "status (idle/busy/offline)\n"
            "- TaskQueue that accepts tasks with required_capabilities\n"
            "- Dispatcher that matches tasks to agents based on capability "
            "overlap and availability\n"
            "- Support task priorities (high/medium/low) with priority queue\n"
            "- Implement a simple round-robin load balancer when multiple agents "
            "match\n"
            "- Add task timeout handling: if an agent doesn't complete within "
            "a deadline, reassign the task\n"
            "- Event system: emit events on task_assigned, task_completed, "
            "task_timeout, agent_status_change\n"
            "- Use asyncio for concurrent agent execution\n"
            "- Write comprehensive tests with at least 10 test cases\n"
            "- All functions must have type annotations\n"
            "- Provide a # file: header for every file"
        ),
        requirements=[
            "Agent base class with capabilities and status",
            "TaskQueue with priority support",
            "Dispatcher matches by capability",
            "Round-robin load balancing",
            "Task timeout with reassignment",
            "Event system with callbacks",
            "Uses asyncio",
            "10+ test cases",
            "All functions have type hints",
        ],
        entry_point="coordinator.py",
    ),
]


def get_quick_prompts() -> list[BenchmarkPrompt]:
    """Return the subset of prompts used for --quick runs."""
    return list(BENCHMARK_PROMPTS)


def get_prompt_by_id(prompt_id: str) -> BenchmarkPrompt | None:
    """Look up a prompt by its ID."""
    for p in BENCHMARK_PROMPTS:
        if p.id == prompt_id:
            return p
    return None

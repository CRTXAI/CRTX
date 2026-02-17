# Triad Orchestrator — Project Context

## Overview

Multi-model AI orchestration platform built by NexusAI. Coordinates multiple LLMs into a structured code generation pipeline with consensus mechanisms, role-based specialization, and an independent adversarial Arbiter layer.

## Architecture

- **Pipeline:** 4 stages — Architect → Implement → Refactor → Verify
- **Modes:** Sequential, Parallel Exploration, Debate
- **Arbiter:** Independent adversarial review (APPROVE / FLAG / REJECT / HALT)
- **Routing:** Fitness-based model selection (quality / cost / speed / hybrid)
- **Consensus:** Cross-domain suggestions, escalation voting, tiebreaker

## Tech Stack

- Python 3.12+, LiteLLM, Pydantic v2, Typer + Rich, TOML config, asyncio
- Testing: pytest + pytest-asyncio
- Linting: Ruff
- License: Apache 2.0

## Code Conventions

- All schemas use Pydantic v2 BaseModel with strict validation
- Async-first: all provider calls and pipeline stages are async
- Type hints on everything — no `Any` unless absolutely necessary
- Docstrings on all public classes and functions
- All provider calls go through LiteLLM, never direct SDK calls
- Prompts stored as Markdown in prompts/, loaded at runtime with Jinja2
- Tests: pytest with fixtures for mock providers, schema validation, pipeline execution
- Keep CLI output beautiful — Rich panels, color-coded verdicts, live status

## Project Structure

```
triad/
├── cli.py                  # Typer entry point
├── orchestrator.py         # Pipeline engine (sequential, parallel, debate)
├── planner.py              # Task planner
├── providers/              # LiteLLM adapter + model registry
├── routing/                # Smart routing engine + strategies
├── arbiter/                # Adversarial review + feedback + reconciliation
├── consensus/              # Voting protocol + suggestion escalation
├── context/                # Codebase scanner + context builder
├── persistence/            # SQLite session storage
├── ci/                     # CI/CD review integration
├── dashboard/              # Real-time WebSocket dashboard (optional)
├── schemas/                # Pydantic v2 models
├── prompts/                # Jinja2 role prompts
├── output/                 # File writer + Markdown renderer
└── config/                 # TOML configuration files
```

## When Generating Code

- Follow the project structure above
- Use the established tech stack — don't introduce new dependencies without discussion
- All new code gets tests
- All schemas use Pydantic v2
- All provider calls go through LiteLLM
- Example tasks must be generic (REST APIs, CLI tools, data pipelines)
- The `config/domain/` directory is .gitignore'd for private domain rules

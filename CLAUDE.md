# Triad Orchestrator — Claude Code Project Context

## Project Overview

You are working on the **Triad Orchestrator**, an open-source multi-model AI orchestration platform built by NexusAI. Triad coordinates multiple frontier LLMs (Claude, GPT-4, Gemini, Grok, DeepSeek, Llama, Mistral, and any future model) into a structured code generation pipeline with consensus mechanisms, role-based specialization, and an independent adversarial Arbiter layer.

The founder and sole developer is Adam (Founder & CTO of NexusAI). He is an experienced full-stack developer working with Python, FastAPI, React Native/Expo, and uses Claude Code CLI tools in his development workflow. He works fast, prefers concise communication, and expects production-quality code.

## Architecture Summary

Triad uses a **sequential pipeline** with 4 stages: **Architect → Implement → Refactor → Verify**. Models are assigned to roles dynamically based on fitness benchmarks via a model-agnostic plugin system powered by LiteLLM.

The **Arbiter Layer** sits above the pipeline as an independent referee. It reviews stage outputs using a cross-model enforcement rule (arbiter model ≠ generator model) and issues one of four verdicts: APPROVE, FLAG, REJECT (with structured feedback injection + retry), or HALT (circuit breaker for human review). The default review mode is "Bookend" (Arbiter reviews Architect output and final Verify output).

The **Implementation Summary Reconciliation (ISR)** is an opt-in final Arbiter pass (`--reconcile` flag) where the Verifier produces a structured ImplementationSummary, and a cross-model Arbiter compares it against the original TaskSpec and Architect scaffold to catch spec drift, missing requirements, and silently dropped features.

### Project Structure
```
triad/
├── __init__.py
├── cli.py                  # Typer entry point
├── orchestrator.py         # Pipeline engine (sequential)
├── providers/
│   ├── __init__.py
│   ├── base.py             # ModelProvider ABC
│   ├── litellm_provider.py # Universal LiteLLM adapter
│   └── registry.py         # Model discovery + config loading
├── arbiter/
│   ├── __init__.py
│   ├── arbiter.py          # Arbiter review engine
│   ├── feedback.py         # Structured feedback injection
│   └── reconciler.py       # Implementation Summary Reconciliation
├── schemas/
│   ├── __init__.py
│   ├── messages.py         # AgentMessage, Suggestion
│   ├── arbiter.py          # ArbiterReview, Verdict, Issue
│   ├── reconciliation.py   # ImplementationSummary, Deviation
│   └── pipeline.py         # PipelineConfig, TaskSpec
├── prompts/
│   ├── architect.md
│   ├── implementer.md
│   ├── refactorer.md
│   ├── verifier.md
│   ├── arbiter.md
│   └── reconciler.md       # Reconciliation Arbiter prompt
├── output/
│   ├── renderer.py         # Markdown summary generator
│   └── writer.py           # File output handler
└── config/
    ├── models.toml         # Model registry
    ├── defaults.toml       # Default pipeline config
    └── domain/             # .gitignore'd — private rules
        └── (alia_patterns.toml — never committed)

# Repo root also includes:
# CLAUDE.md           (this file — auto-read by Claude Code)
# LICENSE             (Apache 2.0)
# CLA.md              (Contributor License Agreement — required for all external PRs)
# CONTRIBUTING.md     (contribution guidelines, references CLA)
# README.md           (quick start, architecture overview, GIF demo)
# pyproject.toml      (package config, dependencies, CLI entry point)
# docs/               (full specs: build-spec.md, architecture.md, arbiter.md, model-agnostic.md)
# .github/workflows/  (CI: pytest, ruff lint, CLA check)
```

## Tech Stack

- **Language:** Python 3.12+
- **LLM Adapter:** LiteLLM (universal provider interface, 100+ models)
- **Schema Validation:** Pydantic v2
- **CLI Framework:** Typer + Rich (beautiful terminal UI)
- **Config Format:** TOML (models.toml, defaults.toml)
- **Async Runtime:** asyncio (stdlib)
- **Testing:** pytest + pytest-asyncio
- **Linting:** Ruff
- **Package Manager:** uv (with pip fallback)
- **License:** Apache 2.0 (pending patent attorney review)

## Supported Providers (MVP)

- **Anthropic:** Claude Opus 4.5, Claude Sonnet 4.5, Claude Haiku 4.5
- **OpenAI:** GPT-4o, o3-mini
- **Google:** Gemini 2.5 Pro, Gemini 2.5 Flash
- **xAI:** Grok 4, Grok 3

Adding a new provider = adding a TOML entry in models.toml. LiteLLM handles the rest.

## Code Conventions

- All schemas use **Pydantic v2 BaseModel** with strict validation
- Async-first: all provider calls and pipeline stages are async
- Type hints on everything — no `Any` unless absolutely necessary
- Docstrings on all public classes and functions
- Error handling: explicit exceptions, never silent failures
- Config loading: TOML parsed at startup, validated against Pydantic config models
- Prompts: stored as Markdown files in prompts/, loaded at runtime, support Jinja2-style variable injection for task context and domain rules
- Output: structured directories per pipeline run (code/, tests/, reports/, logs/)
- Tests: pytest with fixtures for mock providers, schema validation, pipeline execution

## Key Schemas

### AgentMessage
The universal message envelope between pipeline agents:
- `role`: PipelineStage (architect | implement | refactor | verify)
- `model`: str (which model produced this)
- `content`: str (the actual output)
- `code_blocks`: list[CodeBlock] (extracted code with language, filepath, content)
- `suggestions`: list[Suggestion] (cross-domain suggestions for other roles)
- `confidence`: float (0.0-1.0)
- `token_usage`: TokenUsage (prompt_tokens, completion_tokens, cost)

### ArbiterReview
- `stage_reviewed`: PipelineStage
- `reviewed_model`: str
- `arbiter_model`: str
- `verdict`: Verdict (APPROVE | FLAG | REJECT | HALT)
- `issues`: list[Issue] (severity, category, location, description, suggestion, evidence)
- `alternatives`: list[Alternative] (description, rationale, code_sketch, confidence)
- `confidence`: float
- `reasoning`: str
- `token_cost`: float

### ImplementationSummary (ISR)
- `task_echo`: str (Verifier's restatement of the original task)
- `endpoints_implemented`: list[str]
- `schemas_created`: list[str]
- `files_created`: list[str]
- `files_modified`: list[str]
- `behaviors_implemented`: list[str]
- `test_coverage`: list[str]
- `deviations`: list[Deviation] (what, reason, stage)
- `omissions`: list[str]

### Verdict Flow
- **APPROVE:** Continue to next stage. Log any warnings.
- **FLAG:** Continue, but inject flags into next agent's context.
- **REJECT:** Re-run current stage with Arbiter feedback injected. Max 2 retries.
- **HALT:** Stop pipeline. Present analysis for human decision.

## CRITICAL: Disclosure Rules

This is an **open-source project**. All code, docs, README content, example tasks, blog posts, and any generated content must follow these rules:

### NEVER include in any public-facing content:
- The words: insurance, agency, AMS, broker, underwriting, policy, premium, COI, renewal, certificate of insurance, binding, carrier
- Any ALIA-specific schemas: Evidence v2.1, THR-YYYY-NNNN, Connect threading, Classification Intelligence, Binding Trail, Client Vault, Renewal Workbench
- Any ALIA architecture details: FastAPI backend structure, LiveKit/Deepgram/ElevenLabs voice stack, M365 email integration, iOS widgets
- Fundraising details: round size, valuation, investor names
- Benchmark data from internal ALIA development runs

### ALWAYS use this framing:
- ALIA/Atlas365 is referenced ONLY as a **"financial services operating system"** or **"financial services platform"**
- Dogfooding narrative: "We use Triad to build a financial services OS"
- Domain verification: "Triad supports custom verification rules for domain-specific patterns"
- Test coverage: "Our platform has 2,900+ tests" (no breakdown by module)

### Example tasks in the repo must be GENERIC:
- "Build a REST API with authentication"
- "Refactor a data pipeline for streaming"
- "Add WebSocket support to an existing server"
- "Create a CLI tool with argument parsing"
- NEVER use insurance/financial domain examples

### The private domain config directory:
- `config/domain/` is in `.gitignore` and NEVER committed
- ALIA-specific Arbiter rules (alia_patterns.toml) live here
- ALIA context packages live here
- This separation is the architectural boundary between public Triad and private ALIA

## Business Strategy Context

Triad serves a **dual strategic purpose** for NexusAI:

1. **ALIA development accelerator (primary):** Triad is the tool used to build a proprietary financial services operating system called ALIA (Atlas365 OS). Routing ALIA development through Triad's pipeline produces higher-quality code faster. ALIA has a **July 2026 launch target** — this deadline takes priority over all Triad work.

2. **Fundraising and market leverage (secondary):** Open-source traction strengthens the NexusAI fundraising narrative (currently raising a pre-seed round), provides recruiting signal, and creates optionality for a second revenue stream.

### Priority Hierarchy for All Decisions
1. Does this help ALIA ship in July 2026?
2. Does this increase open-source adoption?
3. Does this protect IP (disclosure rules, CLA, private/public separation)?
4. Does this lay groundwork for future monetization?
5. Does this improve Triad for Triad's sake? (lowest priority — let community build these)

### IP Protection Requirements
- **License:** Apache 2.0 (includes patent grant)
- **CLA:** Required from Day 1 for all external contributors. Implemented via CLA Assistant bot on GitHub. No PR from an external contributor should be merged without a signed CLA.
- **Patents:** Provisional patent filings may be made before open-source release for the Arbiter protocol, cross-domain suggestion escalation, and fitness-based role assignment. Patent attorney consultation is pending.

## Build Plan

The MVP is a 10-day sprint targeting mid-March 2026 open-source release. Full build spec with day-by-day deliverables is in `docs/build-spec.md`.

## Success Criteria

1. 5-minute quick start (install → configure → first pipeline run)
2. 4 providers work (Anthropic, OpenAI, Google, xAI)
3. Arbiter catches a real bug in at least one test task
4. Cross-model enforcement holds (arbiter ≠ generator always)
5. Output is clean and usable (code files, tests, Markdown summary)
6. CLI looks great in a GIF/screenshot for the README
7. Zero private keyword leakage (grep audit passes)
8. Tests pass (schemas, routing, pipeline, Arbiter verdicts)
9. Reconciliation catches a spec gap in at least one test task with `--reconcile` enabled

## When Generating Code

- Follow the project structure above exactly
- Use the established tech stack — don't introduce new dependencies without discussion
- All new code gets tests
- All schemas use Pydantic v2
- All provider calls go through LiteLLM, never direct SDK calls
- Run disclosure keyword checks before suggesting any example content
- Keep the CLI output beautiful — Rich panels, color-coded verdicts, live status
- When in doubt about public vs. private content, it's private

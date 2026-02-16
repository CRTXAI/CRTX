# Triad Orchestrator — MVP Build Specification

## 2-Week Sprint to Open-Source Release

*Ship fast. Ship clean. Let the community build the rest.*

**NexusAI** | February 2026

---

## 1. MVP Scope — What Ships in 2 Weeks

The MVP is a **working CLI tool** that a developer can install, configure their API keys, and run a multi-model coding pipeline on a real task within 5 minutes. Everything else is future scope. The goal is a compelling open-source release, not a commercial product.

### 1.1 In Scope (Must Ship)

| Component | MVP Specification |
|---|---|
| **Provider Layer** | ModelProvider ABC + LiteLLM universal adapter. TOML-based model registry. Pre-configured for: Anthropic (Claude Opus/Sonnet/Haiku), OpenAI (GPT-4o, o3-mini), Google (Gemini 2.5 Pro/Flash), xAI (Grok 4/3). Adding new models = adding a TOML entry. |
| **Pipeline Engine** | Sequential pipeline only for MVP. 4 stages: Architect → Implement → Refactor → Verify. Async execution with configurable per-agent timeouts. Basic retry with exponential backoff. |
| **Consensus (Lite)** | Simplified consensus: after the pipeline runs, the Verifier evaluates the final output and produces a confidence score. Cross-domain suggestions are collected but not voted on (logged for future implementation). No debate mode in MVP. |
| **Arbiter (Bookend)** | Arbiter reviews Architect output and Verify output only (bookend mode). Cross-model enforcement (arbiter ≠ generator). Four verdicts with REJECT retry (max 2). Structured feedback injection on reject. |
| **Implementation Summary Reconciliation (Opt-in)** | When `--reconcile` is enabled, the Verifier produces a structured ImplementationSummary after the Verify stage. A cross-model Arbiter pass compares the summary against the original TaskSpec and Architect scaffold output. Same four-verdict logic (APPROVE/FLAG/REJECT/HALT). Cross-model enforcement applies (reconciliation arbiter ≠ verifier). Single REJECT retry triggers rewind to Implement or Refactor with reconciliation feedback injected. Off by default; enabled via `--reconcile` flag. Works independently of the main Arbiter depth setting. |
| **Schemas** | Pydantic v2 models: AgentMessage, ArbiterReview, Suggestion, CodeBlock, PipelineConfig, TaskSpec, ImplementationSummary, Deviation. Structured output parsing with fallback to text extraction. |
| **System Prompts** | 4 role-based prompts (architect.md, implementer.md, refactorer.md, verifier.md) + arbiter.md + reconciler.md. Generic prompts with no domain-specific content. Slot for user-provided domain context. |
| **CLI** | Typer-based CLI. Commands: run (execute pipeline), models (list/add/remove), config (show/edit). Rich-powered live output showing agent status, streaming responses, and Arbiter verdicts in real-time. `--reconcile` flag on `run` command. |
| **Output** | Pipeline output as structured directory: code files, test files, arbiter reviews, reconciliation report (when enabled), session log (JSON). Markdown summary report generated automatically. |

### 1.2 Out of Scope (Post-MVP)

- Parallel exploration and debate pipeline modes
- Full consensus voting with cross-domain suggestion escalation
- Tournament mode and fitness benchmarking system
- Smart routing engine (MVP uses explicit model assignment)
- React dashboard / WebSocket UI
- Session persistence / audit trail database
- CI/CD GitHub Actions integration
- Context injection from codebase (manual context for MVP)
- Full reconciliation against mid-pipeline consensus decisions and accepted cross-domain suggestions (MVP reconciles against TaskSpec + Architect output only; suggestion tracking is post-MVP)

---

## 2. Architecture

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
        └── (domain_rules.toml — never committed)
```

---

## 3. Day-by-Day Build Plan

10 working days. Each day has a clear deliverable. The build is ordered so that the pipeline is functional as early as Day 4, with remaining days adding the Arbiter, reconciliation, polish, and release prep.

| Day | Focus | Deliverables | Milestone |
|---|---|---|---|
| **1** | Foundation | Project scaffold (pyproject.toml, uv/pip, directory structure). Pydantic schemas: AgentMessage, CodeBlock, PipelineConfig, TaskSpec, ImplementationSummary, Deviation. TOML config loader for model registry. | Schemas parse. Config loads. |
| **2** | Provider Layer | ModelProvider ABC. LiteLLM adapter with structured output parsing. Model registry with TOML auto-discovery. Pre-configured entries for Claude, GPT-4o, Gemini, Grok. | Can call all 4 providers. |
| **3** | System Prompts | Write all 6 role prompts (architect, implementer, refactorer, verifier, arbiter, reconciler). Generic prompt templates with domain context injection slot. Output schema instructions per role. | Prompts reviewed and tested manually. |
| **4** | Pipeline Engine | Sequential pipeline orchestrator. Stage execution with async model calls. Output passing between stages (Architect output → Implementer input, etc.). Basic timeout + retry. | **PIPELINE RUNS END-TO-END** |
| **5** | Arbiter Core | ArbiterReview schema. Arbiter engine with cross-model enforcement. Four verdicts. Bookend review configuration (review Architect + Verify stages). | Arbiter reviews and returns verdicts. |
| **6** | Feedback Loop + Reconciliation | REJECT handling: structured feedback injection into re-generation prompt. Retry budget (max 2). HALT handling: pipeline stops and presents analysis. FLAG handling: warnings injected into next stage context. **Implementation Summary Reconciliation (opt-in):** Verifier produces ImplementationSummary schema after Verify stage. When `--reconcile` is enabled, cross-model Arbiter compares summary against TaskSpec + Architect output. Same four-verdict logic. Single REJECT retry triggers rewind with reconciliation feedback injected. | Arbiter can reject and trigger re-gen. Reconciliation catches spec gaps when enabled. |
| **7** | CLI Interface | Typer CLI: triad run, triad models, triad config. `--reconcile` flag on run command. Rich live display: agent status panels, streaming output, Arbiter verdicts with color-coded severity, reconciliation report panel. Progress tracking. | **CLI IS BEAUTIFUL + FUNCTIONAL** |
| **8** | Output + Logging | File output handler: writes code files, test files to structured directory. Markdown summary report (includes reconciliation results when enabled). JSON session log with all messages, verdicts, reconciliation verdicts, token usage, costs. | Complete audit trail per run. |
| **9** | Testing + Docs | Pytest suite for schemas, provider layer, pipeline logic, Arbiter verdicts, reconciliation verdicts. README.md with quick start, architecture overview, configuration guide (including `--reconcile` flag). 5 generic example tasks. | Tests pass. README is compelling. |
| **10** | Release Prep | Disclosure audit (grep for private-tier keywords). LICENSE file. CONTRIBUTING.md. CLA.md. GitHub Actions for CI (pytest, ruff, CLA check). Final keyword scan. GIF/screenshot of CLI in action for README (include one run with reconciliation enabled). | **SHIP IT** |

---

## 4. Technical Decisions

| Decision | Choice | Rationale |
|---|---|---|
| Language | Python 3.12+ | LiteLLM, Pydantic, Typer, Rich all Python-native. Matches our platform backend. |
| Package manager | uv (with pip fallback) | Fast, modern, gaining adoption. pip install still works for those who prefer it. |
| LLM adapter | LiteLLM | 100+ provider support. OpenAI-compatible interface. Battle-tested. Same choice CrewAI made. |
| Schema validation | Pydantic v2 | Industry standard. Fast. Great error messages. Industry standard patterns. |
| CLI framework | Typer + Rich | Typer for argument parsing, Rich for beautiful terminal output. Both well-maintained. |
| Config format | TOML | Human-readable, Python-native (tomllib), standard for Python projects. |
| Async runtime | asyncio (stdlib) | No external deps. Sufficient for sequential pipeline. Scales to parallel later. |
| Testing | pytest + pytest-asyncio | Standard. Standard patterns. Easy for contributors. Easy for contributors. |
| License | Apache 2.0 (pending patent attorney advice) | Maximum adoption. Patent grant included. Permissive. May adjust based on attorney consultation. |

---

## 5. Dependencies

```toml
[project]
name = "triad-orchestrator"
version = "0.1.0"
requires-python = ">=3.12"
dependencies = [
    "litellm>=1.55",    # Universal LLM adapter
    "pydantic>=2.0",    # Schema validation
    "typer>=0.12",      # CLI framework
    "rich>=13.0",       # Terminal UI
    "tomli>=2.0",       # TOML parsing (stdlib in 3.11+)
    "aiofiles>=24.0",   # Async file I/O
]

[project.optional-dependencies]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.5",        # Linting
]

[project.scripts]
triad = "triad.cli:app"
```

---

## 6. Success Criteria

The MVP is ready to ship when all of the following are true:

1. **5-minute quick start:** A developer can pip install, set API keys, and run their first pipeline in under 5 minutes.
2. **4 providers work:** Anthropic, OpenAI, Google, and xAI all route correctly through LiteLLM.
3. **Arbiter catches a real bug:** In at least one test task, the Arbiter successfully REJECTs output and the re-generation fixes the issue.
4. **Cross-model enforcement holds:** The system prevents the Arbiter from being the same model as the generator in all configurations.
5. **Output is useful:** The generated code files, test files, and Markdown summary are clean and usable without manual extraction.
6. **CLI is beautiful:** Rich output with live agent status, color-coded Arbiter verdicts, and cost summary looks good in a GIF/screenshot for the README.
7. **Zero private leakage:** grep for all private-tier keywords returns zero results in the public repo.
8. **Tests pass:** pytest suite covers schemas, provider routing, pipeline execution, and Arbiter verdict logic.
9. **Reconciliation catches a spec gap:** In at least one test task with `--reconcile` enabled, the reconciliation Arbiter detects a missing or deviated requirement and the rewind addresses it.

---

*This build spec targets the minimum viable open-source release. The full feature set from the Triad Orchestrator Spec v1.0, Addendum A, and Addendum B will be implemented incrementally post-release based on community feedback and internal development needs.*

**10 days. Ship it. Let the community tell us what to build next.**

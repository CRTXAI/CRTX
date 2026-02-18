# Triad Orchestrator — Build Specification

**Full-Featured Open-Source Release**

3-Week Sprint · 15 Working Days

*Ship everything. Ship it right. Give the community something worth forking.*

**TriadAI** | February 2026

---

## 1. Release Scope

The release is a **full-featured CLI platform** that a developer can install, configure API keys, and run multi-model coding pipelines — sequential, parallel, or debate — with smart routing, adversarial Arbiter review, consensus voting, session persistence, and codebase context injection. Plus a task planner, CI/CD integration, and a real-time dashboard.

This is not an MVP. This is the product.

### 1.1 What Ships

| Component | Specification |
|-----------|--------------|
| **Provider Layer** | ModelProvider ABC + LiteLLM universal adapter. TOML-based model registry. Pre-configured for: Anthropic (Claude Opus/Sonnet/Haiku), OpenAI (GPT-4o, o3-mini), Google (Gemini 2.5 Pro/Flash), xAI (Grok 4/3). Adding new models = adding a TOML entry. |
| **Pipeline Engine** | Three pipeline modes: Sequential (Architect → Implement → Refactor → Verify), Parallel Exploration (all agents solve independently, cross-review, merge), Debate (position papers → rebuttals → judgment). |
| **Smart Routing** | Dynamic model-to-role assignment based on fitness benchmarks. Four routing modes: quality-first, cost-optimized, speed-first, hybrid. Per-stage overrides. Cost estimation before run. |
| **Consensus** | Full consensus voting with cross-domain suggestion escalation. Suggestion schema with domain, rationale, confidence, code sketch, impact assessment. Primary role-holder evaluates; rejected suggestions escalate to group vote. Configurable tiebreaker. |
| **Arbiter Layer** | Independent adversarial review at configurable depth (off/final-only/bookend/full). Cross-model enforcement (arbiter ≠ generator). Four verdicts (APPROVE/FLAG/REJECT/HALT). Structured feedback injection on REJECT. Retry budget (max 2). Implementation Summary Reconciliation (ISR) via --reconcile flag. |
| **Session Persistence** | SQLite audit trail with full session history. JSON export. Queryable by task, model, verdict, cost, date. Session replay for debugging. |
| **Context Injection** | Auto-scan project files and feed relevant code into agent context. Configurable include/exclude patterns. AST-aware for Python. Respects .gitignore. |
| **Task Planner** | `triad plan` command: expand a rough idea into a structured task spec with one LLM call. Interactive mode with clarifying questions. Pipe directly to `triad run`. |
| **CLI** | Typer + Rich. Commands: run, plan, models, config, sessions, estimate. Live agent status panels, streaming output, Arbiter verdicts with color-coded severity, cost summary. |
| **CI/CD Integration** | GitHub Actions workflow. `triad review` as a PR check: parallel review with consensus comments. Configurable trigger (PR, push, manual). |
| **Dashboard** | React + WebSocket real-time pipeline visualization. Agent status, message flow, Arbiter verdicts, cost tracking, session history. Optional — CLI is the primary interface. |
| **Output** | Structured directory: code files, test files, arbiter reviews, session log (JSON + SQLite). Markdown summary report. Cost breakdown per stage and model. |
| **Schemas** | Pydantic v2 models: AgentMessage, ArbiterReview, Suggestion, CodeBlock, PipelineConfig, TaskSpec, PipelineResult, ConsensusResult, RoutingDecision, SessionRecord. Structured output parsing with fallback to text extraction. |

---

## 2. Architecture

```
triad/
├── __init__.py
├── cli.py                    # Typer entry point — all commands
├── orchestrator.py           # Pipeline engine (sequential, parallel, debate)
├── planner.py                # Task planner (triad plan)
├── providers/
│   ├── __init__.py
│   ├── base.py               # ModelProvider ABC
│   ├── litellm_provider.py   # Universal LiteLLM adapter
│   └── registry.py           # Model discovery + config loading
├── routing/
│   ├── __init__.py
│   ├── engine.py             # Smart routing — model-to-role assignment
│   └── strategies.py         # Quality/cost/speed/hybrid routing logic
├── arbiter/
│   ├── __init__.py
│   ├── arbiter.py            # Arbiter review engine
│   ├── feedback.py           # Structured feedback injection
│   └── reconciler.py         # ISR — Implementation Summary Reconciliation
├── consensus/
│   ├── __init__.py
│   ├── protocol.py           # Consensus decision engine
│   ├── suggestions.py        # Cross-domain suggestion handler + escalation
│   └── voting.py             # Vote collection + tiebreak
├── context/
│   ├── __init__.py
│   ├── scanner.py            # Project file scanner (respects .gitignore)
│   ├── builder.py            # Dynamic context assembly
│   └── pruner.py             # Context pruning to fit model limits
├── persistence/
│   ├── __init__.py
│   ├── session.py            # Session recording + replay
│   ├── database.py           # SQLite schema + queries
│   └── export.py             # JSON/Markdown export
├── schemas/
│   ├── __init__.py
│   ├── messages.py           # AgentMessage, Suggestion, CodeBlock
│   ├── arbiter.py            # ArbiterReview, Verdict, Issue
│   ├── reconciliation.py     # ImplementationSummary, Deviation
│   ├── pipeline.py           # PipelineConfig, TaskSpec, PipelineResult
│   ├── consensus.py          # ConsensusResult, Vote, SuggestionEscalation
│   ├── routing.py            # RoutingDecision, RoutingStrategy
│   └── session.py            # SessionRecord, SessionQuery
├── prompts/
│   ├── __init__.py           # render_prompt() utility
│   ├── architect.md
│   ├── implementer.md
│   ├── refactorer.md
│   ├── verifier.md
│   ├── arbiter.md
│   ├── reconciler.md
│   └── planner.md
├── output/
│   ├── renderer.py           # Markdown summary generator
│   └── writer.py             # File output handler
├── dashboard/                # Optional React dashboard
│   ├── server.py             # WebSocket server (FastAPI)
│   ├── static/               # Built React app
│   └── events.py             # Pipeline event emitter
└── config/
    ├── models.toml            # Model registry
    ├── defaults.toml          # Default pipeline config
    ├── routing.toml           # Routing policies
    └── domain/                # .gitignore'd — private rules
        └── (domain_patterns.toml — never committed)
```

---

## 3. Day-by-Day Build Plan

15 working days. Each day has a clear deliverable. The build is ordered so dependencies flow forward — nothing is built before what it depends on is ready.

### Week 1 — Core Pipeline (Days 1–5)

| Day | Focus | Deliverables | Milestone |
|-----|-------|-------------|-----------|
| **1** ✅ | Foundation | Project scaffold, Pydantic schemas (AgentMessage, CodeBlock, ArbiterReview, PipelineConfig, TaskSpec, ImplementationSummary), TOML config loader, model registry. | Schemas parse. Config loads. |
| **2** ✅ | Provider Layer | ModelProvider ABC. LiteLLM adapter with structured output parsing, code block extraction, retry with exponential backoff. Model registry with TOML auto-discovery. | Can call all 4 providers. |
| **3** ✅ | System Prompts | 6 role prompts (architect, implementer, refactorer, verifier, arbiter, reconciler) with Jinja2 templates. Domain context injection slots. Arbiter feedback slots. Prompt loader utility. | Prompts render correctly. |
| **4** ✅ | Pipeline Engine | Sequential pipeline orchestrator. Stage execution with async model calls via LiteLLMProvider. Output passing between stages. Cross-domain suggestion collection + downstream injection. Confidence extraction. PipelineResult aggregation (cost, tokens, duration). | **PIPELINE RUNS END-TO-END** |
| **5** | Arbiter Core | Arbiter review engine with cross-model enforcement (arbiter ≠ generator). Four verdicts (APPROVE/FLAG/REJECT/HALT). Bookend review configuration. REJECT handling: structured feedback injection, retry budget (max 2). FLAG propagation to next stage context. HALT: pipeline stops with analysis. ISR reconciliation pass (opt-in, --reconcile). | Arbiter reviews and can reject/re-gen. |

### Week 2 — Advanced Pipeline + Routing (Days 6–10)

| Day | Focus | Deliverables | Milestone |
|-----|-------|-------------|-----------|
| **6** | Parallel + Debate Modes | Extend orchestrator with parallel exploration (fan-out via asyncio.gather, cross-review, scoring, merge) and debate mode (position papers, rebuttals, final arguments, judgment). Mode selection via `--mode sequential\|parallel\|debate`. | All 3 pipeline modes work. |
| **7** | Smart Routing Engine | Routing engine with 4 strategies (quality-first, cost-optimized, speed-first, hybrid). Fitness-based model selection per stage. Cost estimation (`triad estimate`). Routing decisions logged. TOML routing policy config. | `triad run --route hybrid` works. |
| **8** | Consensus Protocol | Full consensus voting: suggestion evaluation by primary role-holder, rejection → escalation to group vote, majority wins, configurable tiebreaker. ConsensusResult schema. Integrates with all 3 pipeline modes. | Agents can disagree, suggest, vote, reach consensus. |
| **9** | Session Persistence | SQLite database schema (sessions, stages, messages, verdicts). Session recording during pipeline runs. JSON export. Query by task/model/verdict/cost/date. `triad sessions` CLI commands (list, show, export, replay). | Full audit trail persisted and queryable. |
| **10** | CLI Interface | Typer CLI: `triad run`, `triad plan`, `triad models`, `triad config`, `triad sessions`, `triad estimate`. Rich live display: agent status panels, streaming output, Arbiter verdicts with color-coded severity, cost summary, progress tracking. Beautiful terminal output. | **CLI IS COMPLETE AND BEAUTIFUL** |

### Week 3 — Ecosystem + Release (Days 11–15)

| Day | Focus | Deliverables | Milestone |
|-----|-------|-------------|-----------|
| **11** | Context Injection | Project file scanner (respects .gitignore, configurable include/exclude globs). AST-aware Python scanning (extract classes, functions, imports). Dynamic context builder: assembles relevant code for each stage. Context pruner: fits within model context limits. `--context-dir` CLI flag. | Pipeline reads existing codebase for context. |
| **12** | Task Planner | `triad plan` command: single LLM call to expand rough idea → structured task spec. `triad plan --interactive` for clarifying questions. Save to file, edit in $EDITOR, or pipe to `triad run`. Planner prompt (prompts/planner.md). PlannerResult schema. | `triad plan "build X" --run` works end-to-end. |
| **13** | CI/CD Integration | GitHub Actions workflow file (.github/workflows/triad-review.yml). `triad review` command: accepts diff input, runs parallel review, outputs consensus comments. Configurable trigger. PR comment formatting. Exit codes for CI pass/fail. | Triad runs as automated PR reviewer. |
| **14** | Dashboard | FastAPI WebSocket server. Pipeline event emitter (stage start/complete, arbiter verdict, cost update). React frontend: agent cards, pipeline flow visualization, message viewer, cost tracker, session history. Served via `triad dashboard`. | Real-time pipeline visualization in browser. |
| **15** | Release Prep | Full pytest suite (target: 300+ tests). README.md with quick start, architecture overview, configuration guide, GIF/screenshot. Disclosure audit (grep for private-tier keywords). LICENSE (Apache 2.0). CONTRIBUTING.md + CLA. GitHub repo setup. GitHub Actions CI. Final keyword scan. | **SHIP IT** |

---

## 4. Technical Decisions

| Decision | Choice | Rationale |
|----------|--------|-----------|
| Language | Python 3.12+ | LiteLLM, Pydantic, Typer, Rich all Python-native. |
| Package manager | uv (with pip fallback) | Fast, modern, gaining adoption. pip install still works. |
| LLM adapter | LiteLLM | 100+ provider support. OpenAI-compatible interface. Battle-tested. |
| Schema validation | Pydantic v2 | Industry standard. Fast. Great error messages. |
| CLI framework | Typer + Rich | Typer for argument parsing, Rich for beautiful terminal output. |
| Config format | TOML | Human-readable, Python-native (tomllib), standard for Python projects. |
| Async runtime | asyncio (stdlib) | No external deps. Sufficient for sequential, scales to parallel. |
| Session storage | SQLite | Zero config, file-based, ships with Python, queryable. |
| Dashboard | FastAPI + React | WebSocket native in FastAPI. React for component-based UI. Optional dep. |
| Template engine | Jinja2 | Industry standard, clean syntax, conditional blocks for optional sections. |
| Testing | pytest + pytest-asyncio | Standard. Easy for contributors. |
| License | Apache 2.0 (pending patent attorney) | Maximum adoption. Patent grant included. Permissive. |

---

## 5. Dependencies

```toml
[project]
name = "triad-orchestrator"
version = "0.1.0"
requires-python = ">=3.12"

dependencies = [
    "litellm>=1.55",          # Universal LLM adapter
    "pydantic>=2.0",          # Schema validation
    "typer>=0.12",            # CLI framework
    "rich>=13.0",             # Terminal UI
    "tomli>=2.0",             # TOML parsing
    "aiofiles>=24.0",         # Async file I/O
    "jinja2>=3.0",            # Prompt template rendering
    "aiosqlite>=0.20",        # Async SQLite for session persistence
]

[project.optional-dependencies]
dashboard = [
    "fastapi>=0.110",         # WebSocket server for dashboard
    "uvicorn>=0.30",          # ASGI server
    "websockets>=12.0",       # WebSocket support
]
dev = [
    "pytest>=8.0",
    "pytest-asyncio>=0.24",
    "ruff>=0.5",
]
ci = [
    "pygithub>=2.0",          # GitHub API for PR comments
]

[project.scripts]
triad = "triad.cli:app"
```

---

## 6. Feature Dependency Graph

Features build on each other. This ordering is why the day-by-day plan is sequenced the way it is:

```
Day 1: Schemas ─────────────────────┐
Day 2: Providers ───────────────────┤
Day 3: Prompts ─────────────────────┤
Day 4: Sequential Pipeline ─────────┤─── Core (everything depends on this)
Day 5: Arbiter ─────────────────────┘
         │
         ├── Day 6: Parallel + Debate (extends orchestrator)
         │     │
         │     └── Day 8: Consensus (works across all modes)
         │
         ├── Day 7: Smart Routing (works with all pipeline modes)
         │
         ├── Day 9: Session Persistence (records any pipeline run)
         │     │
         │     └── Day 14: Dashboard (reads session data + live events)
         │
         ├── Day 10: CLI (wires everything together)
         │
         ├── Day 11: Context Injection (feeds into any pipeline mode)
         │
         ├── Day 12: Task Planner (pre-pipeline, feeds into run)
         │
         └── Day 13: CI/CD (wraps triad review command)
```

---

## 7. Pipeline Mode Specifications

### 7.1 Sequential (Default)

Architect → Implement → Refactor → Verify. Each stage builds on the previous output. Arbiter reviews at configured depth (bookend by default: Architect + Verify).

### 7.2 Parallel Exploration

All assigned agents receive the identical task simultaneously. Each produces an independent solution. Cross-review phase: each agent scores the other solutions (architecture, implementation, quality — 1-10 each). Vote for best approach (no self-voting). 2-of-3 wins; three-way split → tiebreaker decides. Winning approach enhanced with best elements from others, then verified.

### 7.3 Debate Mode

Each agent proposes its preferred approach with explicit tradeoff analysis. Rebuttals: each agent receives the other proposals and writes structured critiques. Final arguments: agents update proposals incorporating valid criticisms. Judgment: tiebreaker evaluates all positions and produces a reasoned decision document. Result passes through verification.

---

## 8. Smart Routing Modes

| Mode | Behavior | Use Case |
|------|----------|----------|
| **quality-first** | Highest-fitness model per role regardless of cost | Production code, critical features |
| **cost-optimized** | Cheapest model above minimum fitness threshold (default 0.70) | Prototyping, internal tooling, batch tasks |
| **speed-first** | Lowest-latency models, Flash/mini variants preferred | Live coding, rapid iteration |
| **hybrid** (default) | Quality-first for critical stages (refactor + verify), cost-optimized for earlier stages | Best cost/quality balance |

CLI:
```
triad run --task "..." --route hybrid
triad run --task "..." --route quality-first
triad estimate --task "..." --compare-routes
```

---

## 9. Success Criteria

The release is ready when all of the following are true:

1. **5-minute quick start**: pip install, set API keys, run first pipeline in under 5 minutes.
2. **4 providers work**: Anthropic, OpenAI, Google, xAI all route correctly.
3. **3 pipeline modes work**: Sequential, parallel, and debate all produce valid output.
4. **Smart routing selects correctly**: Quality-first picks highest-fitness, cost-optimized picks cheapest above threshold.
5. **Arbiter catches a real bug**: In at least one test task, the Arbiter REJECTs and re-generation fixes the issue.
6. **Cross-model enforcement holds**: Arbiter is never the same model as the generator.
7. **Consensus resolves disagreements**: In parallel/debate mode, agents vote and reach resolution.
8. **Reconciliation catches spec drift**: With --reconcile, the ISR Arbiter detects at least one gap in a test task.
9. **Sessions persist**: Every pipeline run is recorded in SQLite and can be queried/exported.
10. **Context injection works**: Pipeline reads existing project files and uses them in agent context.
11. **Task planner expands prompts**: `triad plan "build X"` produces a structured spec and can pipe to run.
12. **CI/CD runs as PR check**: GitHub Action triggers `triad review` and posts consensus comments.
13. **Dashboard shows live pipeline**: WebSocket server streams events, React UI renders agent status in real-time.
14. **CLI is beautiful**: Rich output with live panels, color-coded verdicts, and cost summary.
15. **Output is useful**: Generated code, tests, and reports are clean and usable.
16. **Zero private leakage**: grep for all private-tier keywords returns zero results.
17. **Tests pass**: pytest suite with 300+ tests covering all components.

---

## 10. Cost Estimates

Per-task costs for a medium-complexity feature (new API endpoint with tests):

| Mode | Estimated Cost | Notes |
|------|---------------|-------|
| Sequential | ~$4.30 | 4 stages, one model each |
| Sequential + Bookend Arbiter | ~$5.80 | +2 arbiter review passes |
| Sequential + Full Arbiter | ~$7.30 | +4 arbiter review passes |
| Sequential + Reconciliation | ~$6.10–$6.40 | Bookend + ISR pass |
| Parallel | ~$8.60 | 3 independent solutions + cross-review + merge |
| Debate | ~$10–12 | Position papers + rebuttals + judgment |

At 15 tasks/week: ~$87–180/month for sequential+arbiter, ~$130–180 for parallel/debate.

---

## 11. Risk Mitigation

| Risk | Mitigation | Fallback |
|------|-----------|----------|
| API rate limits / outages | Exponential backoff with jitter. Circuit breaker per provider. | Degrade to fewer agents. Priority model always gets retried. |
| Consensus deadlock | Max 3 deliberation rounds. After round 3, tiebreaker decides unilaterally. | Developer presented all proposals to decide manually. |
| Dashboard scope creep | Dashboard is optional dep. Core pipeline works without it. Ship CLI-first. | If Day 14 runs long, ship dashboard as v0.1.1 patch. |
| Context window overflow | Pruner trims context to fit model limits. Largest context model gets most. | Fall back to manual context via --context flag. |
| SQLite concurrency | Single-writer, multiple-reader. WAL mode. Sufficient for local CLI tool. | JSON file fallback if SQLite causes issues. |
| Token cost spiraling | Per-session budget caps. Context pruning. Cost estimation before run. | Auto-switch to cheaper variants for non-critical passes. |
| 15-day timeline slips | Days 13-14 (CI/CD, Dashboard) are independent and can ship as fast-follows. | Core pipeline (Days 1-12) ships on time regardless. |

---

## 12. Post-Release Roadmap

Planned for v0.2.0+ after community feedback:

- **`triad advise`**: Full architectural consultation with adversarial review (see docs/advisor.md)
- **v0.3 — Advanced multi-model modes (coming soon)**
- **Codebase audit**: `triad audit` scans existing code with multiple models independently, cross-validates findings through the Arbiter, produces tiered report (critical/important/suggestion/dismissed), verifies fixes on re-run (see docs/audit.md)
- **Tournament mode**: Run all registered models on same task, score and rank outputs
- **Fitness benchmarking system**: Standardized task suites to measure model strengths
- **Agent memory**: Persistent memory of past decisions across sessions
- **MCP server integration**: Expose Triad as MCP server for Claude Code
- **GitHub PR review bot**: Automated consensus review on every pull request
- **Model hot-swap**: Live model replacement without pipeline restart
- **Model auto-discovery**: `triad models discover --provider anthropic` queries provider model list APIs, generates TOML config entries for new models, and optionally runs fitness benchmarks. Reduces "new model dropped" to a single command instead of manual config editing.
- **Template library**: Pre-built task specs for common patterns (REST API, CLI, data pipeline)
- **Codebase-aware advising**: Scan existing project to inform architectural recommendations

---

*15 days. Full-featured. Let the community tell us what's next.*

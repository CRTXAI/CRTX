# CRTX SDK Map — Engine vs CLI Analysis

> Prerequisite for v0.3.0: extracting the engine so consumers can
> `from crtx import Loop, Arbiter, Router`.
>
> **Read-only analysis** — no code was changed.

---

## 1. Engine Components (belongs in `crtx` package)

These are the components a third-party consumer (e.g. ClawBucks) needs when
calling `from crtx import ...`. Zero CLI, Rich, or Typer dependencies.

### 1.1 Orchestrators — `triad/orchestrator.py` (2912 lines)

The entire file is engine logic. Zero CLI imports, zero `print()` calls, zero
Rich rendering. All state is instance-scoped (no mutable module globals).

| Component | Lines | What it does | External deps |
|-----------|-------|--------------|---------------|
| `PipelineOrchestrator` | 110–825 | Sequential 4-stage pipeline (Architect → Implement → Refactor → Verify) with Arbiter integration, retries, ISR reconciliation, and suggestion evaluation | `litellm` (via providers), `pydantic`, `jinja2` (via prompts) |
| `ParallelOrchestrator` | 995–1510 | Fan-out to top-N models, cross-review scoring, consensus vote, synthesis, completion pass, arbiter review | same |
| `DebateOrchestrator` | 1516–1953 | Structured debate: propose → rebuttal → final argument → judge decision, arbiter review | same |
| `ReviewOrchestrator` | 1959–2281 | Multi-model code review: independent analysis → cross-review → synthesis, arbiter review | same |
| `ImproveOrchestrator` | 2287–2745 | Multi-model code improvement: independent improvements → cross-review → consensus → synthesis | same |
| `run_pipeline()` | 2751–2812 | Top-level dispatch: routes to the correct orchestrator based on `PipelineMode`, handles context injection and session persistence | same |
| `_save_session()` | 2815–2865 | Persist a `PipelineResult` to SQLite | `aiosqlite` |
| `_inject_context()` | 2868–2912 | Scan project dir and prepend codebase context to task spec | none (internal) |
| `_select_top_models()` | 901–947 | Select top-N models with provider diversity enforcement | none (pure logic) |
| `_extract_confidence()` | 828–840 | Parse `CONFIDENCE: <float>` from model output | none |
| `_format_suggestions()` | 843–867 | Format cross-domain suggestions for prompt injection | none |
| `_extract_scores()` | 877–890 | Parse architecture/implementation/quality scores from review | none |
| `_get_tiebreaker_key()` | 893–895 | Select highest-fitness verifier as tiebreaker | none |
| `_detect_incomplete_sections()` | 950–992 | Detect placeholder/lazy code in synthesized output | none |

**Note:** `_call_model()` is implemented independently (and near-identically) in
`ParallelOrchestrator`, `DebateOrchestrator`, `ReviewOrchestrator`, and
`ImproveOrchestrator` (~80 lines each, same retry/fallback pattern). Strong
candidate for extraction into a shared base class or mixin during SDK refactor.

### 1.2 Loop Engine — `triad/loop/`

Already cleanly separated from CLI via callback hooks. The `LoopOrchestrator`
accepts `on_route`, `on_generate`, `on_test`, `on_fix`, `on_review`, and
`on_escalation` callbacks — it never imports the presenter.

| Component | File | Lines | What it does | External deps |
|-----------|------|-------|--------------|---------------|
| `LoopOrchestrator` | `orchestrator.py` | 121–678 | Core generate → test → fix → review cycle with 3-tier gap-closing escalation | `litellm` (via providers) |
| `LoopResult` | `orchestrator.py` | 107–118 | Result dataclass for a loop run | `dataclasses` |
| `LoopStats` | `orchestrator.py` | 91–103 | Statistics dataclass (iterations, tokens, cost) | `dataclasses` |
| `GapResult` | `orchestrator.py` | 79–87 | Result of gap-closing escalation (3 tiers: diagnose, minimal fix, second opinion) | `dataclasses` |
| `CodeGenerator` | `generator.py` | 66–143 | Single-model code generation from prompt | `litellm` (via providers) |
| `GenerationResult` | `generator.py` | 53–63 | Generation output dataclass | `dataclasses` |
| `extract_files_from_content()` | `generator.py` | 146–193 | Two-pass parser: code-block regex then `# file:` header splitting | none |
| `CodeFixer` | `fixer.py` | 42–251 | Fix code from test failures, identify broken files, detect phantom test imports via AST | `litellm` (via providers) |
| `FixResult` | `fixer.py` | 31–39 | Fix output dataclass | `dataclasses` |
| `ArbiterReviewer` | `reviewer.py` | 30–127 | Post-loop arbiter review; triggers one more fix cycle on REJECT/HALT | `litellm` (via providers) |
| `ReviewResult` | `reviewer.py` | 19–27 | Review output dataclass | `dataclasses` |
| `TaskRouter` | `router.py` | 48–96 | Rule-based prompt classifier: SIMPLE/MEDIUM/COMPLEX/SAFETY tiers via keyword matching | none (pure logic, zero deps) |
| `TaskComplexity` | `router.py` | 9–13 | Complexity enum (StrEnum) | none |
| `RouteDecision` | `router.py` | 17–24 | Routing decision: complexity, model, debate flag, max fix iterations, arbiter required | `dataclasses` |
| `TestRunner` | `test_runner.py` | 65–416 | Execute parse checks, import validation, pyflakes, pytest, entry-point on generated code | `subprocess` |
| `TestReport` | `test_runner.py` | 28–62 | Test execution report dataclass | `dataclasses` |
| `CheckResult` | `test_runner.py` | 20–24 | Single check result dataclass | `dataclasses` |

**Note:** `_find_model_config()` is duplicated in both `generator.py:131` and
`fixer.py:242`. Factor out to a shared utility during extraction.

### 1.3 Providers — `triad/providers/`

100% engine logic. Single choke-point for all LLM calls in the entire codebase.

| Component | File | Lines | What it does | External deps |
|-----------|------|-------|--------------|---------------|
| `ModelProvider` | `base.py` | 18–177 | Abstract base class for all LLM providers (complete, streaming, cost calc) | `pydantic` |
| `LiteLLMProvider` | `litellm_provider.py` | 77–467 | Universal provider adapter via `litellm.acompletion()` with retry, streaming, structured output parsing | `litellm` |
| `extract_code_blocks()` | `litellm_provider.py` | 470–526 | Parse fenced code blocks from model output with filepath hint detection | none |
| `_merge_filepath_blocks()` | `litellm_provider.py` | 529–566 | Merge `# file:` annotated code blocks with following code block | none |
| `_filter_untitled_fragments()` | `litellm_provider.py` | 569–596 | Remove orphan code fragments | none |
| `_short_error_reason()` | `litellm_provider.py` | 54–74 | Map LiteLLM error status codes to short reason strings (for logging, not display) | none |
| `ProviderHealth` | `health.py` | 17–52 | Track unhealthy models with cooldown-based recovery (fully self-contained, zero imports) | `time` (stdlib) |
| `load_models()` | `registry.py` | 26–61 | Load model registry from `models.toml` → `dict[str, ModelConfig]` | `tomllib` |
| `load_pipeline_config()` | `registry.py` | 64–104 | Load pipeline config from `defaults.toml` → `PipelineConfig` | `tomllib` |
| `get_best_model_for_role()` | `registry.py` | 107–132 | Find highest-fitness model for a given pipeline role | none |

**Note:** `litellm_provider.py:20` has a module-level side effect:
`litellm.suppress_debug_info = True`. This suppresses LiteLLM banners globally
at import time. SDK consumers should be aware.

### 1.4 Arbiter — `triad/arbiter/`

100% engine logic. No CLI constructs anywhere.

| Component | File | Lines | What it does | External deps |
|-----------|------|-------|--------------|---------------|
| `ArbiterEngine` | `arbiter.py` | 47–296 | Cross-model quality review: APPROVE / FLAG / REJECT / HALT verdicts with fallback and confidence downgrade | `litellm` (via providers), `jinja2` |
| `ReconciliationEngine` | `reconciler.py` | 27–165 | Post-pipeline ISR reconciliation: compare Verifier output against TaskSpec + Architect scaffold for spec drift | `litellm` (via providers), `jinja2` |
| `format_arbiter_feedback()` | `feedback.py` | 12–75 | Format arbiter review into structured Markdown for retry prompt injection (LLM-facing, not human-facing) | none |
| `_extract_verdict()` | `arbiter.py` | 299–312 | Parse Verdict from arbiter response; fallback to FLAG | none |
| `_extract_confidence()` | `arbiter.py` | 315–324 | Parse confidence float (0.0–1.0); default 0.5 | none |
| `_parse_issues()` | `arbiter.py` | 350–392 | Parse `## Issues` section into typed `Issue` objects | none |
| `_parse_alternatives()` | `arbiter.py` | 395–437 | Parse `## Alternatives` section into `Alternative` objects | none |

**Note:** `reconciler.py` imports 4 private functions (`_extract_verdict`, etc.)
directly from `arbiter.py`. These shared parsers should be promoted to a public
parsing utility during SDK extraction.

### 1.5 Routing — `triad/routing/`

100% engine logic. Pure model-selection algorithms.

| Component | File | Lines | What it does | External deps |
|-----------|------|-------|--------------|---------------|
| `RoutingEngine` | `engine.py` | 90–326 | Smart model-to-role assignment with per-stage overrides, health-aware filtering, fallback, and diversity enforcement | none (pure logic) |
| `estimate_cost()` | `engine.py` | 327–365 | Estimate pipeline run cost for a routing strategy using per-stage token estimates | none |
| `quality_first()` | `strategies.py` | 34–54 | Rank models by fitness for role (highest wins) | none |
| `cost_optimized()` | `strategies.py` | 57–94 | Rank by cost with minimum fitness threshold | none |
| `speed_first()` | `strategies.py` | 97–126 | Rank by context window as latency proxy | none |
| `hybrid()` | `strategies.py` | 129+ | Quality-first for critical stages (architect, verify), cost-optimized for others | none |

### 1.6 Consensus — `triad/consensus/`

100% engine logic.

| Component | File | Lines | What it does | External deps |
|-----------|------|-------|--------------|---------------|
| `ConsensusEngine` | `protocol.py` | 35+ | Evaluate suggestions, escalate disagreements, resolve via tiebreaker | `litellm` (via providers), `jinja2` |
| `SuggestionEvaluator` | `suggestions.py` | 74+ | Score individual suggestions against multiple models | `litellm` (via providers), `jinja2` |
| `format_suggestion_for_evaluation()` | `suggestions.py` | 34 | Format a suggestion for LLM evaluation prompt | none |
| `parse_evaluation()` | `suggestions.py` | 55 | Parse DECISION: ACCEPT|REJECT from model output | none |
| `tally_votes()` | `voting.py` | 21 | Count votes, identify winner or tie | none |
| `select_tiebreaker()` | `voting.py` | 56 | Choose highest verifier-fitness model for tiebreaking | none |
| `extract_winner()` | `voting.py` | 73 | Parse WINNER: key from tiebreaker output | none |
| `build_consensus_result()` | `voting.py` | 79 | Assemble final `ConsensusResult` from votes and tally | none |

### 1.7 Context — `triad/context/`

100% engine logic. Zero external deps beyond stdlib.

| Component | File | What it does | External deps |
|-----------|------|--------------|---------------|
| `CodeScanner` | `scanner.py` | Walk project dir, parse Python files via AST, apply include/exclude globs, respect `.gitignore` | none |
| `ContextBuilder` | `builder.py` | Multi-factor relevance scoring (keyword overlap, structural richness, entry-point bonus), assemble within token budget | none |
| `ContextPruner` | `pruner.py` | Prune assembled context to fit model's available window (minus 4000 reserved tokens) | none |

### 1.8 Apply — `triad/apply/`

Engine-side components for code application (file writing, patching, git ops).

| Component | File | What it does | External deps |
|-----------|------|--------------|---------------|
| `ConflictDetector` | `conflict.py` | Detect file changes between scan-time and apply-time via mtime + SHA-256 | none (standalone) |
| `GitSafetyGate` | `git.py` | Pre-apply git safety checks: branch protection, dirty worktree, commit, rollback | `subprocess` (git) |
| `FilePathResolver` | `resolver.py` | 4-tier resolution cascade: exact join → basename match → fuzzy match → create new | none |
| `extract_code_blocks_from_result()` | `resolver.py` | Merge structured + regex-parsed code blocks from `PipelineResult` | none |
| `PostApplyVerifier` | `verify.py` | Run post-apply tests with 300s timeout, rollback from backups on failure | `subprocess` |
| `ASTPatcher` | `patcher.py` | AST-aware structured patcher for Python files: 7 ops, 3-tier anchor resolution | none |

**Note:** `DiffPreview` and `ConflictResolver` (`diff.py`) are **CLI**, not
engine — they use Rich panels, interactive prompts, and `rich.console.Console`.
See §2. `ApplyEngine` (`engine.py`) is **ambiguous** — see §3.

### 1.9 Persistence — `triad/persistence/`

100% engine logic. No CLI dependencies.

| Component | File | What it does | External deps |
|-----------|------|--------------|---------------|
| `SessionStore` | `session.py` | Full CRUD for session records in SQLite (save, get, list, delete, export, prefix-match) | `aiosqlite` |
| `init_db()` / `close_db()` | `database.py` | Initialize SQLite with WAL mode + foreign keys, DDL for 4 tables | `aiosqlite` |
| `export_json()` | `export.py` | Export `SessionRecord` to JSON string | none |
| `export_markdown()` | `export.py` | Export `SessionRecord` to structured Markdown report | none |

### 1.10 Schemas — `triad/schemas/`

All pure Pydantic data models. **Every file is engine.** 13 files, zero CLI deps.

| File | Key types |
|------|-----------|
| `pipeline.py` | `PipelineConfig`, `ModelConfig`, `PipelineResult`, `TaskSpec`, `PipelineMode`, `ArbiterMode`, `RoleFitness`, `StageConfig` |
| `messages.py` | `AgentMessage`, `PipelineStage`, `MessageType`, `TokenUsage`, `CodeBlock`, `Suggestion`, `Objection` |
| `arbiter.py` | `ArbiterReview`, `Verdict`, `Issue`, `Alternative`, `Severity`, `IssueCategory` |
| `routing.py` | `RoutingDecision`, `RoutingStrategy`, `CostEstimate` |
| `consensus.py` | `ConsensusResult`, `ParallelResult`, `DebateResult`, `ImproveResult`, `ReviewResult`, `SuggestionDecision`, `SuggestionVerdict`, `VoteTally`, `EscalationResult` |
| `session.py` | `SessionRecord`, `StageRecord`, `SessionSummary`, `SessionQuery` |
| `context.py` | `ScannedFile`, `ProjectProfile`, `ContextResult`, `FunctionSignature` |
| `apply.py` | `ApplyConfig`, `ApplyResult`, `ResolvedFile`, `FileAction`, `GitState`, `StructuredPatch`, `PatchResult`, `FileConflict`, `Resolution` |
| `reconciliation.py` | `Deviation`, `ImplementationSummary` |
| `planner.py` | `PlannerResult` |
| `streaming.py` | `StreamChunk` |
| `ci.py` | `ReviewConfig`, `ReviewFinding`, `ModelAssessment`, `ReviewResult` |

### 1.11 Other Engine Modules

| Component | File | What it does | External deps |
|-----------|------|--------------|---------------|
| `TaskPlanner` | `planner.py` | Expand rough task idea into structured `TaskSpec` via LLM, with quick and interactive modes | `litellm` (via providers), `jinja2` |
| `render_prompt()` | `prompts/__init__.py` | Load and render Jinja2 prompt templates from `.md` files | `jinja2` |
| `write_pipeline_output()` | `output/writer.py` | Write pipeline output files to structured session dirs (`code/`, `tests/`, `reviews/`, `summary.md`, `session.json`) | none |
| `render_summary()` | `output/renderer.py` | Render Markdown summary from `PipelineResult` covering all 5 pipeline modes | none |
| `PipelineEventEmitter` | `dashboard/events.py` | Event types (17 types), Pydantic event model, async emitter with listener registration. No server deps. | `pydantic` |
| `ReviewRunner` | `ci/reviewer.py` | Multi-model parallel code review with cross-validation and consensus recommendation | `litellm` (via providers), `jinja2` |
| `BenchmarkRunner` | `benchmark/runner.py` | Benchmark suite: runs single-model and CRTX pipeline conditions across prompts with verification | `litellm` (via providers) |
| `BenchmarkScorer` | `benchmark/scorer.py` | Score generated code on 5 dimensions: parse_rate, runs, tests, type_hints, imports | `subprocess` |
| `SingleModelRunner` | `benchmark/single_model.py` | Single-model baseline benchmark runner | `litellm` (via providers) |
| `BenchmarkPrompt` | `benchmark/prompts.py` | 3 tiered benchmark prompts (simple/medium/complex) | none |
| `ProAgent` | `pro/agent.py` | Event batching and async HTTP delivery to CRTX Pro API (0.1s window, max 200 events) | `urllib` (stdlib) |

### 1.12 Config Files (ship with SDK)

| File | Purpose |
|------|---------|
| `config/defaults.toml` | Pipeline defaults (arbiter_mode, timeouts, retries) |
| `config/models.toml` | Model registry (all supported models with fitness scores) |
| `config/routing.toml` | Routing policy (strategy, min_fitness, token estimates) |
| `prompts/*.md` | 28 Jinja2 prompt templates (arbiter, architect, debate_*, evaluate_suggestion, implementer, improve_*, parallel_*, planner, reconciler, refactorer, review_*, tiebreak, verifier) |

---

## 2. CLI-Only Components (stays in CLI)

These depend on `typer`, `rich`, interactive I/O, or are CLI-specific UX.
A library consumer never needs these.

### 2.1 CLI Entry Point — `triad/cli.py` (2933 lines)

All 30+ Typer commands and 14 helper functions. No classes defined — all
module-level functions registered as Typer commands.

| Group | Functions | Purpose |
|-------|-----------|---------|
| **App setup** | `app`, `main`, `repl`, `_version_callback` | Typer app, entry point, REPL launcher |
| **Core commands** | `run` (438–841), `plan` (1282–1441), `review_code` (2233–2459), `improve` (2465–2677), `loop` (2779–2859), `demo` (423–433), `benchmark` (2683–2773) | CLI wrappers that construct config → call engine → display results |
| **Model/config** | `models_list`, `models_show`, `models_test`, `config_show`, `config_path` | Registry/config inspection |
| **Sessions** | `sessions_list`, `sessions_show`, `sessions_export`, `sessions_delete` | Session CRUD via CLI |
| **Display helpers** | `_display_completion` (901–1157), `_display_task_panel`, `_display_apply_result`, `_display_plan_result`, `_display_result` | Rich panels, tables, formatting |
| **Utilities** | `_load_registry`, `_load_config`, `_verdict_style`, `_format_duration`, `_display_name_from_litellm_id` | CLI wrappers around engine calls |
| **Setup** | `setup` (202–373), `_setup_check` (376–417) | Interactive API key configuration |
| **Dashboard** | `dashboard` (2865–2927) | Standalone uvicorn dashboard server command |
| **Show** | `show` (1199–1239), `_find_latest_session`, `_find_session_by_prefix`, `_load_session_result` | View previous run outputs |
| **File reading** | `_read_source_files` (2211–2227) | Read source files for review/improve commands |

**Engine logic embedded in CLI** (must be extracted — see §3):

The following pattern is repeated in `run`, `_run_from_plan`, `review_code`,
and `improve`:

1. Resolve preset → validate enums → build `PipelineConfig` + `TaskSpec`
2. Pre-flight model reachability check via `_select_top_models()`
3. Create `PipelineEventEmitter` → attach dashboard relay → call `run_pipeline()`
4. Write output → display completion → interactive viewer

This should become a single SDK function like
`build_and_run(task, preset, overrides)`.

### 2.2 Display Modules

| File | Lines | What it does |
|------|-------|--------------|
| `cli_display.py` | ~1000 | Rich terminal display: ASCII logos, `ConfigScreen` (interactive pre-run config), `PipelineDisplay` (real-time Rich Live stage progress), `CompletionSummary` (post-run summary panel), brand color constants |
| `cli_streaming_display.py` | ~530 | `ScrollingPipelineDisplay`: real-time streaming token display with pinned status bar, `StreamBuffer` for text accumulation and file-boundary detection |
| `post_run_viewer.py` | ~420 | `PostRunViewer`: interactive post-run menu with keypress navigation — summary, code files, arbiter reviews, diffs, improve/apply actions |
| `loop/presenter.py` | ~300 | `LoopPresenter`: Rich terminal rendering for Loop progress callbacks — the **only** CLI file in `triad/loop/` |
| `apply/diff.py` | ~370 | `DiffPreview` (unified diff with interactive file toggle) and `ConflictResolver` (per-file conflict resolution prompts). Uses Rich panels and interactive input. |
| `benchmark/reporter.py` | ~300 | Rich tables for benchmark results, JSON export, color-coded score display |

### 2.3 Other CLI Modules

| File | What it does |
|------|--------------|
| `keys.py` | API key management: `load_keys_env()`, `save_keys()`, `validate_key()`, `KEYS_FILE`, `PROVIDERS` registry with signup URLs |
| `demo.py` | Guided 60-second demo: `run_demo()`, `select_demo_models()`, hardcoded demo task, Rich output with educational annotations |
| `repl.py` | Interactive REPL session: `TriadREPL` with command dispatch, mode/route/arbiter state, provider health, dashboard integration |
| `ci/formatter.py` | CI output formatting: `format_github_comments()`, `format_summary()`, `format_exit_code()` for CI/CD systems |
| `dashboard/server.py` | FastAPI WebSocket server + REST API for real-time dashboard visualization, `DashboardServer` background thread |

---

## 3. Shared / Ambiguous

| Component | Current location | Recommendation | Reasoning |
|-----------|-----------------|----------------|-----------|
| `presets.py` | `triad/presets.py` | **SDK** | Presets define pipeline configurations — a library consumer benefits from `resolve_preset("fast")` just like the CLI does. Move to SDK, keep CLI importing it. |
| `keys.py` | `triad/keys.py` | **CLI** | Key management is filesystem I/O with `~/.crtx/keys.env` and `.env`. SDK consumers manage their own API keys via env vars or constructor args. `validate_key()` is mildly engine-adjacent but coupled to `KEYS_FILE` and `PROVIDERS`. |
| `ApplyEngine` | `triad/apply/engine.py` | **Split** | The 8-step orchestration logic (git safety → resolve → baseline → write → verify) is engine. But it instantiates `DiffPreview` and `ConflictResolver` with a `rich.console.Console` — those are CLI. Extract the engine steps; inject a preview/conflict interface. |
| `_attach_dashboard_relay()` | `triad/cli.py:161–196` | **SDK** | The event relay pattern (HTTP POST to dashboard) is reusable monitoring infrastructure. Extract to `dashboard/events.py`; keep server startup in CLI. |
| `run_pipeline()` | `triad/orchestrator.py:2751–2812` | **SDK** | The top-level dispatcher is the natural SDK entry point. It handles context injection, pipeline mode dispatch, and session persistence — all engine concerns. The ambiguity is only that CLI currently adds a `PipelineEventEmitter` before calling it. |
| `dashboard/events.py` | `triad/dashboard/events.py` | **SDK** | Event types, `PipelineEvent` model, and `PipelineEventEmitter` are engine-side hooks. The server (`server.py`) that consumes them is CLI. |
| `ci/reviewer.py` | `triad/ci/reviewer.py` | **SDK** | `ReviewRunner` does parallel review, cross-validation, consensus — pure engine. But the CLI `review` command handles `git diff` via subprocess. SDK should accept a diff string; CLI handles `git diff`. |
| `ci/formatter.py` | `triad/ci/formatter.py` | **CLI** | Formats review output for GitHub PR comments and CI exit codes. Presentation layer for CI systems. |
| `output/renderer.py` | `triad/output/renderer.py` | **SDK** | `render_summary()` generates Markdown from `PipelineResult` — consumed by both CLI display and file output. No CLI deps. |
| `output/writer.py` | `triad/output/writer.py` | **SDK** | `write_pipeline_output()` writes structured dirs to disk. No CLI deps. Consumers wanting disk persistence need this. |
| `persistence/` | `triad/persistence/` | **SDK** | Session storage is useful for any consumer wanting run history. No CLI deps. Make `aiosqlite` an optional extra. |
| `_format_duration()` | `triad/cli.py:877–883` | **SDK** | Pure utility. Move to a shared utils module. |
| `_display_name_from_litellm_id()` | `triad/cli.py:862–874` & `orchestrator.py:164–169` | **SDK** | Duplicated in CLI and orchestrator. Extract to `providers/registry.py` as a public utility. |
| `loop` command monkey-patch | `triad/cli.py:2830–2844` | **SDK** | CLI monkey-patches `orchestrator._router.classify` to override `max_fix_iterations`. This should be a proper parameter on `LoopOrchestrator` (e.g. `max_fix_override`). |

---

## 4. Dependency Graph

```
run_pipeline()
├── PipelineOrchestrator
│   ├── RoutingEngine ──── strategies (quality_first, cost_optimized, speed_first, hybrid)
│   ├── ArbiterEngine ──── format_arbiter_feedback()
│   ├── ReconciliationEngine
│   ├── ConsensusEngine ── SuggestionEvaluator, voting (tally_votes, extract_winner)
│   ├── ProviderHealth
│   ├── LiteLLMProvider ── ModelProvider (base)
│   ├── render_prompt() ── prompts/*.md (Jinja2 templates)
│   └── _select_top_models()
│
├── ParallelOrchestrator
│   ├── ArbiterEngine, ConsensusEngine
│   ├── LiteLLMProvider, ProviderHealth
│   └── _select_top_models()
│
├── DebateOrchestrator
│   ├── ArbiterEngine, ConsensusEngine
│   ├── LiteLLMProvider, ProviderHealth
│   └── _select_top_models()
│
├── ReviewOrchestrator
│   ├── ArbiterEngine
│   ├── LiteLLMProvider, ProviderHealth
│   └── _select_top_models()
│
├── ImproveOrchestrator
│   ├── ArbiterEngine, ConsensusEngine
│   ├── LiteLLMProvider, ProviderHealth
│   └── _select_top_models()
│
├── _inject_context()
│   ├── CodeScanner
│   └── ContextBuilder ── ContextPruner
│
└── _save_session()
    └── SessionStore ── init_db()

LoopOrchestrator
├── TaskRouter
├── CodeGenerator ──── LiteLLMProvider
├── CodeFixer ─────── LiteLLMProvider
├── ArbiterReviewer ── ArbiterEngine, CodeFixer, TestRunner
├── TestRunner ─────── subprocess (pytest, pyflakes)
└── (callbacks: on_route, on_generate, on_test, on_fix, on_review, on_escalation)

ApplyEngine
├── FilePathResolver ── extract_code_blocks_from_result()
├── ConflictDetector
├── DiffPreview ─────── [CLI: rich.console]
├── ConflictResolver ── [CLI: rich.console]
├── GitSafetyGate ───── subprocess (git)
├── ASTPatcher
└── PostApplyVerifier ── subprocess

TaskPlanner ── LiteLLMProvider, render_prompt()

ReviewRunner (CI) ── LiteLLMProvider, render_prompt()

BenchmarkRunner ── SingleModelRunner, LoopOrchestrator, run_pipeline(), TestRunner
```

### Flat dependency table

| Component | Depends on |
|-----------|-----------|
| `schemas/*` | `pydantic` (external only) |
| `ProviderHealth` | nothing (zero imports) |
| `ModelProvider` | `schemas` |
| `LiteLLMProvider` | `ModelProvider`, `schemas`, `litellm` |
| `load_models()` / `load_pipeline_config()` | `schemas`, `tomllib` |
| `routing/strategies` | `schemas` |
| `RoutingEngine` | `strategies`, `schemas`, `ProviderHealth` |
| `format_arbiter_feedback()` | `schemas` |
| `ArbiterEngine` | `LiteLLMProvider`, `ProviderHealth`, `schemas`, `render_prompt()` |
| `ReconciliationEngine` | `ArbiterEngine` (parsing helpers), `LiteLLMProvider`, `schemas`, `render_prompt()` |
| `SuggestionEvaluator` | `LiteLLMProvider`, `schemas`, `render_prompt()` |
| `voting` | `schemas` |
| `ConsensusEngine` | `SuggestionEvaluator`, `voting`, `LiteLLMProvider`, `schemas`, `render_prompt()` |
| `CodeScanner` | `schemas.context` |
| `ContextBuilder` | `schemas.context` |
| `ContextPruner` | `schemas.context` |
| `TestRunner` | `subprocess` (fully self-contained within `triad/loop`) |
| `TaskRouter` | nothing (zero imports beyond stdlib) |
| `CodeGenerator` | `TaskRouter`, `LiteLLMProvider`, `schemas` |
| `CodeFixer` | `CodeGenerator`, `TaskRouter`, `TestRunner`, `LiteLLMProvider`, `schemas` |
| `ArbiterReviewer` | `ArbiterEngine`, `CodeFixer`, `TaskRouter`, `TestRunner`, `schemas` |
| `LoopOrchestrator` | `TaskRouter`, `CodeGenerator`, `CodeFixer`, `ArbiterReviewer`, `TestRunner`, `LiteLLMProvider` |
| `ConflictDetector` | nothing (standalone) |
| `PostApplyVerifier` | `subprocess` (standalone) |
| `GitSafetyGate` | `subprocess`, `schemas.apply` |
| `FilePathResolver` | `schemas.apply`, `schemas.messages` |
| `ASTPatcher` | `schemas.apply` |
| `ApplyEngine` | `FilePathResolver`, `ConflictDetector`, `DiffPreview` [CLI], `GitSafetyGate`, `PostApplyVerifier`, `schemas` |
| `render_prompt()` | `jinja2` |
| `TaskPlanner` | `LiteLLMProvider`, `render_prompt()`, `schemas` |
| `PipelineOrchestrator` | `RoutingEngine`, `ArbiterEngine`, `ReconciliationEngine`, `ConsensusEngine`, `LiteLLMProvider`, `ProviderHealth`, `render_prompt()`, `schemas` |
| `run_pipeline()` | All 5 orchestrators, `SessionStore`, `CodeScanner`, `ContextBuilder` |
| `SessionStore` | `aiosqlite`, `schemas` |
| `write_pipeline_output()` | `render_summary()`, `schemas` |
| `render_summary()` | `schemas`, `load_models()` |
| `PipelineEventEmitter` | `pydantic` |
| `ReviewRunner` | `LiteLLMProvider`, `render_prompt()`, `schemas` |
| `BenchmarkRunner` | `run_pipeline()`, `LoopOrchestrator`, `SingleModelRunner`, `TestRunner`, `schemas` |
| `ProAgent` | `keys` (for `PRO_KEY_ENV`), `urllib` |

---

## 5. Recommended Extraction Order

Extract leaves first, work toward the root. Each phase can be a single PR.

### Phase 1 — Leaf data models (zero internal deps)

```
triad/schemas/ → crtx/schemas/
```

Everything else imports these. Extract first, change nothing else.

**Files:** All 13 files in `triad/schemas/`.
**Risk:** None. Pure data definitions.

### Phase 2 — Provider layer

```
triad/providers/ → crtx/providers/
```

Depends only on `schemas`. The entire codebase calls LLMs through this layer.

**Files:** `base.py`, `litellm_provider.py`, `health.py`, `registry.py`,
`__init__.py`.
**Risk:** `registry.py` reads from `triad/config/` via `__file__`-relative path.
Must update to use `importlib.resources` or make config dir configurable.

### Phase 3 — Prompt templates + config files

```
triad/prompts/ → crtx/prompts/
triad/config/  → crtx/config/
```

Prompt templates are Jinja2 `.md` files loaded by `render_prompt()`. Config
TOMLs define model registry and pipeline defaults. Both are pure data.

**Files:** `prompts/__init__.py`, 28 `.md` templates, 3 `.toml` configs.
**Risk:** None. Static data + one loader function.

### Phase 4 — Routing + Arbiter + Consensus (middle tier)

```
triad/routing/   → crtx/routing/
triad/arbiter/   → crtx/arbiter/
triad/consensus/ → crtx/consensus/
```

Depends on `schemas` + `providers` + `prompts`. These are the quality-control
layer that all orchestrators share.

**Files:** 3 routing files, 3 arbiter files, 3 consensus files (9 total).
**Action:** Promote `arbiter.py`'s shared parsing helpers (`_extract_verdict`,
`_parse_issues`, etc.) to a public `arbiter.parsing` module.

### Phase 5 — Context + Apply + Persistence + Output (utility engines)

```
triad/context/     → crtx/context/
triad/apply/       → crtx/apply/       (engine files only, not diff.py)
triad/persistence/ → crtx/persistence/
triad/output/      → crtx/output/
```

Independent subsystems. Context and Apply have no provider deps.
Persistence depends only on `schemas` + `aiosqlite`.

**Files:** 3 context, 5 apply (excluding `diff.py`), 3 persistence, 2 output (13 total).
**Action:** Split `apply/engine.py` — extract engine orchestration, inject
preview/conflict resolution as an interface. `DiffPreview` and `ConflictResolver`
stay in CLI.

### Phase 6 — Loop engine

```
triad/loop/ → crtx/loop/   (exclude presenter.py)
```

Depends on `providers`, `arbiter`, `schemas`, and its own `router`.
`LoopPresenter` stays in CLI — it's the only CLI file in this package.

**Files:** `orchestrator.py`, `generator.py`, `fixer.py`, `reviewer.py`,
`router.py`, `test_runner.py` (6 files). Exclude `presenter.py`.
**Action:** Factor out duplicated `_find_model_config()` from generator and fixer.
Add `max_fix_override` parameter to `LoopOrchestrator` to replace CLI monkey-patch.

### Phase 7 — Pipeline orchestrators + planner + presets (the root)

```
triad/orchestrator.py → crtx/orchestrator.py
triad/planner.py      → crtx/planner.py
triad/presets.py      → crtx/presets.py
```

These are the top-level entry points. Depends on everything in phases 1–6.
After this, `from crtx import run_pipeline, LoopOrchestrator, TaskPlanner` works.

**Action:** Extract the repeated PipelineConfig-building pattern from `cli.py`
(preset resolution → enum validation → config construction → pre-flight checks)
into an SDK-level `build_pipeline_config()` or `PipelineBuilder` helper.
Also factor out the duplicated `_call_model()` across the 4 parallel orchestrators
into a shared base class or mixin.

### Phase 8 — Public API surface (`crtx/__init__.py`)

Wire up the public exports:

```python
from crtx.loop.orchestrator import LoopOrchestrator, LoopResult
from crtx.orchestrator import run_pipeline, PipelineOrchestrator
from crtx.providers import LiteLLMProvider, ModelProvider, load_models, load_pipeline_config
from crtx.routing import RoutingEngine, estimate_cost
from crtx.arbiter import ArbiterEngine
from crtx.consensus import ConsensusEngine
from crtx.schemas.pipeline import PipelineConfig, PipelineResult, TaskSpec, ModelConfig
from crtx.planner import TaskPlanner
from crtx.apply.engine import ApplyEngine
from crtx.presets import resolve_preset, PRESETS
```

### Phase 9 — CLI migration

Repoint all CLI imports from `triad.*` to `crtx.*`. CLI becomes a thin
consumer of the SDK. Reorganize CLI-only files:

```
triad/cli.py                   → triad/cli/main.py  (or keep flat)
triad/cli_display.py           → triad/cli/display.py
triad/cli_streaming_display.py → triad/cli/streaming.py
triad/post_run_viewer.py       → triad/cli/viewer.py
triad/keys.py                  → triad/cli/keys.py
triad/demo.py                  → triad/cli/demo.py
triad/repl.py                  → triad/cli/repl.py
triad/loop/presenter.py        → triad/cli/loop_presenter.py
triad/apply/diff.py            → triad/cli/diff_preview.py
triad/dashboard/server.py      → triad/cli/dashboard.py
triad/benchmark/reporter.py    → triad/cli/benchmark_reporter.py
triad/ci/formatter.py          → triad/cli/ci_formatter.py
```

---

## 6. Open Questions

1. **Package naming:** Is the SDK package `crtx` (replacing `triad` in
   `pyproject.toml`) or a new `crtx-sdk` alongside the existing `crtx` CLI
   package? Monorepo with two packages, or single package with extras?

2. **Event emitter protocol:** All five orchestrators accept `event_emitter:
   object | None`. Should the SDK define a formal `EventEmitter` protocol/ABC,
   or keep it duck-typed? A formal protocol would let consumers plug in their
   own monitoring without studying the codebase.

3. **Config loading strategy:** `load_models()` and `load_pipeline_config()`
   read from bundled TOML files using `__file__`-relative paths. SDK consumers
   may want to pass their own config dicts instead. Should the SDK accept both
   TOML paths and raw dicts?

4. **`aiosqlite` as SDK dependency:** Persistence is optional (guarded by
   `config.persist_sessions`). Should `aiosqlite` be a required SDK dep, or
   an optional extra (`crtx[persistence]`)?

5. **`subprocess` in `TestRunner` and `ApplyEngine`:** These shell out to
   `pytest`, `pyflakes`, and `git`. Is this acceptable for SDK consumers, or
   should the SDK define injectable test/git interfaces?

6. **Stream callback typing:** `stream_callback: Callable | None` is untyped.
   Should the SDK define a `StreamCallback` protocol with a clear signature?

7. **`_select_top_models()` visibility:** Currently a module-level private
   function used by all parallel orchestrators and imported by `cli.py`.
   Promote to `crtx.routing.select_top_models()`?

8. **`ApplyEngine` CLI coupling:** `ApplyEngine` instantiates `DiffPreview`
   and `ConflictResolver` directly with a `Console` arg. SDK extraction
   requires injecting a preview/conflict interface. Define an ABC, or accept
   callbacks?

9. **Duplicated `_call_model()` across orchestrators:** Four near-identical
   ~80-line implementations. Extract to a shared base class, mixin, or
   standalone helper? If base class, what's the inheritance hierarchy?

10. **`ci/reviewer.py` git coupling:** `ReviewRunner` is pure engine, but the
    CLI `review` command reads diffs via `subprocess.run(["git", "diff", ...])`.
    Should the SDK `ReviewRunner.review()` accept a diff string parameter, with
    CLI responsible for obtaining the diff?

# Changelog

All notable changes to CRTX will be documented in this file.

## [0.2.1] - 2026-02-20

### Fixed

- **PyPI links** — Homepage and Repository URLs now point to the correct locations (crtx-ai.com and github.com/CRTXAI/CRTX)

## [0.2.0] - 2026-02-20

### CRTX Loop

A new autonomous code generation pipeline: **generate → test → fix → review**.

- **Loop orchestrator** (`crtx loop`) — single-command code generation with automated quality assurance. Routes tasks by complexity, generates code, runs local tests, and iterates fixes until tests pass.
- **Smart routing** — classifies prompts as simple/medium/complex/safety and selects the appropriate model, fix iteration budget, and timeout tier.
- **Test runner** — local quality gate: AST parse → import check → pyflakes → pytest → entry point execution. Per-file pytest fallback on collection failures.
- **Code fixer** — targeted fix prompts from structured test failures. Detects pytest collection failures and phantom API references in test files.
- **Test-generation fallback** — if generation produces zero test files, a second call generates comprehensive pytest tests so the fix cycle always has tests to work with.

### Three-Tier Gap Closing

When the normal fix cycle can't resolve failures, three escalation tiers activate before giving up:

- **Tier 1 — Diagnose then fix** (~$0.08): Two calls to the primary model. First: "Do NOT write code — analyze the root cause." Second: feed the diagnosis back with the code and ask for a targeted fix.
- **Tier 2 — Minimal context retry** (~$0.05): Strip context to ONLY the failing test file and the single source file it imports. Nothing else. Fresh perspective with less noise.
- **Tier 3 — Second opinion** (~$0.08): Escalate to a different model (prefers o3 if primary was Sonnet, Sonnet if primary was o3, Gemini as fallback). Includes the primary model's diagnosis: "they diagnosed this but couldn't fix it — what do you see?"

### Arbiter Review in Loop

- After the test-fix cycle converges, an independent model reviews the code via the existing Arbiter layer (APPROVE/FLAG/REJECT/HALT verdicts).
- On REJECT, triggers one targeted fix cycle and retests.
- Cross-model enforcement — the arbiter always uses a different model than the generator.
- `--no-arbiter` flag to skip the review step.

### Benchmark Improvements

- **Verified scoring dimension** — runs TestRunner on all non-loop conditions after scoring to produce Verified (Yes/No), Tests Run (X/Y), and Dev Time (estimated developer minutes to production) columns.
- **Dev time estimation** — formula: test failures × 3 min + import errors × 2 min + 20 min if entry fails + 5 min per syntax error, capped at 45 min for complex tasks.
- **Tier-aware timeouts** — pytest and entry point timeouts scale by prompt tier (simple=60s, medium=90s, complex=120s).
- **Type hints scoring fix** — excludes test functions and Test* class methods from type hint coverage calculation.
- **Per-file pytest fallback** in test runner — when combined pytest crashes on collection, runs each test file individually.
- **Phantom API detection** in fixer — test files importing names that don't exist in source are flagged for reconciliation.

### Loop Generation Quality

- **Direct imports rule** — generation prompt now instructs "Use direct imports (`from models import User`), NOT relative imports." Prevents import resolution failures.
- **Fixer collection-failure awareness** — on pytest collection failure, all test files are included in the broken set with a reconciliation instruction.

## [0.1.1] - 2026-02-19

### Fixed

- **Arbiter parser** now extracts structured issues and alternatives from reviews (was returning empty lists despite model output containing them)
- **Debate judge prompt** reframed to produce code, not essays — output is 90%+ code with `# file:` headers instead of comparative analysis
- **max_output_tokens** now explicitly set per model in `models.toml` (was silently defaulting to ~4K via LiteLLM, truncating large outputs)
- **Arbiter model selection** uses `verifier_fitness` tiebreaker and falls back through remaining models on empty/malformed responses
- **File extractor** handles `# file:` headers outside fenced code blocks and unfenced raw code sections (debate judge output was producing code that the extractor couldn't find)

## [0.1.0] - 2026-02-18

### Initial Release

CRTX v0.1.0 — multi-model AI pipeline orchestration with adversarial verification.

### Pipeline Modes

- **Sequential** — Architect → Implement → Refactor → Verify with output chaining between stages
- **Parallel** — All models solve independently, cross-review, score, and merge the best approach
- **Debate** — Position papers → rebuttals → final arguments → judgment

### Arbiter Layer

- Independent adversarial review using a different model than the generator (cross-model enforcement)
- Four verdicts: APPROVE, FLAG, REJECT, HALT
- Configurable depth: `--arbiter off|final_only|bookend|full`
- Structured feedback injection on REJECT with retry budget (max 2)
- Confidence floor — APPROVE below 0.50 is downgraded to FLAG
- Implementation Summary Reconciliation via `--reconcile`

### Smart Routing

- Four strategies: `quality_first`, `cost_optimized`, `speed_first`, `hybrid`
- Fitness-based model selection per stage from TOML config
- Cross-stage diversity — max 2 stages per model to prevent monoculture
- Cost estimation via `crtx estimate`

### Auto-Fallback

- Automatic model substitution on provider outages (rate limits, timeouts)
- ProviderHealth tracker with 5-minute cooldown per model
- Health check before attempting primary model on each stage

### Streaming Display

- Real-time token-by-token output with syntax-highlighted code blocks
- Scrolling file headers with pinned status bar (stage progress, cost, tokens)
- Stage state symbols: ○ pending, ◉ active, ● complete, ⚠ fallback, ✗ failed

### Apply Mode

- Write generated code to disk with git safety gates
- File path resolution (exact → basename → fuzzy → create)
- Interactive diff preview with per-file selection
- AST-aware structured patching (7 operations)
- Post-apply test runner with automatic rollback on failure
- Conflict detection via mtime + SHA-256

### Context Injection

- AST-aware Python project scanner (classes, functions, imports, docstrings)
- Dynamic context builder with relevance scoring
- Configurable token budget (default: 20,000)
- Respects `.gitignore`

### CLI

- `crtx run` — main pipeline command (task is a positional argument)
- `crtx demo` — guided 60-second first-run experience with cross-model generation and review
- `crtx review-code` — multi-model code review on files or git diffs
- `crtx improve` — review → improve pipeline with cross-model consensus
- `crtx repl` — interactive shell with session history
- `crtx setup` — API key configuration wizard
- `crtx models` — list registered models with fitness scores
- `crtx sessions` — session history with short ID support
- `crtx replay` / `crtx show` — re-run or view past sessions
- `crtx estimate` — cost estimation before a run
- `crtx dashboard` — real-time React dashboard with WebSocket
- Six presets: `balanced`, `fast`, `thorough`, `cheap`, `explore`, `debate`

### Post-Run Viewer

- Interactive menu: `[s]` Summary, `[c]` Code, `[r]` Reviews, `[d]` Diffs, `[a]` Apply
- Syntax-highlighted code viewer with file selection
- Arbiter review display with verdict badges and issue categorization
- Diff rendering with color-coded additions/deletions

### Dashboard

- React UI served via FastAPI on `localhost:8420`
- WebSocket real-time event streaming
- Cross-process HTTP POST relay (CLI → dashboard bridge)
- Pipeline view with stage cards and activity feed
- Session history browser

### Infrastructure

- 1,045 tests passing
- SQLite session persistence with full audit trail
- CI/CD with GitHub Actions (Python 3.12 + 3.13, ruff lint, keyword scan)
- Pydantic v2 schemas throughout
- LiteLLM adapter supporting 100+ providers
- 11 models across 4 providers (Anthropic, OpenAI, Google, xAI)
- Apache 2.0 license

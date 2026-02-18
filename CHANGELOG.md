# Changelog

All notable changes to CRTX will be documented in this file.

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

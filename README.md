<p align="center">
  <img src="assets/banner.svg" alt="CRTX" width="800">
</p>

<p align="center">
  <strong>Generate. Test. Fix. Review. One command, verified output.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#the-problem">The Problem</a> •
  <a href="#the-loop">The Loop</a> •
  <a href="#benchmarks">Benchmarks</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#commands">Commands</a> •
  <a href="#supported-models">Supported Models</a>
</p>

<p align="center">
  <img src="https://img.shields.io/pypi/pyversions/crtx" alt="python 3.12+">
  <img src="https://img.shields.io/github/license/CRTXAI/CRTX" alt="license Apache 2.0">
  <img src="https://img.shields.io/pypi/v/crtx" alt="PyPI version">
</p>

---

## What is CRTX?

CRTX is an AI development intelligence tool that generates, tests, fixes, and reviews code automatically. One command in, verified output out.

It works with any model — Claude, GPT, Gemini, Grok, DeepSeek — and picks the right one for each task. You don't configure pipelines or choose models. You describe what you want and CRTX handles the rest.

```bash
crtx loop "Build a REST API with FastAPI, SQLite, search and pagination"
```

## The Problem

Single AI models generate code that *looks* correct but often has failing tests, broken imports, and missed edge cases. Developers spend 10–30 minutes per generation debugging and fixing AI output before it actually works.

Multi-model pipelines cost 10–15x more without meaningfully improving quality. Four models reviewing each other's prose doesn't catch a broken import statement.

The issue isn't the model. It's the lack of verification. Nobody runs the code before handing it to you.

## The Loop

CRTX solves this with the Loop: **Generate → Test → Fix → Review**.

1. **Generate** — The best model for the task writes the code
2. **Test** — CRTX runs the code locally: AST parse, import check, pyflakes, pytest, entry point execution
3. **Fix** — Failures feed back to the model with structured error context for targeted fixes
4. **Review** — An independent Arbiter (always a *different* model) reviews the final output

Every output is tested before you see it. If tests fail, CRTX fixes them. If the fix cycle stalls, three escalation tiers activate before giving up. If the Arbiter rejects the code, one more fix cycle runs.

The result: code that passes its own tests, has been reviewed by a second model, and comes with a verification report.

## Benchmarks

Same 12 prompts, same scoring rubric. CRTX Loop vs. single models vs. multi-model debate:

| Condition | Avg Score | Min | Spread | Avg Dev Time | Cost |
|-----------|-----------|-----|--------|--------------|------|
| Single Sonnet | 94% | 92% | 4 pts | 10 min | $0.36 |
| Single o3 | 81% | 54% | 41 pts | 4 min | $0.44 |
| Multi-model Debate | 88% | 75% | 25 pts | 9 min | $5.59 |
| **CRTX Loop** | **99%** | **98%** | **2 pts** | **2 min** | **$1.80** |

**Dev Time** = estimated developer minutes to get the output to production (based on test failures, import errors, and entry point issues). **Spread** = max score minus min score across all prompts.

The Loop scores higher, more consistently, with less post-generation work than any other condition — at a fraction of the cost of multi-model pipelines.

Run the benchmark yourself:

```bash
crtx benchmark --quick
```

## How It Works

```
  ┌─────────┐    ┌──────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐    ┌─────────┐
  │  Route  │ ─→ │ Generate │ ─→ │  Test   │ ─→ │   Fix   │ ─→ │ Review  │ ─→ │ Present │
  └─────────┘    └──────────┘    └─────────┘    └─────────┘    └─────────┘    └─────────┘
       │                              │              │
       │                              └──────────────┘
       │                               ↑ loop until pass
       │
       ├── simple  → fast model, 2 fix iterations
       ├── medium  → balanced model, 3 fix iterations
       └── complex → best model, 5 fix iterations + architecture debate
```

**Route** — Classifies your prompt by complexity (simple/medium/complex) and selects the model, fix budget, and timeout tier.

**Generate** — Produces source files and test files. If no tests are generated, a second call creates comprehensive pytest tests so the fix cycle always has something to verify against.

**Test** — Five-stage local quality gate: AST parse → import check → pyflakes → pytest → entry point execution. Per-file pytest fallback on collection failures.

**Fix** — Feeds structured test failures back to the model for targeted fixes. Detects phantom API references (tests importing functions that don't exist in source) and pytest collection failures.

**Three-tier gap closing** — When the normal fix cycle can't resolve failures:
- **Tier 1** — Diagnose then fix: "analyze the root cause without writing code," then feed the diagnosis back for a targeted fix
- **Tier 2** — Minimal context retry: strip context to only the failing test and its source file, fresh perspective
- **Tier 3** — Second opinion: escalate to a different model with the primary model's diagnosis

**Review** — An independent Arbiter (always a different model than the generator) reviews for logic errors, security issues, and design problems. On REJECT, triggers one more fix cycle and retests.

**Present** — Final results with verification report, file list, and cost breakdown.

## Key Features

**Smart routing** — Classifies prompts by complexity and picks the right model, fix budget, and timeout for each task. Simple tasks get fast models. Complex tasks get the best model plus an architecture debate.

**Three-tier gap closing** — When fixes stall, CRTX escalates: root cause diagnosis, minimal context retry, then a second opinion from a different model. Most stuck cases resolve at tier 1 or 2.

**Independent Arbiter review** — Every run gets reviewed by a model that didn't write the code. Cross-model review catches errors that self-review misses. Skip with `--no-arbiter`.

**Verified scoring** — Every output is tested locally before you see it. The verification report shows exactly which checks passed, how many tests ran, and estimated developer time to production.

**Auto-fallback** — If a provider goes down mid-run (rate limit, timeout, outage), CRTX substitutes the next best model and keeps going. A 5-minute cooldown prevents hammering a struggling provider.

**Apply mode** — Write generated code directly to your project with `--apply`. Interactive diff preview, git branch protection, conflict detection, AST-aware patching, and automatic rollback if post-apply tests fail.

**Context injection** — Scan your project and inject relevant code into the generation prompt with `--context .`. AST-aware Python analysis extracts class signatures, function definitions, and import graphs within a configurable token budget.

## Quick Start

```bash
pip install crtx
crtx setup        # configure your API keys
```

Then run:

```bash
crtx loop "Build a CLI password generator with strength validation and clipboard support"
```

## Commands

| Command | What it does |
|---------|-------------|
| `crtx loop "task"` | Generate, test, fix, and review code (default) |
| `crtx run "task"` | Run a multi-model pipeline (sequential/parallel/debate) |
| `crtx benchmark` | Run the built-in benchmark suite |
| `crtx repl` | Interactive shell with session history |
| `crtx review-code` | Multi-model code review on files or git diffs |
| `crtx improve` | Review → improve pipeline with cross-model consensus |
| `crtx setup` | API key configuration |
| `crtx models` | List available models with fitness scores |
| `crtx estimate "task"` | Cost estimate before running |
| `crtx sessions` | Browse past runs |
| `crtx replay <id>` | Re-display a previous session |
| `crtx dashboard` | Real-time web dashboard |

## Supported Models

CRTX works with any model supported by LiteLLM — that's 100+ providers. Out of the box, it's configured for:

| Provider | Models |
|----------|--------|
| Anthropic | Claude Opus 4, Sonnet 4 |
| OpenAI | GPT-4o, o3 |
| Google | Gemini 2.5 Pro, Flash |
| xAI | Grok |
| DeepSeek | DeepSeek R1 |

Add any LiteLLM-compatible model in `~/.crtx/config.toml`.

### API Key Setup

Run `crtx setup` to configure your keys interactively, or set them as environment variables:

```bash
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
export GEMINI_API_KEY=...
export XAI_API_KEY=xai-...
export DEEPSEEK_API_KEY=sk-...
```

CRTX only needs one provider to work. More providers means more model diversity for routing and Arbiter review.

## Contributing

Contributions are welcome. Fork the repo, create a branch, and submit a PR.

The test suite has 1,096 tests — run them with `pytest`. Linting is `ruff check .`.

## License

Apache 2.0. See [LICENSE](LICENSE) for details.

---

<p align="center">
  Built by <a href="https://www.crtx-ai.com">TriadAI</a>
</p>

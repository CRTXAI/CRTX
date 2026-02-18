<p align="center">
  <img src="assets/banner.svg" alt="CRTX" width="800"/>
</p>

<!-- TODO: Replace with terminal recording (asciinema/VHS) showing:
     crtx setup → crtx (REPL + logo) → task → config screen → live display → summary -->
<!-- <p align="center"><img src="assets/demo.gif" alt="CRTX CLI demo" width="700"/></p> -->

<p align="center">
  <strong>Multi-model AI that learns your codebase.</strong>
</p>

<p align="center">
  <a href="#quick-start">Quick Start</a> •
  <a href="#getting-started">Getting Started</a> •
  <a href="#how-it-works">How It Works</a> •
  <a href="#the-arbiter">The Arbiter</a> •
  <a href="#supported-models">Supported Models</a> •
  <a href="docs/architecture.md">Architecture</a> •
  <a href="CONTRIBUTING.md">Contributing</a>
</p>

<p align="center">
  <img src="https://img.shields.io/badge/python-3.12+-00ff88?style=flat-square&logo=python&logoColor=00ff88" alt="Python 3.12+"/>
  <img src="https://img.shields.io/badge/license-Apache%202.0-00dd77?style=flat-square" alt="License"/>
  <img src="https://img.shields.io/badge/models-any%20LLM-66ff88?style=flat-square" alt="Any LLM"/>
</p>

---

## The Problem

You paste code into an AI model. It looks right. You ship it. Then you find the hallucinated import, the missed edge case, the pattern violation that cascades through your codebase.

Single-model code generation has a blindspot problem. Every model has biases, gaps, and failure modes — and the same model that wrote the bug can't reliably find it.

## The Fix

CRTX routes your coding task through **multiple AI models in specialized roles**, with an **independent referee** that catches mistakes before they reach your codebase.
```
Task -> [Architect] -> [Implementer] -> [Refactor] -> [Verify] -> Production Code
             |              |              |             |
         Arbiter         Arbiter        Arbiter       Arbiter
      (different model reviews each stage)
```

Each model does what it's best at. A different model checks the work. The code that survives is production-ready.

---

## Quick Start

```bash
pip install crtx
crtx setup          # Interactive API key configuration
crtx                # Launch interactive session
```

That's it. `crtx setup` walks you through API key configuration with live validation. `crtx` launches an interactive session with a branded terminal UI, real-time pipeline status, and a persistent REPL.

---

## Getting Started

### First-Time Setup

```bash
crtx setup
```

Interactive wizard that prompts for API keys (Anthropic, OpenAI, Google, xAI), validates each key against its provider, and saves them to `~/.crtx/keys.env`. You need at least one provider configured. For parallel and debate modes, you'll need at least two.

```bash
crtx setup --check    # Validate existing keys without re-prompting
crtx setup --reset    # Clear saved keys and reconfigure
```

### Interactive Session (REPL)

```bash
crtx
```

Launches the interactive REPL. The REPL maintains session state — set your mode, routing strategy, and arbiter depth once, then run multiple tasks without repeating flags.

```
crtx ▸ mode parallel
  Mode set to parallel

crtx ▸ route quality_first
  Route set to quality_first

crtx ▸ Build a REST API with JWT authentication and rate limiting
  # → Interactive config screen → real-time pipeline display → completion summary

crtx ▸ status
  Mode:    parallel
  Route:   quality_first
  Arbiter: bookend
```

Type `help` for all commands, `exit` or Ctrl+C to quit.

### Direct Execution

```bash
# Run with interactive config screen (choose mode/route/arbiter before launch)
crtx run "Build a REST API with JWT authentication and rate limiting"

# Run with explicit flags (skips config screen)
crtx run "Build a REST API" --mode sequential --route hybrid --arbiter bookend
```

When you run without explicit flags, CRTX shows an interactive config screen where you can cycle through modes, routing strategies, and arbiter settings with single keypresses before confirming. With explicit flags, the pipeline starts immediately.

---

## How It Works

CRTX uses a **sequential pipeline** with four stages. Each stage is handled by whichever model scores highest for that role:

| Stage | Role | What It Does |
|-------|------|-------------|
| **Architect** | Design the solution | Produces a technical scaffold: file structure, interfaces, data models, dependency map. |
| **Implement** | Write the code | Takes the scaffold and produces complete, working implementation with error handling. |
| **Refactor** | Improve and test | Restructures for clarity, adds edge case handling, writes comprehensive test suite. |
| **Verify** | Validate everything | Reviews the complete output for correctness, security, and pattern compliance. |

Models don't just hand off and move on — any model can **suggest improvements** outside its assigned role. The Architect can flag an implementation concern. The Implementer can propose a structural change. Suggestions are tracked, evaluated, and either accepted or escalated to consensus.

---

## The Arbiter

The Arbiter is what makes CRTX fundamentally different from running the same prompt through multiple models.

**It's an independent referee.** The Arbiter never writes code. It never proposes architecture. Its only job is to find what's wrong with other models' work.

**It's always a different model.** If Claude wrote the code, GPT-4 or Grok arbitrates. If Gemini designed the architecture, Claude checks it. The system enforces this automatically — the same model never grades its own work.

**It assumes there are bugs.** The Arbiter's prompt starts from skepticism: *"Assume there are errors until proven otherwise."* This inverts the typical AI review pattern where models default to "looks good" and hedge with minor suggestions.

**It can stop the pipeline.** Four verdicts:

| Verdict | Action |
|---------|--------|
| **APPROVE** | Continue. Output is sound. |
| **FLAG** | Continue, but inject warnings for the next stage to address. |
| **REJECT** | Re-run this stage with structured feedback. Max 2 retries. |
| **HALT** | Stop everything. Present analysis for human decision. |

When the Arbiter rejects, it doesn't just say "this is wrong." It provides structured feedback with severity, category, exact location, evidence, and a suggested fix — all injected into the retry prompt so the generating model knows exactly what to address.

### Configurable Review Depth

Not every task needs full review. Choose your safety level:
```bash
crtx run "..." --arbiter full       # Review every stage (critical features)
crtx run "..." --arbiter bookend    # Review architecture + final output (default)
crtx run "..." --arbiter final      # Review final output only (prototypes)
crtx run "..." --arbiter off        # No review (rapid iteration)
```

Or in the REPL: `arbiter full` sets the depth for all subsequent tasks in the session.

---

## Supported Models

CRTX is **model-agnostic**. Any LLM that supports chat completions works. Add a new model by adding a TOML entry — no code changes required.

### Pre-Configured Providers

| Provider | Models | Best At |
|----------|--------|---------|
| **Anthropic** | Claude Opus, Sonnet, Haiku | Refactoring, verification, nuanced review |
| **OpenAI** | GPT-4o, o3-mini | Fast implementation, broad language support |
| **Google** | Gemini 2.5 Pro, Flash | Architecture, large context reasoning |
| **xAI** | Grok 4, Grok 3 | Independent analysis, alternative perspectives |

### Adding Models
```toml
# config/models.toml
[models.deepseek-v3]
provider = "deepseek"
model = "deepseek-chat"
roles = ["implement", "refactor"]
cost_per_1k_input = 0.0001
cost_per_1k_output = 0.0002
```

DeepSeek, Llama, Mistral, Ollama (local), vLLM (self-hosted) — if LiteLLM supports it, CRTX supports it.

---

## Presets

Instead of specifying `--mode`, `--route`, and `--arbiter` on every command, use a preset:

| Preset | Mode | Route | Arbiter | Use Case |
|--------|------|-------|---------|----------|
| **balanced** (default) | sequential | hybrid | bookend | Standard development. Best cost/quality balance. |
| **fast** | sequential | speed-first | off | Rapid iteration. Cheapest models, no review. |
| **cheap** | sequential | cost-optimized | off | Budget-conscious. Lowest cost above fitness threshold. |
| **thorough** | sequential | quality-first | full | Maximum quality. Best models, every stage reviewed. |
| **explore** | parallel | hybrid | bookend | Fan out to 3+ models, cross-review, synthesize the best. |
| **debate** | debate | quality-first | full | Structured debate. Best for architecture decisions and tradeoffs. |

```bash
crtx run "Build a REST API" --preset explore
crtx run "Build a REST API" --preset fast

# Override any part of a preset
crtx run "Build a REST API" --preset explore --arbiter full
```

In the REPL:
```
crtx [balanced] ▸ preset explore
  Mode set to parallel, route hybrid, arbiter bookend

crtx [explore] ▸ preset fast
  Mode set to sequential, route speed-first, arbiter off
```

No preset flag defaults to `balanced`. If you manually change mode/route/arbiter after selecting a preset, the prompt shows the current settings instead of a preset name.

---

## Presets

Most users never need to touch `--mode`, `--route`, or `--arbiter` directly. Presets bundle them:

| Preset | Mode | Routing | Arbiter | Use Case |
|--------|------|---------|---------|----------|
| **balanced** (default) | sequential | hybrid | bookend | Standard development. Best cost/quality balance. |
| **fast** | sequential | speed-first | off | Rapid iteration. Cheapest models, no review. |
| **cheap** | sequential | cost-optimized | off | Budget-conscious. Cheapest models above fitness threshold. |
| **thorough** | sequential | quality-first | full | Critical features. Best models, every stage reviewed. |
| **explore** | parallel | hybrid | bookend | Fan out to 3+ models, cross-review, synthesize the best. |
| **debate** | debate | quality-first | full | Structured debate between models. Architecture decisions. |

```bash
crtx run "Build a REST API" --preset explore
crtx run "Build a REST API" --preset fast
crtx run "Build a REST API"                    # balanced (default)
```

Presets are starting points — override any part:
```bash
crtx run "Build a REST API" --preset explore --arbiter full
```

In the REPL:
```
crtx [balanced] ▸ preset explore
  Mode set to parallel, route hybrid, arbiter bookend

crtx [explore] ▸ preset fast
  Mode set to sequential, route speed-first, arbiter off
```

---

| Mode | How It Works | Best For |
|------|-------------|----------|
| **Sequential** (default) | Architect → Implement → Refactor → Verify, each building on the last | Standard development, most tasks |
| **Parallel** | All models solve independently, cross-review, score, merge best approach | Complex problems with multiple valid solutions |
| **Debate** | Position papers → rebuttals → final arguments → judgment | Architectural decisions, tradeoff analysis |

```bash
crtx run "..." --mode sequential   # Default
crtx run "..." --mode parallel     # Fan-out + consensus
crtx run "..." --mode debate       # Structured debate
```

Or in the REPL: `mode parallel` sets the mode for all subsequent tasks in the session.

---

## Smart Routing

CRTX assigns models to pipeline roles based on fitness benchmarks — each model is scored on how well it performs as Architect, Implementer, Refactorer, and Verifier. Four routing strategies let you optimize for what matters:

| Strategy | Behavior |
|----------|----------|
| **quality-first** | Best model per role regardless of cost |
| **cost-optimized** | Cheapest model above fitness threshold |
| **speed-first** | Lowest-latency models preferred |
| **hybrid** (default) | Quality for critical stages, cost-optimized for early stages |

```bash
crtx run "..." --route hybrid          # Default
crtx run "..." --route quality-first   # Max quality
crtx estimate "..." --compare-routes   # Compare costs
```

Or in the REPL: `route quality_first` sets the strategy for all subsequent tasks.

---

## Configuration

### API Keys

The recommended way to configure API keys is `crtx setup`, which validates keys and saves them for future sessions. Keys are loaded in this order (highest priority first):

1. **Environment variables** — `ANTHROPIC_API_KEY`, `OPENAI_API_KEY`, `GOOGLE_API_KEY`, `XAI_API_KEY`
2. **`~/.crtx/keys.env`** — User-level keys saved by `crtx setup`
3. **`.env` in current directory** — Project-level overrides

You only need keys for the providers you want to use. At least one provider must be configured.

```bash
# Recommended: interactive setup with validation
crtx setup

# Or set environment variables directly
export ANTHROPIC_API_KEY=sk-ant-...
export OPENAI_API_KEY=sk-...
```

### Pipeline Defaults

Pipeline defaults (mode, routing strategy, arbiter depth, timeout) are configured in `config/defaults.toml`. These can be overridden per-run via CLI flags or the interactive config screen.

---

## CLI Commands

| Command | Description |
|---------|-------------|
| `crtx` | Launch interactive session (REPL mode) |
| `crtx setup` | Configure API keys interactively |
| `crtx setup --check` | Validate existing API keys |
| `crtx setup --reset` | Clear keys and reconfigure |
| `crtx run` | Run a full pipeline on a task |
| `crtx plan` | Expand a rough idea into a structured task spec |
| `crtx estimate` | Estimate cost before running |
| `crtx review` | Multi-model PR review (CI/CD integration) |
| `crtx review-code` | Multi-model review of existing code files |
| `crtx improve` | Multi-model improvement of existing code |
| `crtx models list` | Show registered models with fitness scores |
| `crtx models check` | Verify API key connectivity |
| `crtx config show` | Display current pipeline configuration |
| `crtx sessions list` | Browse past pipeline runs |
| `crtx sessions show` | View full session details |
| `crtx dashboard` | Launch real-time browser visualization |

```bash
# Interactive session — persistent state, branded UI
crtx

# Run a task with interactive config screen
crtx run "Add WebSocket support to the existing Express server"

# Run with explicit flags (skips config screen)
crtx run "Add WebSocket support" --mode sequential --route hybrid --arbiter bookend

# Plan first, then run
crtx plan "Build a data processing pipeline" --run

# Review a PR diff
crtx review --diff changes.patch --fail-on critical

# Review existing code with multiple models
crtx review-code src/middleware.py --preset thorough

# Improve existing code
crtx improve src/rate_limiter/ --focus "error handling, type safety"

# Launch the real-time dashboard
crtx dashboard --port 8420
```

The CLI uses [Rich](https://github.com/Textualize/rich) for a premium terminal experience — branded ASCII art, interactive config screens, real-time pipeline status with stage-by-stage progress, color-coded Arbiter verdicts, and a post-completion summary with export actions.

---

## Review & Improve Existing Code

CRTX doesn't just generate code — it can review and improve code you've already written.

### Multi-Model Review

Have 3+ models independently review your code, cross-check each other's findings, and produce a ranked report:

```bash
crtx review-code src/middleware.py
crtx review-code src/rate_limiter/ --preset thorough
```

Each model finds bugs, security issues, and design problems independently. Then they review each other's findings — agreeing, disagreeing, and catching what others missed. Issues found by multiple models rank highest. Single-source findings are flagged as lower confidence.

### Multi-Model Improve

Have 3+ models each produce an improved version of your code, vote on the best, and synthesize:

```bash
crtx improve src/middleware.py
crtx improve src/rate_limiter/ --focus "error handling, type safety"
```

Like parallel mode, but starting from your existing code instead of a task description. The Arbiter reviews the final improvement against your original. You see a diff before anything is written.

---

CRTX supports domain-specific verification rules that the Arbiter checks in addition to general code quality:
```toml
# config/domain/my_rules.toml
[rules.schema_consistency]
description = "All database models must use integer primary keys"
severity = "critical"
pattern = "UUIDField|uuid4"
action = "reject"

[rules.test_coverage]
description = "Every new service must have corresponding test file"
severity = "warning"
```

We use CRTX to build a financial services platform — our custom rules enforce schema patterns, threading conventions, and audit trail requirements specific to our domain. You can do the same for yours.

---

## How We Use It

We built CRTX because we needed it. Our team uses CRTX as the primary development workflow for a financial services operating system with 2,900+ tests. Every new feature, every module, every refactor goes through the pipeline. The Arbiter has caught schema mismatches, hallucinated dependencies, over-engineered abstractions, and integration failures — all before code review.

CRTX isn't a research project. It's a production tool that we bet our own codebase on every day.

---

## Cost

CRTX adds model calls, which cost tokens. Here's what a typical task looks like:

| Configuration | Est. Cost per Task | Use Case |
|--------------|-------------------|----------|
| No Arbiter | ~$4.30 | Rapid iteration |
| Final Only | ~$5.10 | Prototyping |
| **Bookend (default)** | **~$5.80** | Standard development |
| Full Arbiter | ~$7.30 | Critical features |

At the default Bookend depth and ~15 tasks/week, the Arbiter adds about **$90/month**. One production bug it catches pays for a year of reviews.

---

## Documentation

| Document | Description |
|----------|-------------|
| [Architecture](docs/architecture.md) | Core pipeline design, consensus protocol, technology stack |
| [Model-Agnostic System](docs/model-agnostic.md) | Plugin architecture, LiteLLM adapter, dynamic role assignment |
| [Arbiter Layer](docs/arbiter.md) | Independent review system, verdicts, feedback injection |
| [Build Spec](docs/build-spec.md) | MVP scope, day-by-day build plan, technical decisions |

---

## Architecture

```
triad/
├── cli.py                  # Typer + Rich terminal interface
├── cli_display.py          # Branded UI: logos, config screen, live display, summary
├── repl.py                 # Interactive REPL with session state
├── orchestrator.py         # Pipeline engine (sequential, parallel, debate)
├── planner.py              # Task planner (crtx plan)
├── providers/              # LiteLLM adapter + model registry
├── routing/                # Fitness-based model-to-role assignment
├── arbiter/                # Independent adversarial review engine
├── consensus/              # Cross-domain suggestions + voting protocol
├── context/                # AST-aware codebase scanner + context builder
├── persistence/            # SQLite session storage + export
├── ci/                     # Multi-model PR review for CI/CD
├── dashboard/              # Real-time WebSocket visualization (optional)
├── schemas/                # Pydantic v2 models (all data contracts)
├── prompts/                # Jinja2 role prompt templates
├── output/                 # File writer + Markdown report renderer
└── config/                 # TOML configuration (models, defaults, routing)
```

---

## Contributing

We welcome contributions! Please read [CONTRIBUTING.md](CONTRIBUTING.md) before submitting a PR.

**Important:** All contributors must sign our [Contributor License Agreement](CLA.md) before their first PR can be merged. This is handled automatically via CLA Assistant — you'll be prompted when you open your first PR.

---

## License

Apache 2.0 — see [LICENSE](LICENSE) for details.

---

<p align="center">
  <sub>Built by <a href="https://github.com/triad-ai">TriadAI</a> — Every session smarter than the last.</sub>
</p>

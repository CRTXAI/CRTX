# Codebase Audit

**Multi-Model Architectural Review of Existing Code**

*Three models read your codebase. Each finds what the others miss. The Arbiter confirms what's real.*

**Status:** v0.2.0 · Depends on: Context Injection (Day 11), Arbiter Layer (Day 5)

---

## Problem

Developers accumulate technical debt, architectural drift, and latent bugs over time — not because they're careless, but because no single reviewer can hold the full context of a growing codebase while simultaneously thinking about security, performance, patterns, edge cases, and maintainability. Code review catches issues in new code. Nothing systematically reviews what's already there.

Static analysis tools (pylint, mypy, SonarQube) catch syntactic and type-level issues. But they can't reason about architectural decisions, domain logic correctness, abstraction boundaries, or whether the patterns in module A conflict with the patterns in module B. That requires understanding intent, and that's what LLMs are good at.

The problem with asking a single LLM to review your codebase: it has the same blindspot issue as a single developer. Claude might catch every edge case but miss that the architecture is over-engineered. GPT-4 might flag the over-engineering but miss a subtle race condition. Gemini might see how the modules interact at scale but miss the security issue in the auth flow.

Multi-model audit fixes this. Each model reviews independently, findings are cross-validated through the Arbiter, and the output is a tiered, confirmed report — not a list of maybes.

## How It Works

### Phase 1: Scan

The context injection engine (Day 11) scans the target directory and builds a map of the codebase. For Python projects: modules, classes, functions, imports, dependencies, test coverage mapping. For JS/TS: components, hooks, routes, API calls, package dependencies. Language-agnostic fallback: file tree, file sizes, import/require patterns.

The scan produces a **Codebase Profile** — a structured summary that fits within model context limits:

```
Project: triad-orchestrator
Language: Python 3.12
Framework: asyncio + Pydantic + Typer
Files: 47 Python files, 12 test files
Lines: ~4,200 LOC (excluding tests)
Entry points: cli.py, orchestrator.py
Key patterns: ABC providers, Pydantic schemas, Jinja2 templates
Dependencies: litellm, pydantic, typer, rich, jinja2, aiosqlite
Test coverage: 238 tests across 6 test files
```

For larger codebases, the scanner uses AST analysis to extract the structural skeleton (class/function signatures, imports, docstrings) without sending full file contents. Full file contents are sent only for files flagged during the review phase.

### Phase 2: Independent Review

Each model receives the codebase profile and relevant code, then reviews independently using a structured audit prompt. Each model reviews through its cognitive strengths:

**Model A (broad context — e.g., Gemini):**
- Architectural coherence: Do the modules fit together cleanly?
- Dependency health: Circular imports, unnecessary coupling, missing abstractions?
- Scalability concerns: What breaks at 10x scale?
- Redundancy: Duplicated logic across modules?

**Model B (practical implementation — e.g., GPT-4):**
- Implementation quality: Dead code, unreachable branches, incomplete error handling?
- API surface: Inconsistent interfaces, missing validation, undocumented contracts?
- Performance: N+1 queries, unnecessary allocations, blocking calls in async code?
- Dependency risks: Outdated packages, known vulnerabilities, unnecessary dependencies?

**Model C (careful analysis — e.g., Claude):**
- Edge cases: What inputs break this? What states are unhandled?
- Security: Injection vectors, auth bypasses, data exposure, unsafe defaults?
- Correctness: Does the logic actually match the intent? Subtle bugs in conditionals?
- Test gaps: What critical paths have no test coverage?

Each model produces a structured findings list with severity, category, location, description, evidence, and suggested fix.

### Phase 3: Cross-Validation

This is where it gets interesting. Raw findings from three models contain duplicates, false positives, and varying severity assessments. The cross-validation phase resolves this:

**Agreement mapping:**
- Finding reported by 2+ models → HIGH CONFIDENCE (likely real)
- Finding reported by 1 model → NEEDS VERIFICATION
- Contradictory assessments → DISPUTED (one model says it's fine, another says it's broken)

**Arbiter verification:**
Each NEEDS VERIFICATION finding is sent to the Arbiter (a different model than the one that reported it) with the specific code in question. The Arbiter either:
- CONFIRMS: The finding is real. Include in report with evidence.
- DISMISSES: False positive. Exclude from report with explanation.
- ESCALATES: Can't determine without more context. Flag for human review.

**Disputed findings:**
When models disagree, the Arbiter reviews both positions and makes a ruling, similar to debate mode judgment. The ruling and reasoning are included in the report.

### Phase 4: Tiered Report

The final output is a structured report with findings organized by severity and confidence:

```
# Codebase Audit Report
## Project: triad-orchestrator
## Audited by: Gemini 2.5 Pro, GPT-4o, Claude Opus
## Date: 2026-02-20

---

### TIER 1 — CRITICAL (Act Now)
Multi-model agreement + Arbiter confirmed. These are real issues
that should be fixed before the next release.

  1. [SECURITY] SQL injection vector in session query builder
     Location: triad/persistence/database.py:87-92
     Found by: Claude, GPT-4 (confirmed by Gemini arbiter)
     Evidence: User-provided session_id interpolated directly
               into query string without parameterization.
     Fix: Use parameterized query via aiosqlite execute().
     Effort: Low (single function change)

  2. [ARCHITECTURE] Circular dependency between orchestrator and arbiter
     Location: triad/orchestrator.py ↔ triad/arbiter/arbiter.py
     Found by: Gemini, Claude (confirmed by GPT-4 arbiter)
     Evidence: Orchestrator imports ArbiterEngine, ArbiterEngine
               imports schemas used by orchestrator. Currently works
               due to lazy imports but will break if restructured.
     Fix: Extract shared schemas to a separate module or use
          dependency injection.
     Effort: Medium (refactoring required)

---

### TIER 2 — IMPORTANT (Plan to Fix)
High-confidence findings that aren't urgent but will cause
problems if left unaddressed.

  3. [PERFORMANCE] Synchronous file I/O in async pipeline
     Location: triad/output/writer.py:34-56
     Found by: GPT-4 (confirmed by Claude arbiter)
     Evidence: Uses open() instead of aiofiles.open() inside
               async function, blocking the event loop during
               file writes.
     Fix: Replace with aiofiles for async I/O (already a dependency).
     Effort: Low

  4. [TEST GAP] No tests for arbiter retry exhaustion edge case
     Location: tests/test_orchestrator_arbiter.py
     Found by: Claude (confirmed by Gemini arbiter)
     Evidence: Tests cover retry success and halt, but not the
               case where retries exhaust and pipeline continues
               with degraded output.
     Fix: Add test for max_retries exhaustion path.
     Effort: Low

---

### TIER 3 — SUGGESTIONS (Consider)
Single-model findings that the Arbiter confirmed as valid but
non-urgent improvements.

  5. [MAINTAINABILITY] Magic numbers in fitness calculation
     Location: triad/arbiter/arbiter.py:95
     Found by: Claude (confirmed by GPT-4 arbiter)
     Evidence: Average fitness divides by 4 (hardcoded). If roles
               expand, this silently produces wrong averages.
     Fix: Use len(fitness.model_fields) or a method on RoleFitness.
     Effort: Low

---

### TIER 4 — DISMISSED
Findings that were reported but dismissed by the Arbiter as false
positives or non-issues. Included for transparency.

  6. [PERFORMANCE] "LiteLLM adapter creates new client per call"
     Reported by: GPT-4
     Arbiter ruling: DISMISSED (Gemini)
     Reason: LiteLLM internally pools connections. Creating a new
             provider instance is lightweight and doesn't create
             new HTTP connections. No performance impact.

---

### AUDIT SUMMARY
| Metric | Value |
|--------|-------|
| Files scanned | 47 |
| Models used | 3 (Gemini, GPT-4, Claude) |
| Total findings reported | 14 |
| Arbiter confirmed | 8 |
| Arbiter dismissed | 4 |
| Arbiter escalated (needs human) | 2 |
| Tier 1 (Critical) | 2 |
| Tier 2 (Important) | 3 |
| Tier 3 (Suggestions) | 3 |
| Audit cost | $8.40 |
| Audit duration | 4m 22s |
```

### Phase 5: Fix Verification (Optional)

After the developer fixes issues from the report, they can run a targeted re-audit:

```
triad audit --verify-fixes --previous-report audit-2026-02-20.json
```

This re-scans only the files that contained findings, sends the updated code to the Arbiter, and produces a verification report:

```
# Fix Verification Report
## Based on: audit-2026-02-20.json

| # | Finding | Status | Verified By |
|---|---------|--------|-------------|
| 1 | SQL injection in session query | FIXED ✓ | Claude |
| 2 | Circular dependency | FIXED ✓ | GPT-4 |
| 3 | Sync file I/O in async pipeline | FIXED ✓ | Claude |
| 4 | Missing retry exhaustion test | FIXED ✓ | Gemini |
| 5 | Magic numbers in fitness calc | NOT FIXED | GPT-4 |

4 of 5 targeted fixes verified. 1 remaining.
```

## CLI Interface

```bash
# Full audit of current directory
triad audit

# Audit specific directory
triad audit --path ./backend/app

# Audit with specific focus areas
triad audit --focus security,performance,architecture

# Quick audit (single model + arbiter confirmation, cheaper)
triad audit --quick

# Audit with all models + full cross-validation (most thorough)
triad audit --thorough

# Output formats
triad audit --format markdown    # default, human-readable
triad audit --format json        # machine-readable, for CI integration
triad audit --format sarif       # SARIF format for GitHub code scanning

# Verify fixes from a previous audit
triad audit --verify-fixes --previous-report audit-2026-02-20.json

# Audit only files changed since last commit
triad audit --changed-only

# Audit only files in a PR diff
triad audit --diff origin/main..HEAD

# Cost estimate before running
triad audit --estimate
```

## Architecture

```
triad/
├── audit/
│   ├── __init__.py
│   ├── scanner.py          # Codebase scanning + profile generation
│   ├── reviewer.py         # Per-model independent review orchestration
│   ├── cross_validator.py  # Agreement mapping + arbiter verification
│   ├── reporter.py         # Tiered report generation (md, json, sarif)
│   └── verifier.py         # Fix verification against previous report
├── prompts/
│   ├── audit_review.md     # "Review this codebase for issues in [focus]"
│   ├── audit_verify.md     # "Is this finding real? Review the evidence."
│   ├── audit_dispute.md    # "Two models disagree. Review both positions."
│   └── audit_fix_check.md  # "Was this specific issue fixed in the updated code?"
```

## Schemas

```python
class AuditConfig(BaseModel):
    """Configuration for a codebase audit run."""
    path: str = "."
    focus_areas: list[str] = []           # security, performance, architecture, etc.
    depth: str = "standard"                # quick | standard | thorough
    output_format: str = "markdown"        # markdown | json | sarif
    changed_only: bool = False
    diff_ref: str | None = None            # git diff reference
    previous_report: str | None = None     # for fix verification

class AuditFinding(BaseModel):
    """A single finding from the audit."""
    id: int
    severity: str                          # critical | important | suggestion
    category: str                          # security | architecture | performance | test_gap | maintainability | correctness
    title: str
    description: str
    location: str                          # file:line_range
    evidence: str
    suggested_fix: str
    effort: str                            # low | medium | high
    reported_by: list[str]                 # model keys that found this
    confirmed_by: str | None               # arbiter model that confirmed
    confidence: str                        # high | needs_verification | disputed
    arbiter_ruling: str | None             # confirmed | dismissed | escalated
    arbiter_reasoning: str | None

class AuditReport(BaseModel):
    """Complete audit report."""
    project: str
    path: str
    timestamp: datetime
    models_used: list[str]
    files_scanned: int
    total_findings: int
    tier_1_critical: list[AuditFinding]
    tier_2_important: list[AuditFinding]
    tier_3_suggestions: list[AuditFinding]
    tier_4_dismissed: list[AuditFinding]
    escalated: list[AuditFinding]          # needs human review
    total_cost: float
    duration_seconds: float

class FixVerification(BaseModel):
    """Result of verifying fixes from a previous audit."""
    finding_id: int
    original_finding: AuditFinding
    status: str                            # fixed | not_fixed | partially_fixed | regressed
    verified_by: str                       # model that verified
    notes: str

class CodebaseProfile(BaseModel):
    """Structural summary of a scanned codebase."""
    project_name: str
    language: str
    framework: str | None
    files: int
    test_files: int
    total_lines: int
    entry_points: list[str]
    key_patterns: list[str]
    dependencies: list[str]
    module_map: dict[str, list[str]]       # module → [classes/functions]
```

## Audit Depth Modes

| Mode | Models | Cross-Validation | Arbiter | Cost (medium project) | Time |
|------|--------|-----------------|---------|----------------------|------|
| **Quick** | 1 model | None | Confirms top findings only | ~$2–4 | ~2 min |
| **Standard** | 3 models | Agreement mapping | Full arbiter on unconfirmed | ~$8–15 | ~5 min |
| **Thorough** | All registered | Full cross-review + scoring | Full arbiter on everything | ~$20–40 | ~10 min |

## CI Integration

The audit can run as a CI step, failing the build if Tier 1 findings are detected:

```yaml
# .github/workflows/triad-audit.yml
- name: Triad Codebase Audit
  run: triad audit --format sarif --changed-only --diff origin/main..HEAD
  
- name: Upload SARIF
  uses: github/codeql-action/upload-sarif@v3
  with:
    sarif_file: triad-audit.sarif

- name: Fail on Critical
  run: |
    CRITICAL=$(triad audit --format json --changed-only | jq '.tier_1_critical | length')
    if [ "$CRITICAL" -gt 0 ]; then exit 1; fi
```

## What Makes This Different

**vs. Static analysis (pylint, SonarQube):** Static tools check syntax and known patterns. Audit reasons about intent, architecture, and domain correctness. They're complementary — run both.

**vs. Single-model code review:** One model reviewing code is useful but has blindspots. Three models independently reviewing, then cross-validating through an adversarial Arbiter, produces higher-confidence findings with fewer false positives.

**vs. Manual code review:** Humans are better at judging business logic correctness and UX implications. Models are better at exhaustively checking every file, finding subtle patterns across modules, and never getting fatigued. Audit handles the exhaustive scan; humans handle the judgment calls (escalated findings).

**vs. GitHub Copilot code review:** Copilot reviews diffs (new code). Audit reviews the entire codebase (existing code). Different scope, different purpose, complementary.

## Relationship to Other Features

- **Context injection (Day 11)** provides the scanning infrastructure
- **Arbiter layer (Day 5)** provides the cross-validation mechanism
- **Parallel mode (Day 6)** provides the independent multi-model execution pattern
- **Session persistence (Day 9)** stores audit history for trend tracking
- **CI/CD integration (Day 13)** provides the GitHub Actions framework
- **`triad audit --verify-fixes`** is essentially a targeted re-run of the Arbiter on specific findings

The audit feature is a natural composition of infrastructure already being built in v0.1. The new pieces are the scanning/profiling layer, the agreement mapping logic, and the tiered report generator.

## Scope

**v0.2.0:** Core audit with Python support, 3 tiers, markdown output, fix verification
**v0.3.0:** Multi-language support (JS/TS, Go, Rust), SARIF output, CI integration, `--changed-only` and `--diff` modes
**v0.4.0:** Trend tracking across audits, regression detection, custom rule definitions, team audit policies

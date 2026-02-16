# Triad Orchestrator — Addendum B: The Arbiter Layer

## Independent Referee, Adversarial Reviewer & Circuit Breaker

*Formalizing the pattern: never trust the builder to grade their own work.*

**NexusAI** | Version 1.2 | February 2026

---

## B1. The Problem This Solves

In development workflows, a pattern has emerged organically that produces consistently better results: when an AI model outputs a plan or generates code, routing that output through a separate review pass catches mistakes, suggests alternatives, and identifies edge cases that would have been missed with auto-approval. This is a manual arbitration step — and it works because the reviewing instance has **no investment in the original output**. It didn't write the code, so it has no cognitive bias toward defending it.

The Triad Orchestrator already has a Verification Gate (Section 6.3 of v1.0), but that gate is a quality checkpoint *after* consensus. What's missing is a mechanism that operates **throughout the pipeline** — an independent referee that can intervene at any stage, challenge any model's output, and act as a circuit breaker before bad code propagates downstream. This addendum formalizes that mechanism as the **Arbiter Layer**.

---

## B2. Design Principles

The Arbiter Layer is governed by five principles that distinguish it from the existing pipeline roles:

1. **Independence:** The Arbiter never participates in generation. It never writes the code, proposes the architecture, or produces the implementation. Its only job is to evaluate, challenge, and verify other models' work. This separation is essential — the same model that wrote the code cannot objectively review it because LLMs exhibit self-consistency bias.

2. **Adversarial posture:** The Arbiter's system prompt explicitly instructs it to look for problems, not to confirm quality. It asks: "What's wrong with this? What edge cases are missed? What would break in production? What pattern violations exist?" This is the opposite of the pipeline agents, whose prompts focus on building and improving.

3. **Different model from the generator:** The Arbiter MUST be a different model (or at minimum a different instance with a different system prompt and no shared conversation context) than the model that produced the output being reviewed. Cross-model review catches blindspots that intra-model review misses.

4. **Stage-gated, not just final:** Unlike the Verification Gate which runs once at the end, the Arbiter can be configured to review after every pipeline stage. It can catch a bad architectural scaffold before the Implementer wastes tokens implementing it, or flag a flawed implementation before the Refactorer spends expensive tokens refactoring it.

5. **Circuit breaker authority:** The Arbiter can halt the pipeline, rewind to a previous stage, or force a re-generation with specific feedback. It doesn't just log concerns — it can act on them before bad output cascades.

---

## B3. Arbiter Architecture

### B3.1 Where the Arbiter Sits

The Arbiter is a new layer that wraps the existing pipeline. It does not replace any existing role — it observes and intervenes:

```
┌──────────────────────────────────────┐
│         ⚠ ARBITER LAYER ⚠            │
│  Independent • Adversarial • Cross-model  │
└───┬────────┬────────┬────────┬───────┘
    │        │        │        │
  reviews  reviews  reviews  reviews
    │        │        │        │
Task ──▶ [Architect] ─▶ [Implement] ─▶ [Refactor] ─▶ [Verify] ─▶ Output
           │           │           │
           └─ HALT? ───┘─ HALT? ──┘
             rewind      rewind
```

The Arbiter sits *above* the pipeline, not inside it. It receives a copy of every stage's output and can intervene between stages.

### B3.2 The Arbiter Schema

```python
class ArbiterReview(BaseModel):
    """Output from the Arbiter after reviewing a stage."""

    stage_reviewed: PipelineStage   # architect | implement | refactor | verify
    reviewed_model: str             # Which model produced the output
    arbiter_model: str              # Which model is arbitrating
    verdict: Verdict                # APPROVE | FLAG | REJECT | HALT
    issues: list[Issue]             # Structured list of problems found
    # Each Issue has:
    #   severity: critical | warning | suggestion
    #   category: logic | pattern | security | performance |
    #             edge_case | hallucination
    #   location: file path + line range (if applicable)
    #   description: What's wrong
    #   suggestion: How to fix it
    #   evidence: Why the Arbiter believes this is an issue
    alternatives: list[Alternative] # Better approaches the Arbiter suggests
    # Each Alternative has:
    #   description: What to do differently
    #   rationale: Why it's better
    #   code_sketch: Optional implementation hint
    #   confidence: 0.0-1.0
    confidence: float               # 0.0-1.0 overall confidence in verdict
    reasoning: str                  # Full chain-of-thought explanation
    token_cost: float               # Cost of this review pass
```

### B3.3 The Four Verdicts

| Verdict | Meaning | Pipeline Action | When Used |
|---|---|---|---|
| **APPROVE** | Output is sound. No critical issues. Pipeline proceeds to next stage. | Continue to next stage. Warnings (if any) are logged but don't block. | Clean output with at most minor suggestions. |
| **FLAG** | Issues detected but not pipeline-breaking. Output proceeds with flagged items attached for downstream agents to address. | Continue, but inject flags into the next agent's context so it knows what to fix. | Moderate issues: missing edge cases, suboptimal patterns, minor security concerns. |
| **REJECT** | Significant problems found. The current stage must re-generate with the Arbiter's feedback injected into the prompt. | Rewind to current stage. Re-run the same agent with Arbiter feedback appended. Max 2 retries before escalation. | Logic errors, hallucinated APIs, pattern violations, broken interfaces. |
| **HALT** | Critical failure. The output is fundamentally unsalvageable at this stage. Pipeline rewinds to a previous stage or stops entirely for human review. | Stop pipeline. Present Arbiter's analysis to developer with options: rewind N stages, change model assignment, or abort. | Architectural unsoundness, fundamental misunderstanding of task, security vulnerabilities that would cascade. |

---

## B4. Cross-Model Enforcement

The critical rule: **the Arbiter must always be a different model than the one it's reviewing.** This prevents self-consistency bias — the well-documented tendency for LLMs to validate their own output even when it's wrong. The routing engine enforces this automatically.

| Stage Model | Arbiter Model | Why This Pairing |
|---|---|---|
| **Gemini (Architect)** | **Claude or Grok** | Claude catches over-abstraction and pattern violations. Grok provides alternative architectural perspectives with strong reasoning. |
| **GPT-4o (Implementer)** | **Claude or Gemini** | Claude excels at finding edge cases in implementation code. Gemini can verify the implementation matches its own architectural intent. |
| **Claude (Refactorer)** | **GPT-4o or Grok** | GPT-4 catches when Claude over-engineers or introduces unnecessary complexity. Grok provides independent reasoning validation. |
| **Claude (Verifier)** | **Grok or Gemini** | Final safety net: a completely independent model validates Claude's own verification. Catches any blindspots in Claude's review. |

The Arbiter pairing is configurable and can be optimized by the Smart Routing Engine. The only hard constraint is: `arbiter_model != stage_model`.

---

## B5. The Arbiter System Prompt

The Arbiter's system prompt is fundamentally different from the pipeline agents. While pipeline agents are prompted to *build and improve*, the Arbiter is prompted to **challenge and break**:

```markdown
# ARBITER SYSTEM PROMPT (abbreviated)

You are the Arbiter. You did NOT write this code. You have no
investment in it. Your job is to find what's wrong.

## Your Mandate
- ASSUME there are bugs until proven otherwise
- LOOK FOR hallucinated imports, APIs, or methods
- CHECK every interface contract against actual usage
- VERIFY error handling covers all failure modes
- QUESTION every architectural decision: is there a simpler way?
- TEST mentally: what happens with null input? Empty list?
  Concurrent access? Network failure? Max-size payload?

## What You Must NOT Do
- Do NOT rewrite the code. You are not a generator.
- Do NOT approve just because the code 'looks right'
- Do NOT defer to the generating model's authority
- Do NOT rubber-stamp. If you have no issues, explain WHY
  you're confident, not just that you are.

## Suggesting Alternatives
When you find a problem, don't just flag it. Suggest a better
approach with rationale and a confidence score. Your alternatives
will be injected into the pipeline for the generating model to
consider on retry.
```

The key behavioral instruction is **"assume there are bugs until proven otherwise."** This inverts the typical LLM review pattern where models default to saying "this looks good" and then hedge with minor suggestions.

---

## B6. Configurable Review Depth

Not every task needs full Arbiter review at every stage. The system supports configurable review depth to balance quality against cost and speed:

| Mode | Stages Reviewed | Cost Impact | When to Use |
|---|---|---|---|
| **Full Arbiter** | Every stage | ~60–80% cost increase over base pipeline. | Production-critical features, security-sensitive code, complex domain logic. |
| **Bookend Arbiter** | Architect + final Verify | ~25–35% cost increase. | Standard feature development. Best cost/quality balance. **Recommended default.** |
| **Final Only** | Verify stage only | ~15–20% cost increase. | Internal tooling, prototypes, tasks with low blast radius. |
| **Off** | None | No additional cost. | Rapid iteration, exploration, benchmarking runs. |

CLI usage:

```bash
# Full Arbiter on critical feature
$ triad run --task '...' --arbiter full

# Bookend (recommended default)
$ triad run --task '...' --arbiter bookend

# Force specific Arbiter model
$ triad run --task '...' --arbiter full --arbiter-model grok-4

# Override: use Grok as Arbiter for architecture, Claude for implementation
$ triad run --task '...' --arbiter full \
    --arbiter-architect grok-4 \
    --arbiter-implement claude-opus-4-5
```

---

## B7. Feedback Injection Protocol

When the Arbiter issues a REJECT verdict, its feedback must be injected into the re-generation prompt in a way that the generating model can act on:

1. **Structured injection:** The Arbiter's issues and alternatives are appended to the original task prompt as a structured block, not as free-text. This ensures the generating model receives specific, parseable feedback rather than vague critique.

2. **Priority ordering:** Issues are sorted by severity (critical first), so if the model's context window is tight, the most important feedback is guaranteed to be seen.

3. **Diff-aware:** For REJECT verdicts on implementation or refactoring stages, the Arbiter's feedback references specific code locations (file + line range), so the generating model knows exactly where to focus.

4. **Retry budget:** Each stage gets a maximum of 2 Arbiter-triggered retries. If the stage still fails after 2 retries, the pipeline escalates: either HALT for human review, or rewind to the previous stage with the accumulated feedback so a different model can attempt a different approach.

The feedback injection template:

```markdown
## ARBITER FEEDBACK (Retry {n} of 2)

Your previous output was REJECTED by the independent Arbiter.
You MUST address the following issues in your revised output.

### CRITICAL ISSUES (must fix)
1. [category] description
   Location: file:line_range
   Evidence: why this is wrong
   Suggested fix: specific guidance

### WARNINGS (should fix)
1. [category] description ...

### ALTERNATIVES TO CONSIDER
The Arbiter suggests these alternative approaches:
1. description (confidence: 0.85)
   Rationale: why this might be better
   Sketch: optional code hint
```

---

## B8. Arbiter Cost Impact

The Arbiter adds review passes that cost tokens. Real-world cost impact per routing mode for a medium-complexity task:

| Configuration | Base Cost | Arbiter Cost | Total | ROI Argument |
|---|---|---|---|---|
| No Arbiter | ~$4.30 | $0 | **$4.30** | Baseline |
| Final Only | ~$4.30 | ~$0.80 | **$5.10** | Catches output-level issues |
| **Bookend (rec.)** | ~$4.30 | ~$1.50 | **$5.80** | Best balance: catches bad foundations early + validates output |
| Full Arbiter | ~$4.30 | ~$3.00 | **$7.30** | Maximum safety for critical paths |

At Bookend depth and 15 tasks/week, the Arbiter adds approximately **$90/month** to the pipeline cost. The Arbiter pays for itself the first time it catches a single critical bug before it ships.

---

## B9. Updated Pipeline Summary

With the Arbiter Layer integrated, the complete pipeline:

| # | Stage | Agent (pluggable) | Arbiter (cross-model) | Gate Condition |
|---|---|---|---|---|
| 1 | Architect | Best-fit architect model | Different model → review | APPROVE or FLAG to proceed. REJECT triggers re-gen with feedback. HALT stops pipeline. |
| 2 | Implement | Best-fit implementer | Different model → review | Same gate logic. Arbiter feedback includes scaffold context from stage 1. |
| 3 | Refactor | Best-fit refactorer | Different model → review | Arbiter checks for over-engineering, unnecessary complexity. |
| 4 | Consensus | All pipeline agents vote | Arbiter observes, no vote | Arbiter can FLAG the consensus itself if it sees groupthink. |
| 5 | Verify | Best-fit verifier | Different model → final check | Arbiter validates the Verifier's own review. Final confidence score. |

---

## B10. Cross-Model Enforcement in the MVP

For the MVP (bookend mode), the Arbiter reviews only two stages:

1. **After Architect:** Catches bad scaffolds, wrong patterns, and structural issues before implementation begins.
2. **After Verify:** Final safety net before output delivery.

Cross-model enforcement is a hard rule in all configurations: `arbiter_model != stage_model`. The MVP validates this at pipeline startup and raises an error if the configuration violates it.

---

## B11. Implementation Summary Reconciliation

### B11.1 The Gap This Closes

The existing Arbiter checks validate code *quality* — is it correct, secure, well-structured, and pattern-compliant? What they don't validate is code *completeness* — did the pipeline actually deliver everything the task specified? In a multi-model pipeline where each stage reinterprets the task through its own lens, scope drift is a real failure mode. The Architect might design five endpoints but the Implementer only wires four. The Refactorer might remove a feature it considers unnecessary. The Verifier might confirm the code works without checking whether it works *as specified*.

The Implementation Summary Reconciliation (ISR) adds a final Arbiter pass that answers one question: **does the output match the input?**

### B11.2 How It Works

After the Verify stage completes, the Verifier produces a structured Implementation Summary — a machine-parseable manifest of what was built. The Arbiter then compares this summary against three reference documents:

1. **The original TaskSpec:** What the developer asked for.
2. **The Architect's scaffold:** What was planned, including any scope decisions made during architecture.
3. **Accepted suggestions:** Any cross-domain suggestions that were adopted during consensus and should be reflected in the final output.

The Arbiter issues a reconciliation verdict using the same four-verdict system (APPROVE / FLAG / REJECT / HALT), but the review criteria shift from code correctness to **spec compliance.**

**MVP scope:** The MVP reconciles against TaskSpec + Architect output only. Full reconciliation against mid-pipeline consensus decisions and accepted cross-domain suggestions is post-MVP.

### B11.3 The Implementation Summary Schema

The Verifier produces this structured summary as part of its output:

```python
class ImplementationSummary(BaseModel):
    """Structured manifest of what was actually built."""

    task_echo: str                    # Verifier's restatement of the original task
    endpoints_implemented: list[str]  # API routes / CLI commands delivered
    schemas_created: list[str]        # Data models / Pydantic schemas
    files_created: list[str]          # File paths produced
    files_modified: list[str]         # Existing files changed (if applicable)
    behaviors_implemented: list[str]  # Functional behaviors (business logic, handlers)
    test_coverage: list[str]          # What's tested (by name/description)
    deviations: list[Deviation]       # Intentional departures from the spec
    # Each Deviation has:
    #   what: str            — what was changed/omitted
    #   reason: str          — why the agent deviated
    #   stage: PipelineStage — which stage introduced the deviation
    omissions: list[str]              # Items from spec NOT implemented (if any)
```

### B11.4 Reconciliation Review Criteria

| Check | What the Arbiter Asks |
|---|---|
| **Completeness** | Does every requirement in the TaskSpec have a corresponding entry in the summary? Are there spec items with no matching implementation? |
| **Scope Fidelity** | Did the pipeline add features not in the spec? Are deviations justified with clear rationale, or did scope creep silently? |
| **Architectural Alignment** | Does the final implementation match the Architect's scaffold? If it diverged, was the divergence approved via consensus or suggestion? |
| **Suggestion Integration** | Were accepted cross-domain suggestions reflected in the output? Did any accepted suggestions get silently dropped? (Post-MVP) |
| **Test-to-Spec Mapping** | Does every specified behavior have at least one corresponding test? Are there untested requirements? |

### B11.5 Reconciliation Verdicts

| Verdict | Meaning | Pipeline Action |
|---|---|---|
| **APPROVE** | Implementation matches spec. All requirements delivered, deviations justified, tests aligned. | Pipeline completes. Summary and reconciliation report included in output. |
| **FLAG** | Minor gaps detected — e.g., a non-critical behavior missing test coverage, or an undocumented deviation that appears intentional. | Pipeline completes with flags. Gaps listed in the output report for developer awareness. |
| **REJECT** | Meaningful spec gap — a required endpoint missing, a specified behavior unimplemented, or an accepted suggestion silently dropped. | Rewind to the appropriate stage (Implement or Refactor) with the reconciliation feedback injected. Max 1 retry (single REJECT retry for reconciliation). |
| **HALT** | Fundamental misalignment — the output addresses a different task than what was specified, or critical requirements are entirely absent. | Pipeline stops. Reconciliation analysis presented to developer with options: rewind, reassign models, or abort. |

### B11.6 Configuration

The ISR is **opt-in** and controlled via the `--reconcile` flag. It can be enabled independently of the main Arbiter depth setting:

```bash
# Enable reconciliation (works with any Arbiter mode, including off)
$ triad run --task '...' --arbiter bookend --reconcile

# Reconciliation without other Arbiter checks
$ triad run --task '...' --arbiter off --reconcile

# Full Arbiter + reconciliation (maximum safety)
$ triad run --task '...' --arbiter full --reconcile

# Force specific reconciliation Arbiter model
$ triad run --task '...' --reconcile --reconcile-model grok-4
```

The cross-model enforcement rule applies: the reconciliation Arbiter must be a different model than the Verifier.

### B11.7 Cost Impact

The reconciliation pass is a single additional model call comparing the summary against the spec — it's lightweight because it reviews a structured manifest, not full code. Estimated cost: **~$0.30–$0.60 per task** depending on the Arbiter model selected. At 15 tasks/week, this adds roughly **$20–$35/month**.

---

*The Arbiter Layer transforms the Triad from a collaborative pipeline into an adversarial-collaborative pipeline — models build together, but an independent referee ensures they build correctly. The Implementation Summary Reconciliation ensures they build completely.*

They build. **The Arbiter breaks.** The reconciler confirms. The code that survives is **production-ready**.

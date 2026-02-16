# Triad Orchestrator — Technical Architecture

## Multi-Model AI Development Platform

**NexusAI** | Version 1.0 | February 2026

---

## 1. Overview

The Triad Orchestrator is a multi-model AI development platform that coordinates frontier language models into a unified coding pipeline. Rather than using a single AI assistant, Triad assigns each model to the role where it demonstrably excels while granting every agent the **autonomy to propose ideas outside its primary role**. A consensus protocol ensures models agree on approach before code is finalized, and an independent Arbiter Layer catches mistakes before they propagate.

---

## 2. Problem Statement

Using a single AI model for all development tasks means every task inherits that model's specific blindspots:

1. **Model-specific blindspots:** Each model has strengths and weaknesses. One excels at careful refactoring but over-engineers simple tasks. Another writes functional code rapidly but cuts corners on edge cases. A third has outstanding architectural vision but may lack depth in granular implementation.

2. **No cross-pollination:** When one model writes code, the unique insights another model might offer — an alternative design pattern, a performance optimization, a security consideration — are completely lost.

3. **No adversarial review:** If the assigned model hallucinates an API, uses a deprecated pattern, or misunderstands the domain model, there's no built-in check until human review.

4. **Complex codebases demand specialization:** Architecture, implementation, and quality assurance are three distinct cognitive skills. No single model consistently outperforms in all three.

---

## 3. Pipeline Stages & Role Assignment

Each model receives a **primary role** based on empirical benchmarks, plus **secondary capabilities** allowing it to contribute ideas outside its lane. Every agent can suggest anything, but primary roles determine default task routing.

### Pipeline Stages

| # | Stage | Responsibility | Output |
|---|---|---|---|
| 1 | **Architect** | Scaffolding, structure, system design, file layout, interfaces, data models | Scaffold + file structure + interfaces |
| 2 | **Implement** | Wiring, business logic, API handlers, state management, integration | Working code with logic wired |
| 3 | **Refactor** | Code quality, optimization, testing, edge cases, maintainability | Optimized code + tests + improvement log |
| 4 | **Verify** | Final validation, confidence scoring, integration risk assessment | Final deliverable + confidence score |

Between stages, the **Arbiter Layer** (see `docs/arbiter.md`) reviews outputs and can APPROVE, FLAG, REJECT, or HALT the pipeline.

### Dynamic Role Assignment

Roles are not hardcoded to specific models. The orchestrator uses fitness benchmarks — each registered model is periodically scored against a standard task set for each role. The model with the highest fitness for a role gets default assignment. Assignments change automatically as models are updated or new models are added.

---

## 4. System Architecture

### 4.1 High-Level Layers

1. **Orchestration Layer:** Manages the pipeline lifecycle, routes tasks to agents, enforces timeouts (configurable per-agent, default 120s), handles retries with exponential backoff, and maintains the session audit trail.

2. **Provider Layer:** Model-agnostic plugin system. Any LLM with an API can participate via a single abstract interface. LiteLLM provides the universal adapter for 100+ providers. Adding a new model is a TOML config change, not a code change.

3. **Arbiter Layer:** Independent referee that sits above the pipeline. Reviews stage outputs using cross-model enforcement (arbiter ≠ generator). Issues verdicts (APPROVE/FLAG/REJECT/HALT). See `docs/arbiter.md` for full specification.

4. **Consensus Layer:** Deliberation protocol that passes proposals between agents, collects votes, handles objections, manages cross-domain suggestion evaluation, and invokes a tiebreaker when models cannot agree.

5. **Output Layer:** Structured file output (code, tests, reports, logs), Markdown summary generation, JSON session logs with full audit trail.

### 4.2 Technology Stack

| Component | Technology |
|---|---|
| Runtime | Python 3.12+ with asyncio, Pydantic v2 for all schemas |
| LLM Adapter | LiteLLM (universal interface to 100+ providers) |
| CLI Interface | Typer + Rich for terminal UI with live agent status panels |
| Config | TOML (models.toml, defaults.toml) |
| Testing | pytest + pytest-asyncio |
| Linting | Ruff |

### 4.3 Agent Message Envelope

All inter-agent communication uses a standardized Pydantic model:

```python
class AgentMessage(BaseModel):
    message_id: str             # UUID v4
    from_agent: PipelineStage   # architect | implement | refactor | verify
    to_agent: PipelineStage     # target or 'orchestrator'
    msg_type: MessageType       # proposal | implementation | review |
                                # objection | suggestion | vote |
                                # consensus | verification
    content: str                # The actual code / analysis
    code_blocks: list[CodeBlock]  # Parsed structured code
    confidence: float           # 0.0-1.0 self-assessed confidence
    suggestions: list[Suggestion] # Cross-domain ideas
    objections: list[Objection]   # Disagreements with reasons
    metadata: dict              # Token usage, latency, model version
    timestamp: datetime
```

The `suggestions` field is the key enabler of cross-domain autonomy. Any agent can attach suggestions to any message, even when that suggestion falls outside its primary role.

---

## 5. Pipeline Workflows

### 5.1 Sequential Pipeline (Default / MVP)

The standard workflow for well-defined feature tasks. Each stage builds on the previous output:

```
Task → [Architect] → [Implement] → [Refactor] → [Verify] → Output
          ↑              ↑              ↑            ↑
       Arbiter        Arbiter        Arbiter      Arbiter
```

### 5.2 Parallel Exploration (Post-MVP)

For architectural decisions or when the best approach is unclear. All models independently solve the same task, then cross-review:

1. **Fan-out:** All agents receive the identical task prompt simultaneously.
2. **Independent work:** Each agent produces a complete solution without seeing others' output.
3. **Cross-review:** Each agent reviews the other solutions and produces scored ratings.
4. **Consensus vote:** Each agent votes for the best overall approach (cannot self-vote). Majority wins. Tie → designated tiebreaker decides.
5. **Synthesis:** Winning approach is enhanced with best elements from other proposals, then passes through verification.

### 5.3 Debate Mode (Post-MVP)

For contentious design decisions requiring explicit tradeoff analysis:

1. **Position papers:** Each agent proposes its preferred approach with explicit tradeoff analysis.
2. **Rebuttals:** Each agent receives the other proposals and writes a structured rebuttal.
3. **Final arguments:** Each agent updates its proposal incorporating valid criticisms.
4. **Judgment:** Designated tiebreaker evaluates all positions, rebuttals, and final arguments to produce a reasoned decision document.

---

## 6. Consensus Protocol & Cross-Domain Autonomy

### 6.1 The Suggestion Mechanism

Every agent message can include a `suggestions[]` array containing ideas outside the agent's primary role. Each suggestion includes:

- **domain:** Which role's territory the suggestion enters (architecture, implementation, quality)
- **rationale:** Detailed reasoning for why this idea is better than the current approach
- **confidence:** Self-assessed confidence score (0.0–1.0)
- **code_sketch:** Optional code snippet demonstrating the alternative
- **impact_assessment:** What changes downstream if this suggestion is adopted

### 6.2 Consensus Decision Tree

| Scenario | Resolution |
|---|---|
| All agents agree | Proceed immediately. No deliberation needed. |
| Majority agrees | Majority wins, but dissenter's objection is logged. Dissenting agent gets one rebuttal. If rebuttal is compelling (per tiebreaker's judgment), re-vote. |
| All disagree | Each agent submits a position paper. Tiebreaker evaluates all three, selects winner with written justification. |
| Cross-domain suggestion | Primary role-holder for that domain evaluates. If rejected, suggester can escalate to a full consensus vote. |

---

## 7. Model-Agnostic Plugin Architecture

See `docs/model-agnostic.md` for the full specification. Key points:

- Any model with an API can participate via the `ModelProvider` abstract interface
- LiteLLM provides the universal adapter — adding a model is a TOML config entry
- Smart Routing Engine selects optimal models per role based on fitness benchmarks and cost/quality/speed priorities
- Routing modes: Quality-First, Cost-Optimized, Speed-First, Hybrid, Tournament

---

## 8. Cost & Token Economics

Estimated per-task costs for a medium-complexity feature (new API endpoint with tests):

| Configuration | Est. Input Tokens | Est. Output Tokens | Est. Cost |
|---|---|---|---|
| Base pipeline (no Arbiter) | ~150K | ~35K | ~$4.30 |
| + Bookend Arbiter | ~190K | ~45K | ~$5.80 |
| + Full Arbiter | ~250K | ~60K | ~$7.30 |
| + Reconciliation (opt-in) | +~20K | +~5K | +~$0.30–$0.60 |

At 10–20 tasks per week, monthly cost ranges from approximately $170–$580 depending on Arbiter configuration.

---

## 9. Risk Mitigation

| Risk | Mitigation | Fallback |
|---|---|---|
| API rate limits / outages | Exponential backoff with jitter per-agent. Circuit breaker: 3 failures = agent offline for 5 min. | Degrade to fewer agents if one is down. |
| Consensus deadlock | Max 3 deliberation rounds. After round 3, tiebreaker decides unilaterally with written justification. | Developer presented all proposals to make the call manually. |
| Hallucinated APIs | Arbiter runs static import checks. Cross-model review catches single-model hallucinations. | Flag low-confidence outputs for human review before merge. |
| Token cost spiraling | Per-session budget caps. Context pruning keeps inputs under model limits. Usage dashboard with alerts. | Auto-switch to cheaper model variants for non-critical passes. |
| Conflicting code styles | Shared coding conventions in context package. Ruff/Black formatting applied to all outputs post-consensus. | Style enforcement is automated, not dependent on model agreement. |

---

*This specification is a living document. As the Triad Orchestrator moves through implementation phases, each section will be updated with implementation details, benchmarks, and lessons learned.*

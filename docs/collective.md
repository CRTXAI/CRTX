# Collective Intelligence Mode

**Multi-Model Reasoning Fusion**

*Every model thinking together. The output is smarter than any individual.*

**Status:** Post-Release · Target: v0.3.0
**Depends on:** Parallel + Debate modes (v0.1.0), Smart Routing (v0.1.0), Consensus Protocol (v0.1.0)

---

## Problem

Triad's current pipeline modes are structured and role-based. Sequential assigns one model per job. Parallel asks each model to solve the whole problem independently. Debate has models argue for their preferred approach. In all three modes, models work within defined lanes with fixed interaction patterns.

But frontier LLMs don't have uniform intelligence — they have different cognitive profiles. Claude is meticulous about edge cases. Gemini reasons across massive contexts. GPT-4 generates working code fast. Grok brings lateral reasoning. Each model sees things the others miss, not because of its assigned role, but because of how it thinks.

The current modes don't capture this. When Claude catches a subtle race condition during refactoring, that insight only matters for the refactoring stage. But what if Claude's careful reasoning could improve the architecture before it's even built? What if Gemini's ability to hold the full project in context could enhance every model's output at every stage? What if the models could genuinely think together instead of taking turns?

That's what collective intelligence mode is: structured fusion of multiple models' reasoning on the same problem, producing a composite output that no single model — and no single-model-per-role pipeline — could match.

## Core Patterns

Collective intelligence mode isn't a single technique. It's a family of four composition patterns that the orchestrator can apply based on task characteristics. Each pattern combines models differently depending on what the task needs.

### Pattern 1: Chain-of-Thought Fusion

Models extend each other's reasoning chains instead of producing independent outputs.

**How it works:**

```
Task: "Design a rate limiting system that handles multi-tenant isolation,
       burst tolerance, and distributed state across 3 regions."

Step 1 — Gemini starts the reasoning chain (large-context strength):
  "Given 3 regions with eventual consistency, the core challenge is
   distributed counting without a central authority. Options include:
   sliding window counters with Redis, token bucket with regional
   quotas, or leaky bucket with cross-region sync..."
   [Gemini explores 4 approaches at high level, identifies tradeoffs]

Step 2 — Claude extends the chain (careful analysis strength):
  "Gemini's analysis of the token bucket approach missed a critical
   edge case: during a region failover, the quota redistribution
   creates a window where the global limit can be exceeded by up to
   Nx the regional quota. The sliding window approach handles this
   more gracefully because..."
   [Claude deepens the analysis, finds edge cases, narrows to 2 options]

Step 3 — GPT-4 extends further (practical implementation strength):
  "Between the two remaining approaches, the sliding window with
   Redis Cluster has a concrete implementation advantage: we can use
   Redis MULTI/EXEC for atomic increment-and-check, and the TTL-based
   expiry handles window rotation without a background job. Here's
   the implementation skeleton..."
   [GPT-4 turns the refined reasoning into concrete code structure]

Step 4 — Synthesis: The final output is the complete chain — Gemini's
  broad exploration → Claude's deep analysis → GPT-4's implementation.
  Each model built on the previous model's strongest thinking.
```

**When to use:** Complex design problems where breadth of exploration, depth of analysis, and practical implementation are all critical. Architecture decisions, algorithm design, system design interviews.

**Key insight:** Each model starts where the previous model's cognitive strengths peak. Gemini explores breadth. Claude analyzes depth. GPT-4 builds concretely. The chain captures all three strengths in a single reasoning flow.

### Pattern 2: Ensemble Consensus

All models answer independently, then a synthesis pass extracts the best reasoning from each — not just picking a winner, but fusing insights.

**How it works:**

```
Task: "Review this authentication module for security vulnerabilities."

Phase 1 — Independent analysis (all models in parallel):
  Claude: Finds 3 issues — SQL injection in login query, missing rate
          limiting on /auth/reset, JWT secret hardcoded in config.
  Gemini: Finds 3 issues — JWT secret hardcoded, no CSRF protection
          on session endpoints, token expiry set to 30 days.
  GPT-4:  Finds 2 issues — SQL injection in login query, missing
          input sanitization on email field.

Phase 2 — Agreement mapping:
  HIGH CONFIDENCE (2+ models agree):
  - SQL injection in login query (Claude + GPT-4)
  - JWT secret hardcoded (Claude + Gemini)

  UNIQUE INSIGHTS (only one model caught it):
  - Missing rate limiting on /auth/reset (Claude only)
  - No CSRF protection on session endpoints (Gemini only)
  - Token expiry too long at 30 days (Gemini only)
  - Missing input sanitization on email field (GPT-4 only)

Phase 3 — Synthesis:
  The synthesizer (strongest reviewer model) produces a unified
  security report that includes ALL findings, weighted by:
  - Multi-model agreement → highest confidence
  - Unique insights → verified by synthesizer before including
  - Contradictions → synthesizer analyzes and resolves

Final output: 6 verified security issues, ranked by severity,
  with remediation guidance drawn from whichever model provided
  the clearest fix for each issue.
```

**When to use:** Review tasks, bug hunting, security audits, code review — anywhere "more eyes" genuinely helps and different models catch different things.

**Key insight:** The value isn't in the majority vote. It's in the unique insights — the thing only one model caught. Ensemble consensus ensures those don't get lost.

### Pattern 3: Iterative Refinement Loop

Two models build on each other's work in alternating passes, with each pass improving the output. Not adversarial (that's debate mode) — collaborative.

**How it works:**

```
Task: "Implement a WebSocket chat system with rooms, presence,
       and message history."

Round 1:
  GPT-4 (builder): Produces working implementation — WebSocket
    handlers, room management, basic message broadcasting.

  Claude (refiner): Takes GPT-4's code, identifies 4 improvements:
    adds connection heartbeat for presence detection, extracts room
    logic into a service class, adds message ordering guarantees
    with sequence numbers, strengthens error handling on disconnect.

Round 2:
  GPT-4 (builder): Takes Claude's refined version, integrates
    message history persistence (which Claude's refactoring made
    easier by extracting the service class), adds reconnection
    logic that replays missed messages using sequence numbers.

  Claude (refiner): Reviews the additions, catches a race condition
    in the reconnection replay (messages could arrive during replay),
    adds a buffering mechanism, writes comprehensive tests.

Output: A WebSocket implementation that went through 4 passes —
  each pass building on the genuine strengths of the previous model's
  contribution. GPT-4 builds fast, Claude refines carefully, and the
  output has both speed and quality.
```

**When to use:** Implementation tasks where rapid generation + careful refinement produces better results than either approach alone. Code generation, content creation, data pipeline design.

**Key insight:** This is different from sequential (where each role does a different job) because both models are doing the same job — building the implementation — but their alternating perspectives catch each other's weaknesses.

### Pattern 4: Cognitive Sub-Routing

A complex task is decomposed into sub-problems, and each sub-problem is routed to the model whose cognitive strengths best match that specific type of thinking.

**How it works:**

```
Task: "Build a document processing pipeline that handles PDF extraction,
       entity recognition, classification, and search indexing."

Step 1 — Decomposition (any model, or the planner):
  Sub-problems identified:
  a) Architecture: How do the components connect? (broad reasoning)
  b) PDF extraction: OCR + layout analysis (practical implementation)
  c) Entity recognition: NLP pipeline design (careful analysis)
  d) Classification: ML model selection + training data (reasoning)
  e) Search indexing: Elasticsearch schema + query design (implementation)
  f) Error handling: What happens when each component fails? (edge cases)

Step 2 — Cognitive routing:
  a) Architecture → Gemini (full-system reasoning across components)
  b) PDF extraction → GPT-4 (practical library integration)
  c) Entity recognition → Claude (careful NLP pipeline design)
  d) Classification → Grok (reasoning about model selection tradeoffs)
  e) Search indexing → GPT-4 (practical Elasticsearch patterns)
  f) Error handling → Claude (edge case analysis)

Step 3 — Integration:
  Each sub-solution is assembled by the synthesizer model, which
  resolves interface mismatches, adds glue code, and ensures the
  components integrate cleanly.

Step 4 — Cross-validation:
  Each model reviews the sub-solutions produced by other models
  (standard Arbiter pattern) to catch integration issues.
```

**When to use:** Large, multi-faceted tasks where different sub-problems require fundamentally different types of thinking. Full-stack features, system migrations, complex data pipelines.

**Key insight:** This goes beyond role-based assignment (architect/implementer/refactorer) to cognitive-profile-based assignment (broad reasoner/fast builder/careful analyzer/lateral thinker). The routing is by thinking style, not job title.

## Implementation Design

### Mode Selection

```
# Automatic — orchestrator picks the best pattern based on task analysis
triad run --task "..." --mode collective

# Explicit pattern selection
triad run --task "..." --mode collective:fusion
triad run --task "..." --mode collective:ensemble
triad run --task "..." --mode collective:refinement
triad run --task "..." --mode collective:subroute

# Combined — decompose into sub-tasks, use fusion within each
triad run --task "..." --mode collective:subroute+fusion
```

### Architecture

```
triad/
├── collective/
│   ├── __init__.py
│   ├── orchestrator.py      # Collective mode coordinator
│   ├── fusion.py            # Chain-of-thought fusion logic
│   ├── ensemble.py          # Ensemble consensus + agreement mapping
│   ├── refinement.py        # Iterative refinement loop
│   ├── subrouter.py         # Cognitive sub-routing + decomposition
│   └── synthesizer.py       # Output synthesis + integration
├── prompts/
│   ├── collective_fusion.md      # "Continue this reasoning chain..."
│   ├── collective_ensemble.md    # "Analyze independently, then..."
│   ├── collective_refine.md      # "Improve this, building on..."
│   ├── collective_decompose.md   # "Break this task into sub-problems..."
│   └── collective_synthesize.md  # "Integrate these sub-solutions..."
```

### Schema Additions

```python
class CollectivePattern(str, Enum):
    """Which collective intelligence pattern to use."""
    FUSION = "fusion"           # Chain-of-thought fusion
    ENSEMBLE = "ensemble"       # Independent + synthesis
    REFINEMENT = "refinement"   # Iterative build/refine loop
    SUBROUTE = "subroute"       # Cognitive sub-routing
    AUTO = "auto"               # Orchestrator picks based on task

class ReasoningChain(BaseModel):
    """A fused reasoning chain from multiple models."""
    steps: list[ReasoningStep]
    models_involved: list[str]
    total_tokens: int
    total_cost: float

class ReasoningStep(BaseModel):
    """One model's contribution to a reasoning chain."""
    model: str
    content: str
    cognitive_role: str         # "explorer", "analyzer", "builder", etc.
    built_on: str | None        # model ID of previous step
    insights_added: list[str]   # what this step contributed
    token_usage: TokenUsage

class EnsembleResult(BaseModel):
    """Result of ensemble consensus."""
    individual_outputs: dict[str, str]  # model → output
    agreements: list[str]               # findings 2+ models share
    unique_insights: dict[str, list[str]]  # model → unique findings
    contradictions: list[str]           # where models disagree
    synthesized_output: str             # final fused result

class SubRouteMap(BaseModel):
    """Task decomposition with cognitive routing."""
    sub_tasks: list[SubTask]
    routing_rationale: str

class SubTask(BaseModel):
    """A sub-problem routed to a specific model."""
    description: str
    cognitive_need: str          # "broad reasoning", "careful analysis", etc.
    assigned_model: str
    rationale: str               # why this model for this sub-task
    output: str | None           # filled after execution
    depends_on: list[int]        # sub-task indices this depends on
```

### Cognitive Profiles

The routing engine needs a richer model of each LLM's strengths than the current role-based fitness scores. Cognitive profiles describe how a model thinks, not what job it does:

```toml
[models.claude-opus.cognitive]
careful_analysis = 0.95      # edge cases, subtle bugs, deep reasoning
broad_exploration = 0.75     # brainstorming, divergent thinking
fast_generation = 0.70       # quick working code, boilerplate
pattern_matching = 0.90      # recognizing structures, conventions
lateral_thinking = 0.80      # unexpected connections, creative solutions
instruction_following = 0.95 # precise adherence to complex specs

[models.gemini-pro.cognitive]
careful_analysis = 0.75
broad_exploration = 0.92     # massive context, full-codebase reasoning
fast_generation = 0.80
pattern_matching = 0.85
lateral_thinking = 0.78
instruction_following = 0.80

[models.gpt-4o.cognitive]
careful_analysis = 0.75
broad_exploration = 0.78
fast_generation = 0.92       # fastest high-quality generation
pattern_matching = 0.82
lateral_thinking = 0.75
instruction_following = 0.85
```

These profiles inform sub-routing decisions: a sub-problem that needs careful analysis gets Claude, one that needs broad exploration gets Gemini, one that needs fast generation gets GPT-4.

### Auto-Pattern Selection

When `--mode collective` is used without specifying a pattern, the orchestrator analyzes the task and selects:

| Task Signal | Pattern Selected | Reasoning |
|-------------|-----------------|-----------|
| Complex design decision with tradeoffs | Fusion | Needs breadth → depth → implementation reasoning chain |
| Review/audit/find-bugs task | Ensemble | More eyes catch more issues; unique insights matter |
| "Build X" with clear spec | Refinement | Fast generation + careful polish = best code |
| Large multi-faceted task | Sub-route | Different sub-problems need different thinking styles |
| Ambiguous — could go either way | Ensemble | Safest default; produces multiple perspectives |

## Cost Profile

Collective mode uses more tokens than standard modes because models see each other's reasoning. Estimated costs for a medium-complexity task:

| Pattern | Models Used | Estimated Cost | vs. Sequential |
|---------|------------|---------------|----------------|
| Fusion (3-step chain) | 3 | ~$6–9 | 1.5–2x |
| Ensemble (3 models + synthesis) | 3+1 | ~$10–14 | 2.5–3x |
| Refinement (2 rounds of 2 models) | 2 | ~$8–11 | 2–2.5x |
| Sub-route (4 sub-tasks + synthesis) | 3–4 | ~$12–18 | 3–4x |

The cost premium is justified when the task is complex enough that a single model's output would require multiple rounds of human review and correction. For simple tasks, sequential or parallel modes remain more cost-effective.

## What Makes This Different

**vs. Triad's existing modes:** Sequential, parallel, and debate assign models to roles. Collective assigns models to cognitive functions. The difference is between "you're the architect" and "you're the broad thinker on this specific sub-problem."

**vs. Mixture of Experts (MoE):** MoE routes at the token level within a single model. Collective routes at the task/reasoning level across multiple models. MoE is internal architecture; collective is external orchestration.

**vs. Multi-agent frameworks (CrewAI, AutoGen):** These frameworks provide the plumbing for agents to talk to each other. Collective provides the strategy — which model should think about what, when, and how to fuse the results. The orchestration protocol is the value, not the message passing.

**vs. Just asking the same question to multiple models:** The key difference is structured composition. Ensemble doesn't just take the best answer — it extracts unique insights from each and fuses them. Fusion doesn't just concatenate — each model builds on the previous one's strongest thinking. The synthesis is the product.

## Scope Boundaries

**In scope for v0.3:**
- Fusion and Ensemble patterns (highest value, most distinct from existing modes)
- Cognitive profile extension to model config
- Auto-pattern selection based on task analysis
- Integration with existing Arbiter (cross-validates collective output)

**v0.4+:**
- Refinement loop pattern
- Cognitive sub-routing with task decomposition
- Combined patterns (subroute + fusion within each sub-task)
- Learning from history (which patterns work best for which task types)
- Cognitive profile auto-calibration from benchmark data

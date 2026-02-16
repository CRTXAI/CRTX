# Triad Orchestrator — Addendum A: Model-Agnostic Plugin Architecture

## From Three Models to Infinite Models

*Any model with an API. Any role in the pipeline. Plug and play.*

**NexusAI** | Version 1.1 | February 2026

---

## A1. Strategic Rationale

The original Triad spec assigns three fixed models to three fixed roles. This addendum transforms the architecture from a hardcoded three-model system into a **model-agnostic plugin platform** where any LLM with an API can be registered, benchmarked, and assigned to any pipeline role.

1. **The frontier is expanding:** New models from xAI, DeepSeek, Meta, Mistral, and others now offer frontier-level coding capabilities. Locking into three models means missing these capabilities.

2. **Models leapfrog constantly:** Every 2–3 months, a new model claims the top benchmark spot in some category. A plugin architecture lets you hot-swap the leader into the pipeline without code changes.

3. **Cost optimization demands flexibility:** Different models have dramatically different price/performance ratios. The orchestrator should route to the best value, not a fixed assignment.

4. **Commercial viability requires provider-agnosticism:** Enterprise customers need to bring their own model contracts, use private deployments, or comply with data sovereignty requirements.

---

## A2. Plugin Architecture

### A2.1 The Model Provider Interface

Every model — cloud API, local deployment, or custom fine-tune — implements a single abstract interface:

```python
class ModelProvider(ABC):
    """Any LLM that can participate in the orchestrator."""

    # Identity
    provider_id: str        # 'openai', 'anthropic', 'xai', 'local'
    model_id: str           # 'gpt-4o', 'claude-opus-4-5', 'grok-4'
    display_name: str       # Human-friendly name

    # Capabilities (self-declared + benchmarked)
    context_window: int
    supports_tools: bool
    supports_structured: bool
    supports_vision: bool
    supports_thinking: bool
    cost_per_1m_input: float
    cost_per_1m_output: float

    # Role fitness scores (0.0-1.0, updated by benchmarks)
    fitness: RoleFitness    # architect, implementer, refactorer,
                            # reviewer, verifier scores

    @abstractmethod
    async def complete(
        self, messages: list[Message],
        system: str, tools: list[Tool] | None,
        output_schema: type[BaseModel] | None,
    ) -> AgentMessage:
        """The only method a provider must implement."""
```

The orchestrator doesn't care *how* a model generates its response. As long as it implements `complete()` and declares its capabilities, it can participate.

### A2.2 LiteLLM as the Universal Adapter

Rather than writing a separate SDK integration for every provider, the plugin layer uses **LiteLLM** as its universal adapter. Adding a new model is a configuration change, not a code change:

```toml
# Adding Grok 4 to the orchestrator = one config entry
[models.grok4]
provider = "xai"
model = "grok-4-0709"
api_key_env = "XAI_API_KEY"
api_base = "https://api.x.ai/v1"
context_window = 131072
supports_tools = true
supports_thinking = true
cost_input = 3.00   # per 1M tokens
cost_output = 15.00

[models.deepseek_v3]
provider = "deepseek"
model = "deepseek-chat"
api_key_env = "DEEPSEEK_API_KEY"
context_window = 65536
cost_input = 0.27
cost_output = 1.10

[models.local_llama4]
provider = "ollama"
model = "llama4:scout"
api_base = "http://localhost:11434"
context_window = 131072
cost_input = 0.00   # Free - runs locally
cost_output = 0.00
```

### A2.3 Dynamic Role Assignment

Roles are no longer hardcoded. The orchestrator uses a **Role Assignment Engine** that matches models to roles based on three factors:

1. **Fitness benchmarks:** Each registered model is periodically benchmarked against a standard task set for each role. The model with the highest fitness for a role gets default assignment.

2. **Task-specific overrides:** A developer can override defaults per-task.

3. **Cost-aware routing:** The orchestrator can be configured to optimize for cost, quality, or a balance. In cost mode, cheaper models handle simple tasks while frontier models handle complex domain logic — same pipeline, dynamically routed.

---

## A3. Supported Providers

The following providers are supported at launch through LiteLLM integration. New models can be added via a single TOML config entry with no code changes:

| Provider | Models | Context | Cost Tier | Best At |
|---|---|---|---|---|
| **Anthropic** | Opus 4.5, Sonnet 4.5, Haiku 4.5 | 200K | $$$ | Refactoring, verification, reasoning, testing |
| **OpenAI** | GPT-4o, o3, o3-mini, GPT-4o-mini | 128K–400K | $$–$$$ | Fast implementation, broad capability, speed |
| **Google** | Gemini 2.5 Pro, Flash, Nano | 1M | $–$$ | Architecture, full-codebase reasoning, multimodal |
| **xAI** | Grok 4, Grok 3 | 131K | $$–$$$ | Reasoning, coding, real-time data |
| **DeepSeek** | V3, R1, Coder V2 | 64K–128K | $ | Cost-efficient coding, high-volume tasks |
| **Meta** | Llama 4 Scout, Maverick | 128K–10M | Free–$ | Local/private deployment, data sovereignty |
| **Mistral** | Large 3, Codestral, Devstral | 32K–256K | $–$$ | Code completion, IDE-speed generation, multilingual |
| **Custom** | Any GGUF, any vLLM-served model | Varies | Free | Domain-specific tasks |

---

## A4. Smart Routing Engine

The Smart Routing Engine replaces fixed model assignment with dynamic, context-aware routing:

| Mode | Behavior | Use Case |
|---|---|---|
| **Quality-First** | Assigns highest-fitness model to each role regardless of cost. | Production-critical features, client-facing code. |
| **Cost-Optimized** | Uses cheapest model that meets a minimum fitness threshold (configurable, default 0.70). | Internal tooling, prototyping, high-volume batch tasks. |
| **Speed-First** | Prioritizes lowest-latency models. Uses Flash/mini variants where available. | Live coding sessions, rapid iteration. |
| **Hybrid** | Quality-first for critical stages (refactor + verify), cost-optimized for earlier stages (architect + implement). | **Default recommended mode.** Best cost/quality balance. |
| **Tournament** | Run all registered models on the same task, score outputs, use the winner. Expensive but produces benchmarking data. | Benchmarking new models, evaluating upgrades. |

CLI interface:

```bash
# Use default smart routing (hybrid mode)
$ triad run --task 'Build entity detection' --route hybrid

# Force specific models per role
$ triad run --task '...' \
    --architect grok-4 \
    --implementer gpt-4o \
    --refactorer claude-opus-4-5 \
    --verifier claude-opus-4-5

# Add a new model (no code change needed)
$ triad models add --provider xai --model grok-4 \
    --api-key-env XAI_API_KEY --benchmark

# Run tournament to benchmark all models
$ triad benchmark --task-suite standard --all-models

# Cost comparison across routing modes
$ triad estimate --task '...' --compare-routes
```

---

## A5. Consensus Protocol for N Models

With a plugin architecture supporting N models, the consensus protocol generalizes:

### A5.1 Pipeline Consensus (3–5 models)

When the pipeline has 3–5 models in distinct roles, consensus works as before: each model can suggest, object, and vote. Majority rules with a designated tiebreaker (configurable, defaults to the Verifier model).

### A5.2 Tournament Consensus (N models)

When running Tournament mode with many models, consensus scales via a scoring protocol:

1. **Independent generation:** All N models produce solutions independently.
2. **Cross-scoring:** Each model scores a sample of other models' solutions (O(N) not O(N²) to control costs).
3. **Weighted aggregation:** Scores are aggregated with weights proportional to the scorer's own fitness score for the review role.
4. **Top-K synthesis:** The top 2–3 scoring solutions are passed to the Verifier for final synthesis.

### A5.3 The Verifier Role Remains Privileged

Regardless of how many models participate, the **Verifier role retains special authority**. The Verifier has unilateral veto on pattern violations, produces the final confidence score, and serves as tiebreaker. The highest-fitness model for verification gets this role by default. If a different model surpasses it on verification benchmarks, the routing engine will reassign automatically.

---

*This addendum supersedes the fixed model assignments in the original spec. All other sections — pipeline workflows, consensus protocol, and implementation phases — remain valid and are enhanced by the model-agnostic architecture described here.*

Any model. Any role. **The protocol is the product.**

# Debate — Rebuttal Round

You are participating in a structured debate between AI models. You have already submitted your own proposal. Now you must review the other proposals and write structured rebuttals.

## Original Task

{{ task }}

{% if context %}
## Additional Context

{{ context }}
{% endif %}

{% if domain_context %}
## Domain-Specific Rules

{{ domain_context }}
{% endif %}

## Your Original Proposal

{{ own_proposal }}

## Other Proposals

{% for model_key, proposal in other_proposals.items() %}
### Proposal from {{ model_key }}

{{ proposal }}

{% endfor %}

## Your Mandate

For each proposal above, write a structured rebuttal. Be fair but rigorous:

1. **Acknowledge Strengths**: What does this approach get right? (1-2 points)
2. **Identify Weaknesses**: Where does this approach fail or underperform? Be specific.
3. **Compare to Your Approach**: On which dimensions is your approach superior?
4. **Concrete Concerns**: Name specific failure modes, edge cases, or maintenance burdens this approach introduces.

You MUST engage with the substance of each proposal. Generic criticism ("this is too complex") without evidence is not acceptable.

## Output Format

For each proposal, structure your rebuttal as:

```
## Rebuttal: <model_key>

**Strengths**: ...
**Weaknesses**: ...
**Comparison**: ...
**Concerns**: ...
```

CONFIDENCE: <0.0-1.0>

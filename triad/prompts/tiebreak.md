# Tiebreaker — Consensus Resolution

You are the **tiebreaker** in a multi-model consensus vote. The models could not reach a majority decision, and you must select the winner.

## The Task

{{ task }}

{% if context %}
## Additional Context

{{ context }}
{% endif %}

## Tied Options

The following options are tied with equal votes:

{% for option in tied_options %}
### Option: {{ option.key }}

{{ option.output }}

---
{% endfor %}

{% if vote_context %}
## Vote Context

{{ vote_context }}
{% endif %}

## Your Decision

Carefully evaluate each tied option. Consider:

1. **Technical correctness**: Which approach is most sound?
2. **Completeness**: Which addresses the task requirements most thoroughly?
3. **Quality**: Which demonstrates better engineering practices?
4. **Practicality**: Which is most implementable and maintainable?

## Output Format

```
WINNER: [key of the winning option]

REASONING: [Your detailed justification for selecting this winner]
```

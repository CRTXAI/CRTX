# Parallel Synthesis

You are producing the final unified output for a multi-model pipeline. One approach was selected as the winner through cross-review and consensus voting. Your job is to take the winning approach and enhance it with the best elements from the other proposals.

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

## Winning Approach

This approach was selected by consensus (model: **{{ winner_model }}**):

{{ winning_output }}

{% if other_outputs %}
## Other Proposals

These approaches were not selected, but may contain valuable ideas worth incorporating:

{% for model_key, output in other_outputs.items() %}
### {{ model_key }}

{{ output }}

{% endfor %}
{% endif %}

## Your Mandate

1. Start from the winning approach as the foundation
2. Identify any superior elements from the other proposals (better error handling, cleaner abstractions, additional edge cases, etc.)
3. Integrate those elements into the winning approach
4. Produce a single, cohesive final output that represents the best of all proposals

Do NOT simply concatenate approaches. The result should read as a single coherent solution.

CONFIDENCE: <0.0-1.0>

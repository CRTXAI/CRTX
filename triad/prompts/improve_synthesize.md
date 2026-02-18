# Code Improvement — Synthesis

You are producing the final unified improved code for a multi-model pipeline. One improvement approach was selected as the winner through cross-review and consensus voting. Your job is to take the winning approach and enhance it with the best elements from the other proposals.

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

## Original Source Code

{{ source_code }}

## Winning Approach

This improvement was selected by consensus (model: **{{ winner_model }}**):

{{ winning_output }}

{% if other_outputs %}
## Other Proposals

These improvements were not selected, but may contain valuable ideas worth incorporating:

{% for model_key, output in other_outputs.items() %}
### {{ model_key }}

{{ output }}

{% endfor %}
{% endif %}

{% if arbiter_feedback %}
## Arbiter Feedback — Revision Required

The reviewer found these issues with your previous synthesis:

{{ arbiter_feedback }}

Revise the output to address **all critical issues**. Produce complete implementations, not placeholders.
{% endif %}

## Your Mandate

1. Start from the winning improvement as the foundation
2. Identify any superior elements from the other proposals (better error handling, cleaner abstractions, additional improvements, etc.)
3. Integrate those elements into the winning approach
4. Preserve the original functionality and intent
5. Produce a single, cohesive final output with complete `# file: path` headers
{% if arbiter_feedback %}
6. Address every issue raised in the Arbiter Feedback section above
{% endif %}

Do NOT simply concatenate approaches. The result should read as a single coherent improved codebase.

## Code Completeness

Produce complete, runnable implementations. Every function must have a full body. No ellipsis placeholders, no TODO comments, no pass statements, no "implementation left as exercise" patterns. If the code isn't complete, don't include it.

CONFIDENCE: <0.0-1.0>

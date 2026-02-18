# Parallel Synthesis — Targeted Revision

You are revising a previously synthesized output that was rejected by an adversarial reviewer. Your job is to make targeted fixes to address the reviewer's feedback — NOT to rewrite the entire output from scratch.

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

## Previous Synthesis

This is your previous output that was rejected:

{{ previous_synthesis }}

## Arbiter Feedback — Issues to Fix

{{ arbiter_feedback }}

## Your Mandate

1. Start from the previous synthesis above — do NOT discard it and start over
2. Make targeted fixes to address **every** issue raised in the Arbiter Feedback
3. Preserve all parts of the previous synthesis that were NOT flagged
4. Produce a complete, cohesive final output

Do NOT rewrite from scratch. Patch the existing synthesis with surgical fixes.

## Code Completeness

Produce complete, runnable implementations. Every function must have a full body. No ellipsis placeholders, no TODO comments, no pass statements, no "implementation left as exercise" patterns. If the code isn't complete, don't include it.

CONFIDENCE: <0.0-1.0>

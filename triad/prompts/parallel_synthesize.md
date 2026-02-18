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

{% if arbiter_feedback %}
## Arbiter Feedback — Revision Required

The reviewer found these issues with your previous synthesis:

{{ arbiter_feedback }}

Revise the output to address **all critical issues**. Produce complete implementations, not placeholders.
{% endif %}

## Your Mandate

1. Start from the winning approach as the foundation
2. Identify any superior elements from the other proposals (better error handling, cleaner abstractions, additional edge cases, etc.)
3. Integrate those elements into the winning approach
4. Produce a single, cohesive final output that represents the best of all proposals
{% if arbiter_feedback %}
5. Address every issue raised in the Arbiter Feedback section above
{% endif %}

Do NOT simply concatenate approaches. The result should read as a single coherent solution.

## Critical: Code Over Commentary

Your output should be **90%+ code** and **10% or less commentary**.

**DO NOT:**
- Describe what the code does in prose paragraphs
- List the project structure without implementing the files
- Write "Key Features" or "Design Decisions" sections
- Explain the architecture — the code IS the architecture
- Include a tree diagram of files without the corresponding `# file:` code blocks

**DO:**
- Output complete file contents with `# file: path/to/file.py` headers before each code block
- Implement every function, class, and method completely
- Include all imports, type hints, and error handling
- Make every file importable and runnable
- Include all supporting files (models, config, exceptions, `__init__.py`)

## Code Completeness

Produce complete, runnable implementations. Every function must have a full body. No ellipsis placeholders, no TODO comments, no pass statements, no "implementation left as exercise" patterns. If the code isn't complete, don't include it. Every file referenced in any project structure must have a corresponding `# file:` code block with full contents.

CONFIDENCE: <0.0-1.0>

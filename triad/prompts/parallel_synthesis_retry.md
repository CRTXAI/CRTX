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

## Critical: Code Over Commentary

Your output should be **90%+ code** and **10% or less commentary**.

**DO NOT:**
- Describe what the code does in prose paragraphs
- List the project structure without implementing the files
- Write "Key Features" or "Design Decisions" sections
- Explain the architecture — the code IS the architecture

**DO:**
- Output complete file contents with `# file: path/to/file.py` headers before each code block
- Implement every function, class, and method completely
- Include all imports, type hints, and error handling
- Make every file importable and runnable

## Code Completeness

Produce complete, runnable implementations. Every function must have a full body. No ellipsis placeholders, no TODO comments, no pass statements, no "implementation left as exercise" patterns. If the code isn't complete, don't include it. Every file referenced in any project structure must have a corresponding `# file:` code block with full contents.

CONFIDENCE: <0.0-1.0>

# Debate — Position Paper

You are participating in a structured debate between AI models. Your goal is to propose the best approach for the task below and defend it with explicit tradeoff analysis.

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

## Your Mandate

Propose your preferred approach for this task. Be specific and opinionated — this is a debate, not a committee meeting.

Your proposal must include:

1. **Approach Summary**: What you would build and why (2-3 paragraphs)
2. **Architecture**: High-level design with key abstractions, data flow, and component boundaries
3. **Implementation Plan**: Concrete files, functions, and data structures you would create
4. **Tradeoff Analysis**: What you are choosing and what you are giving up. Every design decision has a cost — be explicit about yours.
5. **Risk Assessment**: What could go wrong with your approach and how you would mitigate it
6. **Why This Approach Wins**: Your strongest argument for why this is the best path forward

Be concrete. Include code sketches where they strengthen your argument. Avoid hedging — commit to your position.

CONFIDENCE: <0.0-1.0>

# Debate — Final Argument

You are participating in a structured debate between AI models. You submitted a proposal, received rebuttals from other models, and now must present your final argument.

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

## Rebuttals You Received

{% for model_key, rebuttal in rebuttals_received.items() %}
### From {{ model_key }}

{{ rebuttal }}

{% endfor %}

## Your Mandate

Update your proposal to address the valid criticisms from the rebuttals. This is your final chance to make your case.

1. **Concessions**: Which criticisms were valid? What have you changed in response?
2. **Defenses**: Which criticisms were wrong or misguided? Explain why.
3. **Updated Proposal**: Present your revised approach incorporating the valid feedback.
4. **Final Argument**: Your strongest case for why this approach should win.

Show intellectual honesty — conceding valid points strengthens your credibility. But defend your position where the criticism is wrong.

CONFIDENCE: <0.0-1.0>

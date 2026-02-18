# Code Improvement — Cross-Review

You are reviewing an improved version of source code produced by a different AI model.

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

## Improved Version Under Review

This improvement was produced by **{{ reviewer_model }}**:

{{ output_under_review }}

## Your Review

Score this improvement on three dimensions. Be honest and specific — your scores directly influence which approach gets selected.

For each dimension, provide:
1. A score from 1-10
2. A brief justification (1-2 sentences)

### Scoring Criteria

- **Architecture (1-10)**: Are the structural improvements sound? Are abstractions appropriate? Is the result more maintainable?
- **Implementation (1-10)**: Are the changes correct? Do they preserve original functionality? Are new bugs introduced?
- **Quality (1-10)**: Is the improved code clean, well-documented, and following best practices? Are the improvements meaningful?

## Output Format

```
ARCHITECTURE: <score>
<justification>

IMPLEMENTATION: <score>
<justification>

QUALITY: <score>
<justification>

VOTE: <YES or NO>
<Would you recommend this as the winning approach? One sentence explaining why.>
```

CONFIDENCE: <0.0-1.0>

# Parallel Cross-Review

You are reviewing a solution produced by a different AI model for the following task.

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

## Solution Under Review

This solution was produced by **{{ reviewed_model }}**:

{{ solution }}

## Your Review

Score this solution on three dimensions. Be honest and specific — your scores directly influence which approach gets selected.

For each dimension, provide:
1. A score from 1-10
2. A brief justification (1-2 sentences)

### Scoring Criteria

- **Architecture (1-10)**: Is the overall design sound? Are abstractions appropriate? Is the structure maintainable and extensible?
- **Implementation (1-10)**: Is the code correct? Are edge cases handled? Is error handling adequate?
- **Quality (1-10)**: Is the code clean, well-documented, and following best practices? Are tests included?

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

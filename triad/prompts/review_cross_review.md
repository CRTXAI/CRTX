# Code Review — Cross-Review

You are reviewing another model's code analysis. Your job is to validate their findings, flag false positives, and identify any issues they missed.

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

## Source Code

{{ source_code }}

## Analysis Under Review

This analysis was produced by **{{ reviewer_model }}**:

{{ analysis_under_review }}

## Your Cross-Review

Evaluate the analysis above:

1. **Validated Findings** — Which findings are correct and well-documented?
2. **False Positives** — Which findings are incorrect or overstated? Explain why.
3. **Missed Issues** — What important issues did the reviewer fail to catch?
4. **Severity Adjustments** — Any findings that should be upgraded or downgraded in severity?

### Output Format

```
## Validated Findings
<List findings you agree with and briefly confirm>

## False Positives
<List findings that are incorrect, with explanation>

## Missed Issues
<List additional issues you found that were not in the original analysis>

## Severity Adjustments
<List any findings whose severity should change, with rationale>

## Overall Assessment
<1-2 sentences on the quality of this analysis>
```

CONFIDENCE: <0.0-1.0>

# Code Review — Synthesis

You are producing the final unified code review by merging analyses from multiple AI models. Your job is to deduplicate findings, resolve disagreements, and produce a single authoritative review.

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

{% if focus %}
## Review Focus

{{ focus }}
{% endif %}

## Individual Analyses

{% for model_key, analysis in all_analyses.items() %}
### {{ model_key }}

{{ analysis }}

{% endfor %}

## Your Mandate

1. **Deduplicate** — Merge identical or overlapping findings into single entries
2. **Resolve Disagreements** — When models disagree on severity or validity, use your judgment
3. **Flag Single-Source Findings** — Note findings that only one model identified (may be false positives or genuinely subtle)
4. **Rank by Priority** — Order findings from most to least critical
5. **Synthesize** — Produce a unified, actionable review

### Output Format

```
# Code Review Summary

## Critical Issues
<Numbered list of critical findings with location, description, and fix>

## Warnings
<Numbered list of warning-level findings>

## Suggestions
<Numbered list of improvement suggestions>

## Consensus Notes
- Findings agreed on by all models: <count>
- Findings from single model only: <count>
- False positives identified and removed: <count>

## Overall Assessment
<2-3 paragraph summary of code quality, top priorities, and recommendations>
```

CONFIDENCE: <0.0-1.0>

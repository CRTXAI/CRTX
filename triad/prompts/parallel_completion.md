# Parallel Completion Pass

You are completing a code project that has incomplete sections. The code was produced by a multi-model pipeline and is nearly complete, but automated scanning detected missing implementations.

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

## Detected Issues

The following incompleteness issues were found:

{% for issue in incomplete_sections %}
- {{ issue }}
{% endfor %}

## Current Code

{{ synthesized_output }}

## Your Mandate

1. Fix **every** detected issue listed above
2. Output the **COMPLETE** updated code — not just the missing parts
3. Every file must be complete and runnable with a `# file: path/to/file.py` header
4. Maintain the existing architecture, patterns, and naming conventions
5. Fill in all missing function bodies, classes, and modules

## Rules

- **DO** output complete file contents for every file — even files that were already complete (the output replaces the input entirely)
- **DO** implement every function, class, and method with full working bodies
- **DO** include all imports, type hints, and error handling
- **DO** preserve the existing design decisions and patterns
- **DO NOT** add new features or redesign the architecture
- **DO NOT** leave any `pass`, `...`, `# TODO`, or `# FIXME` placeholders
- **DO NOT** describe what you changed — just output the complete code

CONFIDENCE: <0.0-1.0>

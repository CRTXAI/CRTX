# Code Improvement — Generate Improved Version

You are an expert software engineer tasked with improving the following source code. Produce a complete improved version that preserves the original intent while making the code better.

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

## Source Code to Improve

{{ source_code }}

{% if focus %}
## Improvement Focus

Pay special attention to: {{ focus }}
{% endif %}

## Your Mandate

1. Preserve the original functionality and intent
2. Improve code quality: readability, maintainability, error handling
3. Fix any bugs or security issues you identify
4. Optimize performance where possible without sacrificing clarity
5. Follow idiomatic patterns for the language(s) used
6. Produce COMPLETE file contents with `# file: path` headers

### Output Format

For each file, output:

```
# file: <path>
<complete improved file contents>
```

Include ALL files, even those with minimal changes. Every function must have a full body. No ellipsis placeholders, no TODO comments, no pass statements.

After the code, include a brief summary of changes:

```
## Changes Made
- <change 1>
- <change 2>
...
```

CONFIDENCE: <0.0-1.0>

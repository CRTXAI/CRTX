"""Prompt template loader for pipeline role prompts.

Loads Markdown prompt templates from the prompts/ directory and renders
them with Jinja2 variable substitution. Used by the orchestrator to
build system prompts for each pipeline stage.
"""

from __future__ import annotations

from pathlib import Path

from jinja2 import BaseLoader, Environment

# Directory containing the .md prompt template files
_PROMPTS_DIR = Path(__file__).parent


def render_prompt(template_name: str, **variables: object) -> str:
    """Load a prompt template and render it with Jinja2 variables.

    Args:
        template_name: Name of the template file (without .md extension).
                       Must correspond to a file in the prompts/ directory.
        **variables: Template variables to inject (task, context,
                     domain_context, previous_output, etc.).

    Returns:
        The fully rendered prompt string.

    Raises:
        FileNotFoundError: If the template file does not exist.
    """
    path = _PROMPTS_DIR / f"{template_name}.md"
    if not path.exists():
        raise FileNotFoundError(f"Prompt template not found: {path}")

    template_text = path.read_text(encoding="utf-8")

    # Jinja2 env with default Undefined (falsy, renders as empty string)
    # so {% if optional_var %} blocks are skipped when var is not provided
    env = Environment(loader=BaseLoader(), keep_trailing_newline=True)
    template = env.from_string(template_text)
    return template.render(**variables)

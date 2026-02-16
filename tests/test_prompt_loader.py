"""Tests for triad.prompts — Prompt template loading and rendering."""

import pytest

from triad.prompts import render_prompt


class TestRenderPrompt:
    def test_loads_and_renders_architect_template(self):
        result = render_prompt("architect", task="Build a REST API")
        assert "Build a REST API" in result
        assert "Architect" in result

    def test_loads_and_renders_implementer_template(self):
        result = render_prompt(
            "implementer",
            task="Build a REST API",
            previous_output="scaffold goes here",
        )
        assert "Build a REST API" in result
        assert "scaffold goes here" in result

    def test_optional_vars_omitted_gracefully(self):
        """Templates use {% if var %} guards — missing vars should not error."""
        result = render_prompt("architect", task="Build something")
        # Should not contain domain context section when not provided
        assert "Domain-Specific Rules" not in result

    def test_domain_context_rendered_when_provided(self):
        result = render_prompt(
            "architect",
            task="Build a CLI",
            domain_context="Always validate inputs",
        )
        assert "Always validate inputs" in result
        assert "Domain-Specific Rules" in result

    def test_context_rendered_when_provided(self):
        result = render_prompt(
            "architect",
            task="Build a CLI",
            context="This is a Python project using Click",
        )
        assert "This is a Python project using Click" in result

    def test_nonexistent_template_raises(self):
        with pytest.raises(FileNotFoundError, match="Prompt template not found"):
            render_prompt("nonexistent_role", task="anything")

    def test_reconciliation_conditional_in_verifier(self):
        result_disabled = render_prompt(
            "verifier",
            task="Build something",
            previous_output="code",
            architect_output="scaffold",
            reconciliation_enabled=False,
        )
        assert "Implementation Summary" not in result_disabled

        result_enabled = render_prompt(
            "verifier",
            task="Build something",
            previous_output="code",
            architect_output="scaffold",
            reconciliation_enabled=True,
        )
        assert "Implementation Summary" in result_enabled

    def test_arbiter_feedback_rendered_when_provided(self):
        result = render_prompt(
            "architect",
            task="Build something",
            arbiter_feedback="Fix the interface contract",
            retry_number=1,
        )
        assert "Fix the interface contract" in result
        assert "Retry 1 of 2" in result

    def test_arbiter_feedback_omitted_when_not_provided(self):
        result = render_prompt("architect", task="Build something")
        assert "REJECTED" not in result

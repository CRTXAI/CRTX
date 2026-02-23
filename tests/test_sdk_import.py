"""Smoke test: verify the public SDK API imports without error."""

import sys


def test_loop_import():
    from crtx import Loop, LoopResult, LoopStats
    assert Loop is not None
    assert LoopResult is not None
    assert LoopStats is not None


def test_arbiter_import():
    from crtx import Arbiter, ArbiterResult
    assert Arbiter is not None
    assert ArbiterResult is not None


def test_router_import():
    from crtx import Router, RouteDecision, TaskComplexity
    assert Router is not None
    assert RouteDecision is not None
    assert TaskComplexity is not None


def test_providers_import():
    from crtx import ModelProvider, LiteLLMProvider
    assert ModelProvider is not None
    assert LiteLLMProvider is not None


def test_config_import():
    from crtx.config import load_models, load_pipeline_config
    assert load_models is not None
    assert load_pipeline_config is not None


def test_version():
    from crtx import __version__
    assert __version__ == "0.3.0"


def test_no_cli_imports():
    """crtx package must not import typer, rich, or click at module level."""
    cli_packages = {"typer", "rich", "click"}
    pre_existing = cli_packages & set(sys.modules.keys())

    import crtx  # noqa: F401

    newly_loaded = (cli_packages & set(sys.modules.keys())) - pre_existing
    assert not newly_loaded, f"crtx imported CLI packages: {newly_loaded}"

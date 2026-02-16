# Contributing to Triad Orchestrator

Thank you for your interest in contributing to Triad Orchestrator! This document provides guidelines for contributing to the project.

## Contributor License Agreement (CLA)

**All contributors must sign our CLA before any pull request can be merged.** This is enforced automatically via CLA Assistant on GitHub.

When you open your first PR, the CLA Assistant bot will post a comment asking you to sign. It's a one-time process — once signed, it covers all future contributions.

The full CLA text is available in [CLA.md](./CLA.md).

## Getting Started

1. Fork the repository
2. Clone your fork locally
3. Create a virtual environment: `python -m venv .venv && source .venv/bin/activate` (or `.venv\Scripts\Activate.ps1` on Windows)
4. Install dependencies: `pip install -e ".[dev]"`
5. Create a branch for your changes: `git checkout -b your-feature-name`

## Development Setup

Triad requires Python 3.12+ and API keys for at least one supported provider.

```bash
# Copy the example env file and add your keys
cp .env.example .env

# Run the test suite
pytest

# Run the linter
ruff check triad/ tests/
```

## Code Standards

- **Pydantic v2** for all schemas — strict validation, no `Any` types unless absolutely necessary
- **Async-first** — all provider calls and pipeline stages are async
- **Type hints on everything** — full type annotations on all function signatures
- **Docstrings** on all public classes and functions
- **Tests required** — all new code must include pytest tests
- **Ruff** for linting — run `ruff check` before submitting

## Project Structure

All code lives under the `triad/` package. See `CLAUDE.md` for the full project structure and architecture overview.

Key conventions:
- Provider calls always go through LiteLLM, never direct SDK calls
- Schemas live in `triad/schemas/`
- System prompts live in `triad/prompts/` as Markdown files
- Config files use TOML format

## Pull Request Process

1. **Sign the CLA** (automated via CLA Assistant on your first PR)
2. Ensure your code passes `pytest` and `ruff check`
3. Include tests for any new functionality
4. Update documentation if your change affects the public API or CLI
5. Keep PRs focused — one feature or fix per PR
6. Write a clear PR description explaining what and why

## What to Contribute

We welcome contributions in these areas:

- **Bug fixes** — with a test that reproduces the issue
- **New provider integrations** — via TOML config + LiteLLM (usually zero code needed)
- **Pipeline mode implementations** — parallel, debate, tournament
- **CLI improvements** — Rich output enhancements, new commands
- **Documentation** — examples, guides, tutorials
- **Test coverage** — additional test cases for edge cases

## Reporting Issues

- Use GitHub Issues for bug reports and feature requests
- Include reproduction steps for bugs
- For security issues, email adam@nexusinsure.ai directly — do not open a public issue

## Code of Conduct

Be respectful. Be constructive. We're building something useful together.

## License

By contributing to Triad Orchestrator, you agree that your contributions will be licensed under the Apache License 2.0, subject to the terms of the CLA.

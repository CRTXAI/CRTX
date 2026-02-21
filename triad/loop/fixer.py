"""Code fixer — targeted fix prompts from test failures."""

from __future__ import annotations

import ast
import time
from dataclasses import dataclass
from pathlib import Path

from triad.loop.generator import extract_files_from_content
from triad.loop.router import RouteDecision
from triad.loop.test_runner import TestReport
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.pipeline import ModelConfig


FIX_SYSTEM_PROMPT = """\
You are an expert Python debugger. Fix the specific issues reported below.

Rules:
1. Fix ONLY the issues listed. Do not restructure, rename, or reorganize.
2. Output the complete fixed files using # file: path headers.
3. Include the FULL file content, not just changed lines.
4. Do NOT output files that don't need changes.
5. Preserve all existing type hints, tests, and functionality.
6. No explanations — just the fixed code.
"""


@dataclass
class FixResult:
    """Result of a fix iteration."""

    files: dict[str, str]
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    error: str = ""


class CodeFixer:
    """Generate targeted fixes from test failure reports."""

    async def fix(
        self,
        files: dict[str, str],
        test_report: TestReport,
        route: RouteDecision,
        registry: dict[str, ModelConfig],
        iteration: int,
    ) -> FixResult:
        """Feed failure context to the model and get fixed files back.

        Only sends broken files as "to fix" and the rest as read-only
        context, keeping the prompt focused and cheap.
        """
        model_config = self._find_model_config(route.model, registry)
        if model_config is None:
            return FixResult(
                files=files,
                error=f"Model '{route.model}' not found in registry",
            )

        failure_summary = test_report.failure_summary()
        is_collection_failure = self._is_collection_failure(test_report)
        phantom_tests = self._find_phantom_tests(files)
        broken_files = self._identify_broken_files(files, test_report)
        context_files = {k: v for k, v in files.items() if k not in broken_files}

        prompt_parts = [
            f"Fix iteration {iteration + 1}. The following code has failures.\n",
        ]
        if is_collection_failure:
            prompt_parts.append(
                "IMPORTANT: Multiple test files exist with incompatible APIs. "
                "Reconcile all test files to match the source module APIs.\n\n"
            )
        if phantom_tests:
            names = ", ".join(sorted(phantom_tests))
            prompt_parts.append(
                f"WARNING: These test files reference APIs that don't exist in "
                f"the source: {names}. Either fix the tests to match the source "
                f"API or remove the incompatible test file.\n\n"
            )
        prompt_parts.extend([
            "FAILURES:\n",
            failure_summary,
            "\n\nFILES THAT NEED FIXING:\n",
        ])
        for fp, code in broken_files.items():
            prompt_parts.append(f"\n# file: {fp}\n```python\n{code}\n```\n")

        if context_files:
            prompt_parts.append("\nFULL CONTEXT (read-only — modify only if an import is missing):\n")
            for fp, code in context_files.items():
                prompt_parts.append(f"\n# file: {fp}\n```python\n{code}\n```\n")

        user_content = "".join(prompt_parts)

        provider = LiteLLMProvider(model_config)
        messages = [{"role": "user", "content": user_content}]

        start = time.monotonic()
        try:
            agent_msg = await provider.complete(
                messages, FIX_SYSTEM_PROMPT, timeout=300,
            )
        except Exception as exc:
            return FixResult(files=files, error=str(exc),
                             duration_seconds=time.monotonic() - start)
        elapsed = time.monotonic() - start

        # Extract fixed files and merge with unchanged
        fixed_files = extract_files_from_content(agent_msg.content)
        for block in agent_msg.code_blocks:
            from triad.loop.generator import _sanitise_filepath
            fp = _sanitise_filepath(block.filepath)
            if fp not in fixed_files and block.content.strip():
                fixed_files[fp] = block.content

        merged = {**files}
        for name, content in fixed_files.items():
            merged[name] = content

        token_usage = agent_msg.token_usage
        return FixResult(
            files=merged,
            cost=token_usage.cost if token_usage else 0.0,
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            duration_seconds=elapsed,
        )

    @staticmethod
    def _is_collection_failure(report: TestReport) -> bool:
        """Check if the test report indicates a pytest collection failure."""
        if report.tests.passed:
            return False
        details = report.tests.details
        return "COLLECTION FAILURE" in details or "pytest crashed" in details

    @staticmethod
    def _identify_broken_files(
        files: dict[str, str], report: TestReport,
    ) -> dict[str, str]:
        """Determine which files need fixing from the test report."""
        broken: set[str] = set()

        # On collection failure, include ALL test files — they likely have
        # incompatible APIs that need reconciliation against the source.
        collection_failure = CodeFixer._is_collection_failure(report)
        if collection_failure:
            for fp in files:
                if "test" in fp.lower() and fp.endswith(".py"):
                    broken.add(fp)

        # Parse error details to find filenames
        for check in (report.parse, report.imports, report.tests, report.runs):
            if check.passed:
                continue
            for line in check.details.splitlines():
                # Match patterns like "filename.py:10:" or "filename.py:"
                for fp in files:
                    if fp in line:
                        broken.add(fp)

        # If nothing specific found, include all non-test files
        if not broken:
            broken = {
                fp for fp in files
                if fp.endswith(".py") and "test" not in fp.lower()
            }

        # Always include test files if tests failed
        if not report.tests.passed:
            for fp in files:
                if "test" in fp.lower() and fp.endswith(".py"):
                    broken.add(fp)

        # Check for phantom API references in test files
        phantom = CodeFixer._find_phantom_tests(files)
        broken.update(phantom)

        return {k: v for k, v in files.items() if k in broken}

    @staticmethod
    def _find_phantom_tests(files: dict[str, str]) -> set[str]:
        """Find test files that reference APIs not present in the source.

        Returns set of test file paths that have phantom references.
        """
        # Build an index of all names exported by source files
        source_exports: dict[str, set[str]] = {}  # module_stem -> {names}
        for fp, code in files.items():
            if not fp.endswith(".py") or "test" in fp.lower():
                continue
            stem = Path(fp).stem
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue
            names: set[str] = set()
            for node in ast.iter_child_nodes(tree):
                if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    names.add(node.name)
                elif isinstance(node, ast.ClassDef):
                    names.add(node.name)
                elif isinstance(node, ast.Assign):
                    for target in node.targets:
                        if isinstance(target, ast.Name):
                            names.add(target.id)
            source_exports[stem] = names

        if not source_exports:
            return set()

        phantom_files: set[str] = set()

        for fp, code in files.items():
            if not fp.endswith(".py") or "test" not in fp.lower():
                continue
            try:
                tree = ast.parse(code)
            except SyntaxError:
                continue

            # Find imports from local source modules
            for node in ast.walk(tree):
                if isinstance(node, ast.ImportFrom) and node.module and node.level == 0:
                    module_stem = node.module.split(".")[0]
                    if module_stem not in source_exports:
                        continue
                    available = source_exports[module_stem]
                    for alias in node.names:
                        if alias.name != "*" and alias.name not in available:
                            phantom_files.add(fp)
                            break

        return phantom_files

    @staticmethod
    def _find_model_config(
        model_id: str, registry: dict[str, ModelConfig],
    ) -> ModelConfig | None:
        for key, cfg in registry.items():
            if cfg.model == model_id:
                return cfg
        if model_id in registry:
            return registry[model_id]
        return None

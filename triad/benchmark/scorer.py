"""Benchmark scoring engine.

Evaluates generated code across five dimensions and produces a weighted
composite score:

  parse_rate  (20%) — fraction of .py files that pass ast.parse
  runs        (20%) — entry point executes without error (configurable timeout)
  tests       (25%) — pytest pass rate (configurable timeout)
  type_hints  (15%) — fraction of non-test functions with return annotations
  imports     (20%) — fraction of files whose imports resolve
"""

from __future__ import annotations

import ast
import os
import re
import subprocess
import sys
import tempfile
from dataclasses import dataclass, field
from pathlib import Path


# Weights for composite score (must sum to 1.0)
_WEIGHTS = {
    "parse_rate": 0.20,
    "runs": 0.20,
    "tests": 0.25,
    "type_hints": 0.15,
    "imports": 0.20,
}


@dataclass
class ScoreBreakdown:
    """Detailed scoring results for one benchmark run."""

    parse_rate: float = 0.0       # 0.0–1.0
    runs: float = 0.0             # 0.0 or 1.0
    tests: float = 0.0            # 0.0–1.0 (pass_count / total_count)
    test_passed: int = 0
    test_failed: int = 0
    test_errors: int = 0
    type_hints: float = 0.0       # 0.0–1.0
    imports: float = 0.0          # 0.0–1.0
    composite: float = 0.0        # weighted total
    errors: list[str] = field(default_factory=list)


class BenchmarkScorer:
    """Score a set of generated files for a benchmark prompt."""

    def __init__(
        self,
        files: dict[str, str],
        entry_point: str = "",
        *,
        pytest_timeout: int = 60,
        entry_timeout: int = 30,
    ) -> None:
        self._files = files
        self._entry_point = entry_point
        self._pytest_timeout = pytest_timeout
        self._entry_timeout = entry_timeout

    def score_all(self) -> ScoreBreakdown:
        """Run all checks and compute the composite score."""
        breakdown = ScoreBreakdown()

        breakdown.parse_rate = self.check_parse_rate(breakdown)
        breakdown.runs = self.check_runs(breakdown)
        breakdown.tests = self.check_tests(breakdown)
        breakdown.type_hints = self.check_type_hints(breakdown)
        breakdown.imports = self.check_imports(breakdown)

        breakdown.composite = (
            _WEIGHTS["parse_rate"] * breakdown.parse_rate
            + _WEIGHTS["runs"] * breakdown.runs
            + _WEIGHTS["tests"] * breakdown.tests
            + _WEIGHTS["type_hints"] * breakdown.type_hints
            + _WEIGHTS["imports"] * breakdown.imports
        )
        return breakdown

    def check_parse_rate(self, breakdown: ScoreBreakdown) -> float:
        """Try ast.parse on every .py file, return fraction that parse."""
        py_files = {k: v for k, v in self._files.items() if k.endswith(".py")}
        if not py_files:
            return 0.0

        parsed = 0
        for filepath, content in py_files.items():
            try:
                ast.parse(content, filename=filepath)
                parsed += 1
            except SyntaxError as exc:
                breakdown.errors.append(f"SyntaxError in {filepath}: {exc}")

        return parsed / len(py_files)

    def check_runs(self, breakdown: ScoreBreakdown) -> float:
        """Execute the entry point with a configurable timeout. Returns 1.0 or 0.0."""
        if not self._entry_point:
            return 0.0

        # Find the entry point file — try exact match first, then auto-detect
        entry_filepath = self._find_entry_point()

        if entry_filepath is None:
            breakdown.errors.append(
                f"Entry point '{self._entry_point}' not found in generated files"
            )
            return 0.0

        return self._run_in_temp(entry_filepath, breakdown, timeout=self._entry_timeout)

    def _find_entry_point(self) -> str | None:
        """Locate the entry point file by name or auto-detection.

        Search order:
        1. Exact basename match (e.g. "coordinator.py")
        2. Exact full-path match
        3. Auto-detect: file containing ``if __name__ == "__main__"``
        4. Auto-detect: file with the most imports from other generated files
        """
        py_files = {k: v for k, v in self._files.items() if k.endswith(".py")}

        # 1-2: Exact match on the configured entry_point name
        for filepath, content in py_files.items():
            if Path(filepath).name == self._entry_point:
                return filepath
            if filepath == self._entry_point:
                return filepath

        # 3: Find file with if __name__ == "__main__" (skip test files)
        main_guard_re = re.compile(
            r'''if\s+__name__\s*==\s*['"]__main__['"]'''
        )
        candidates = []
        for filepath, content in py_files.items():
            if "test" in filepath.lower():
                continue
            if main_guard_re.search(content):
                candidates.append(filepath)

        if len(candidates) == 1:
            return candidates[0]

        # 4: Among candidates (or all non-test files), pick the one that
        # imports the most other generated modules — likely the top-level entry
        if not candidates:
            candidates = [
                fp for fp in py_files
                if "test" not in fp.lower()
                and "__init__" not in fp
            ]

        local_modules = self._build_local_module_set()

        best_fp = None
        best_count = -1
        for filepath in candidates:
            content = py_files[filepath]
            try:
                tree = ast.parse(content, filename=filepath)
            except SyntaxError:
                continue
            count = 0
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        if alias.name.split(".")[0] in local_modules:
                            count += 1
                elif isinstance(node, ast.ImportFrom) and node.module:
                    if node.module.split(".")[0] in local_modules:
                        count += 1
            if count > best_count:
                best_count = count
                best_fp = filepath

        return best_fp

    def check_tests(self, breakdown: ScoreBreakdown) -> float:
        """Run pytest on test files with a configurable timeout, return pass rate."""
        test_files = {
            k: v for k, v in self._files.items()
            if "test" in k.lower() and k.endswith(".py")
        }
        if not test_files:
            breakdown.errors.append("No test files found")
            return 0.0

        timeout = self._pytest_timeout

        # Write all files to a temp dir and run pytest
        with tempfile.TemporaryDirectory(prefix="crtx_bench_") as tmpdir:
            tmp = Path(tmpdir)
            for filepath, content in self._files.items():
                dest = tmp / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            # Build PYTHONPATH so package-relative imports resolve
            env = self._build_env(tmp)

            try:
                result = subprocess.run(
                    [sys.executable, "-m", "pytest", str(tmp), "-v", "--tb=short", "-q"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(tmp),
                    env=env,
                )
            except subprocess.TimeoutExpired:
                breakdown.errors.append(f"pytest timed out ({timeout}s)")
                return 0.0
            except FileNotFoundError:
                breakdown.errors.append("pytest not found")
                return 0.0

            # Parse pytest output for pass/fail counts
            return self._parse_pytest_output(result, breakdown)

    def check_type_hints(self, breakdown: ScoreBreakdown) -> float:
        """AST scan: fraction of non-test function defs with return annotations.

        Skips functions named ``test_*`` and methods inside classes named
        ``Test*``, since ``-> None`` on test methods is cosmetic rather
        than a meaningful quality signal.
        """
        py_files = {k: v for k, v in self._files.items() if k.endswith(".py")}
        if not py_files:
            return 0.0

        total_funcs = 0
        annotated_funcs = 0

        for filepath, content in py_files.items():
            try:
                tree = ast.parse(content, filename=filepath)
            except SyntaxError:
                continue

            for node in _walk_non_test_funcs(tree):
                total_funcs += 1
                if node.returns is not None:
                    annotated_funcs += 1

        if total_funcs == 0:
            return 0.0
        return annotated_funcs / total_funcs

    def check_imports(self, breakdown: ScoreBreakdown) -> float:
        """Check that imports in each .py file resolve to local files or stdlib.

        Uses AST-based resolution: extracts top-level module names from
        Import/ImportFrom nodes and checks whether they correspond to files
        in the generated output (as modules, packages, or directory names)
        or are available as stdlib/installed packages.
        """
        py_files = {k: v for k, v in self._files.items() if k.endswith(".py")}
        if not py_files:
            return 0.0

        local_modules = self._build_local_module_set()

        succeeded = 0
        for filepath, content in py_files.items():
            try:
                tree = ast.parse(content, filename=filepath)
            except SyntaxError:
                continue

            all_ok = True
            for node in ast.walk(tree):
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top = alias.name.split(".")[0]
                        if not self._module_available(top, local_modules):
                            all_ok = False
                            breakdown.errors.append(
                                f"Import '{alias.name}' in {filepath} not available"
                            )
                elif isinstance(node, ast.ImportFrom):
                    if node.module:
                        top = node.module.split(".")[0]
                        if not self._module_available(top, local_modules):
                            all_ok = False
                            breakdown.errors.append(
                                f"Import '{node.module}' in {filepath} not available"
                            )
            if all_ok:
                succeeded += 1

        return succeeded / len(py_files)

    # ── Helpers ──────────────────────────────────────────────────

    def _run_in_temp(
        self, entry_filepath: str, breakdown: ScoreBreakdown, timeout: int,
    ) -> float:
        """Write code + supporting files to a temp dir and run the entry point."""
        with tempfile.TemporaryDirectory(prefix="crtx_bench_") as tmpdir:
            tmp = Path(tmpdir)

            # Write all generated files so imports resolve
            for filepath, content in self._files.items():
                dest = tmp / filepath
                dest.parent.mkdir(parents=True, exist_ok=True)
                dest.write_text(content, encoding="utf-8")

            entry_path = tmp / entry_filepath
            if not entry_path.exists():
                breakdown.errors.append(f"Entry point file not written: {entry_filepath}")
                return 0.0

            # Build PYTHONPATH so package-relative imports resolve
            env = self._build_env(tmp)

            try:
                result = subprocess.run(
                    [sys.executable, str(entry_path), "--help"],
                    capture_output=True,
                    text=True,
                    timeout=timeout,
                    cwd=str(tmp),
                    env=env,
                )
                if result.returncode == 0:
                    return 1.0
                breakdown.errors.append(
                    f"Entry point exited with code {result.returncode}: "
                    f"{result.stderr[:200]}"
                )
                return 0.0
            except subprocess.TimeoutExpired:
                breakdown.errors.append(f"Entry point timed out ({timeout}s)")
                return 0.0

    def _parse_pytest_output(
        self, result: subprocess.CompletedProcess, breakdown: ScoreBreakdown,
    ) -> float:
        """Parse pytest -q output to extract pass/fail counts."""
        output = result.stdout + "\n" + result.stderr

        # pytest -q summary line: "5 passed, 2 failed" or "5 passed"
        passed_match = re.search(r"(\d+)\s+passed", output)
        failed_match = re.search(r"(\d+)\s+failed", output)
        error_match = re.search(r"(\d+)\s+error", output)

        passed = int(passed_match.group(1)) if passed_match else 0
        failed = int(failed_match.group(1)) if failed_match else 0
        errors = int(error_match.group(1)) if error_match else 0

        breakdown.test_passed = passed
        breakdown.test_failed = failed
        breakdown.test_errors = errors

        total = passed + failed + errors
        if total == 0:
            if result.returncode != 0:
                breakdown.errors.append(
                    f"pytest failed with no test results: {result.stderr[:300]}"
                )
            return 0.0

        return passed / total

    def _build_local_module_set(self) -> set[str]:
        """Build the set of top-level module/package names from generated files.

        Recognises both flat modules (``agent.py`` → ``agent``) and packages
        (``src/multi_agent_system/__init__.py`` → ``src``, ``multi_agent_system``).
        Also adds every directory name that contains a .py file, since models
        often generate package structures.
        """
        modules: set[str] = set()
        for filepath in self._files:
            if not filepath.endswith(".py"):
                continue
            p = Path(filepath)
            # Flat module: stem of the file itself
            modules.add(p.stem)
            # Every directory component is a potential package root
            for part in p.parts[:-1]:
                modules.add(part)
        return modules

    @staticmethod
    def _build_env(output_dir: Path) -> dict[str, str]:
        """Build an env dict with PYTHONPATH covering all source roots.

        Adds the output directory itself plus any ``src/`` subdirectory
        so that both ``import agent`` and ``from multi_agent_system import ...``
        resolve regardless of how the model structured its output.
        """
        env = os.environ.copy()
        paths = [str(output_dir)]
        src_dir = output_dir / "src"
        if src_dir.is_dir():
            paths.append(str(src_dir))
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))
        return env

    @staticmethod
    def _module_available(module_name: str, local_modules: set[str]) -> bool:
        """Check if a module exists locally or as stdlib/installed package.

        Checks the local generated file set first (cheap), then falls back
        to importlib for stdlib and installed packages.
        """
        if module_name in local_modules:
            return True
        import importlib.util

        try:
            spec = importlib.util.find_spec(module_name)
            return spec is not None
        except (ModuleNotFoundError, ValueError):
            return False


def _walk_non_test_funcs(tree: ast.Module) -> list[ast.FunctionDef | ast.AsyncFunctionDef]:
    """Return function defs that are NOT test methods.

    Skips ``test_*`` functions at any level, and ALL methods inside
    ``Test*`` classes. Handles the parent-class context that plain
    ``ast.walk`` cannot provide.
    """
    results: list[ast.FunctionDef | ast.AsyncFunctionDef] = []

    for node in ast.iter_child_nodes(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            if not node.name.startswith("test_"):
                results.append(node)
        elif isinstance(node, ast.ClassDef):
            is_test_class = node.name.startswith("Test")
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if is_test_class or child.name.startswith("test_"):
                        continue
                    results.append(child)
        else:
            # Recurse into other top-level constructs (if blocks, etc.)
            for child in ast.walk(node):
                if isinstance(child, (ast.FunctionDef, ast.AsyncFunctionDef)):
                    if not child.name.startswith("test_"):
                        results.append(child)
    return results

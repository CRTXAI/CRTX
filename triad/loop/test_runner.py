"""Test runner — local quality checks on generated code.

Runs in order: parse → imports → pyflakes → pytest → entry point.
All checks are free (local execution, no API calls).
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


@dataclass
class CheckResult:
    """Result of a single check."""

    passed: bool = True
    details: str = ""


@dataclass
class TestReport:
    """Aggregated results from all checks."""

    parse: CheckResult = field(default_factory=CheckResult)
    imports: CheckResult = field(default_factory=CheckResult)
    static: CheckResult = field(default_factory=CheckResult)
    tests: CheckResult = field(default_factory=CheckResult)
    runs: CheckResult = field(default_factory=CheckResult)
    test_passed: int = 0
    test_failed: int = 0
    test_errors: int = 0

    @property
    def all_pass(self) -> bool:
        return (
            self.parse.passed
            and self.imports.passed
            and self.tests.passed
            and self.runs.passed
        )

    def failure_summary(self) -> str:
        """Structured summary of only what failed — feeds into the Fix phase."""
        lines: list[str] = []
        if not self.parse.passed:
            lines.append(f"SYNTAX ERRORS:\n{self.parse.details}")
        if not self.imports.passed:
            lines.append(f"IMPORT ERRORS:\n{self.imports.details}")
        if not self.static.passed:
            lines.append(f"STATIC ANALYSIS:\n{self.static.details}")
        if not self.tests.passed:
            lines.append(f"TEST FAILURES:\n{self.tests.details}")
        if not self.runs.passed:
            lines.append(f"ENTRY POINT FAILED:\n{self.runs.details}")
        return "\n\n".join(lines)


class TestRunner:
    """Run all automated quality checks on generated code."""

    def __init__(
        self,
        *,
        pytest_timeout: int = 60,
        entry_timeout: int = 30,
    ) -> None:
        self._pytest_timeout = pytest_timeout
        self._entry_timeout = entry_timeout

    async def run_all(
        self, files: dict[str, str], work_dir: Path | None = None,
    ) -> TestReport:
        """Write files to a temp dir and run all checks."""
        report = TestReport()

        if not files:
            report.parse = CheckResult(passed=False, details="No files to test")
            return report

        # Use a temp dir if no work_dir provided
        if work_dir is not None:
            self._write_files(files, work_dir)
            self._create_init_files(work_dir)
            self._run_checks(files, work_dir, report)
        else:
            with tempfile.TemporaryDirectory(prefix="crtx_loop_") as tmpdir:
                tmp = Path(tmpdir)
                self._write_files(files, tmp)
                self._create_init_files(tmp)
                self._run_checks(files, tmp, report)

        return report

    def _run_checks(
        self, files: dict[str, str], work_dir: Path, report: TestReport,
    ) -> None:
        """Run all checks in order."""
        # 1. Parse check
        report.parse = self._check_parse(files)
        if not report.parse.passed:
            # If files don't parse, other checks will fail meaninglessly
            report.imports = CheckResult(passed=False, details="Skipped (parse errors)")
            report.tests = CheckResult(passed=False, details="Skipped (parse errors)")
            report.runs = CheckResult(passed=False, details="Skipped (parse errors)")
            return

        # 2. Import check
        report.imports = self._check_imports(files)

        # 3. Static analysis (pyflakes)
        report.static = self._run_pyflakes(work_dir)

        # 4. Pytest
        test_result = self._run_pytest(files, work_dir, pytest_timeout=self._pytest_timeout)
        report.tests = test_result
        report.test_passed = _extract_count(test_result.details, "passed")
        report.test_failed = _extract_count(test_result.details, "failed")
        report.test_errors = _extract_count(test_result.details, "error")

        # 5. Entry point execution
        report.runs = self._run_entry_point(files, work_dir, entry_timeout=self._entry_timeout)

    @staticmethod
    def _write_files(files: dict[str, str], work_dir: Path) -> None:
        """Write all generated files to the work directory."""
        for filepath, content in files.items():
            dest = work_dir / filepath
            dest.parent.mkdir(parents=True, exist_ok=True)
            dest.write_text(content, encoding="utf-8")

    @staticmethod
    def _create_init_files(work_dir: Path) -> None:
        """Create missing __init__.py files so package imports resolve."""
        for py_file in work_dir.rglob("*.py"):
            parent = py_file.parent
            while parent != work_dir:
                init = parent / "__init__.py"
                if not init.exists():
                    init.write_text("", encoding="utf-8")
                parent = parent.parent

    @staticmethod
    def _check_parse(files: dict[str, str]) -> CheckResult:
        """Try ast.parse on every .py file."""
        py_files = {k: v for k, v in files.items() if k.endswith(".py")}
        if not py_files:
            return CheckResult(passed=True, details="No Python files")

        errors: list[str] = []
        for filepath, content in py_files.items():
            try:
                ast.parse(content, filename=filepath)
            except SyntaxError as exc:
                errors.append(f"{filepath}:{exc.lineno}: {exc.msg}")

        if errors:
            return CheckResult(passed=False, details="\n".join(errors))
        return CheckResult(passed=True, details=f"{len(py_files)} files parsed OK")

    @staticmethod
    def _check_imports(files: dict[str, str]) -> CheckResult:
        """AST-based import validation against the generated file set."""
        py_files = {k: v for k, v in files.items() if k.endswith(".py")}
        if not py_files:
            return CheckResult(passed=True)

        # Build set of local module/package names
        local_modules: set[str] = set()
        for filepath in files:
            if not filepath.endswith(".py"):
                continue
            p = Path(filepath)
            local_modules.add(p.stem)
            for part in p.parts[:-1]:
                local_modules.add(part)

        import importlib.util

        errors: list[str] = []
        for filepath, content in py_files.items():
            try:
                tree = ast.parse(content, filename=filepath)
            except SyntaxError:
                continue

            for node in ast.walk(tree):
                top_module = None
                if isinstance(node, ast.Import):
                    for alias in node.names:
                        top_module = alias.name.split(".")[0]
                elif isinstance(node, ast.ImportFrom):
                    if node.module and node.level == 0:
                        top_module = node.module.split(".")[0]

                if top_module is None:
                    continue
                if top_module in local_modules:
                    continue
                # Check stdlib / installed
                try:
                    spec = importlib.util.find_spec(top_module)
                    if spec is not None:
                        continue
                except (ModuleNotFoundError, ValueError):
                    pass
                errors.append(f"{filepath}: import '{top_module}' not found")

        if errors:
            return CheckResult(passed=False, details="\n".join(errors))
        return CheckResult(passed=True, details="All imports resolve")

    @staticmethod
    def _run_pyflakes(work_dir: Path) -> CheckResult:
        """Run pyflakes for undefined names and unused imports."""
        try:
            result = subprocess.run(
                [sys.executable, "-m", "pyflakes", str(work_dir)],
                capture_output=True,
                text=True,
                timeout=30,
                cwd=str(work_dir),
            )
        except (subprocess.TimeoutExpired, FileNotFoundError):
            return CheckResult(passed=True, details="pyflakes unavailable")

        output = result.stdout.strip()
        if not output:
            return CheckResult(passed=True, details="No pyflakes issues")

        # Filter to only real errors (undefined names), skip unused imports
        real_errors = [
            line for line in output.splitlines()
            if "undefined name" in line
        ]
        if real_errors:
            return CheckResult(passed=False, details="\n".join(real_errors))
        return CheckResult(passed=True, details=output[:200])

    @staticmethod
    def _run_pytest(
        files: dict[str, str], work_dir: Path, *, pytest_timeout: int = 60,
    ) -> CheckResult:
        """Run pytest with PYTHONPATH set for package imports."""
        test_files = [k for k in files if "test" in k.lower() and k.endswith(".py")]
        if not test_files:
            return CheckResult(passed=True, details="No test files (0 passed)")

        env = os.environ.copy()
        paths = [str(work_dir)]
        src_dir = work_dir / "src"
        if src_dir.is_dir():
            paths.append(str(src_dir))
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(work_dir),
                 "-v", "--tb=short", "-q", "--no-header"],
                capture_output=True,
                text=True,
                timeout=pytest_timeout,
                cwd=str(work_dir),
                env=env,
            )
        except subprocess.TimeoutExpired:
            return CheckResult(passed=False, details=f"pytest timed out ({pytest_timeout}s)")
        except FileNotFoundError:
            return CheckResult(passed=True, details="pytest not installed")

        output = result.stdout + "\n" + result.stderr

        passed = _extract_count(output, "passed")
        failed = _extract_count(output, "failed")
        errors = _extract_count(output, "error")
        total = passed + failed + errors

        summary = f"{passed} passed, {failed} failed, {errors} errors"
        if total == 0 and result.returncode != 0:
            # pytest crashed before collecting tests — fall back to
            # running each test file individually so we get per-file
            # diagnostics the fixer can act on.
            return _run_pytest_per_file(test_files, work_dir, env, pytest_timeout)

        if failed > 0 or errors > 0:
            # Include the failure output for the fixer
            return CheckResult(passed=False, details=f"{summary}\n\n{output[-2000:]}")
        return CheckResult(passed=True, details=summary)

    @staticmethod
    def _run_entry_point(
        files: dict[str, str], work_dir: Path, *, entry_timeout: int = 30,
    ) -> CheckResult:
        """Find and execute the entry point."""
        # Find entry point: file with if __name__ == "__main__"
        main_re = re.compile(r'''if\s+__name__\s*==\s*['"]__main__['"]''')
        py_files = {k: v for k, v in files.items()
                    if k.endswith(".py") and "test" not in k.lower()}

        entry = None
        for filepath, content in py_files.items():
            if main_re.search(content):
                entry = filepath
                break

        if entry is None:
            # Try common names
            for name in ("main.py", "app.py", "cli.py"):
                for filepath in py_files:
                    if Path(filepath).name == name:
                        entry = filepath
                        break
                if entry:
                    break

        if entry is None:
            return CheckResult(passed=True, details="No entry point found (skipped)")

        env = os.environ.copy()
        paths = [str(work_dir)]
        src_dir = work_dir / "src"
        if src_dir.is_dir():
            paths.append(str(src_dir))
        existing = env.get("PYTHONPATH", "")
        env["PYTHONPATH"] = os.pathsep.join(paths + ([existing] if existing else []))

        entry_path = work_dir / entry
        try:
            result = subprocess.run(
                [sys.executable, str(entry_path), "--help"],
                capture_output=True,
                text=True,
                timeout=entry_timeout,
                cwd=str(work_dir),
                env=env,
            )
            if result.returncode == 0:
                return CheckResult(passed=True, details=f"{entry} runs OK")
            return CheckResult(
                passed=False,
                details=f"{entry} exited {result.returncode}: {result.stderr[:300]}",
            )
        except subprocess.TimeoutExpired:
            return CheckResult(passed=False, details=f"{entry} timed out ({entry_timeout}s)")


def _run_pytest_per_file(
    test_files: list[str],
    work_dir: Path,
    env: dict[str, str],
    pytest_timeout: int = 60,
) -> CheckResult:
    """Run pytest on each test file individually and aggregate results.

    Called when a combined pytest run collects 0 tests (collection failure).
    Reports per-file pass/fail so the fixer knows which file is broken.
    """
    total_passed = 0
    total_failed = 0
    total_errors = 0
    per_file_details: list[str] = []

    for tf in test_files:
        tf_path = work_dir / tf
        if not tf_path.exists():
            continue

        try:
            result = subprocess.run(
                [sys.executable, "-m", "pytest", str(tf_path),
                 "-v", "--tb=short", "-q", "--no-header"],
                capture_output=True,
                text=True,
                timeout=pytest_timeout,
                cwd=str(work_dir),
                env=env,
            )
        except subprocess.TimeoutExpired:
            per_file_details.append(f"{tf}: COLLECTION FAILURE\npytest timed out ({pytest_timeout}s)")  # noqa: E501
            continue

        output = result.stdout + "\n" + result.stderr
        p = _extract_count(output, "passed")
        f = _extract_count(output, "failed")
        e = _extract_count(output, "error")
        file_total = p + f + e

        if file_total == 0 and result.returncode != 0:
            # This specific file failed to collect
            detail = result.stderr[:500] if result.stderr else result.stdout[:500]
            per_file_details.append(f"{tf}: COLLECTION FAILURE\n{detail}")
            total_errors += 1
        elif f > 0 or e > 0:
            per_file_details.append(
                f"{tf}: {p} passed, {f} failed, {e} errors\n{output[-1000:]}"
            )
            total_passed += p
            total_failed += f
            total_errors += e
        else:
            total_passed += p
            per_file_details.append(f"{tf}: {p} passed")

    summary = f"{total_passed} passed, {total_failed} failed, {total_errors} errors"
    all_details = summary + "\n\n" + "\n\n".join(per_file_details)

    if total_failed > 0 or total_errors > 0 or not per_file_details:
        return CheckResult(passed=False, details=all_details)
    return CheckResult(passed=True, details=summary)


def _extract_count(text: str, label: str) -> int:
    """Extract a count like '5 passed' or '2 failed' from pytest output."""
    match = re.search(rf"(\d+)\s+{label}", text)
    return int(match.group(1)) if match else 0

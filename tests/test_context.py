"""Tests for the context injection module (Day 11).

Covers: CodeScanner, ContextBuilder, ContextPruner, schemas,
orchestrator integration, and CLI flags.
"""

from __future__ import annotations

import textwrap
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock, patch

import pytest

from triad.context.builder import ContextBuilder
from triad.context.pruner import ContextPruner
from triad.context.scanner import CodeScanner
from triad.schemas.context import (
    ContextResult,
    FunctionSignature,
    ProjectProfile,
    ScannedFile,
)
from triad.schemas.pipeline import PipelineConfig, TaskSpec

# ── Fixtures ──────────────────────────────────────────────────────


@pytest.fixture()
def sample_project(tmp_path: Path) -> Path:
    """Create a minimal project structure for scanning."""
    # Python file with classes and functions
    (tmp_path / "app.py").write_text(
        textwrap.dedent('''\
        """Main application module."""

        from fastapi import FastAPI

        app = FastAPI()


        class UserService:
            """Handles user operations."""

            async def get_user(self, user_id: int) -> dict:
                """Get a user by ID."""
                return {"id": user_id}

            def list_users(self) -> list[dict]:
                """List all users."""
                return []


        @app.get("/health")
        async def health_check() -> dict:
            return {"status": "ok"}
        '''),
        encoding="utf-8",
    )

    # Python __init__.py
    (tmp_path / "__init__.py").write_text(
        '"""Package init."""\n',
        encoding="utf-8",
    )

    # Python test file
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    (tests_dir / "test_app.py").write_text(
        textwrap.dedent('''\
        """Tests for app."""
        import pytest

        def test_health():
            assert True
        '''),
        encoding="utf-8",
    )

    # Config file (non-Python)
    (tmp_path / "config.toml").write_text(
        '[server]\nhost = "0.0.0.0"\nport = 8000\n',
        encoding="utf-8",
    )

    # Markdown README
    (tmp_path / "README.md").write_text(
        "# My Project\n\nA sample project.\n",
        encoding="utf-8",
    )

    # JavaScript file
    (tmp_path / "script.js").write_text(
        'console.log("hello");\n',
        encoding="utf-8",
    )

    # .gitignore
    (tmp_path / ".gitignore").write_text(
        "*.pyc\n__pycache__/\n.env\nbuild/\n",
        encoding="utf-8",
    )

    # Ignored directory
    pycache = tmp_path / "__pycache__"
    pycache.mkdir()
    (pycache / "app.cpython-312.pyc").write_bytes(b"\x00\x01\x02")

    # Build directory (should be gitignored)
    build_dir = tmp_path / "build"
    build_dir.mkdir()
    (build_dir / "output.txt").write_text("compiled", encoding="utf-8")

    return tmp_path


# ── Schema Tests ──────────────────────────────────────────────────


class TestSchemas:
    """Test context schema validation."""

    def test_function_signature_defaults(self):
        sig = FunctionSignature(name="foo")
        assert sig.name == "foo"
        assert sig.args == []
        assert sig.return_type is None
        assert sig.is_async is False
        assert sig.decorators == []

    def test_function_signature_full(self):
        sig = FunctionSignature(
            name="get_user",
            args=["self", "user_id"],
            return_type="dict",
            is_async=True,
            decorators=["app.get"],
        )
        assert sig.is_async is True
        assert len(sig.args) == 2

    def test_scanned_file_defaults(self):
        f = ScannedFile(path="src/main.py")
        assert f.language == "unknown"
        assert f.size_bytes == 0
        assert f.classes == []
        assert f.functions == []
        assert f.imports == []
        assert f.docstring is None
        assert f.preview is None
        assert f.relevance_score == 0.0

    def test_scanned_file_validation(self):
        with pytest.raises(ValueError):
            ScannedFile(path="test.py", size_bytes=-1)

    def test_project_profile_defaults(self):
        p = ProjectProfile(root_path="/tmp/project")
        assert p.total_files == 0
        assert p.languages == {}
        assert p.entry_points == []

    def test_context_result_defaults(self):
        profile = ProjectProfile(root_path="/tmp")
        cr = ContextResult(profile=profile)
        assert cr.context_text == ""
        assert cr.files_included == 0
        assert cr.truncated is False

    def test_pipeline_config_context_defaults(self):
        cfg = PipelineConfig()
        assert cfg.context_dir is None
        assert cfg.context_include == ["*.py"]
        assert cfg.context_exclude == []
        assert cfg.context_token_budget == 8000

    def test_pipeline_config_context_custom(self):
        cfg = PipelineConfig(
            context_dir="/my/project",
            context_include=["*.py", "*.ts"],
            context_exclude=["test_*"],
            context_token_budget=16000,
        )
        assert cfg.context_dir == "/my/project"
        assert len(cfg.context_include) == 2
        assert cfg.context_token_budget == 16000


# ── Scanner Tests ─────────────────────────────────────────────────


class TestCodeScanner:
    """Test the CodeScanner file discovery and AST parsing."""

    def test_scan_discovers_files(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        paths = {f.path for f in files}
        assert "app.py" in paths
        assert "__init__.py" in paths
        assert "config.toml" in paths

    def test_scan_respects_gitignore(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        paths = {f.path for f in files}
        # __pycache__ and build/ should be excluded
        assert not any("__pycache__" in p for p in paths)
        assert not any("build/" in p for p in paths)

    def test_scan_default_ignores(self, sample_project: Path):
        # Create a .venv directory that should be ignored by default
        venv = sample_project / ".venv"
        venv.mkdir()
        (venv / "lib.py").write_text("x = 1", encoding="utf-8")

        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        paths = {f.path for f in files}
        assert not any(".venv" in p for p in paths)

    def test_scan_include_filter(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        for f in files:
            assert f.path.endswith(".py")

    def test_scan_exclude_filter(self, sample_project: Path):
        scanner = CodeScanner(
            sample_project,
            include=["*"],
            exclude=["*.md"],
        )
        files = scanner.scan()
        paths = {f.path for f in files}
        assert "README.md" not in paths

    def test_scan_python_ast_classes(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        assert "UserService" in app_file.classes

    def test_scan_python_ast_functions(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        func_names = [fn.name for fn in app_file.functions]
        assert "get_user" in func_names
        assert "list_users" in func_names
        assert "health_check" in func_names

    def test_scan_python_async_detection(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        get_user = next(fn for fn in app_file.functions if fn.name == "get_user")
        assert get_user.is_async is True
        list_users = next(fn for fn in app_file.functions if fn.name == "list_users")
        assert list_users.is_async is False

    def test_scan_python_return_types(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        get_user = next(fn for fn in app_file.functions if fn.name == "get_user")
        assert get_user.return_type == "dict"

    def test_scan_python_imports(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        assert any("fastapi" in imp.lower() for imp in app_file.imports)

    def test_scan_python_docstring(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        assert app_file.docstring == "Main application module."

    def test_scan_non_python_preview(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.toml"])
        files = scanner.scan()
        config_file = next(f for f in files if f.path == "config.toml")
        assert config_file.preview is not None
        assert "host" in config_file.preview

    def test_scan_language_detection(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        assert app_file.language == "python"
        js_file = next(f for f in files if f.path == "script.js")
        assert js_file.language == "javascript"
        toml_file = next(f for f in files if f.path == "config.toml")
        assert toml_file.language == "toml"

    def test_scan_file_size(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        assert app_file.size_bytes > 0

    def test_scan_nonexistent_directory(self, tmp_path: Path):
        scanner = CodeScanner(tmp_path / "does_not_exist")
        files = scanner.scan()
        assert files == []

    def test_scan_empty_directory(self, tmp_path: Path):
        empty = tmp_path / "empty"
        empty.mkdir()
        scanner = CodeScanner(empty)
        files = scanner.scan()
        assert files == []

    def test_scan_syntax_error_fallback(self, tmp_path: Path):
        """Files with syntax errors should get a preview instead of AST."""
        (tmp_path / "bad.py").write_text(
            "def broken(:\n  pass\n",
            encoding="utf-8",
        )
        scanner = CodeScanner(tmp_path, include=["*.py"])
        files = scanner.scan()
        bad = next(f for f in files if f.path == "bad.py")
        assert bad.preview is not None
        assert bad.classes == []

    def test_scan_binary_file_skipped(self, tmp_path: Path):
        """Binary files exceeding 1 MB should be skipped."""
        big = tmp_path / "huge.dat"
        big.write_bytes(b"\x00" * 1_100_000)
        scanner = CodeScanner(tmp_path, include=["*"])
        files = scanner.scan()
        paths = {f.path for f in files}
        assert "huge.dat" not in paths

    def test_scan_decorators_extracted(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        app_file = next(f for f in files if f.path == "app.py")
        health = next(
            fn for fn in app_file.functions if fn.name == "health_check"
        )
        assert any("app.get" in d for d in health.decorators)


# ── Builder Tests ─────────────────────────────────────────────────


class TestContextBuilder:
    """Test the ContextBuilder scoring and assembly."""

    def test_build_returns_context_result(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project), token_budget=8000)
        result = builder.build(files, "add user authentication")
        assert isinstance(result, ContextResult)
        assert result.files_included > 0
        assert result.files_scanned == len(files)

    def test_build_profile_languages(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project))
        result = builder.build(files, "build an API")
        assert "python" in result.profile.languages

    def test_build_profile_entry_points(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project))
        result = builder.build(files, "task")
        assert "app.py" in result.profile.entry_points

    def test_build_profile_key_patterns(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project))
        result = builder.build(files, "task")
        assert "FastAPI" in result.profile.key_patterns

    def test_build_context_text_contains_files(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project))
        result = builder.build(files, "user auth")
        assert "app.py" in result.context_text
        assert "UserService" in result.context_text

    def test_build_relevance_scoring(self, sample_project: Path):
        scanner = CodeScanner(sample_project, include=["*.py"])
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project))
        builder.build(files, "user authentication")
        # app.py should score higher than test file due to test penalty
        app_file = next(f for f in files if f.path == "app.py")
        test_file = next(
            (f for f in files if "test" in f.path), None,
        )
        if test_file:
            assert app_file.relevance_score >= test_file.relevance_score

    def test_build_token_budget_respected(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        # Very small budget
        builder = ContextBuilder(str(sample_project), token_budget=50)
        result = builder.build(files, "task")
        assert result.token_estimate <= 60  # small tolerance
        assert result.truncated is True

    def test_build_includes_profile_header(self, sample_project: Path):
        scanner = CodeScanner(sample_project)
        files = scanner.scan()
        builder = ContextBuilder(str(sample_project))
        result = builder.build(files, "task")
        assert "## Project Profile" in result.context_text

    def test_build_empty_files_list(self):
        builder = ContextBuilder("/tmp/empty")
        result = builder.build([], "task")
        assert result.files_included == 0
        assert result.files_scanned == 0


# ── Pruner Tests ──────────────────────────────────────────────────


class TestContextPruner:
    """Test the ContextPruner token-limit enforcement."""

    def _make_context(self, text: str, token_est: int) -> ContextResult:
        return ContextResult(
            profile=ProjectProfile(root_path="/tmp"),
            context_text=text,
            files_included=5,
            files_scanned=10,
            token_estimate=token_est,
            truncated=False,
        )

    def test_prune_no_change_when_fits(self):
        ctx = self._make_context("Small context", 100)
        pruner = ContextPruner(model_context_window=200_000)
        result = pruner.prune(ctx)
        assert result.context_text == "Small context"
        assert result.truncated is False

    def test_prune_truncates_large_context(self):
        # Create a context that exceeds the available char budget
        # window=10000, reserved=4000, available=6000, char_limit=24000
        # So text needs to be > 24000 chars
        big_text = "\n### file1.py\n" + ("x" * 200 + "\n") * 200
        token_est = len(big_text) // 4
        ctx = self._make_context(big_text, token_est)
        pruner = ContextPruner(model_context_window=10000)
        result = pruner.prune(ctx)
        assert result.truncated is True
        assert result.token_estimate < token_est
        assert "[Context truncated" in result.context_text

    def test_prune_tiny_window(self):
        ctx = self._make_context("Some content", 100)
        pruner = ContextPruner(model_context_window=3000)
        result = pruner.prune(ctx)
        # With 3000 - 4000 reserved = negative, should return empty
        assert result.context_text == ""
        assert result.truncated is True
        assert result.files_included == 0

    def test_prune_preserves_profile(self):
        profile = ProjectProfile(
            root_path="/my/project",
            total_files=42,
            languages={"python": 30},
        )
        ctx = ContextResult(
            profile=profile,
            context_text="x" * 100000,
            files_included=10,
            files_scanned=42,
            token_estimate=25000,
            truncated=False,
        )
        pruner = ContextPruner(model_context_window=10000)
        result = pruner.prune(ctx)
        assert result.profile.root_path == "/my/project"
        assert result.profile.total_files == 42


# ── Orchestrator Integration Tests ────────────────────────────────


class TestOrchestratorContextInjection:
    """Test that context injection integrates with run_pipeline."""

    def test_inject_context_builds_task(self, sample_project: Path):
        from triad.orchestrator import _inject_context

        task = TaskSpec(task="add user auth", output_dir="out")
        config = PipelineConfig(
            context_dir=str(sample_project),
            context_include=["*.py"],
            context_exclude=[],
            context_token_budget=8000,
        )
        new_task = _inject_context(task, config)
        assert "UserService" in new_task.context
        assert "app.py" in new_task.context

    def test_inject_context_preserves_existing(self, sample_project: Path):
        from triad.orchestrator import _inject_context

        task = TaskSpec(
            task="add auth",
            context="Use JWT tokens",
            output_dir="out",
        )
        config = PipelineConfig(
            context_dir=str(sample_project),
            context_include=["*.py"],
        )
        new_task = _inject_context(task, config)
        assert "Use JWT tokens" in new_task.context
        assert "app.py" in new_task.context

    def test_inject_context_preserves_domain_rules(self, sample_project: Path):
        from triad.orchestrator import _inject_context

        task = TaskSpec(
            task="build API",
            domain_rules="Follow REST conventions",
            output_dir="out",
        )
        config = PipelineConfig(
            context_dir=str(sample_project),
            context_include=["*.py"],
        )
        new_task = _inject_context(task, config)
        assert new_task.domain_rules == "Follow REST conventions"

    def test_inject_context_no_files(self, tmp_path: Path):
        from triad.orchestrator import _inject_context

        empty = tmp_path / "empty_project"
        empty.mkdir()
        task = TaskSpec(task="do something", output_dir="out")
        config = PipelineConfig(
            context_dir=str(empty),
            context_include=["*.py"],
        )
        # Should return original task unchanged
        result = _inject_context(task, config)
        assert result.task == "do something"
        assert result.context == ""

    def test_run_pipeline_calls_inject(self, sample_project: Path):
        """run_pipeline should call _inject_context when context_dir is set."""
        from triad.orchestrator import run_pipeline

        task = TaskSpec(task="build auth", output_dir="out")
        config = PipelineConfig(
            context_dir=str(sample_project),
            context_include=["*.py"],
            persist_sessions=False,
        )

        # Mock the actual pipeline run
        mock_result = MagicMock()
        mock_result.session_id = ""
        mock_result.config = config

        with patch(
            "triad.orchestrator.PipelineOrchestrator"
        ) as mock_orch:
            mock_inst = MagicMock()
            mock_inst.run = AsyncMock(return_value=mock_result)
            mock_orch.return_value = mock_inst

            import asyncio

            asyncio.run(run_pipeline(task, config, {}))

            # Verify the orchestrator was called with enriched task
            call_args = mock_orch.call_args
            enriched_task = call_args[0][0]
            assert "app.py" in enriched_task.context

    def test_run_pipeline_skips_when_no_context_dir(self):
        """run_pipeline should not inject context when context_dir is None."""
        from triad.orchestrator import run_pipeline

        task = TaskSpec(task="do something", output_dir="out")
        config = PipelineConfig(persist_sessions=False)

        mock_result = MagicMock()
        mock_result.session_id = ""
        mock_result.config = config

        with patch(
            "triad.orchestrator.PipelineOrchestrator"
        ) as mock_orch:
            mock_inst = MagicMock()
            mock_inst.run = AsyncMock(return_value=mock_result)
            mock_orch.return_value = mock_inst

            import asyncio

            asyncio.run(run_pipeline(task, config, {}))

            call_args = mock_orch.call_args
            original_task = call_args[0][0]
            assert original_task.context == ""


# ── CLI Integration Tests ─────────────────────────────────────────


class TestCLIContextFlags:
    """Test the CLI --context-dir, --include, --exclude flags."""

    def test_run_help_shows_context_flags(self):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()
        result = runner.invoke(app, ["run", "--help"])
        assert "--context-dir" in result.output
        assert "--include" in result.output
        assert "--exclude" in result.output
        assert "--context-budget" in result.output

    def test_run_with_context_dir(self, sample_project: Path):
        from typer.testing import CliRunner

        from triad.cli import app

        runner = CliRunner()

        with patch("triad.cli._load_registry") as mock_reg, \
             patch("triad.cli._load_config") as mock_cfg, \
             patch("triad.orchestrator.run_pipeline") as mock_run, \
             patch("triad.output.writer.write_pipeline_output"):

            mock_reg.return_value = {}
            mock_cfg.return_value = PipelineConfig()

            mock_result = MagicMock()
            mock_result.halted = False
            mock_result.success = True
            mock_result.total_cost = 0.01
            mock_result.total_tokens = 100
            mock_result.duration_seconds = 1.5
            mock_result.session_id = "test-123"
            mock_result.arbiter_reviews = []
            mock_result.routing_decisions = []
            mock_run.return_value = mock_result

            result = runner.invoke(app, [
                "run", "build auth",
                "--context-dir", str(sample_project),
                "--include", "*.py",
                "--exclude", "test_*",
                "--context-budget", "4000",
            ])

            assert result.exit_code == 0
            # Verify config was built with context options
            call_args = mock_run.call_args
            config = call_args[0][1]
            assert config.context_dir == str(sample_project)
            assert config.context_include == ["*.py"]
            assert config.context_exclude == ["test_*"]
            assert config.context_token_budget == 4000

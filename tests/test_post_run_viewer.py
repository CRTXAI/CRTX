"""Tests for the post-run interactive viewer."""

from __future__ import annotations

from pathlib import Path
from unittest.mock import MagicMock, patch

import pytest
from rich.console import Console

from triad.post_run_viewer import PostRunViewer


@pytest.fixture
def session_dir(tmp_path):
    """Create a mock session directory with test files."""
    code_dir = tmp_path / "code"
    code_dir.mkdir()
    tests_dir = tmp_path / "tests"
    tests_dir.mkdir()
    reviews_dir = tmp_path / "reviews"
    reviews_dir.mkdir()

    # Code file
    (code_dir / "app.py").write_text(
        "from fastapi import FastAPI\n\napp = FastAPI()\n",
        encoding="utf-8",
    )
    (code_dir / "models.py").write_text(
        "class User:\n    name: str\n",
        encoding="utf-8",
    )

    # Test file
    (tests_dir / "test_app.py").write_text(
        "def test_health():\n    assert True\n",
        encoding="utf-8",
    )

    # Summary
    (tmp_path / "summary.md").write_text(
        "# Pipeline Summary\n\nBuilt a REST API.\n",
        encoding="utf-8",
    )

    # Session JSON (minimal)
    (tmp_path / "session.json").write_text(
        '{"session_id": "abc12345"}',
        encoding="utf-8",
    )

    return tmp_path


@pytest.fixture
def mock_result():
    """Create a mock pipeline result."""
    result = MagicMock()
    result.arbiter_reviews = []
    result.stages = {}
    result.total_cost = 0.5
    result.total_tokens = 10000
    result.duration_seconds = 120.0
    result.success = True
    result.halted = False
    return result


class TestPostRunViewerInit:
    def test_create_viewer(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        assert viewer.console is console
        assert viewer.session_dir == session_dir
        assert viewer.result is mock_result


class TestSummaryView:
    def test_show_summary_renders_markdown(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        # No input needed — _wait_for_back removed
        viewer._show_summary()

    def test_show_summary_missing_file(self, tmp_path, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, tmp_path, mock_result)
        viewer._show_summary()


class TestCodeView:
    def test_show_code_lists_files(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        # Code viewer has its own input loop — 'b' to go back
        with patch("builtins.input", return_value="b"):
            viewer._show_code()

    def test_show_code_view_file(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        # Select file 1, then go back
        with patch("builtins.input", side_effect=["1", "b"]):
            viewer._show_code()

    def test_show_code_invalid_number(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch("builtins.input", side_effect=["99", "b"]):
            viewer._show_code()

    def test_show_code_empty_dir(self, tmp_path, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, tmp_path, mock_result)
        # No input needed — returns immediately when no files
        viewer._show_code()

    def test_show_code_by_index(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer._show_code_by_index(1)

    def test_show_code_by_index_out_of_range(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer._show_code_by_index(999)

    def test_view_file_displays_syntax(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        code_file = session_dir / "code" / "app.py"
        viewer._view_file(code_file, session_dir)


class TestReviewsView:
    def test_show_reviews_empty(self, session_dir, mock_result):
        console = Console(quiet=True)
        mock_result.arbiter_reviews = []
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer._show_reviews()

    def test_show_reviews_with_data(self, session_dir):
        console = Console(quiet=True)

        review = MagicMock()
        review.verdict = MagicMock(value="flag")
        review.stage_reviewed = MagicMock(value="architect")
        review.confidence = 0.88
        review.arbiter_model = "claude-opus"
        review.reasoning = "Some issues found."
        review.issues = []

        result = MagicMock()
        result.arbiter_reviews = [review]
        result.stages = {}

        viewer = PostRunViewer(console, session_dir, result)
        viewer._show_reviews()


class TestDiffsView:
    def test_show_diffs_empty(self, session_dir, mock_result):
        console = Console(quiet=True)
        mock_result.stages = {}
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer._show_diffs()

    def test_show_diffs_with_refactor_output(self, session_dir):
        console = Console(quiet=True)

        refactor_msg = MagicMock()
        refactor_msg.content = (
            "--- a/app.py\n+++ b/app.py\n@@ -1,3 +1,5 @@\n"
            " from fastapi import FastAPI\n+import logging\n"
        )

        stage_key = MagicMock(value="refactor")
        result = MagicMock()
        result.arbiter_reviews = []
        result.stages = {stage_key: refactor_msg}

        viewer = PostRunViewer(console, session_dir, result)
        viewer._show_diffs()


class TestRunDirect:
    def test_run_direct_summary(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer.run_direct("summary")

    def test_run_direct_code(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch("builtins.input", return_value="b"):
            viewer.run_direct("code")

    def test_run_direct_reviews(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer.run_direct("reviews")

    def test_run_direct_diffs(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        viewer.run_direct("diffs")


class TestRunLoop:
    def test_run_enter_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch("builtins.input", return_value=""):
            result = viewer.run()
            assert result is None

    def test_run_q_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch("builtins.input", return_value="q"):
            result = viewer.run()
            assert result is None

    def test_run_s_then_exit(self, session_dir, mock_result):
        """Pressing 's' shows summary then menu reappears for next input."""
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        # First input: 's' (view summary), second: '' (exit)
        with patch("builtins.input", side_effect=["s", ""]):
            result = viewer.run()
            assert result is None

    def test_run_multiple_views_then_exit(self, session_dir, mock_result):
        """User can view multiple subviews before exiting."""
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        # summary → reviews → diffs → exit
        with patch("builtins.input", side_effect=["s", "r", "d", ""]):
            result = viewer.run()
            assert result is None

    def test_run_eof_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch("builtins.input", side_effect=EOFError):
            result = viewer.run()
            assert result is None

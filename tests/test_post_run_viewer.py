"""Tests for the post-run interactive viewer."""

from __future__ import annotations

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


_READ_KEY = "triad.cli_display._read_key"


class TestRunLoop:
    def test_run_enter_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch(_READ_KEY, return_value="enter"):
            result = viewer.run()
            assert result is None

    def test_run_q_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch(_READ_KEY, return_value="q"):
            result = viewer.run()
            assert result is None

    def test_run_escape_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch(_READ_KEY, return_value="escape"):
            result = viewer.run()
            assert result is None

    def test_run_s_then_exit(self, session_dir, mock_result):
        """Pressing 's' shows summary then menu reappears for next input."""
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        # First keypress: 's' (view summary), second: 'enter' (exit)
        with patch(_READ_KEY, side_effect=["s", "enter"]):
            result = viewer.run()
            assert result is None

    def test_run_multiple_views_then_exit(self, session_dir, mock_result):
        """User can view multiple subviews before exiting."""
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        # summary → reviews → diffs → exit
        with patch(_READ_KEY, side_effect=["s", "r", "d", "enter"]):
            result = viewer.run()
            assert result is None

    def test_run_eof_exits(self, session_dir, mock_result):
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch(_READ_KEY, side_effect=EOFError):
            result = viewer.run()
            assert result is None


class TestImproveApplyOptions:
    """Tests for the [i] Improve and [a] Apply menu options."""

    def _make_review_result(self):
        """Create a mock review_result with a synthesized_review."""
        rr = MagicMock()
        rr.synthesized_review = "Found security issues in auth module."
        return rr

    def _make_improve_result(self):
        """Create a mock improve_result."""
        return MagicMock()

    def test_menu_shows_improve_when_review_result(self, session_dir, mock_result):
        """[i] appears in menu when on_improve callback and review_result present."""
        mock_result.review_result = self._make_review_result()
        mock_result.improve_result = None
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_improve=lambda focus: None,
        )
        menu = viewer._build_menu()
        assert "[i]" in menu
        assert "[a]" not in menu

    def test_menu_hides_improve_when_no_review_result(self, session_dir, mock_result):
        """[i] absent when review_result is None."""
        mock_result.review_result = None
        mock_result.improve_result = None
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_improve=lambda focus: None,
        )
        menu = viewer._build_menu()
        assert "[i]" not in menu

    def test_menu_hides_improve_when_no_callback(self, session_dir, mock_result):
        """[i] absent when on_improve not provided."""
        mock_result.review_result = self._make_review_result()
        mock_result.improve_result = None
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)
        menu = viewer._build_menu()
        assert "[i]" not in menu

    def test_menu_shows_apply_when_improve_result(self, session_dir, mock_result):
        """[a] appears in menu when on_apply callback and improve_result present."""
        mock_result.review_result = None
        mock_result.improve_result = self._make_improve_result()
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_apply=lambda result: None,
        )
        menu = viewer._build_menu()
        assert "[a]" in menu
        assert "[i]" not in menu

    def test_menu_hides_apply_when_no_improve_result(self, session_dir, mock_result):
        """[a] absent when improve_result is None."""
        mock_result.review_result = None
        mock_result.improve_result = None
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_apply=lambda result: None,
        )
        menu = viewer._build_menu()
        assert "[a]" not in menu

    def test_pressing_i_calls_improve_callback(self, session_dir, mock_result):
        """Callback invoked with synthesized_review text, result swapped."""
        mock_result.review_result = self._make_review_result()
        mock_result.improve_result = None

        new_result = MagicMock()
        new_result.improve_result = self._make_improve_result()
        new_result.review_result = None
        new_path = str(session_dir / "improved")
        (session_dir / "improved").mkdir()

        callback = MagicMock(return_value=(new_result, new_path))
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_improve=callback,
        )

        with patch(_READ_KEY, side_effect=["i", "enter"]):
            viewer.run()

        callback.assert_called_once_with("Found security issues in auth module.")
        assert viewer.result is new_result

    def test_pressing_a_calls_apply_callback(self, session_dir, mock_result):
        """Callback invoked with current result."""
        mock_result.improve_result = self._make_improve_result()
        mock_result.review_result = None

        callback = MagicMock()
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_apply=callback,
        )

        with patch(_READ_KEY, side_effect=["a", "enter"]):
            viewer.run()

        callback.assert_called_once_with(mock_result)

    def test_improve_then_apply_chain(self, session_dir, mock_result):
        """Press i → a → enter: improve called, then apply with improved result."""
        mock_result.review_result = self._make_review_result()
        mock_result.improve_result = None

        improved_result = MagicMock()
        improved_result.improve_result = self._make_improve_result()
        improved_result.review_result = None
        new_path = str(session_dir / "improved")
        (session_dir / "improved").mkdir()

        improve_cb = MagicMock(return_value=(improved_result, new_path))
        apply_cb = MagicMock()
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_improve=improve_cb,
            on_apply=apply_cb,
        )

        with patch(_READ_KEY, side_effect=["i", "a", "enter"]):
            viewer.run()

        improve_cb.assert_called_once()
        apply_cb.assert_called_once_with(improved_result)

    def test_pressing_i_without_callback_is_noop(self, session_dir, mock_result):
        """No crash when i pressed with no callback."""
        mock_result.review_result = self._make_review_result()
        mock_result.improve_result = None
        console = Console(quiet=True)
        viewer = PostRunViewer(console, session_dir, mock_result)

        with patch(_READ_KEY, side_effect=["i", "enter"]):
            result = viewer.run()
            assert result is None
        # result unchanged
        assert viewer.result is mock_result

    def test_improve_failure_keeps_original_result(self, session_dir, mock_result):
        """When callback returns None, result unchanged."""
        mock_result.review_result = self._make_review_result()
        mock_result.improve_result = None

        callback = MagicMock(return_value=None)
        console = Console(quiet=True)
        viewer = PostRunViewer(
            console, session_dir, mock_result,
            on_improve=callback,
        )

        with patch(_READ_KEY, side_effect=["i", "enter"]):
            viewer.run()

        callback.assert_called_once()
        assert viewer.result is mock_result

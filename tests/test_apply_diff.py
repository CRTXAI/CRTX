"""Tests for the diff preview display."""

from __future__ import annotations

from rich.console import Console

from triad.apply.diff import DiffPreview
from triad.schemas.apply import FileAction, ResolvedFile


def _make_file(action=FileAction.CREATE, content="new", existing=None):
    return ResolvedFile(
        source_filepath="src/main.py",
        resolved_path="/project/src/main.py",
        action=action,
        content=content,
        language="python",
        existing_content=existing,
        match_confidence=1.0,
    )


class TestDiffPreview:
    def test_non_interactive_returns_files(self):
        console = Console(quiet=True)
        files = [_make_file()]
        preview = DiffPreview(console, files, interactive=False)
        result = preview.show()
        assert len(result) == 1
        assert result[0].selected is True

    def test_empty_file_list(self):
        console = Console(quiet=True)
        preview = DiffPreview(console, [], interactive=False)
        result = preview.show()
        assert len(result) == 0

    def test_skipped_files_excluded_from_active(self):
        console = Console(quiet=True)
        files = [
            _make_file(action=FileAction.SKIP),
            _make_file(action=FileAction.CREATE),
        ]
        preview = DiffPreview(console, files, interactive=False)
        result = preview.show()
        # All files returned, but skip status preserved
        skipped = [f for f in result if f.action == FileAction.SKIP]
        assert len(skipped) == 1

"""Tests for the conflict detector and conflict resolution UI."""

from __future__ import annotations

import time
from pathlib import Path
from unittest.mock import patch

import pytest
from rich.console import Console

from triad.apply.conflict import ConflictDetector
from triad.apply.diff import ConflictResolver
from triad.schemas.apply import ConflictAction, FileConflict, Resolution


class TestConflictDetector:
    def test_no_conflict_unchanged(self, tmp_path):
        filepath = tmp_path / "test.py"
        filepath.write_text("original")
        detector = ConflictDetector()
        detector.snapshot(str(filepath))
        assert detector.check(str(filepath)) is False

    def test_conflict_modified(self, tmp_path):
        filepath = tmp_path / "test.py"
        filepath.write_text("original")
        detector = ConflictDetector()
        detector.snapshot(str(filepath))
        # Wait to ensure mtime changes
        time.sleep(0.05)
        filepath.write_text("modified")
        assert detector.check(str(filepath)) is True

    def test_conflict_deleted(self, tmp_path):
        filepath = tmp_path / "test.py"
        filepath.write_text("original")
        detector = ConflictDetector()
        detector.snapshot(str(filepath))
        filepath.unlink()
        assert detector.check(str(filepath)) is True

    def test_no_snapshot_no_conflict(self, tmp_path):
        detector = ConflictDetector()
        assert detector.check(str(tmp_path / "unknown.py")) is False

    def test_check_all(self, tmp_path):
        file1 = tmp_path / "a.py"
        file2 = tmp_path / "b.py"
        file1.write_text("content a")
        file2.write_text("content b")

        detector = ConflictDetector()
        detector.snapshot(str(file1))
        detector.snapshot(str(file2))

        time.sleep(0.05)
        file2.write_text("modified b")

        conflicts = detector.check_all()
        assert str(file1) not in conflicts
        assert str(file2) in conflicts


class TestConflictResolver:
    def _make_conflict(
        self, filepath: str = "/project/src/main.py",
    ) -> FileConflict:
        return FileConflict(
            filepath=filepath,
            scan_time=time.time() - 300,
            current_mtime=time.time(),
            total_operations=5,
            clean_operations=3,
            conflicting_operations=2,
            scanned_content="# original\nprint('old')\n",
            current_content="# modified\nprint('new')\n",
        )

    def test_non_interactive_defaults_to_skip(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=False)
        conflicts = [self._make_conflict()]
        resolutions = resolver.resolve(conflicts)
        assert len(resolutions) == 1
        assert resolutions[0].action == ConflictAction.SKIP

    def test_empty_conflicts_returns_empty(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        resolutions = resolver.resolve([])
        assert resolutions == []

    def test_interactive_force(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        conflict = self._make_conflict()
        with patch.object(console, "input", return_value="f"):
            resolutions = resolver.resolve([conflict])
        assert len(resolutions) == 1
        assert resolutions[0].action == ConflictAction.FORCE

    def test_interactive_skip(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        conflict = self._make_conflict()
        with patch.object(console, "input", return_value="s"):
            resolutions = resolver.resolve([conflict])
        assert len(resolutions) == 1
        assert resolutions[0].action == ConflictAction.SKIP

    def test_interactive_apply_clean(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        conflict = self._make_conflict()
        with patch.object(console, "input", return_value="p"):
            resolutions = resolver.resolve([conflict])
        assert len(resolutions) == 1
        assert resolutions[0].action == ConflictAction.APPLY_CLEAN

    def test_interactive_cancel(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        conflict = self._make_conflict()
        with patch.object(console, "input", return_value="q"):
            resolutions = resolver.resolve([conflict])
        assert resolutions == []

    def test_interactive_view_then_skip(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        conflict = self._make_conflict()
        # First input "v" to view, then "s" to skip
        with patch.object(console, "input", side_effect=["v", "s"]):
            resolutions = resolver.resolve([conflict])
        assert len(resolutions) == 1
        assert resolutions[0].action == ConflictAction.SKIP

    def test_multiple_conflicts(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        c1 = self._make_conflict("/project/a.py")
        c2 = self._make_conflict("/project/b.py")
        # Force first, skip second
        with patch.object(console, "input", side_effect=["f", "s"]):
            resolutions = resolver.resolve([c1, c2])
        assert len(resolutions) == 2
        assert resolutions[0].action == ConflictAction.FORCE
        assert resolutions[0].filepath == "/project/a.py"
        assert resolutions[1].action == ConflictAction.SKIP
        assert resolutions[1].filepath == "/project/b.py"

    def test_eof_cancels(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=True)
        conflict = self._make_conflict()
        with patch.object(console, "input", side_effect=EOFError):
            resolutions = resolver.resolve([conflict])
        assert resolutions == []

    def test_non_interactive_multiple(self):
        console = Console(quiet=True)
        resolver = ConflictResolver(console, interactive=False)
        c1 = self._make_conflict("/project/a.py")
        c2 = self._make_conflict("/project/b.py")
        resolutions = resolver.resolve([c1, c2])
        assert len(resolutions) == 2
        assert all(r.action == ConflictAction.SKIP for r in resolutions)

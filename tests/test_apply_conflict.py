"""Tests for the conflict detector."""

from __future__ import annotations

import time
from pathlib import Path

import pytest

from triad.apply.conflict import ConflictDetector


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

"""Tests for the post-apply verifier."""

from __future__ import annotations

from pathlib import Path

import pytest

from triad.apply.verify import PostApplyVerifier


class TestPostApplyVerifier:
    def test_no_test_command(self, tmp_path):
        verifier = PostApplyVerifier(tmp_path, "")
        passed, output = verifier.run_test_command()
        assert passed is True
        assert output == ""

    def test_passing_test(self, tmp_path):
        verifier = PostApplyVerifier(tmp_path, "python -c \"print('ok')\"")
        passed, output = verifier.run_test_command()
        assert passed is True

    def test_failing_test(self, tmp_path):
        verifier = PostApplyVerifier(tmp_path, "python -c \"exit(1)\"")
        passed, output = verifier.run_test_command()
        assert passed is False

    def test_command_not_found(self, tmp_path):
        verifier = PostApplyVerifier(tmp_path, "nonexistent_command_xyz")
        passed, output = verifier.run_test_command()
        assert passed is False
        assert "not found" in output.lower() or "error" in output.lower()

    def test_rollback_restores(self, tmp_path):
        # Create a file
        test_file = tmp_path / "test.py"
        test_file.write_text("original content")

        # Modify it
        test_file.write_text("modified content")

        # Rollback
        backups = {str(test_file): "original content"}
        PostApplyVerifier.rollback(backups)

        assert test_file.read_text() == "original content"

    def test_rollback_deletes_new_files(self, tmp_path):
        new_file = tmp_path / "new.py"
        new_file.write_text("new content")

        backups = {str(new_file): None}
        PostApplyVerifier.rollback(backups)

        assert not new_file.exists()

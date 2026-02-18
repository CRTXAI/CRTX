"""Tests for the git safety gate."""

from __future__ import annotations

import subprocess

import pytest

from triad.apply.git import _PROTECTED_BRANCHES, GitSafetyGate
from triad.schemas.apply import GitState


@pytest.fixture
def git_dir(tmp_path):
    """Create a temporary git repository."""
    subprocess.run(
        ["git", "init"], cwd=str(tmp_path),
        capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.email", "test@test.com"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "config", "user.name", "Test"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    # Create initial commit
    (tmp_path / "README.md").write_text("# test\n")
    subprocess.run(
        ["git", "add", "README.md"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    subprocess.run(
        ["git", "commit", "-m", "init"],
        cwd=str(tmp_path), capture_output=True, check=True,
    )
    return tmp_path


class TestGitSafetyGate:
    def test_check_state_non_git(self, tmp_path):
        gate = GitSafetyGate(tmp_path)
        state = gate.check_state()
        assert state.is_git_repo is False

    def test_check_state_git_clean(self, git_dir):
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        assert state.is_git_repo is True
        assert state.is_clean is True
        assert state.head_sha != ""

    def test_check_state_dirty(self, git_dir):
        (git_dir / "dirty.txt").write_text("dirty")
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        assert state.is_clean is False

    def test_ensure_safe_clean(self, git_dir):
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        # Rename branch to avoid protected check
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=str(git_dir), capture_output=True, check=True,
        )
        state = gate.check_state()
        warnings = gate.ensure_safe(state, [])
        assert len(warnings) == 0

    def test_ensure_safe_dirty_warning(self, git_dir):
        (git_dir / "dirty.txt").write_text("dirty")
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        warnings = gate.ensure_safe(state, [])
        assert any("uncommitted" in w for w in warnings)

    def test_ensure_safe_protected_branch(self, git_dir):
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        # Default branch might be main or master
        if state.current_branch in _PROTECTED_BRANCHES:
            warnings = gate.ensure_safe(state, [])
            assert any("protected" in w.lower() for w in warnings)

    def test_ensure_safe_reject_blocks(self, git_dir):
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        warnings = gate.ensure_safe(state, ["reject"])
        assert any("blocked" in w.lower() for w in warnings)

    def test_ensure_safe_halt_blocks(self, git_dir):
        gate = GitSafetyGate(git_dir)
        state = gate.check_state()
        warnings = gate.ensure_safe(state, ["halt"])
        assert any("blocked" in w.lower() for w in warnings)

    def test_create_branch(self, git_dir):
        gate = GitSafetyGate(git_dir)
        name = gate.create_branch("test-branch")
        assert name == "test-branch"
        state = gate.check_state()
        assert state.current_branch == "test-branch"

    def test_commit_changes(self, git_dir):
        gate = GitSafetyGate(git_dir)
        new_file = git_dir / "new.txt"
        new_file.write_text("new content")
        sha = gate.commit_changes([str(new_file)], "test commit")
        assert len(sha) == 40

    def test_ensure_safe_non_git(self, tmp_path):
        gate = GitSafetyGate(tmp_path)
        state = GitState(is_git_repo=False)
        warnings = gate.ensure_safe(state, [])
        assert any("not a git" in w.lower() for w in warnings)

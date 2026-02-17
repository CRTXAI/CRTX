"""Git safety gate for apply mode.

Checks repository state, creates branches, commits changes, and
performs rollbacks. Uses subprocess directly to avoid external
dependencies.
"""

from __future__ import annotations

import logging
import subprocess
from pathlib import Path

from triad.schemas.apply import GitState

logger = logging.getLogger(__name__)

# Branches protected from direct apply
_PROTECTED_BRANCHES = {"main", "master", "develop", "production"}


class GitSafetyGate:
    """Git safety checks and operations for apply mode.

    All operations use subprocess to call git directly, avoiding
    a dependency on GitPython.
    """

    def __init__(self, context_dir: Path) -> None:
        self._cwd = str(context_dir)

    def _run(self, *args: str, check: bool = True) -> subprocess.CompletedProcess:
        """Run a git command and return the result."""
        cmd = ["git", *args]
        return subprocess.run(
            cmd,
            cwd=self._cwd,
            capture_output=True,
            text=True,
            check=check,
            timeout=30,
        )

    def check_state(self) -> GitState:
        """Check the current git repository state.

        Returns:
            GitState with is_git_repo, is_clean, current_branch, head_sha.
        """
        # Check if it's a git repo
        result = self._run("rev-parse", "--is-inside-work-tree", check=False)
        if result.returncode != 0:
            return GitState(is_git_repo=False)

        # Get current branch
        branch_result = self._run("rev-parse", "--abbrev-ref", "HEAD", check=False)
        current_branch = branch_result.stdout.strip() if branch_result.returncode == 0 else ""

        # Get HEAD SHA
        sha_result = self._run("rev-parse", "HEAD", check=False)
        head_sha = sha_result.stdout.strip() if sha_result.returncode == 0 else ""

        # Check if working tree is clean
        status_result = self._run("status", "--porcelain", check=False)
        is_clean = not status_result.stdout.strip()

        return GitState(
            is_git_repo=True,
            is_clean=is_clean,
            current_branch=current_branch,
            head_sha=head_sha,
        )

    def ensure_safe(self, state: GitState, arbiter_verdicts: list[str]) -> list[str]:
        """Check for safety issues before applying.

        Args:
            state: Current git state.
            arbiter_verdicts: List of arbiter verdict strings from the pipeline.

        Returns:
            List of warning/blocker messages. Empty list means safe to proceed.
        """
        warnings: list[str] = []

        if not state.is_git_repo:
            warnings.append(
                "Not a git repository. Changes cannot be tracked or rolled back."
            )
            return warnings

        if not state.is_clean:
            warnings.append(
                "Working tree has uncommitted changes. "
                "Consider committing or stashing before apply."
            )

        if state.current_branch in _PROTECTED_BRANCHES:
            warnings.append(
                f"On protected branch '{state.current_branch}'. "
                f"Use --branch to create a feature branch."
            )

        # Block on REJECT/HALT verdicts
        blocking_verdicts = {"reject", "halt"}
        for v in arbiter_verdicts:
            if v.lower() in blocking_verdicts:
                warnings.append(
                    f"Arbiter issued {v.upper()} verdict. "
                    f"Apply is blocked until issues are resolved."
                )
                break

        return warnings

    def create_branch(self, name: str) -> str:
        """Create and checkout a new branch.

        Args:
            name: Branch name to create.

        Returns:
            The branch name.

        Raises:
            RuntimeError: If branch creation fails.
        """
        try:
            self._run("checkout", "-b", name)
            logger.info("Created branch: %s", name)
            return name
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to create branch '{name}': {e.stderr}") from e

    def commit_changes(self, files: list[str], message: str) -> str:
        """Stage files and create a commit.

        Args:
            files: List of file paths to stage.
            message: Commit message.

        Returns:
            The commit SHA.

        Raises:
            RuntimeError: If commit fails.
        """
        try:
            for f in files:
                self._run("add", f)
            self._run("commit", "-m", message)
            result = self._run("rev-parse", "HEAD")
            sha = result.stdout.strip()
            logger.info("Committed %d files: %s", len(files), sha[:12])
            return sha
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to commit: {e.stderr}") from e

    def rollback_to(self, sha: str) -> None:
        """Hard reset to a previous commit.

        Only used after a failed post-apply test. This is destructive
        and should only be called with user consent.

        Args:
            sha: Commit SHA to reset to.

        Raises:
            RuntimeError: If rollback fails.
        """
        try:
            self._run("reset", "--hard", sha)
            logger.info("Rolled back to %s", sha[:12])
        except subprocess.CalledProcessError as e:
            raise RuntimeError(f"Failed to rollback to {sha}: {e.stderr}") from e

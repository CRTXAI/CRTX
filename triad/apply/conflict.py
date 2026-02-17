"""Conflict detection for apply mode Phase 2.

Compares file modification times and content hashes between
context-scan time and apply time to detect concurrent changes.
"""

from __future__ import annotations

import hashlib
import logging
from pathlib import Path

logger = logging.getLogger(__name__)


class ConflictDetector:
    """Detects concurrent file modifications between scan and apply.

    Records file state (mtime + content hash) at scan time, then
    checks for changes at apply time. Reports which files have
    been modified externally.
    """

    def __init__(self) -> None:
        self._snapshots: dict[str, tuple[float, str]] = {}

    def snapshot(self, filepath: str) -> None:
        """Record the current state of a file.

        Args:
            filepath: Absolute path to the file.
        """
        path = Path(filepath)
        if path.exists():
            mtime = path.stat().st_mtime
            content_hash = self._hash_file(path)
            self._snapshots[filepath] = (mtime, content_hash)

    def check(self, filepath: str) -> bool:
        """Check if a file has changed since its snapshot.

        Args:
            filepath: Absolute path to the file.

        Returns:
            True if the file has been modified, False otherwise.
        """
        if filepath not in self._snapshots:
            return False

        path = Path(filepath)
        if not path.exists():
            # File was deleted — that's a conflict
            return True

        old_mtime, old_hash = self._snapshots[filepath]
        current_mtime = path.stat().st_mtime

        # Quick check: mtime unchanged means no conflict
        if current_mtime == old_mtime:
            return False

        # mtime changed — verify with content hash
        current_hash = self._hash_file(path)
        if current_hash != old_hash:
            logger.warning(
                "Conflict detected: %s modified since scan", filepath,
            )
            return True

        return False

    def check_all(self) -> list[str]:
        """Check all snapshotted files for conflicts.

        Returns:
            List of filepaths that have been modified.
        """
        conflicts: list[str] = []
        for filepath in self._snapshots:
            if self.check(filepath):
                conflicts.append(filepath)
        return conflicts

    @staticmethod
    def _hash_file(path: Path) -> str:
        """Compute a SHA-256 hash of a file's content."""
        content = path.read_bytes()
        return hashlib.sha256(content).hexdigest()

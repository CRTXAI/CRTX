"""Post-apply verification for apply mode.

Runs test commands after file application and handles rollback
if tests fail.
"""

from __future__ import annotations

import logging
import shlex
import subprocess
from pathlib import Path

logger = logging.getLogger(__name__)


class PostApplyVerifier:
    """Runs post-apply test commands and handles rollback.

    Tests are run via subprocess with a timeout. If tests fail and
    rollback is requested, original file contents are restored from
    in-memory backups.
    """

    def __init__(self, cwd: Path, test_command: str) -> None:
        self._cwd = cwd
        self._test_command = test_command

    def run_test_command(self) -> tuple[bool, str]:
        """Run the configured test command.

        Returns:
            Tuple of (passed, output).
        """
        if not self._test_command:
            return True, ""

        try:
            args = shlex.split(self._test_command)
            result = subprocess.run(
                args,
                cwd=str(self._cwd),
                capture_output=True,
                text=True,
                timeout=300,
            )
            output = result.stdout + result.stderr
            passed = result.returncode == 0

            if passed:
                logger.info("Post-apply tests passed")
            else:
                logger.warning(
                    "Post-apply tests failed (exit code %d)", result.returncode
                )

            return passed, output

        except subprocess.TimeoutExpired:
            logger.warning("Post-apply test timed out after 300s")
            return False, "Test command timed out after 300 seconds"
        except FileNotFoundError:
            msg = f"Test command not found: {self._test_command}"
            logger.error(msg)
            return False, msg
        except Exception as e:
            msg = f"Test command error: {e}"
            logger.error(msg)
            return False, msg

    @staticmethod
    def rollback(backups: dict[str, str | None]) -> None:
        """Restore original file contents from in-memory backups.

        Args:
            backups: Mapping of filepath -> original content.
                     None values indicate newly created files that
                     should be deleted.
        """
        for filepath, original_content in backups.items():
            path = Path(filepath)
            if original_content is None:
                # File was newly created — delete it
                if path.exists():
                    path.unlink()
                    logger.info("Deleted new file: %s", filepath)
            else:
                # Restore original content
                path.write_text(original_content, encoding="utf-8")
                logger.info("Restored: %s", filepath)

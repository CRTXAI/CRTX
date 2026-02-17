"""Apply engine — orchestrates the full apply flow.

Coordinates git safety checks, file path resolution, diff preview,
file writing, git commit, post-apply testing, and rollback.
This is a synchronous engine (no LLM calls).
"""

from __future__ import annotations

import logging
from pathlib import Path

from rich.console import Console
from rich.panel import Panel
from rich.text import Text

from triad.apply.conflict import ConflictDetector
from triad.apply.diff import ConflictResolver, DiffPreview
from triad.apply.git import GitSafetyGate
from triad.apply.resolver import FilePathResolver, extract_code_blocks_from_result
from triad.apply.verify import PostApplyVerifier
from triad.schemas.apply import (
    ApplyConfig,
    ApplyResult,
    ConflictAction,
    FileAction,
    FileConflict,
    GitState,
    ResolvedFile,
)
from triad.schemas.pipeline import PipelineResult

logger = logging.getLogger(__name__)


class ApplyEngine:
    """Orchestrates the full apply flow.

    Synchronous engine that takes a PipelineResult and ApplyConfig,
    resolves files, shows diffs, writes to disk, commits, and
    optionally runs tests with rollback.
    """

    def __init__(
        self,
        result: PipelineResult,
        config: ApplyConfig,
        context_dir: str,
        console: Console,
        interactive: bool = True,
    ) -> None:
        self._result = result
        self._config = config
        self._context_dir = Path(context_dir)
        self._console = console
        self._interactive = interactive

    def run(self) -> ApplyResult:
        """Execute the full apply flow.

        Flow:
        1. Git safety gates
        2. Resolve file paths
        3. Baseline test run (if configured)
        4. Diff preview (interactive only)
        5. Backup + write files
        6. Git commit
        7. Post-apply test + rollback on failure
        8. Build ApplyResult

        Returns:
            ApplyResult audit record.
        """
        errors: list[str] = []
        commit_sha = ""
        test_passed: bool | None = None
        test_output = ""
        rolled_back = False

        # ── 1. Git safety gates ──────────────────────────────────
        git = GitSafetyGate(self._context_dir)
        git_state = git.check_state()

        arbiter_verdicts = [
            r.verdict.value for r in self._result.arbiter_reviews
        ]
        warnings = git.ensure_safe(git_state, arbiter_verdicts)

        if warnings:
            self._console.print()
            for w in warnings:
                self._console.print(f"  [yellow]Warning:[/yellow] {w}")

            # Block on REJECT/HALT
            blocking = [w for w in warnings if "blocked" in w.lower()]
            if blocking:
                self._console.print(
                    "\n  [red]Apply blocked.[/red] Fix arbiter issues first."
                )
                return ApplyResult(
                    session_id=self._result.session_id,
                    git_state=git_state,
                    errors=[w for w in warnings],
                )

        # Create branch if requested
        if self._config.branch and git_state.is_git_repo:
            try:
                branch_name = git.create_branch(self._config.branch)
                git_state.created_branch = branch_name
                self._console.print(
                    f"  [green]Created branch:[/green] {branch_name}"
                )
            except RuntimeError as e:
                errors.append(str(e))
                self._console.print(f"  [red]Branch error:[/red] {e}")

        # ── 2. Resolve file paths ────────────────────────────────
        blocks = extract_code_blocks_from_result(self._result)
        if not blocks:
            self._console.print("[dim]No code blocks found to apply.[/dim]")
            return ApplyResult(
                session_id=self._result.session_id,
                git_state=git_state,
                errors=["No code blocks found in pipeline output"],
            )

        resolver = FilePathResolver(self._context_dir, self._config)
        resolved = resolver.resolve(blocks)

        # ── 2b. Conflict detection ────────────────────────────────
        detector = ConflictDetector()
        for f in resolved:
            if f.action == FileAction.OVERWRITE:
                detector.snapshot(f.resolved_path)

        conflict_paths = detector.check_all()
        if conflict_paths:
            file_conflicts = []
            for fp in conflict_paths:
                snap = detector._snapshots.get(fp)
                scan_time = snap[0] if snap else 0.0
                current_mtime = Path(fp).stat().st_mtime if Path(fp).exists() else 0.0
                scanned = ""
                current = ""
                if Path(fp).exists():
                    current = Path(fp).read_text(encoding="utf-8", errors="replace")
                # Find the resolved file to get content at scan time
                for rf in resolved:
                    if rf.resolved_path == fp and rf.existing_content is not None:
                        scanned = rf.existing_content
                        break

                file_conflicts.append(FileConflict(
                    filepath=fp,
                    scan_time=scan_time,
                    current_mtime=current_mtime,
                    scanned_content=scanned,
                    current_content=current,
                ))

            cr = ConflictResolver(self._console, interactive=self._interactive)
            resolutions = cr.resolve(file_conflicts)

            if not resolutions and file_conflicts:
                # User cancelled all
                self._console.print("[dim]Apply cancelled.[/dim]")
                return ApplyResult(
                    session_id=self._result.session_id,
                    git_state=git_state,
                    errors=["Apply cancelled due to conflicts"],
                )

            # Apply resolution decisions
            skip_paths = {
                r.filepath for r in resolutions
                if r.action == ConflictAction.SKIP
            }
            for f in resolved:
                if f.resolved_path in skip_paths:
                    f.selected = False

        # ── 3. Baseline test run ─────────────────────────────────
        if self._config.test_command and self._config.rollback_on_fail:
            self._console.print("\n  [dim]Running baseline tests...[/dim]")
            verifier = PostApplyVerifier(self._context_dir, self._config.test_command)
            baseline_passed, baseline_output = verifier.run_test_command()
            if not baseline_passed:
                self._console.print(
                    "  [yellow]Warning:[/yellow] Baseline tests already failing"
                )

        # ── 4. Diff preview ──────────────────────────────────────
        if self._interactive and self._config.confirm:
            preview = DiffPreview(self._console, resolved, interactive=True)
            resolved = preview.show()
        else:
            preview = DiffPreview(self._console, resolved, interactive=False)
            resolved = preview.show()

        # Filter to selected files only
        to_apply = [f for f in resolved if f.selected and f.action != FileAction.SKIP]
        to_skip = [f for f in resolved if not f.selected or f.action == FileAction.SKIP]

        if not to_apply:
            self._console.print("[dim]No files selected for apply.[/dim]")
            return ApplyResult(
                session_id=self._result.session_id,
                files_skipped=to_skip,
                git_state=git_state,
            )

        # ── 5. Backup + write ────────────────────────────────────
        backups: dict[str, str | None] = {}

        for f in to_apply:
            path = Path(f.resolved_path)

            # Backup existing file content (None for new files)
            if path.exists():
                backups[f.resolved_path] = path.read_text(
                    encoding="utf-8", errors="replace"
                )
            else:
                backups[f.resolved_path] = None

            # Write file
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(f.content, encoding="utf-8")

        self._console.print(
            f"\n  [green]Applied {len(to_apply)} file(s)[/green]"
        )

        # ── 6. Git commit ────────────────────────────────────────
        if git_state.is_git_repo:
            try:
                file_paths = [f.resolved_path for f in to_apply]
                message = (
                    f"triad apply: {len(to_apply)} files "
                    f"(session {self._result.session_id[:8]})"
                )
                commit_sha = git.commit_changes(file_paths, message)
                self._console.print(
                    f"  [green]Committed:[/green] {commit_sha[:12]}"
                )
            except RuntimeError as e:
                errors.append(str(e))
                self._console.print(f"  [red]Commit error:[/red] {e}")

        # ── 7. Post-apply test + rollback ────────────────────────
        if self._config.test_command:
            self._console.print("\n  [dim]Running post-apply tests...[/dim]")
            verifier = PostApplyVerifier(self._context_dir, self._config.test_command)
            test_passed, test_output = verifier.run_test_command()

            if test_passed:
                self._console.print("  [green]Tests passed![/green]")
            else:
                self._console.print("  [red]Tests failed![/red]")
                if self._config.rollback_on_fail:
                    self._console.print("  [yellow]Rolling back changes...[/yellow]")
                    PostApplyVerifier.rollback(backups)
                    # Also git rollback if we committed
                    if commit_sha and git_state.head_sha:
                        try:
                            git.rollback_to(git_state.head_sha)
                            self._console.print(
                                f"  [yellow]Git rolled back to {git_state.head_sha[:12]}[/yellow]"
                            )
                        except RuntimeError as e:
                            errors.append(f"Git rollback failed: {e}")
                    rolled_back = True

        # ── 8. Build result ──────────────────────────────────────
        return ApplyResult(
            session_id=self._result.session_id,
            files_applied=to_apply if not rolled_back else [],
            files_skipped=to_skip,
            git_state=git_state,
            commit_sha=commit_sha if not rolled_back else "",
            test_passed=test_passed,
            test_output=test_output,
            rolled_back=rolled_back,
            errors=errors,
        )

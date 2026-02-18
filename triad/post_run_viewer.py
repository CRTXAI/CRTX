"""Interactive post-run output viewer.

Renders pipeline results in the terminal after a run completes.
Provides sub-views for summary, code files, arbiter reviews, and diffs.
"""

from __future__ import annotations

from pathlib import Path

import rich.box
from rich.console import Console
from rich.markdown import Markdown
from rich.panel import Panel
from rich.syntax import Syntax
from rich.table import Table

_STAGE_COLORS = {
    "architect": "cyan",
    "implement": "green",
    "refactor": "yellow",
    "verify": "magenta",
}

_VERDICT_COLORS = {
    "approve": "green",
    "flag": "yellow",
    "reject": "red",
    "halt": "bold red",
}

_LANG_MAP = {
    ".py": "python", ".js": "javascript", ".ts": "typescript",
    ".toml": "toml", ".json": "json", ".md": "markdown",
    ".yaml": "yaml", ".yml": "yaml", ".html": "html",
    ".css": "css", ".sql": "sql", ".sh": "bash",
    ".rs": "rust", ".go": "go", ".java": "java",
    ".rb": "ruby", ".cpp": "cpp", ".c": "c", ".h": "c",
}


class PostRunViewer:
    """Interactive post-run output viewer.

    Renders pipeline results in the terminal without leaving the CLI.
    Called after the completion panel prints with menu keys.
    """

    def __init__(
        self,
        console: Console,
        session_dir: Path,
        result: object,
    ) -> None:
        self.console = console
        self.session_dir = session_dir
        self.result = result

    def run(self) -> str | None:
        """Main input loop. Returns when user presses Enter/q.

        Uses single-keypress reading (msvcrt on Windows, tty on Unix)
        so the user can press s/c/r/d/Enter without typing + Enter.
        The first iteration skips the menu banner because the completion
        panel already displays the key hints.
        """
        from triad.cli_display import _read_key

        first = True
        while True:
            if not first:
                self.console.print(
                    "\n[green]\\[s][/] Summary  [green]\\[c][/] Code  "
                    "[green]\\[r][/] Reviews  [green]\\[d][/] Diffs  "
                    "[green]\\[Enter][/] Exit"
                )
            first = False

            try:
                key = _read_key()
            except (EOFError, KeyboardInterrupt):
                break

            if key in ("enter", "q", "escape"):
                break
            elif key == "s":
                self._show_summary()
            elif key == "c":
                self._show_code()
            elif key == "r":
                self._show_reviews()
            elif key == "d":
                self._show_diffs()
        return None

    def run_direct(self, view: str) -> None:
        """Show a specific view without the interactive loop."""
        if view == "summary":
            self._show_summary()
        elif view == "code":
            self._show_code()
        elif view == "reviews":
            self._show_reviews()
        elif view == "diffs":
            self._show_diffs()

    # ── Summary view ───────────────────────────────────────────────

    def _show_summary(self) -> None:
        """Render summary.md with Rich Markdown."""
        summary_path = self.session_dir / "summary.md"
        if not summary_path.exists():
            self.console.print("[dim]No summary available.[/dim]")
            return

        content = summary_path.read_text(encoding="utf-8")
        self.console.print()
        self.console.print(Panel(
            Markdown(content),
            title="[bold bright_blue]Summary[/bold bright_blue]",
            border_style="dim",
            box=rich.box.ROUNDED,
        ))

    # ── Code view ──────────────────────────────────────────────────

    def _show_code(self) -> None:
        """List code files, let user pick one to view with syntax highlighting."""
        code_dir = self.session_dir / "code"
        tests_dir = self.session_dir / "tests"

        files: list[Path] = []
        if code_dir.exists():
            files.extend(sorted(f for f in code_dir.rglob("*") if f.is_file()))
        if tests_dir.exists():
            files.extend(sorted(f for f in tests_dir.rglob("*") if f.is_file()))

        if not files:
            self.console.print("[dim]No code files found.[/dim]")
            return

        # Determine base dir for relative paths
        base_dir = self.session_dir

        # File listing table
        table = Table(
            title="Generated Files",
            show_header=True,
            header_style="bold bright_blue",
            border_style="dim",
            box=rich.box.ROUNDED,
        )
        table.add_column("#", style="dim", width=4)
        table.add_column("File", style="bold")
        table.add_column("Language", style="cyan")
        table.add_column("Lines", justify="right")
        table.add_column("Size", justify="right")

        for i, f in enumerate(files, 1):
            rel = f.relative_to(base_dir)
            lang = _LANG_MAP.get(f.suffix, f.suffix.lstrip(".") or "text")
            try:
                lines = len(f.read_text(errors="replace").splitlines())
            except OSError:
                lines = 0
            size = f.stat().st_size
            size_str = f"{size}B" if size < 1024 else f"{size / 1024:.1f}K"
            table.add_row(str(i), str(rel), lang, str(lines), size_str)

        self.console.print()
        self.console.print(table)

        # File selection loop
        while True:
            self.console.print(
                "\n[dim]Enter file number to view (or b to go back):[/dim] ",
                end="",
            )
            try:
                choice = input().strip().lower()
            except (EOFError, KeyboardInterrupt):
                break

            if choice in ("b", ""):
                break

            try:
                idx = int(choice) - 1
                if 0 <= idx < len(files):
                    self._view_file(files[idx], base_dir)
                else:
                    self.console.print("[red]Invalid number.[/red]")
            except ValueError:
                self.console.print("[red]Enter a number or 'b'.[/red]")

    def _show_code_by_index(self, index: int) -> None:
        """View a specific code file by its 1-based index."""
        code_dir = self.session_dir / "code"
        tests_dir = self.session_dir / "tests"

        files: list[Path] = []
        if code_dir.exists():
            files.extend(sorted(f for f in code_dir.rglob("*") if f.is_file()))
        if tests_dir.exists():
            files.extend(sorted(f for f in tests_dir.rglob("*") if f.is_file()))

        idx = index - 1
        if 0 <= idx < len(files):
            self._view_file(files[idx], self.session_dir)
        else:
            self.console.print(f"[red]File #{index} not found.[/red]")

    def _view_file(self, path: Path, base_dir: Path) -> None:
        """Display a single file with syntax highlighting."""
        try:
            code = path.read_text(errors="replace")
        except OSError:
            self.console.print(f"[red]Cannot read {path}[/red]")
            return

        lang = _LANG_MAP.get(path.suffix, "text")
        rel = path.relative_to(base_dir)

        syntax = Syntax(
            code, lang,
            theme="monokai",
            line_numbers=True,
            word_wrap=False,
        )
        self.console.print()
        self.console.print(Panel(
            syntax,
            title=f"[bold]{rel}[/bold]",
            subtitle=f"[dim]{lang} · {len(code.splitlines())} lines[/dim]",
            border_style="bright_blue",
            box=rich.box.ROUNDED,
        ))

    # ── Reviews view ───────────────────────────────────────────────

    def _show_reviews(self) -> None:
        """Show arbiter reviews with verdict badges."""
        reviews = getattr(self.result, "arbiter_reviews", [])
        if not reviews:
            self.console.print("[dim]No arbiter reviews available.[/dim]")
            return

        for review in reviews:
            verdict_val = (
                review.verdict.value
                if hasattr(review.verdict, "value")
                else str(review.verdict)
            )
            stage_val = (
                review.stage_reviewed.value
                if hasattr(review.stage_reviewed, "value")
                else str(review.stage_reviewed)
            )

            v_color = _VERDICT_COLORS.get(verdict_val, "white")
            stage_color = _STAGE_COLORS.get(stage_val, "white")

            header = (
                f"[{stage_color} bold]{stage_val.upper()}[/]  "
                f"[{v_color} bold]{verdict_val.upper()}[/]  "
                f"[dim]conf {review.confidence:.2f} · {review.arbiter_model}[/dim]"
            )

            # Build review body
            body_parts: list[str] = []
            if review.reasoning:
                body_parts.append(review.reasoning)

            if review.issues:
                body_parts.append("\n**Issues:**")
                for issue in review.issues:
                    sev = (
                        issue.severity.value.upper()
                        if hasattr(issue.severity, "value")
                        else str(issue.severity)
                    )
                    cat = (
                        issue.category.value
                        if hasattr(issue.category, "value")
                        else str(issue.category)
                    )
                    loc = f" at `{issue.location}`" if issue.location else ""
                    body_parts.append(f"- **[{sev}]** [{cat}]{loc}: {issue.description}")
                    if issue.suggestion:
                        body_parts.append(f"  - *Fix:* {issue.suggestion}")

            body = "\n".join(body_parts) if body_parts else "No details."

            self.console.print()
            self.console.print(Panel(
                Markdown(body),
                title=header,
                border_style=v_color.split()[-1] if " " in v_color else v_color,
                box=rich.box.ROUNDED,
            ))

    # ── Diffs view ─────────────────────────────────────────────────

    def _show_diffs(self) -> None:
        """Show refactor diffs if available."""
        stages = getattr(self.result, "stages", {})

        # Look for refactor stage output
        diff_content = ""
        for stage_key, stage_msg in stages.items():
            stage_name = stage_key.value if hasattr(stage_key, "value") else str(stage_key)
            if stage_name == "refactor" and stage_msg.content:
                diff_content = stage_msg.content
                break

        if not diff_content:
            self.console.print("[dim]No diffs available.[/dim]")
            return

        self.console.print()
        from triad.cli_streaming_display import _looks_like_diff

        # Extract diff blocks from the content
        lines = diff_content.splitlines()
        in_diff = False
        diff_lines: list[str] = []

        for line in lines:
            if line.startswith("---") or line.startswith("+++") or line.startswith("@@"):
                in_diff = True
            if in_diff or _looks_like_diff("\n".join(lines)):
                diff_lines.append(line)

        if diff_lines:
            self._render_diff("\n".join(diff_lines))
        else:
            # Show the full refactor output as markdown
            self.console.print(Panel(
                Markdown(diff_content[:2000]),
                title="[bold]Refactor Output[/bold]",
                border_style="yellow",
                box=rich.box.ROUNDED,
            ))

    def _render_diff(self, content: str) -> None:
        """Render diff content with colors."""
        from rich.text import Text

        text = Text()
        for line in content.splitlines():
            if line.startswith("+++") or line.startswith("---"):
                text.append(line + "\n", style="bold")
            elif line.startswith("+"):
                text.append(line + "\n", style="green")
            elif line.startswith("-"):
                text.append(line + "\n", style="red")
            elif line.startswith("@@"):
                text.append(line + "\n", style="cyan dim")
            else:
                text.append(line + "\n", style="dim")

        self.console.print(Panel(
            text,
            title="[bold]Diffs[/bold]",
            border_style="yellow",
            box=rich.box.ROUNDED,
        ))


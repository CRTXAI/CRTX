"""Rich diff preview for apply mode.

Renders an interactive preview of file changes before writing,
with unified diff display and per-file selection.
"""

from __future__ import annotations

import difflib

from rich.console import Console
from rich.panel import Panel
from rich.table import Table
from rich.text import Text

from triad.schemas.apply import FileAction, ResolvedFile


class DiffPreview:
    """Interactive diff preview before applying files.

    Shows a summary table of changes and allows viewing unified
    diffs per file. In non-interactive mode, prints the summary
    and proceeds.
    """

    def __init__(
        self,
        console: Console,
        files: list[ResolvedFile],
        interactive: bool = True,
    ) -> None:
        self._console = console
        self._files = files
        self._interactive = interactive

    def show(self) -> list[ResolvedFile]:
        """Display the preview and return the (possibly modified) file list.

        Returns:
            The file list with selected/deselected flags updated.
        """
        active = [f for f in self._files if f.action != FileAction.SKIP]
        if not active:
            self._console.print("[dim]No files to apply.[/dim]")
            return self._files

        self._print_summary(active)

        if not self._interactive:
            return self._files

        return self._interactive_loop(active)

    def _print_summary(self, files: list[ResolvedFile]) -> None:
        """Print the file summary table."""
        table = Table(title="Apply Preview", show_lines=True)
        table.add_column("#", style="dim", justify="right", width=3)
        table.add_column("Action", width=6)
        table.add_column("File", style="cyan")
        table.add_column("Lines", justify="right")
        table.add_column("Confidence", justify="right")

        for i, f in enumerate(files, 1):
            if f.action == FileAction.CREATE:
                action = Text("+ NEW", style="bold green")
            elif f.action == FileAction.OVERWRITE:
                action = Text("* MOD", style="bold yellow")
            else:
                action = Text("- SKIP", style="dim")

            line_count = len(f.content.splitlines())
            conf = f"{f.match_confidence:.0%}" if f.action == FileAction.OVERWRITE else "-"

            selected = "[green]>[/green] " if f.selected else "  "

            table.add_row(
                f"{selected}{i}",
                action,
                f.source_filepath,
                str(line_count),
                conf,
            )

        self._console.print(table)

    def _interactive_loop(self, active: list[ResolvedFile]) -> list[ResolvedFile]:
        """Run the interactive selection loop."""
        self._console.print()
        self._console.print(
            "[dim][a]pply all  [d]iff <#>  [s]elect/deselect <#>  [q]uit[/dim]"
        )

        while True:
            try:
                choice = self._console.input("[bold]apply>[/bold] ").strip().lower()
            except (EOFError, KeyboardInterrupt):
                # Cancel
                for f in self._files:
                    f.selected = False
                return self._files

            if choice == "q":
                for f in self._files:
                    f.selected = False
                return self._files

            if choice == "a":
                return self._files

            if choice.startswith("d"):
                self._show_diff(choice, active)
                continue

            if choice.startswith("s"):
                self._toggle_selection(choice, active)
                continue

            self._console.print("[dim]Unknown command. Use a/d/s/q.[/dim]")

    def _show_diff(self, command: str, active: list[ResolvedFile]) -> None:
        """Show unified diff for a file by index."""
        parts = command.split()
        if len(parts) < 2:
            self._console.print("[dim]Usage: d <number>[/dim]")
            return

        try:
            idx = int(parts[1]) - 1
        except ValueError:
            self._console.print("[dim]Invalid number.[/dim]")
            return

        if idx < 0 or idx >= len(active):
            self._console.print(f"[dim]File #{idx + 1} not found.[/dim]")
            return

        f = active[idx]

        if f.action == FileAction.CREATE:
            # New file — show full content
            text = Text()
            for line in f.content.splitlines():
                text.append(f"+ {line}\n", style="green")
            self._console.print(Panel(
                text,
                title=f"[bold green]NEW[/bold green] {f.source_filepath}",
                border_style="green",
            ))
        elif f.existing_content is not None:
            # Modified — show unified diff
            old_lines = f.existing_content.splitlines(keepends=True)
            new_lines = f.content.splitlines(keepends=True)
            diff = difflib.unified_diff(
                old_lines, new_lines,
                fromfile=f"a/{f.source_filepath}",
                tofile=f"b/{f.source_filepath}",
            )
            text = Text()
            for line in diff:
                line = line.rstrip("\n")
                if line.startswith("+"):
                    text.append(f"{line}\n", style="green")
                elif line.startswith("-"):
                    text.append(f"{line}\n", style="red")
                elif line.startswith("@@"):
                    text.append(f"{line}\n", style="cyan")
                else:
                    text.append(f"{line}\n")
            self._console.print(Panel(
                text,
                title=f"[bold yellow]DIFF[/bold yellow] {f.source_filepath}",
                border_style="yellow",
            ))

    def _toggle_selection(self, command: str, active: list[ResolvedFile]) -> None:
        """Toggle file selection by index."""
        parts = command.split()
        if len(parts) < 2:
            self._console.print("[dim]Usage: s <number>[/dim]")
            return

        try:
            idx = int(parts[1]) - 1
        except ValueError:
            self._console.print("[dim]Invalid number.[/dim]")
            return

        if idx < 0 or idx >= len(active):
            self._console.print(f"[dim]File #{idx + 1} not found.[/dim]")
            return

        f = active[idx]
        f.selected = not f.selected
        state = "[green]selected[/green]" if f.selected else "[red]deselected[/red]"
        self._console.print(f"  {f.source_filepath}: {state}")

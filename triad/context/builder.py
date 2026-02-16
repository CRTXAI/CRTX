"""Context builder for assembling ranked project context.

Takes scanned files from CodeScanner, scores them by relevance to the
task description, sorts by score, and assembles a context string that
fits within a token budget. Produces a ContextResult with profile data.
"""

from __future__ import annotations

import logging
import re

from triad.schemas.context import (
    ContextResult,
    ProjectProfile,
    ScannedFile,
)

logger = logging.getLogger(__name__)

# Rough estimate: 1 token ≈ 4 characters
_CHARS_PER_TOKEN = 4


class ContextBuilder:
    """Builds a ranked context string from scanned project files.

    Scores each file's relevance to the task, selects files until the
    token budget is filled, and assembles them into a formatted context
    string for injection into pipeline prompts.
    """

    def __init__(
        self,
        root_path: str,
        token_budget: int = 8000,
    ) -> None:
        self._root_path = root_path
        self._token_budget = token_budget

    def build(
        self,
        files: list[ScannedFile],
        task: str,
    ) -> ContextResult:
        """Score, rank, and assemble files into a context string.

        Args:
            files: Scanned files from CodeScanner.
            task: The task description used for relevance scoring.

        Returns:
            A ContextResult with the assembled context and metadata.
        """
        profile = self._build_profile(files)

        # Score each file
        for f in files:
            f.relevance_score = self._score_file(f, task)

        # Sort by relevance (highest first)
        ranked = sorted(files, key=lambda f: f.relevance_score, reverse=True)

        # Assemble context within token budget
        context_lines: list[str] = []
        char_budget = self._token_budget * _CHARS_PER_TOKEN
        chars_used = 0
        files_included = 0
        truncated = False

        # Add project profile header
        header = self._format_profile(profile)
        header_chars = len(header)
        if header_chars < char_budget:
            context_lines.append(header)
            chars_used += header_chars

        for f in ranked:
            entry = self._format_file_entry(f)
            entry_chars = len(entry)

            if chars_used + entry_chars > char_budget:
                truncated = True
                break

            context_lines.append(entry)
            chars_used += entry_chars
            files_included += 1

        context_text = "\n".join(context_lines)
        token_estimate = len(context_text) // _CHARS_PER_TOKEN

        return ContextResult(
            profile=profile,
            context_text=context_text,
            files_included=files_included,
            files_scanned=len(files),
            token_estimate=token_estimate,
            truncated=truncated,
        )

    def _build_profile(self, files: list[ScannedFile]) -> ProjectProfile:
        """Build a high-level project profile from scanned files."""
        languages: dict[str, int] = {}
        total_lines = 0
        entry_points: list[str] = []
        key_patterns: list[str] = set()  # type: ignore[assignment]

        for f in files:
            lang = f.language
            languages[lang] = languages.get(lang, 0) + 1

            # Estimate lines from size
            if f.size_bytes > 0:
                total_lines += f.size_bytes // 40  # rough ~40 bytes/line

            # Detect entry points
            name = f.path.split("/")[-1] if "/" in f.path else f.path
            if name in (
                "main.py", "app.py", "server.py", "manage.py",
                "wsgi.py", "asgi.py", "index.js", "index.ts",
                "main.go", "main.rs",
            ):
                entry_points.append(f.path)

            # Detect key patterns from imports
            for imp in f.imports:
                if "fastapi" in imp.lower():
                    key_patterns.add("FastAPI")
                elif "django" in imp.lower():
                    key_patterns.add("Django")
                elif "flask" in imp.lower():
                    key_patterns.add("Flask")
                elif "sqlalchemy" in imp.lower():
                    key_patterns.add("SQLAlchemy")
                elif "pydantic" in imp.lower():
                    key_patterns.add("Pydantic")
                elif "pytest" in imp.lower():
                    key_patterns.add("pytest")
                elif "react" in imp.lower():
                    key_patterns.add("React")
                elif "express" in imp.lower():
                    key_patterns.add("Express")

        return ProjectProfile(
            root_path=self._root_path,
            total_files=len(files),
            total_lines=total_lines,
            languages=languages,
            entry_points=entry_points,
            key_patterns=sorted(key_patterns),
        )

    def _score_file(self, f: ScannedFile, task: str) -> float:
        """Compute a relevance score (0.0–1.0) for a file against the task.

        Scoring heuristics:
        - Keyword overlap between task and file path/content
        - Python files with classes/functions score higher
        - Entry points score higher
        - Test files score lower (but not zero)
        - Larger files get a small penalty
        """
        score = 0.0
        task_lower = task.lower()
        task_words = set(re.findall(r"\w+", task_lower))

        # Path keyword match
        path_lower = f.path.lower()
        path_words = set(re.findall(r"\w+", path_lower))
        overlap = task_words & path_words
        if overlap:
            score += 0.3 * min(len(overlap) / max(len(task_words), 1), 1.0)

        # Content keyword match (docstring, preview, class/function names)
        content_words: set[str] = set()
        if f.docstring:
            content_words.update(re.findall(r"\w+", f.docstring.lower()))
        if f.preview:
            content_words.update(re.findall(r"\w+", f.preview.lower()))
        for cls in f.classes:
            content_words.update(re.findall(r"\w+", cls.lower()))
        for func in f.functions:
            content_words.update(re.findall(r"\w+", func.name.lower()))

        content_overlap = task_words & content_words
        if content_overlap:
            score += 0.3 * min(
                len(content_overlap) / max(len(task_words), 1), 1.0,
            )

        # Structural richness bonus (Python files with classes/funcs)
        if f.classes or f.functions:
            score += 0.15

        # Entry point bonus
        name = f.path.split("/")[-1] if "/" in f.path else f.path
        if name in (
            "main.py", "app.py", "server.py", "manage.py",
            "__init__.py",
        ):
            score += 0.1

        # Test file penalty
        if "test" in path_lower or "spec" in path_lower:
            score *= 0.5

        # Config/schema bonus
        if "schema" in path_lower or "model" in path_lower:
            score += 0.05
        if "config" in path_lower or "settings" in path_lower:
            score += 0.05

        # Size penalty for very large files
        if f.size_bytes > 50_000:
            score *= 0.8
        elif f.size_bytes > 100_000:
            score *= 0.6

        return min(score, 1.0)

    def _format_profile(self, profile: ProjectProfile) -> str:
        """Format the project profile as a context header."""
        lines = [
            "## Project Profile",
            f"- Root: {profile.root_path}",
            f"- Files: {profile.total_files}",
            f"- Approx. lines: {profile.total_lines:,}",
        ]
        if profile.languages:
            lang_str = ", ".join(
                f"{lang}: {count}"
                for lang, count in sorted(
                    profile.languages.items(),
                    key=lambda x: x[1],
                    reverse=True,
                )
            )
            lines.append(f"- Languages: {lang_str}")
        if profile.entry_points:
            lines.append(f"- Entry points: {', '.join(profile.entry_points)}")
        if profile.key_patterns:
            lines.append(f"- Patterns: {', '.join(profile.key_patterns)}")
        lines.append("")
        return "\n".join(lines)

    def _format_file_entry(self, f: ScannedFile) -> str:
        """Format a single file as a context entry."""
        lines = [f"### {f.path} ({f.language}, {f.size_bytes:,} bytes)"]

        if f.docstring:
            lines.append(f"  Docstring: {f.docstring[:200]}")

        if f.classes:
            lines.append(f"  Classes: {', '.join(f.classes)}")

        if f.functions:
            sigs = []
            for func in f.functions[:20]:  # Cap at 20 functions
                sig = func.name
                if func.args:
                    sig += f"({', '.join(func.args[:5])})"
                if func.return_type:
                    sig += f" -> {func.return_type}"
                if func.is_async:
                    sig = f"async {sig}"
                sigs.append(sig)
            lines.append(f"  Functions: {', '.join(sigs)}")

        if f.imports:
            # Show first 10 imports
            imp_list = f.imports[:10]
            if len(f.imports) > 10:
                imp_list.append(f"... +{len(f.imports) - 10} more")
            lines.append(f"  Imports: {', '.join(imp_list)}")

        if f.preview:
            preview = f.preview[:300]
            if len(f.preview) > 300:
                preview += "\n  ..."
            lines.append(f"  Preview:\n  {preview}")

        lines.append("")
        return "\n".join(lines)

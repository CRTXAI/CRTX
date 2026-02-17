"""File path resolver for apply mode.

Maps code block filepath hints to real filesystem paths under the
context directory. Uses a resolution cascade: exact match, basename
match, fuzzy match, or create new.
"""

from __future__ import annotations

import difflib
import fnmatch
import re
from pathlib import Path

from triad.schemas.apply import ApplyConfig, FileAction, ResolvedFile
from triad.schemas.messages import CodeBlock


# Regex for detecting substantive code (function/class definitions)
_HAS_DEFINITION = re.compile(
    r"^\s*(?:def |class |function |const |let |var |pub fn |fn |async def )",
    re.MULTILINE,
)


def extract_code_blocks_from_result(result) -> list[CodeBlock]:
    """Extract code blocks from a PipelineResult.

    Merges structured code_blocks from AgentMessages with regex-parsed
    blocks from raw content. Shared utility used by both writer.py and
    resolver.py.

    Args:
        result: A PipelineResult with stage outputs.

    Returns:
        Deduplicated list of CodeBlock instances.
    """
    # Regex to match code blocks with optional file hints
    code_block_re = re.compile(
        r"```(\w+)\n(?:#\s*file:\s*(.+?)\n)?(.*?)```",
        re.DOTALL,
    )

    blocks: list[CodeBlock] = []
    seen_filenames: set[str] = set()

    # First: structured code blocks from messages
    for msg in result.stages.values():
        for block in msg.code_blocks:
            content = block.content.strip()
            # Skip untitled fragments
            if (
                block.filepath.startswith("untitled")
                and len(content) < 200
                and not _HAS_DEFINITION.search(content)
            ):
                continue
            filename = Path(block.filepath).name
            if filename not in seen_filenames:
                blocks.append(block)
                seen_filenames.add(filename)

    # Then: regex-parsed blocks from the last stage's raw content
    final_content = ""
    for stage_msg in result.stages.values():
        final_content = stage_msg.content

    if final_content:
        counter = 0
        for match in code_block_re.finditer(final_content):
            language = match.group(1)
            filepath_hint = match.group(2)
            code = match.group(3).strip()
            if not code:
                continue

            if filepath_hint:
                filename = Path(filepath_hint.strip()).name
            else:
                counter += 1
                ext = _language_extension(language)
                filename = f"output_{counter}{ext}"

            if filename not in seen_filenames:
                blocks.append(CodeBlock(
                    language=language,
                    filepath=filepath_hint.strip() if filepath_hint else filename,
                    content=code,
                ))
                seen_filenames.add(filename)

    return blocks


def _language_extension(language: str) -> str:
    """Map language identifiers to file extensions."""
    mapping = {
        "python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts", "rust": ".rs", "go": ".go",
        "java": ".java", "toml": ".toml", "yaml": ".yaml", "yml": ".yml",
        "json": ".json", "html": ".html", "css": ".css", "sql": ".sql",
        "bash": ".sh", "sh": ".sh",
    }
    return mapping.get(language.lower(), f".{language}")


class FilePathResolver:
    """Maps code block filepath hints to real filesystem paths.

    Resolution cascade:
    1. Exact join: context_dir / hint_path
    2. Basename match against file tree
    3. Fuzzy match (SequenceMatcher, threshold 0.7)
    4. Create new file at context_dir / hint_path
    """

    _FUZZY_THRESHOLD = 0.7

    def __init__(self, context_dir: Path, config: ApplyConfig) -> None:
        self._context_dir = context_dir
        self._config = config
        self._file_tree = self._scan_file_tree()

    def _scan_file_tree(self) -> list[Path]:
        """Scan the context directory for existing files."""
        files: list[Path] = []
        for path in self._context_dir.rglob("*"):
            if path.is_file() and not any(
                part.startswith(".") for part in path.relative_to(self._context_dir).parts
            ):
                files.append(path)
        return files

    def resolve(self, blocks: list[CodeBlock]) -> list[ResolvedFile]:
        """Resolve a list of code blocks to disk paths.

        Args:
            blocks: Code blocks extracted from pipeline output.

        Returns:
            List of ResolvedFile instances with actions determined.
        """
        resolved: list[ResolvedFile] = []

        for block in blocks:
            hint = block.filepath
            rf = self._resolve_single(hint, block.content, block.language)
            resolved.append(rf)

        # Apply include/exclude filters
        return self._apply_filters(resolved)

    def _resolve_single(
        self, hint: str, content: str, language: str
    ) -> ResolvedFile:
        """Resolve a single filepath hint."""
        # 1. Exact join
        exact_path = self._context_dir / hint
        if exact_path.exists() and exact_path.is_file():
            return ResolvedFile(
                source_filepath=hint,
                resolved_path=str(exact_path),
                action=FileAction.OVERWRITE,
                content=content,
                language=language,
                existing_content=exact_path.read_text(encoding="utf-8", errors="replace"),
                match_confidence=1.0,
            )

        # 2. Basename match
        basename = Path(hint).name
        matches = [f for f in self._file_tree if f.name == basename]
        if len(matches) == 1:
            match_path = matches[0]
            return ResolvedFile(
                source_filepath=hint,
                resolved_path=str(match_path),
                action=FileAction.OVERWRITE,
                content=content,
                language=language,
                existing_content=match_path.read_text(encoding="utf-8", errors="replace"),
                match_confidence=0.9,
            )

        # 3. Fuzzy match
        relative_paths = [
            str(f.relative_to(self._context_dir)) for f in self._file_tree
        ]
        if relative_paths:
            best_matches = difflib.get_close_matches(
                hint, relative_paths, n=1, cutoff=self._FUZZY_THRESHOLD
            )
            if best_matches:
                fuzzy_path = self._context_dir / best_matches[0]
                ratio = difflib.SequenceMatcher(
                    None, hint, best_matches[0]
                ).ratio()
                return ResolvedFile(
                    source_filepath=hint,
                    resolved_path=str(fuzzy_path),
                    action=FileAction.OVERWRITE,
                    content=content,
                    language=language,
                    existing_content=fuzzy_path.read_text(
                        encoding="utf-8", errors="replace"
                    ),
                    match_confidence=ratio,
                )

        # 4. Create new
        new_path = self._context_dir / hint
        return ResolvedFile(
            source_filepath=hint,
            resolved_path=str(new_path),
            action=FileAction.CREATE,
            content=content,
            language=language,
            match_confidence=1.0,
        )

    def _apply_filters(self, files: list[ResolvedFile]) -> list[ResolvedFile]:
        """Apply include/exclude glob patterns."""
        for rf in files:
            rel_path = rf.source_filepath

            # If include patterns specified, file must match at least one
            if self._config.apply_include:
                if not any(
                    fnmatch.fnmatch(rel_path, pat)
                    for pat in self._config.apply_include
                ):
                    rf.action = FileAction.SKIP
                    rf.selected = False
                    continue

            # If exclude patterns specified, file must not match any
            if self._config.apply_exclude:
                if any(
                    fnmatch.fnmatch(rel_path, pat)
                    for pat in self._config.apply_exclude
                ):
                    rf.action = FileAction.SKIP
                    rf.selected = False

        return files

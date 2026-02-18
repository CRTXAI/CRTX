"""File output handler for pipeline results.

Creates structured output directories with extracted code files,
test files, arbiter reviews, a Markdown summary, and a JSON session export.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from triad.output.renderer import _display_name_from_litellm_id, render_summary
from triad.schemas.pipeline import PipelineResult

# Regex to match code blocks with optional file hints
# Matches: ```language\n# file: path/to/file.py\n...```
_CODE_BLOCK_RE = re.compile(
    r"```(\w+)\n(?:#\s*file:\s*(.+?)\n)?(.*?)```",
    re.DOTALL,
)


def write_pipeline_output(result: PipelineResult, output_dir: str) -> str:
    """Write pipeline output to a session-namespaced directory.

    Creates:
        output_dir/{session_id[:8]}/
        ├── code/          # Code files extracted from pipeline output
        ├── tests/         # Test files extracted from pipeline output
        ├── reviews/       # Arbiter review files (one per review)
        ├── summary.md     # Markdown summary report
        └── session.json   # Full session JSON export

    Args:
        result: The completed pipeline result.
        output_dir: Root directory for output.

    Returns:
        The actual output path (session-namespaced).
    """
    base = Path(output_dir) / result.session_id[:8]
    code_dir = base / "code"
    tests_dir = base / "tests"
    reviews_dir = base / "reviews"

    code_dir.mkdir(parents=True, exist_ok=True)
    tests_dir.mkdir(parents=True, exist_ok=True)
    reviews_dir.mkdir(parents=True, exist_ok=True)

    # Extract and write code files from pipeline output
    _extract_code_files(result, code_dir, tests_dir)

    # Write arbiter reviews
    for i, review in enumerate(result.arbiter_reviews):
        review_path = reviews_dir / f"review_{i + 1}_{review.stage_reviewed.value}.md"
        review_content = _format_review(review)
        review_path.write_text(review_content, encoding="utf-8")

    # Write summary
    summary = render_summary(result)
    (base / "summary.md").write_text(summary, encoding="utf-8")

    # Write session JSON
    session_data = result.model_dump(mode="json")
    (base / "session.json").write_text(
        json.dumps(session_data, indent=2),
        encoding="utf-8",
    )

    return str(base)


def _extract_code_files(
    result: PipelineResult,
    code_dir: Path,
    tests_dir: Path,
) -> None:
    """Extract code blocks from stage outputs and write to files.

    Looks for code blocks with "# file: path" hints. Files containing
    "test" in their name go to tests_dir, others to code_dir.
    Falls back to extracting from AgentMessage.code_blocks if available.

    For parallel/debate modes (where result.stages is empty), extracts
    from synthesized_output or judgment respectively.
    """
    # Track written filenames to avoid duplicates between passes
    written: set[str] = set()

    # Regex for detecting substantive code (function/class definitions)
    _has_definition = re.compile(
        r"^\s*(?:def |class |function |const |let |var |pub fn |fn |async def )",
        re.MULTILINE,
    )

    # First try structured code blocks from messages
    for msg in result.stages.values():
        for block in msg.code_blocks:
            content = block.content.strip()
            # Skip untitled fragments: short content with no definitions
            if (
                block.filepath.startswith("untitled")
                and len(content) < 200
                and not _has_definition.search(content)
            ):
                continue
            target = tests_dir if "test" in block.filepath.lower() else code_dir
            filename = Path(block.filepath).name
            file_path = target / filename
            file_path.parent.mkdir(parents=True, exist_ok=True)
            file_path.write_text(block.content, encoding="utf-8")
            written.add(filename)

    # Determine the final content to scan for code blocks
    # For sequential mode: last stage output
    # For parallel mode: synthesized_output
    # For debate mode: judgment
    final_content = ""
    for stage_msg in result.stages.values():
        final_content = stage_msg.content

    if not final_content and result.parallel_result:
        final_content = result.parallel_result.synthesized_output
    if not final_content and result.debate_result:
        final_content = result.debate_result.judgment
    if not final_content and result.review_result:
        final_content = result.review_result.synthesized_review
    if not final_content and result.improve_result:
        final_content = result.improve_result.synthesized_output

    if not final_content:
        return

    file_counter = 0
    for match in _CODE_BLOCK_RE.finditer(final_content):
        language = match.group(1)
        filepath_hint = match.group(2)
        code = match.group(3).strip()

        if not code:
            continue

        if filepath_hint:
            filename = Path(filepath_hint.strip()).name
        else:
            file_counter += 1
            ext = _language_extension(language)
            filename = f"output_{file_counter}{ext}"

        # Skip files already written by the first pass (code_blocks)
        if filename in written:
            continue

        target = tests_dir if "test" in filename.lower() else code_dir
        file_path = target / filename
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(code, encoding="utf-8")


def _language_extension(language: str) -> str:
    """Map language identifiers to file extensions."""
    mapping = {
        "python": ".py",
        "py": ".py",
        "javascript": ".js",
        "js": ".js",
        "typescript": ".ts",
        "ts": ".ts",
        "rust": ".rs",
        "go": ".go",
        "java": ".java",
        "toml": ".toml",
        "yaml": ".yaml",
        "yml": ".yml",
        "json": ".json",
        "html": ".html",
        "css": ".css",
        "sql": ".sql",
        "bash": ".sh",
        "sh": ".sh",
    }
    return mapping.get(language.lower(), f".{language}")


def _format_review(review) -> str:
    """Format an ArbiterReview as a Markdown file."""
    lines: list[str] = []
    lines.append(
        f"# Arbiter Review: {review.stage_reviewed.value.title()}"
    )
    lines.append("")
    lines.append(
        f"**Verdict:** {review.verdict.value.upper()}"
    )
    lines.append(f"**Confidence:** {review.confidence:.2f}")
    lines.append(f"**Reviewed Model:** {_display_name_from_litellm_id(review.reviewed_model)}")
    lines.append(f"**Arbiter Model:** {_display_name_from_litellm_id(review.arbiter_model)}")
    lines.append(f"**Cost:** ${review.token_cost:.4f}")
    lines.append("")

    if review.reasoning:
        lines.append("## Reasoning")
        lines.append("")
        lines.append(review.reasoning)
        lines.append("")

    if review.issues:
        lines.append("## Issues")
        lines.append("")
        for issue in review.issues:
            loc = f" at `{issue.location}`" if issue.location else ""
            lines.append(
                f"- **[{issue.severity.value.upper()}]** "
                f"[{issue.category.value}]{loc}: {issue.description}"
            )
            if issue.suggestion:
                lines.append(f"  - *Fix:* {issue.suggestion}")
        lines.append("")

    if review.alternatives:
        lines.append("## Alternatives")
        lines.append("")
        for alt in review.alternatives:
            lines.append(f"- {alt.description} (confidence: {alt.confidence:.2f})")
            lines.append(f"  - *Rationale:* {alt.rationale}")
            if alt.code_sketch:
                lines.append(f"  - *Code:* `{alt.code_sketch}`")
        lines.append("")

    return "\n".join(lines)

"""Single-model benchmark runner.

Makes one LLM call for a benchmark prompt and extracts code files from
the response, reusing the same extraction logic as the main pipeline.
"""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from triad.benchmark.prompts import BenchmarkPrompt
from triad.providers.litellm_provider import LiteLLMProvider, extract_code_blocks
from triad.schemas.pipeline import ModelConfig

# Standardised system prompt for all single-model benchmark runs.
BENCHMARK_SYSTEM_PROMPT = """\
You are an expert Python software engineer completing a benchmark task.

Rules:
- Produce COMPLETE, RUNNABLE code — no placeholders, no TODOs, no "..."
- Every file MUST be preceded by a `# file: <path>` comment on the line \
before its opening ``` fence
- Include ALL imports and boilerplate needed to run the code
- Write real, meaningful unit tests (not stubs)
- Use type annotations on all function signatures
- Follow PEP 8 style
- Do NOT explain the code — just produce the files
"""

# Regex matching the writer.py pattern for extracting files from LLM output.
_CODE_BLOCK_RE = re.compile(
    r"(?:#\s*file:\s*(.+?)\n\s*)?"      # (1) optional # file: before the block
    r"```(\w*)\n"                         # (2) opening ``` with optional language
    r"(?:#\s*file:\s*(.+?)\n)?"           # (3) optional # file: inside the block
    r"(.*?)"                              # (4) code content
    r"```",                               # closing ```
    re.DOTALL,
)

_FILE_HEADER_LINE_RE = re.compile(
    r"^#\s*file:\s*(.+?)$",
    re.MULTILINE,
)


@dataclass
class SingleModelResult:
    """Result of a single-model benchmark run."""

    prompt_id: str
    model: str
    raw_output: str
    files: dict[str, str]  # filepath -> content
    duration_seconds: float
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost: float = 0.0
    error: str = ""


class SingleModelRunner:
    """Run a benchmark prompt against a single model via LiteLLM."""

    def __init__(self, model_config: ModelConfig) -> None:
        self._provider = LiteLLMProvider(model_config)
        self._model_id = model_config.model
        self._display_name = model_config.display_name

    async def run(self, prompt: BenchmarkPrompt) -> SingleModelResult:
        """Execute one benchmark prompt and return extracted files."""
        messages = [{"role": "user", "content": prompt.prompt_text}]

        start = time.monotonic()
        try:
            agent_msg = await self._provider.complete(
                messages,
                BENCHMARK_SYSTEM_PROMPT,
                timeout=300,
            )
        except Exception as exc:
            return SingleModelResult(
                prompt_id=prompt.id,
                model=self._display_name,
                raw_output="",
                files={},
                duration_seconds=time.monotonic() - start,
                error=str(exc),
            )
        elapsed = time.monotonic() - start

        # Extract files from the response content
        files = _extract_files_from_content(agent_msg.content)

        # Also pull from structured code_blocks (provider-parsed)
        for block in agent_msg.code_blocks:
            fp = _sanitise_filepath(block.filepath)
            if fp not in files and block.content.strip():
                files[fp] = block.content

        token_usage = agent_msg.token_usage
        return SingleModelResult(
            prompt_id=prompt.id,
            model=self._display_name,
            raw_output=agent_msg.content,
            files=files,
            duration_seconds=elapsed,
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            cost=token_usage.cost if token_usage else 0.0,
        )


def _extract_files_from_content(content: str) -> dict[str, str]:
    """Extract code files from LLM markdown output.

    Mirrors the two-pass logic from triad/output/writer.py: first fenced
    code blocks, then fallback to # file: header splitting.
    """
    if not content:
        return {}

    files: dict[str, str] = {}
    counter = 0

    for match in _CODE_BLOCK_RE.finditer(content):
        filepath_before = match.group(1)
        language = match.group(2) or ""
        filepath_inside = match.group(3)
        code = match.group(4).strip()

        if not code:
            continue

        hint = filepath_before or filepath_inside
        if hint:
            rel_path = _sanitise_filepath(hint.strip())
        else:
            counter += 1
            ext = _lang_ext(language) if language else ".txt"
            rel_path = f"output_{counter}{ext}"

        if rel_path not in files:
            files[rel_path] = code

    # Fallback: split on # file: headers when no fences matched
    if not files:
        headers = list(_FILE_HEADER_LINE_RE.finditer(content))
        for i, hdr in enumerate(headers):
            fp = hdr.group(1).strip()
            start = hdr.end()
            end = headers[i + 1].start() if i + 1 < len(headers) else len(content)
            code = content[start:end].strip()
            code = re.sub(r"^```\w*\n?", "", code)
            code = re.sub(r"\n?```\s*$", "", code)
            code = code.strip()
            if code:
                rel_path = _sanitise_filepath(fp)
                if rel_path not in files:
                    files[rel_path] = code

    return files


def _sanitise_filepath(raw: str) -> str:
    """Normalise a filepath hint into a safe relative path."""
    p = Path(raw.replace("\\", "/"))
    parts = [part for part in p.parts if part not in (".", "..", "/", "\\")]
    if not parts:
        return p.name or "output"
    return str(Path(*parts))


def _lang_ext(language: str) -> str:
    """Map language to extension."""
    mapping = {
        "python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts", "rust": ".rs", "go": ".go",
        "json": ".json", "toml": ".toml", "yaml": ".yaml", "bash": ".sh",
    }
    return mapping.get(language.lower(), f".{language}")

"""Code generator — single-model, single-call generation."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field
from pathlib import Path

from triad.loop.router import RouteDecision
from triad.providers.litellm_provider import LiteLLMProvider
from triad.schemas.pipeline import ModelConfig

GENERATION_SYSTEM_PROMPT = """\
You are an expert software engineer. Generate production-quality Python code.

Rules:
1. Output ONLY code. No explanations, no essays, no commentary.
2. Use # file: path/filename.py headers to separate files.
3. Keep the structure simple — prefer fewer files over deep package hierarchies.
   A CLI tool should be 1-3 files. An API should be 3-6 files. Only complex \
systems need more.
4. Use direct imports between files (from models import User), NOT relative imports \
(from .models import User). Files will be in the same directory.
5. Include a main entry point file that can be run directly.
6. Include comprehensive unit tests in a separate test file.
7. Type hints on all function signatures, including test methods (-> None).
8. Include error handling and input validation.
9. No TODOs, no placeholders, no ellipsis, no "implement here" comments. \
Every function must have a complete implementation.
10. No setup.py, pyproject.toml, or package configuration files unless \
specifically requested.
"""

# Regex for file extraction (same as benchmark/single_model.py)
_CODE_BLOCK_RE = re.compile(
    r"(?:#\s*file:\s*(.+?)\n\s*)?"
    r"```(\w*)\n"
    r"(?:#\s*file:\s*(.+?)\n)?"
    r"(.*?)"
    r"```",
    re.DOTALL,
)

_FILE_HEADER_LINE_RE = re.compile(
    r"^#\s*file:\s*(.+?)$",
    re.MULTILINE,
)


@dataclass
class GenerationResult:
    """Result of a single code generation call."""

    files: dict[str, str] = field(default_factory=dict)
    raw_output: str = ""
    model: str = ""
    cost: float = 0.0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    duration_seconds: float = 0.0
    error: str = ""


class CodeGenerator:
    """Generate code from a prompt using a single model call."""

    async def generate(
        self,
        prompt: str,
        route: RouteDecision,
        registry: dict[str, ModelConfig],
        architecture: str | None = None,
    ) -> GenerationResult:
        """Make one API call and extract files from the response."""
        # Find the model config from registry
        model_config = self._find_model_config(route.model, registry)
        if model_config is None:
            return GenerationResult(
                error=f"Model '{route.model}' not found in registry",
            )

        provider = LiteLLMProvider(model_config)

        # Build user message
        user_content = prompt
        if architecture:
            user_content = (
                f"Architecture (already decided — implement this design):\n"
                f"{architecture}\n\n"
                f"Task:\n{prompt}"
            )

        messages = [{"role": "user", "content": user_content}]

        start = time.monotonic()
        try:
            agent_msg = await provider.complete(
                messages,
                GENERATION_SYSTEM_PROMPT,
                timeout=300,
            )
        except Exception as exc:
            return GenerationResult(
                duration_seconds=time.monotonic() - start,
                error=str(exc),
            )
        elapsed = time.monotonic() - start

        # Extract files from the response
        files = extract_files_from_content(agent_msg.content)

        # Also pull from structured code_blocks
        for block in agent_msg.code_blocks:
            fp = _sanitise_filepath(block.filepath)
            if fp not in files and block.content.strip():
                files[fp] = block.content

        token_usage = agent_msg.token_usage
        return GenerationResult(
            files=files,
            raw_output=agent_msg.content,
            model=route.model,
            cost=token_usage.cost if token_usage else 0.0,
            prompt_tokens=token_usage.prompt_tokens if token_usage else 0,
            completion_tokens=token_usage.completion_tokens if token_usage else 0,
            duration_seconds=elapsed,
        )

    @staticmethod
    def _find_model_config(
        model_id: str, registry: dict[str, ModelConfig],
    ) -> ModelConfig | None:
        """Find a ModelConfig by its LiteLLM model identifier or registry key."""
        # Try registry key first
        for _, cfg in registry.items():
            if cfg.model == model_id:
                return cfg
        # Try direct key lookup
        if model_id in registry:
            return registry[model_id]
        return None


def extract_files_from_content(content: str) -> dict[str, str]:
    """Extract code files from LLM markdown output.

    Mirrors the two-pass logic from triad/output/writer.py.
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

    # Fallback: split on # file: headers
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
    p = Path(raw.replace("\\", "/"))
    parts = [part for part in p.parts if part not in (".", "..", "/", "\\")]
    if not parts:
        return p.name or "output"
    return str(Path(*parts))


def _lang_ext(language: str) -> str:
    mapping = {
        "python": ".py", "py": ".py", "javascript": ".js", "js": ".js",
        "typescript": ".ts", "ts": ".ts", "json": ".json", "toml": ".toml",
        "yaml": ".yaml", "bash": ".sh", "sh": ".sh",
    }
    return mapping.get(language.lower(), f".{language}")

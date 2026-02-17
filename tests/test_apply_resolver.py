"""Tests for the file path resolver."""

from __future__ import annotations

from pathlib import Path

import pytest

from triad.apply.resolver import (
    FilePathResolver,
    extract_code_blocks_from_result,
)
from triad.schemas.apply import ApplyConfig, FileAction
from triad.schemas.messages import (
    AgentMessage,
    CodeBlock,
    MessageType,
    PipelineStage,
    TokenUsage,
)
from triad.schemas.pipeline import PipelineConfig, PipelineResult, TaskSpec


@pytest.fixture
def context_dir(tmp_path):
    """Create a temporary context directory with sample files."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# main module\nprint('hello')\n")
    (tmp_path / "src" / "utils.py").write_text("# utils module\ndef helper(): pass\n")
    (tmp_path / "README.md").write_text("# Project\n")
    return tmp_path


@pytest.fixture
def config():
    return ApplyConfig()


class TestFilePathResolver:
    def test_exact_match(self, context_dir, config):
        resolver = FilePathResolver(context_dir, config)
        blocks = [CodeBlock(language="python", filepath="src/main.py", content="new code")]
        resolved = resolver.resolve(blocks)
        assert len(resolved) == 1
        assert resolved[0].action == FileAction.OVERWRITE
        assert resolved[0].match_confidence == 1.0
        assert resolved[0].existing_content is not None

    def test_basename_match(self, context_dir, config):
        resolver = FilePathResolver(context_dir, config)
        blocks = [CodeBlock(language="python", filepath="lib/utils.py", content="new utils")]
        resolved = resolver.resolve(blocks)
        assert len(resolved) == 1
        assert resolved[0].action == FileAction.OVERWRITE
        assert resolved[0].match_confidence == 0.9

    def test_create_new_file(self, context_dir, config):
        resolver = FilePathResolver(context_dir, config)
        blocks = [CodeBlock(language="python", filepath="src/new_module.py", content="# new")]
        resolved = resolver.resolve(blocks)
        assert len(resolved) == 1
        assert resolved[0].action == FileAction.CREATE
        assert resolved[0].match_confidence == 1.0

    def test_include_filter(self, context_dir):
        config = ApplyConfig(apply_include=["*.py"])
        resolver = FilePathResolver(context_dir, config)
        blocks = [
            CodeBlock(language="python", filepath="src/main.py", content="code"),
            CodeBlock(language="markdown", filepath="README.md", content="# doc"),
        ]
        resolved = resolver.resolve(blocks)
        py_file = [f for f in resolved if f.source_filepath == "src/main.py"][0]
        md_file = [f for f in resolved if f.source_filepath == "README.md"][0]
        assert py_file.action != FileAction.SKIP
        assert md_file.action == FileAction.SKIP

    def test_exclude_filter(self, context_dir):
        config = ApplyConfig(apply_exclude=["*.md"])
        resolver = FilePathResolver(context_dir, config)
        blocks = [
            CodeBlock(language="python", filepath="src/main.py", content="code"),
            CodeBlock(language="markdown", filepath="README.md", content="# doc"),
        ]
        resolved = resolver.resolve(blocks)
        md_file = [f for f in resolved if f.source_filepath == "README.md"][0]
        assert md_file.action == FileAction.SKIP


class TestExtractCodeBlocksFromResult:
    def _make_msg(self, content="", code_blocks=None) -> AgentMessage:
        return AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.IMPLEMENTATION,
            content=content,
            code_blocks=code_blocks or [],
            confidence=0.9,
            model="test",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, cost=0.01),
        )

    def test_extracts_from_code_blocks(self):
        blocks = [
            CodeBlock(language="python", filepath="main.py", content="print('hello')"),
        ]
        msg = self._make_msg(code_blocks=blocks)
        result = PipelineResult(
            task=TaskSpec(task="test"),
            config=PipelineConfig(persist_sessions=False),
            stages={PipelineStage.IMPLEMENT: msg},
            success=True,
        )
        extracted = extract_code_blocks_from_result(result)
        assert len(extracted) == 1
        assert extracted[0].filepath == "main.py"

    def test_extracts_from_raw_content(self):
        content = '```python\n# file: src/app.py\ndef main():\n    pass\n```'
        msg = self._make_msg(content=content)
        result = PipelineResult(
            task=TaskSpec(task="test"),
            config=PipelineConfig(persist_sessions=False),
            stages={PipelineStage.IMPLEMENT: msg},
            success=True,
        )
        extracted = extract_code_blocks_from_result(result)
        assert len(extracted) >= 1

    def test_deduplicates(self):
        blocks = [
            CodeBlock(language="python", filepath="main.py", content="print('hello')"),
        ]
        content = '```python\n# file: main.py\nprint("hello")\n```'
        msg = self._make_msg(content=content, code_blocks=blocks)
        result = PipelineResult(
            task=TaskSpec(task="test"),
            config=PipelineConfig(persist_sessions=False),
            stages={PipelineStage.IMPLEMENT: msg},
            success=True,
        )
        extracted = extract_code_blocks_from_result(result)
        filenames = [Path(b.filepath).name for b in extracted]
        assert filenames.count("main.py") == 1

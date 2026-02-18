"""Tests for the apply engine."""

from __future__ import annotations

import pytest
from rich.console import Console

from triad.apply.engine import ApplyEngine
from triad.schemas.apply import ApplyConfig
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
    """Create a context dir with a sample file."""
    (tmp_path / "src").mkdir()
    (tmp_path / "src" / "main.py").write_text("# old content\n")
    return tmp_path


@pytest.fixture
def console():
    return Console(quiet=True)


def _make_result(context_dir, code_content="# new content\n") -> PipelineResult:
    """Create a pipeline result with a code block."""
    msg = AgentMessage(
        from_agent=PipelineStage.IMPLEMENT,
        to_agent=PipelineStage.REFACTOR,
        msg_type=MessageType.IMPLEMENTATION,
        content=f'```python\n# file: src/main.py\n{code_content}```',
        code_blocks=[
            CodeBlock(
                language="python",
                filepath="src/main.py",
                content=code_content,
            ),
        ],
        confidence=0.9,
        model="test-model",
        token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, cost=0.01),
    )
    return PipelineResult(
        session_id="test-session-12345678",
        task=TaskSpec(task="test"),
        config=PipelineConfig(persist_sessions=False),
        stages={PipelineStage.IMPLEMENT: msg},
        success=True,
    )


class TestApplyEngine:
    def test_apply_creates_files(self, context_dir, console):
        result = _make_result(context_dir)
        config = ApplyConfig(enabled=True, confirm=False)
        engine = ApplyEngine(result, config, str(context_dir), console, interactive=False)
        apply_result = engine.run()
        assert len(apply_result.files_applied) >= 1
        assert apply_result.errors == [] or all(
            "protected" not in e.lower() for e in apply_result.errors
        )

    def test_apply_writes_content(self, context_dir, console):
        new_content = "# updated content\nprint('new')\n"
        result = _make_result(context_dir, new_content)
        config = ApplyConfig(enabled=True, confirm=False)
        engine = ApplyEngine(result, config, str(context_dir), console, interactive=False)
        engine.run()
        written = (context_dir / "src" / "main.py").read_text()
        assert "updated content" in written

    def test_no_code_blocks(self, context_dir, console):
        msg = AgentMessage(
            from_agent=PipelineStage.IMPLEMENT,
            to_agent=PipelineStage.REFACTOR,
            msg_type=MessageType.IMPLEMENTATION,
            content="No code here",
            confidence=0.9,
            model="test-model",
            token_usage=TokenUsage(prompt_tokens=100, completion_tokens=50, cost=0.01),
        )
        result = PipelineResult(
            session_id="test-session-12345678",
            task=TaskSpec(task="test"),
            config=PipelineConfig(persist_sessions=False),
            stages={PipelineStage.IMPLEMENT: msg},
            success=True,
        )
        config = ApplyConfig(enabled=True, confirm=False)
        engine = ApplyEngine(result, config, str(context_dir), console, interactive=False)
        apply_result = engine.run()
        assert "No code blocks" in apply_result.errors[0]

    def test_apply_blocked_by_reject(self, console, tmp_path):
        """Apply is blocked when arbiter issued REJECT verdict in a git repo."""
        import subprocess

        from triad.schemas.arbiter import ArbiterReview, Verdict

        # Set up a git repo so ensure_safe actually checks verdicts
        subprocess.run(["git", "init"], cwd=str(tmp_path), capture_output=True, check=True)
        subprocess.run(
            ["git", "config", "user.email", "t@t.com"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "config", "user.name", "T"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )
        (tmp_path / "src").mkdir()
        (tmp_path / "src" / "main.py").write_text("# old\n")
        subprocess.run(
            ["git", "add", "."],
            cwd=str(tmp_path), capture_output=True, check=True,
        )
        subprocess.run(
            ["git", "commit", "-m", "init"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )
        # Move off protected branch
        subprocess.run(
            ["git", "checkout", "-b", "feature"],
            cwd=str(tmp_path), capture_output=True, check=True,
        )

        result = _make_result(tmp_path)
        result.arbiter_reviews = [
            ArbiterReview(
                stage_reviewed=PipelineStage.IMPLEMENT,
                reviewed_model="test",
                arbiter_model="arbiter",
                verdict=Verdict.REJECT,
                confidence=0.9,
                reasoning="Issues found",
                token_cost=0.01,
            ),
        ]
        config = ApplyConfig(enabled=True, confirm=False)
        engine = ApplyEngine(result, config, str(tmp_path), console, interactive=False)
        apply_result = engine.run()
        assert any("blocked" in e.lower() for e in apply_result.errors)

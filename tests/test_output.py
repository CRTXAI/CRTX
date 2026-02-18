"""Tests for output writer and renderer parallel/debate support.

Covers: writer code extraction from parallel synthesized_output and
debate judgment, and renderer sections for parallel/debate results.
"""

from __future__ import annotations

import json
from pathlib import Path

from triad.output.renderer import render_summary
from triad.output.writer import write_pipeline_output
from triad.schemas.arbiter import ArbiterReview, Verdict
from triad.schemas.consensus import DebateResult, ParallelResult
from triad.schemas.messages import (
    AgentMessage,
    MessageType,
    PipelineStage,
    TokenUsage,
)
from triad.schemas.pipeline import (
    PipelineConfig,
    PipelineResult,
    TaskSpec,
)


# ── Factories ──────────────────────────────────────────────────────


def _make_task() -> TaskSpec:
    return TaskSpec(task="Build a REST API", context="Python FastAPI")


def _make_config(**overrides) -> PipelineConfig:
    defaults = {"persist_sessions": False}
    defaults.update(overrides)
    return PipelineConfig(**defaults)


def _make_agent_message(stage: str = "architect") -> AgentMessage:
    return AgentMessage(
        from_agent=stage,
        to_agent="implement",
        msg_type=MessageType.PROPOSAL,
        content=f"Output from {stage}\n\nCONFIDENCE: 0.85",
        confidence=0.85,
        model="test-model-v1",
        token_usage=TokenUsage(
            prompt_tokens=1000,
            completion_tokens=500,
            cost=0.02,
        ),
    )


def _make_parallel_result(**overrides) -> PipelineResult:
    """Create a PipelineResult with parallel_result (no stages)."""
    pr = ParallelResult(
        individual_outputs={
            "model-a": "Solution A",
            "model-b": "Solution B",
            "model-c": "Solution C",
        },
        scores={
            "model-a": {
                "model-b": {"architecture": 8, "implementation": 7, "quality": 9},
                "model-c": {"architecture": 6, "implementation": 5, "quality": 6},
            },
            "model-b": {
                "model-a": {"architecture": 7, "implementation": 7, "quality": 7},
                "model-c": {"architecture": 5, "implementation": 6, "quality": 5},
            },
            "model-c": {
                "model-a": {"architecture": 8, "implementation": 8, "quality": 7},
                "model-b": {"architecture": 6, "implementation": 7, "quality": 6},
            },
        },
        votes={"model-a": "model-b", "model-b": "model-a", "model-c": "model-a"},
        winner="model-a",
        synthesized_output=(
            "Here is the final code:\n\n"
            "```python\n"
            "# file: api/app.py\n"
            "from fastapi import FastAPI\n"
            "app = FastAPI()\n"
            "```\n\n"
            "And the test:\n\n"
            "```python\n"
            "# file: test_app.py\n"
            "def test_app(): pass\n"
            "```"
        ),
    )

    defaults = {
        "session_id": "par-12345678",
        "task": _make_task(),
        "config": _make_config(pipeline_mode="parallel"),
        "stages": {},
        "arbiter_reviews": [],
        "routing_decisions": [],
        "total_cost": 0.50,
        "total_tokens": 12000,
        "duration_seconds": 30.0,
        "success": True,
        "parallel_result": pr,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


def _make_debate_result(**overrides) -> PipelineResult:
    """Create a PipelineResult with debate_result (no stages)."""
    dr = DebateResult(
        proposals={
            "model-a": "Proposal from A: use microservices",
            "model-b": "Proposal from B: use monolith",
            "model-c": "Proposal from C: use serverless",
        },
        rebuttals={
            "model-a": {"model-b": "B is too simple", "model-c": "C is too complex"},
            "model-b": {"model-a": "A is overengineered", "model-c": "C has cold starts"},
            "model-c": {"model-a": "A is fragile", "model-b": "B doesn't scale"},
        },
        final_arguments={
            "model-a": "Final from A",
            "model-b": "Final from B",
            "model-c": "Final from C",
        },
        judgment=(
            "After careful consideration:\n\n"
            "```python\n"
            "# file: service/main.py\n"
            "from flask import Flask\n"
            "app = Flask(__name__)\n"
            "```\n\n"
            "```python\n"
            "# file: test_service.py\n"
            "def test_service(): pass\n"
            "```"
        ),
        judge_model="model-c",
    )

    defaults = {
        "session_id": "deb-12345678",
        "task": _make_task(),
        "config": _make_config(pipeline_mode="debate"),
        "stages": {},
        "arbiter_reviews": [],
        "routing_decisions": [],
        "total_cost": 0.75,
        "total_tokens": 18000,
        "duration_seconds": 60.0,
        "success": True,
        "debate_result": dr,
    }
    defaults.update(overrides)
    return PipelineResult(**defaults)


# ── Writer: Parallel Mode ────────────────────────────────────────


class TestWriterParallel:
    """Tests for write_pipeline_output with parallel results."""

    def test_extracts_code_from_synthesized_output(self, tmp_path):
        """Writer should extract code blocks from synthesized_output."""
        result = _make_parallel_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        code_files = list(Path(actual_path, "code").glob("*"))
        assert len(code_files) >= 1
        # Check that the FastAPI app file was extracted
        filenames = [f.name for f in code_files]
        assert "app.py" in filenames

    def test_extracts_tests_from_synthesized_output(self, tmp_path):
        """Writer should extract test files from synthesized_output."""
        result = _make_parallel_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        test_files = list(Path(actual_path, "tests").glob("*"))
        assert len(test_files) >= 1
        filenames = [f.name for f in test_files]
        assert "test_app.py" in filenames

    def test_creates_summary_md_for_parallel(self, tmp_path):
        """Writer should create summary.md for parallel results."""
        result = _make_parallel_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        summary = (Path(actual_path) / "summary.md").read_text()
        assert "CRTX Pipeline Summary" in summary
        assert "Parallel" in summary

    def test_creates_session_json_for_parallel(self, tmp_path):
        """Writer should create session.json for parallel results."""
        result = _make_parallel_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        session_data = json.loads(
            (Path(actual_path) / "session.json").read_text()
        )
        assert session_data["success"] is True
        assert session_data["parallel_result"] is not None
        assert session_data["parallel_result"]["winner"] == "model-a"

    def test_empty_synthesized_output_still_works(self, tmp_path):
        """Writer should handle empty synthesized_output gracefully."""
        pr = ParallelResult(
            individual_outputs={"model-a": "output"},
            scores={},
            votes={},
            winner="model-a",
            synthesized_output="",
        )
        result = _make_parallel_result(parallel_result=pr)
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        # Should still create the structure without errors
        assert (Path(actual_path) / "summary.md").is_file()


# ── Writer: Debate Mode ──────────────────────────────────────────


class TestWriterDebate:
    """Tests for write_pipeline_output with debate results."""

    def test_extracts_code_from_judgment(self, tmp_path):
        """Writer should extract code blocks from debate judgment."""
        result = _make_debate_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        code_files = list(Path(actual_path, "code").glob("*"))
        assert len(code_files) >= 1
        filenames = [f.name for f in code_files]
        assert "main.py" in filenames

    def test_extracts_tests_from_judgment(self, tmp_path):
        """Writer should extract test files from debate judgment."""
        result = _make_debate_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        test_files = list(Path(actual_path, "tests").glob("*"))
        assert len(test_files) >= 1
        filenames = [f.name for f in test_files]
        assert "test_service.py" in filenames

    def test_creates_summary_md_for_debate(self, tmp_path):
        """Writer should create summary.md for debate results."""
        result = _make_debate_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        summary = (Path(actual_path) / "summary.md").read_text()
        assert "CRTX Pipeline Summary" in summary
        assert "Debate" in summary

    def test_creates_session_json_for_debate(self, tmp_path):
        """Writer should create session.json for debate results."""
        result = _make_debate_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        session_data = json.loads(
            (Path(actual_path) / "session.json").read_text()
        )
        assert session_data["success"] is True
        assert session_data["debate_result"] is not None
        assert session_data["debate_result"]["judge_model"] == "model-c"

    def test_empty_judgment_still_works(self, tmp_path):
        """Writer should handle empty judgment gracefully."""
        dr = DebateResult(
            proposals={"a": "proposal"},
            rebuttals={},
            final_arguments={},
            judgment="",
            judge_model="model-c",
        )
        result = _make_debate_result(debate_result=dr)
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        assert (Path(actual_path) / "summary.md").is_file()


# ── Renderer: Parallel Mode ─────────────────────────────────────


class TestRendererParallel:
    """Tests for render_summary with parallel results."""

    def test_renders_parallel_exploration_section(self):
        """render_summary should include Parallel Exploration Results section."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "## Parallel Exploration Results" in md

    def test_renders_winner(self):
        """render_summary should include the winner."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "model-a" in md
        assert "Winner" in md

    def test_renders_vote_table(self):
        """render_summary should include voting table."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "### Votes" in md
        assert "| Voter | Voted For |" in md
        # Check specific vote entries
        assert "model-a" in md
        assert "model-b" in md

    def test_renders_cross_review_scores(self):
        """render_summary should include cross-review score table."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "### Cross-Review Scores" in md
        assert "| Reviewer | Reviewed | Arch | Impl | Quality |" in md

    def test_renders_synthesized_output_preview(self):
        """render_summary should include synthesized output preview."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "### Synthesized Output" in md
        assert "FastAPI" in md

    def test_renders_cost_summary(self):
        """render_summary should include cost summary."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "## Cost Summary" in md
        assert "$0.5000" in md

    def test_no_stage_summaries_for_parallel(self):
        """render_summary should not have Stage Summaries when stages is empty."""
        result = _make_parallel_result()
        md = render_summary(result)

        assert "## Stage Summaries" not in md


# ── Renderer: Debate Mode ───────────────────────────────────────


class TestRendererDebate:
    """Tests for render_summary with debate results."""

    def test_renders_debate_results_section(self):
        """render_summary should include Debate Results section."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "## Debate Results" in md

    def test_renders_judge_model(self):
        """render_summary should include the judge model."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "model-c" in md
        assert "Judge" in md

    def test_renders_debater_count(self):
        """render_summary should include debater count."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "Debaters" in md
        assert "3" in md

    def test_renders_position_papers(self):
        """render_summary should include position papers section."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "### Position Papers" in md
        assert "#### model-a" in md
        assert "microservices" in md

    def test_renders_judgment_preview(self):
        """render_summary should include judgment preview."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "### Judgment" in md
        assert "careful consideration" in md

    def test_renders_cost_summary(self):
        """render_summary should include cost summary."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "## Cost Summary" in md
        assert "$0.7500" in md

    def test_no_stage_summaries_for_debate(self):
        """render_summary should not have Stage Summaries when stages is empty."""
        result = _make_debate_result()
        md = render_summary(result)

        assert "## Stage Summaries" not in md

    def test_renders_halted_debate(self):
        """render_summary should show HALTED status for halted debate."""
        result = _make_debate_result(
            success=False, halted=True,
            halt_reason="Judgment was flawed",
        )
        md = render_summary(result)

        assert "HALTED" in md
        assert "Judgment was flawed" in md


# ── Writer: Fallback behavior ───────────────────────────────────


class TestWriterFallbackContent:
    """Tests that the writer falls back correctly between content sources."""

    def test_prefers_stages_over_parallel(self, tmp_path):
        """When stages exist alongside parallel_result, use stages content."""
        msg = _make_agent_message("verify")
        msg.content = (
            "```python\n"
            "# file: from_stage.py\n"
            "print('from stage')\n"
            "```"
        )

        pr = ParallelResult(
            individual_outputs={"model-a": "output"},
            scores={},
            votes={},
            winner="model-a",
            synthesized_output=(
                "```python\n"
                "# file: from_parallel.py\n"
                "print('from parallel')\n"
                "```"
            ),
        )

        result = _make_parallel_result(
            stages={PipelineStage.VERIFY: msg},
            parallel_result=pr,
        )
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        code_files = list(Path(actual_path, "code").glob("*"))
        filenames = [f.name for f in code_files]
        # Stage content should be used as final_content
        assert "from_stage.py" in filenames

    def test_parallel_used_when_stages_empty(self, tmp_path):
        """When stages is empty, synthesized_output provides code."""
        result = _make_parallel_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        code_files = list(Path(actual_path, "code").glob("*"))
        assert len(code_files) >= 1

    def test_debate_used_when_stages_empty(self, tmp_path):
        """When stages is empty, judgment provides code."""
        result = _make_debate_result()
        out_dir = str(tmp_path / "output")

        actual_path = write_pipeline_output(result, out_dir)

        code_files = list(Path(actual_path, "code").glob("*"))
        assert len(code_files) >= 1


# ── Renderer: Status line logic ──────────────────────────────────


def _make_arbiter_review(verdict: Verdict) -> ArbiterReview:
    return ArbiterReview(
        stage_reviewed=PipelineStage.VERIFY,
        reviewed_model="test-model-v1",
        arbiter_model="arbiter-model-v1",
        verdict=verdict,
        confidence=0.85,
        reasoning="Test reasoning.",
        token_cost=0.01,
    )


class TestRendererStatus:

    def test_success_with_reject_shows_completed_with_rejections(self):
        """Success + reject arbiter review → COMPLETED WITH REJECTIONS."""
        result = PipelineResult(
            task=_make_task(),
            config=_make_config(),
            success=True,
            halted=False,
            arbiter_reviews=[_make_arbiter_review(Verdict.REJECT)],
        )
        md = render_summary(result)
        assert "COMPLETED WITH REJECTIONS" in md

    def test_success_without_reject_shows_success(self):
        """Success with APPROVE review → SUCCESS."""
        result = PipelineResult(
            task=_make_task(),
            config=_make_config(),
            success=True,
            halted=False,
            arbiter_reviews=[_make_arbiter_review(Verdict.APPROVE)],
        )
        md = render_summary(result)
        assert "## Result: SUCCESS" in md

    def test_halted_still_shows_halted(self):
        """Halted pipeline → HALTED regardless of reviews."""
        result = PipelineResult(
            task=_make_task(),
            config=_make_config(),
            success=False,
            halted=True,
            halt_reason="Critical issue",
            arbiter_reviews=[_make_arbiter_review(Verdict.HALT)],
        )
        md = render_summary(result)
        assert "## Result: HALTED" in md

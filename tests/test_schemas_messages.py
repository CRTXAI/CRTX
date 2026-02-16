"""Tests for triad.schemas.messages — AgentMessage and supporting types."""

import uuid
from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from triad.schemas.messages import (
    AgentMessage,
    CodeBlock,
    MessageType,
    Objection,
    PipelineStage,
    Suggestion,
    TokenUsage,
)


class TestPipelineStage:
    def test_all_stages_exist(self):
        assert PipelineStage.ARCHITECT == "architect"
        assert PipelineStage.IMPLEMENT == "implement"
        assert PipelineStage.REFACTOR == "refactor"
        assert PipelineStage.VERIFY == "verify"

    def test_stage_count(self):
        assert len(PipelineStage) == 4

    def test_stage_from_string(self):
        assert PipelineStage("architect") == PipelineStage.ARCHITECT


class TestMessageType:
    def test_all_types_exist(self):
        assert MessageType.PROPOSAL == "proposal"
        assert MessageType.IMPLEMENTATION == "implementation"
        assert MessageType.REVIEW == "review"
        assert MessageType.OBJECTION == "objection"
        assert MessageType.SUGGESTION == "suggestion"
        assert MessageType.VOTE == "vote"
        assert MessageType.CONSENSUS == "consensus"
        assert MessageType.VERIFICATION == "verification"

    def test_type_count(self):
        assert len(MessageType) == 8


class TestTokenUsage:
    def test_valid_usage(self):
        usage = TokenUsage(prompt_tokens=100, completion_tokens=50, cost=0.005)
        assert usage.prompt_tokens == 100
        assert usage.completion_tokens == 50
        assert usage.cost == 0.005

    def test_zero_values(self):
        usage = TokenUsage(prompt_tokens=0, completion_tokens=0, cost=0.0)
        assert usage.prompt_tokens == 0

    def test_negative_tokens_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=-1, completion_tokens=50, cost=0.0)

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            TokenUsage(prompt_tokens=100, completion_tokens=50, cost=-0.01)


class TestCodeBlock:
    def test_valid_code_block(self):
        block = CodeBlock(
            language="python",
            filepath="src/main.py",
            content="print('hello')",
        )
        assert block.language == "python"
        assert block.filepath == "src/main.py"
        assert block.content == "print('hello')"

    def test_empty_content_allowed(self):
        block = CodeBlock(language="python", filepath="empty.py", content="")
        assert block.content == ""


class TestSuggestion:
    def test_valid_suggestion(self):
        suggestion = Suggestion(
            domain=PipelineStage.ARCHITECT,
            rationale="Use a factory pattern for extensibility",
            confidence=0.85,
            code_sketch="class Factory: ...",
            impact_assessment="Requires new base class",
        )
        assert suggestion.domain == PipelineStage.ARCHITECT
        assert suggestion.confidence == 0.85

    def test_defaults(self):
        suggestion = Suggestion(
            domain=PipelineStage.IMPLEMENT,
            rationale="Consider caching",
            confidence=0.5,
        )
        assert suggestion.code_sketch == ""
        assert suggestion.impact_assessment == ""

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            Suggestion(
                domain=PipelineStage.REFACTOR,
                rationale="test",
                confidence=1.5,
            )

    def test_confidence_below_zero(self):
        with pytest.raises(ValidationError):
            Suggestion(
                domain=PipelineStage.VERIFY,
                rationale="test",
                confidence=-0.1,
            )


class TestObjection:
    def test_valid_objection(self):
        obj = Objection(
            reason="This approach has O(n^2) complexity",
            severity="blocking",
            evidence="The nested loop on line 42",
        )
        assert obj.reason == "This approach has O(n^2) complexity"
        assert obj.severity == "blocking"

    def test_defaults(self):
        obj = Objection(reason="Minor style issue", severity="advisory")
        assert obj.evidence == ""


class TestAgentMessage:
    def test_valid_minimal_message(self):
        msg = AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.PROPOSAL,
            content="Here is the scaffold...",
            confidence=0.9,
        )
        assert msg.from_agent == PipelineStage.ARCHITECT
        assert msg.to_agent == PipelineStage.IMPLEMENT
        assert msg.msg_type == MessageType.PROPOSAL
        assert msg.confidence == 0.9
        assert msg.code_blocks == []
        assert msg.suggestions == []
        assert msg.objections == []
        assert msg.token_usage is None
        assert msg.model == ""

    def test_auto_generated_uuid(self):
        msg = AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.PROPOSAL,
            content="test",
            confidence=0.5,
        )
        # Should be a valid UUID
        uuid.UUID(msg.message_id)

    def test_unique_ids(self):
        msg1 = AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.PROPOSAL,
            content="test",
            confidence=0.5,
        )
        msg2 = AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.PROPOSAL,
            content="test",
            confidence=0.5,
        )
        assert msg1.message_id != msg2.message_id

    def test_auto_timestamp(self):
        before = datetime.now(UTC)
        msg = AgentMessage(
            from_agent=PipelineStage.VERIFY,
            to_agent=PipelineStage.ARCHITECT,
            msg_type=MessageType.VERIFICATION,
            content="verified",
            confidence=0.95,
        )
        after = datetime.now(UTC)
        assert before <= msg.timestamp <= after

    def test_full_message(self):
        msg = AgentMessage(
            from_agent=PipelineStage.IMPLEMENT,
            to_agent=PipelineStage.REFACTOR,
            msg_type=MessageType.IMPLEMENTATION,
            content="def handler(): ...",
            code_blocks=[
                CodeBlock(language="python", filepath="handler.py", content="def handler(): pass")
            ],
            confidence=0.88,
            suggestions=[
                Suggestion(
                    domain=PipelineStage.REFACTOR,
                    rationale="Add type hints",
                    confidence=0.7,
                )
            ],
            objections=[],
            token_usage=TokenUsage(prompt_tokens=1000, completion_tokens=500, cost=0.02),
            model="gpt-4o",
        )
        assert len(msg.code_blocks) == 1
        assert len(msg.suggestions) == 1
        assert msg.token_usage is not None
        assert msg.token_usage.cost == 0.02
        assert msg.model == "gpt-4o"

    def test_confidence_out_of_range(self):
        with pytest.raises(ValidationError):
            AgentMessage(
                from_agent=PipelineStage.ARCHITECT,
                to_agent=PipelineStage.IMPLEMENT,
                msg_type=MessageType.PROPOSAL,
                content="test",
                confidence=2.0,
            )

    def test_invalid_stage(self):
        with pytest.raises(ValidationError):
            AgentMessage(
                from_agent="invalid_stage",
                to_agent=PipelineStage.IMPLEMENT,
                msg_type=MessageType.PROPOSAL,
                content="test",
                confidence=0.5,
            )

    def test_serialization_roundtrip(self):
        msg = AgentMessage(
            from_agent=PipelineStage.ARCHITECT,
            to_agent=PipelineStage.IMPLEMENT,
            msg_type=MessageType.PROPOSAL,
            content="scaffold output",
            confidence=0.9,
            model="gemini-pro",
        )
        data = msg.model_dump()
        restored = AgentMessage(**data)
        assert restored.from_agent == msg.from_agent
        assert restored.content == msg.content
        assert restored.message_id == msg.message_id

"""Message schemas for inter-agent communication in the Triad pipeline.

Defines the universal message envelope (AgentMessage) and all supporting types
used for communication between pipeline stages.
"""

from __future__ import annotations

import uuid
from datetime import UTC, datetime
from enum import StrEnum

from pydantic import BaseModel, Field


class PipelineStage(StrEnum):
    """Pipeline stage identifiers for role assignment and message routing."""

    ARCHITECT = "architect"
    IMPLEMENT = "implement"
    REFACTOR = "refactor"
    VERIFY = "verify"


class MessageType(StrEnum):
    """Types of messages exchanged between pipeline agents."""

    PROPOSAL = "proposal"
    IMPLEMENTATION = "implementation"
    REVIEW = "review"
    OBJECTION = "objection"
    SUGGESTION = "suggestion"
    VOTE = "vote"
    CONSENSUS = "consensus"
    VERIFICATION = "verification"


class TokenUsage(BaseModel):
    """Token consumption and cost tracking for a single model call."""

    prompt_tokens: int = Field(ge=0, description="Number of input tokens consumed")
    completion_tokens: int = Field(ge=0, description="Number of output tokens generated")
    cost: float = Field(ge=0.0, description="Estimated cost in USD for this call")


class CodeBlock(BaseModel):
    """A structured code block extracted from agent output."""

    language: str = Field(description="Programming language identifier (e.g. 'python', 'toml')")
    filepath: str = Field(description="Target file path for this code block")
    content: str = Field(description="The actual source code content")


class Suggestion(BaseModel):
    """A cross-domain suggestion from one agent to another's domain.

    Any agent can attach suggestions to any message, even when the suggestion
    falls outside its primary role. This enables cross-pollination of ideas.
    """

    domain: PipelineStage = Field(description="Which role's territory this suggestion enters")
    rationale: str = Field(
        description="Reasoning for why this idea is better than current approach"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Self-assessed confidence in this suggestion"
    )
    code_sketch: str = Field(
        default="", description="Optional code snippet demonstrating the alternative"
    )
    impact_assessment: str = Field(
        default="", description="What changes downstream if this suggestion is adopted"
    )


class Objection(BaseModel):
    """A structured disagreement raised by an agent during consensus.

    Agents can object to proposals from other agents with specific reasoning
    and evidence to support their position.
    """

    reason: str = Field(description="Why the agent disagrees with the current approach")
    severity: str = Field(description="How critical this objection is: 'blocking' or 'advisory'")
    evidence: str = Field(
        default="", description="Supporting evidence or examples for the objection"
    )


class AgentMessage(BaseModel):
    """Universal message envelope for all inter-agent communication.

    Every message passed between pipeline stages, the orchestrator, and the
    Arbiter uses this schema. The suggestions field enables cross-domain
    autonomy — any agent can propose ideas outside its primary role.
    """

    message_id: str = Field(
        default_factory=lambda: str(uuid.uuid4()),
        description="Unique message identifier (UUID v4)",
    )
    from_agent: PipelineStage = Field(description="Pipeline stage that produced this message")
    to_agent: PipelineStage = Field(description="Target pipeline stage or orchestrator")
    msg_type: MessageType = Field(description="The type of message being sent")
    content: str = Field(description="The actual code, analysis, or review content")
    code_blocks: list[CodeBlock] = Field(
        default_factory=list, description="Parsed structured code blocks from the output"
    )
    confidence: float = Field(
        ge=0.0, le=1.0, description="Self-assessed confidence score for this output"
    )
    suggestions: list[Suggestion] = Field(
        default_factory=list, description="Cross-domain suggestions for other roles"
    )
    objections: list[Objection] = Field(
        default_factory=list, description="Disagreements with reasons"
    )
    token_usage: TokenUsage | None = Field(
        default=None, description="Token consumption and cost for this message"
    )
    model: str = Field(default="", description="Model identifier that produced this message")
    timestamp: datetime = Field(
        default_factory=lambda: datetime.now(UTC),
        description="When this message was created",
    )

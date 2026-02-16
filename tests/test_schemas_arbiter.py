"""Tests for triad.schemas.arbiter — ArbiterReview and supporting types."""

import pytest
from pydantic import ValidationError

from triad.schemas.arbiter import (
    Alternative,
    ArbiterReview,
    Issue,
    IssueCategory,
    Severity,
    Verdict,
)
from triad.schemas.messages import PipelineStage


class TestSeverity:
    def test_all_values(self):
        assert Severity.CRITICAL == "critical"
        assert Severity.WARNING == "warning"
        assert Severity.SUGGESTION == "suggestion"

    def test_count(self):
        assert len(Severity) == 3


class TestIssueCategory:
    def test_all_categories(self):
        assert IssueCategory.LOGIC == "logic"
        assert IssueCategory.PATTERN == "pattern"
        assert IssueCategory.SECURITY == "security"
        assert IssueCategory.PERFORMANCE == "performance"
        assert IssueCategory.EDGE_CASE == "edge_case"
        assert IssueCategory.HALLUCINATION == "hallucination"

    def test_count(self):
        assert len(IssueCategory) == 6


class TestVerdict:
    def test_all_verdicts(self):
        assert Verdict.APPROVE == "approve"
        assert Verdict.FLAG == "flag"
        assert Verdict.REJECT == "reject"
        assert Verdict.HALT == "halt"

    def test_count(self):
        assert len(Verdict) == 4

    def test_from_string(self):
        assert Verdict("reject") == Verdict.REJECT


class TestIssue:
    def test_valid_issue(self):
        issue = Issue(
            severity=Severity.CRITICAL,
            category=IssueCategory.HALLUCINATION,
            location="src/api.py:42-50",
            description="Imports non-existent module 'fastapi.magic'",
            suggestion="Use 'fastapi.responses' instead",
            evidence="Module does not exist in FastAPI >= 0.100",
        )
        assert issue.severity == Severity.CRITICAL
        assert issue.category == IssueCategory.HALLUCINATION
        assert issue.location == "src/api.py:42-50"

    def test_minimal_issue(self):
        issue = Issue(
            severity=Severity.WARNING,
            category=IssueCategory.PERFORMANCE,
            description="N+1 query pattern detected",
        )
        assert issue.location == ""
        assert issue.suggestion == ""
        assert issue.evidence == ""

    def test_invalid_severity(self):
        with pytest.raises(ValidationError):
            Issue(
                severity="extreme",
                category=IssueCategory.LOGIC,
                description="test",
            )


class TestAlternative:
    def test_valid_alternative(self):
        alt = Alternative(
            description="Use connection pooling",
            rationale="Reduces DB connection overhead by 80%",
            code_sketch="pool = create_pool(max_size=10)",
            confidence=0.92,
        )
        assert alt.confidence == 0.92
        assert alt.code_sketch == "pool = create_pool(max_size=10)"

    def test_minimal_alternative(self):
        alt = Alternative(
            description="Use async handlers",
            rationale="Better concurrency",
            confidence=0.7,
        )
        assert alt.code_sketch == ""

    def test_confidence_bounds(self):
        with pytest.raises(ValidationError):
            Alternative(
                description="test",
                rationale="test",
                confidence=1.1,
            )


class TestArbiterReview:
    def test_approve_verdict(self):
        review = ArbiterReview(
            stage_reviewed=PipelineStage.ARCHITECT,
            reviewed_model="gemini-pro",
            arbiter_model="claude-opus",
            verdict=Verdict.APPROVE,
            issues=[],
            alternatives=[],
            confidence=0.95,
            reasoning="Architecture is sound. Clean separation of concerns.",
            token_cost=0.12,
        )
        assert review.verdict == Verdict.APPROVE
        assert review.stage_reviewed == PipelineStage.ARCHITECT
        assert review.reviewed_model != review.arbiter_model

    def test_reject_with_issues(self):
        review = ArbiterReview(
            stage_reviewed=PipelineStage.IMPLEMENT,
            reviewed_model="gpt-4o",
            arbiter_model="claude-sonnet",
            verdict=Verdict.REJECT,
            issues=[
                Issue(
                    severity=Severity.CRITICAL,
                    category=IssueCategory.LOGIC,
                    location="handler.py:15",
                    description="Missing null check on user input",
                    suggestion="Add validation before processing",
                ),
                Issue(
                    severity=Severity.WARNING,
                    category=IssueCategory.EDGE_CASE,
                    description="Empty list not handled in pagination",
                ),
            ],
            alternatives=[
                Alternative(
                    description="Use Pydantic validation at the boundary",
                    rationale="Catches all invalid input shapes automatically",
                    confidence=0.88,
                ),
            ],
            confidence=0.85,
            reasoning="Critical logic error: user input flows unchecked into DB query.",
            token_cost=0.15,
        )
        assert review.verdict == Verdict.REJECT
        assert len(review.issues) == 2
        assert len(review.alternatives) == 1
        assert review.issues[0].severity == Severity.CRITICAL

    def test_halt_verdict(self):
        review = ArbiterReview(
            stage_reviewed=PipelineStage.VERIFY,
            reviewed_model="claude-sonnet",
            arbiter_model="grok-4",
            verdict=Verdict.HALT,
            issues=[
                Issue(
                    severity=Severity.CRITICAL,
                    category=IssueCategory.SECURITY,
                    description="SQL injection vulnerability in search endpoint",
                ),
            ],
            confidence=0.99,
            reasoning="Fundamental security flaw. Cannot proceed.",
            token_cost=0.10,
        )
        assert review.verdict == Verdict.HALT

    def test_confidence_validation(self):
        with pytest.raises(ValidationError):
            ArbiterReview(
                stage_reviewed=PipelineStage.ARCHITECT,
                reviewed_model="test",
                arbiter_model="test2",
                verdict=Verdict.APPROVE,
                confidence=-0.5,
                reasoning="test",
                token_cost=0.0,
            )

    def test_negative_token_cost_rejected(self):
        with pytest.raises(ValidationError):
            ArbiterReview(
                stage_reviewed=PipelineStage.ARCHITECT,
                reviewed_model="test",
                arbiter_model="test2",
                verdict=Verdict.APPROVE,
                confidence=0.9,
                reasoning="test",
                token_cost=-1.0,
            )

    def test_serialization_roundtrip(self):
        review = ArbiterReview(
            stage_reviewed=PipelineStage.REFACTOR,
            reviewed_model="claude-opus",
            arbiter_model="gpt-4o",
            verdict=Verdict.FLAG,
            issues=[
                Issue(
                    severity=Severity.SUGGESTION,
                    category=IssueCategory.PATTERN,
                    description="Consider using dataclass instead of dict",
                ),
            ],
            confidence=0.75,
            reasoning="Minor pattern improvements possible.",
            token_cost=0.08,
        )
        data = review.model_dump()
        restored = ArbiterReview(**data)
        assert restored.verdict == review.verdict
        assert len(restored.issues) == 1
        assert restored.arbiter_model == "gpt-4o"

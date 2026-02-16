"""Tests for triad.schemas.reconciliation — ImplementationSummary and Deviation."""

import pytest
from pydantic import ValidationError

from triad.schemas.messages import PipelineStage
from triad.schemas.reconciliation import Deviation, ImplementationSummary


class TestDeviation:
    def test_valid_deviation(self):
        dev = Deviation(
            what="Removed /admin endpoint",
            reason="Not in original spec, was added by Implementer speculatively",
            stage=PipelineStage.REFACTOR,
        )
        assert dev.what == "Removed /admin endpoint"
        assert dev.stage == PipelineStage.REFACTOR

    def test_invalid_stage(self):
        with pytest.raises(ValidationError):
            Deviation(
                what="test",
                reason="test",
                stage="nonexistent",
            )


class TestImplementationSummary:
    def test_valid_full_summary(self):
        summary = ImplementationSummary(
            task_echo="Build a REST API with user authentication and CRUD operations",
            endpoints_implemented=[
                "POST /auth/login",
                "POST /auth/register",
                "GET /users",
                "GET /users/{id}",
                "PUT /users/{id}",
                "DELETE /users/{id}",
            ],
            schemas_created=["User", "UserCreate", "UserUpdate", "Token"],
            files_created=[
                "src/models/user.py",
                "src/routes/auth.py",
                "src/routes/users.py",
                "tests/test_auth.py",
                "tests/test_users.py",
            ],
            files_modified=["src/main.py", "src/config.py"],
            behaviors_implemented=[
                "JWT authentication with refresh tokens",
                "Password hashing with bcrypt",
                "Role-based access control",
                "Pagination on list endpoints",
            ],
            test_coverage=[
                "Login with valid credentials",
                "Login with invalid credentials",
                "Register new user",
                "CRUD operations on users",
            ],
            deviations=[
                Deviation(
                    what="Added rate limiting to auth endpoints",
                    reason="Security best practice, prevents brute force",
                    stage=PipelineStage.IMPLEMENT,
                ),
            ],
            omissions=["WebSocket notifications for user updates"],
        )
        assert len(summary.endpoints_implemented) == 6
        assert len(summary.schemas_created) == 4
        assert len(summary.files_created) == 5
        assert len(summary.deviations) == 1
        assert len(summary.omissions) == 1

    def test_minimal_summary(self):
        summary = ImplementationSummary(
            task_echo="Create a hello world CLI",
        )
        assert summary.task_echo == "Create a hello world CLI"
        assert summary.endpoints_implemented == []
        assert summary.schemas_created == []
        assert summary.files_created == []
        assert summary.files_modified == []
        assert summary.behaviors_implemented == []
        assert summary.test_coverage == []
        assert summary.deviations == []
        assert summary.omissions == []

    def test_serialization_roundtrip(self):
        summary = ImplementationSummary(
            task_echo="Build a data pipeline",
            endpoints_implemented=["GET /status"],
            schemas_created=["PipelineStatus"],
            files_created=["pipeline.py"],
            behaviors_implemented=["Stream processing with backpressure"],
            test_coverage=["Test backpressure handling"],
        )
        data = summary.model_dump()
        restored = ImplementationSummary(**data)
        assert restored.task_echo == summary.task_echo
        assert restored.endpoints_implemented == summary.endpoints_implemented

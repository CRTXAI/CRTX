"""Tests for triad.schemas.pipeline — PipelineConfig, TaskSpec, ModelConfig, RoleFitness."""

import pytest
from pydantic import ValidationError

from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import (
    ArbiterMode,
    ModelConfig,
    PipelineConfig,
    RoleFitness,
    StageConfig,
    TaskSpec,
)


class TestArbiterMode:
    def test_all_modes(self):
        assert ArbiterMode.OFF == "off"
        assert ArbiterMode.FINAL_ONLY == "final_only"
        assert ArbiterMode.BOOKEND == "bookend"
        assert ArbiterMode.FULL == "full"

    def test_count(self):
        assert len(ArbiterMode) == 4


class TestRoleFitness:
    def test_defaults_to_zero(self):
        fitness = RoleFitness()
        assert fitness.architect == 0.0
        assert fitness.implementer == 0.0
        assert fitness.refactorer == 0.0
        assert fitness.verifier == 0.0

    def test_valid_scores(self):
        fitness = RoleFitness(
            architect=0.9, implementer=0.8, refactorer=0.95, verifier=0.85
        )
        assert fitness.architect == 0.9

    def test_score_above_one_rejected(self):
        with pytest.raises(ValidationError):
            RoleFitness(architect=1.5)

    def test_score_below_zero_rejected(self):
        with pytest.raises(ValidationError):
            RoleFitness(implementer=-0.1)

    def test_boundary_values(self):
        fitness = RoleFitness(architect=0.0, implementer=1.0)
        assert fitness.architect == 0.0
        assert fitness.implementer == 1.0


class TestModelConfig:
    def test_valid_model(self):
        config = ModelConfig(
            provider="anthropic",
            model="claude-opus-4-6",
            display_name="Claude Opus 4.6",
            api_key_env="ANTHROPIC_API_KEY",
            context_window=200000,
            supports_tools=True,
            supports_structured=True,
            supports_vision=True,
            supports_thinking=True,
            cost_input=15.0,
            cost_output=75.0,
            fitness=RoleFitness(architect=0.85, implementer=0.80, refactorer=0.95, verifier=0.90),
        )
        assert config.provider == "anthropic"
        assert config.model == "claude-opus-4-6"
        assert config.context_window == 200000
        assert config.fitness.refactorer == 0.95

    def test_defaults(self):
        config = ModelConfig(
            provider="openai",
            model="gpt-4o",
            display_name="GPT-4o",
            api_key_env="OPENAI_API_KEY",
            context_window=128000,
            cost_input=2.50,
            cost_output=10.00,
        )
        assert config.api_base == ""
        assert config.supports_tools is False
        assert config.supports_vision is False
        assert config.fitness.architect == 0.0

    def test_zero_context_window_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="test",
                model="test",
                display_name="Test",
                api_key_env="TEST_KEY",
                context_window=0,
                cost_input=0.0,
                cost_output=0.0,
            )

    def test_negative_cost_rejected(self):
        with pytest.raises(ValidationError):
            ModelConfig(
                provider="test",
                model="test",
                display_name="Test",
                api_key_env="TEST_KEY",
                context_window=1000,
                cost_input=-1.0,
                cost_output=0.0,
            )


class TestTaskSpec:
    def test_valid_task(self):
        spec = TaskSpec(
            task="Build a REST API with authentication",
            context="Using FastAPI framework",
            domain_rules="Follow repository patterns",
            output_dir="my_output",
        )
        assert spec.task == "Build a REST API with authentication"
        assert spec.context == "Using FastAPI framework"

    def test_minimal_task(self):
        spec = TaskSpec(task="Create a CLI tool")
        assert spec.context == ""
        assert spec.domain_rules == ""
        assert spec.output_dir == "output"

    def test_empty_task_rejected(self):
        with pytest.raises(ValidationError):
            TaskSpec()


class TestStageConfig:
    def test_defaults(self):
        config = StageConfig()
        assert config.model == ""
        assert config.timeout == 120
        assert config.max_retries == 2

    def test_custom_values(self):
        config = StageConfig(model="claude-opus", timeout=300, max_retries=3)
        assert config.model == "claude-opus"
        assert config.timeout == 300

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            StageConfig(timeout=0)

    def test_retries_bounds(self):
        with pytest.raises(ValidationError):
            StageConfig(max_retries=6)


class TestPipelineConfig:
    def test_defaults(self):
        config = PipelineConfig()
        assert config.arbiter_mode == ArbiterMode.BOOKEND
        assert config.reconciliation_enabled is False
        assert config.default_timeout == 120
        assert config.max_retries == 2
        assert config.reconciliation_retries == 1
        assert config.stages == {}
        assert config.arbiter_model == ""
        assert config.reconcile_model == ""

    def test_full_config(self):
        config = PipelineConfig(
            arbiter_mode=ArbiterMode.FULL,
            reconciliation_enabled=True,
            default_timeout=180,
            max_retries=3,
            reconciliation_retries=2,
            stages={
                PipelineStage.ARCHITECT: StageConfig(model="gemini-pro", timeout=240),
                PipelineStage.IMPLEMENT: StageConfig(model="gpt-4o"),
            },
            arbiter_model="claude-opus",
            reconcile_model="grok-4",
        )
        assert config.arbiter_mode == ArbiterMode.FULL
        assert config.reconciliation_enabled is True
        assert len(config.stages) == 2
        assert config.stages[PipelineStage.ARCHITECT].model == "gemini-pro"
        assert config.stages[PipelineStage.ARCHITECT].timeout == 240

    def test_invalid_arbiter_mode(self):
        with pytest.raises(ValidationError):
            PipelineConfig(arbiter_mode="turbo")

    def test_timeout_must_be_positive(self):
        with pytest.raises(ValidationError):
            PipelineConfig(default_timeout=0)

    def test_serialization_roundtrip(self):
        config = PipelineConfig(
            arbiter_mode=ArbiterMode.BOOKEND,
            reconciliation_enabled=True,
            stages={
                PipelineStage.VERIFY: StageConfig(timeout=300),
            },
        )
        data = config.model_dump()
        restored = PipelineConfig(**data)
        assert restored.arbiter_mode == ArbiterMode.BOOKEND
        assert restored.reconciliation_enabled is True
        assert PipelineStage.VERIFY in restored.stages

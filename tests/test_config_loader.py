"""Tests for triad.providers.registry — TOML config loading and model registry."""

from pathlib import Path

import pytest

from triad.providers.registry import (
    get_best_model_for_role,
    load_models,
    load_pipeline_config,
)
from triad.schemas.messages import PipelineStage
from triad.schemas.pipeline import ArbiterMode, ModelConfig, RoleFitness

# Path to the real config files shipped with the package
_CONFIG_DIR = Path(__file__).parent.parent / "triad" / "config"


class TestLoadModels:
    def test_loads_real_config(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        assert len(registry) > 0

    def test_all_expected_models_present(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        expected_keys = [
            "claude-opus",
            "claude-sonnet",
            "claude-haiku",
            "gpt-4o",
            "o3-mini",
            "gemini-pro",
            "gemini-flash",
            "grok-4",
            "grok-3",
        ]
        for key in expected_keys:
            assert key in registry, f"Missing model: {key}"

    def test_model_config_types(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        for key, model in registry.items():
            assert isinstance(model, ModelConfig), f"{key} is not ModelConfig"
            assert model.provider != ""
            assert model.model != ""
            assert model.display_name != ""
            assert model.api_key_env != ""
            assert model.context_window > 0
            assert model.cost_input >= 0.0
            assert model.cost_output >= 0.0

    def test_fitness_scores_loaded(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        claude_opus = registry["claude-opus"]
        assert claude_opus.fitness.architect == 0.85
        assert claude_opus.fitness.refactorer == 0.95

    def test_claude_opus_model_id(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        assert registry["claude-opus"].model == "anthropic/claude-opus-4-6"

    def test_providers_are_correct(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        assert registry["claude-opus"].provider == "anthropic"
        assert registry["gpt-4o"].provider == "openai"
        assert registry["gemini-pro"].provider == "google"
        assert registry["grok-4"].provider == "xai"

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_models(Path("/nonexistent/models.toml"))

    def test_empty_models_section_raises(self, tmp_path):
        bad_toml = tmp_path / "bad.toml"
        bad_toml.write_text("[models]\n")
        # Empty [models] section parses as empty dict — treated as missing
        with pytest.raises(ValueError, match="No \\[models\\] section"):
            load_models(bad_toml)

    def test_no_models_section_raises(self, tmp_path):
        bad_toml = tmp_path / "no_models.toml"
        bad_toml.write_text('[other]\nfoo = "bar"\n')
        with pytest.raises(ValueError, match="No \\[models\\] section"):
            load_models(bad_toml)

    def test_custom_toml(self, tmp_path):
        toml_content = """
[models.test-model]
provider = "test"
model = "test-v1"
display_name = "Test Model"
api_key_env = "TEST_API_KEY"
context_window = 4096
cost_input = 0.5
cost_output = 1.0

[models.test-model.fitness]
architect = 0.9
implementer = 0.8
refactorer = 0.7
verifier = 0.6
"""
        custom = tmp_path / "models.toml"
        custom.write_text(toml_content)

        registry = load_models(custom)
        assert "test-model" in registry
        model = registry["test-model"]
        assert model.provider == "test"
        assert model.fitness.architect == 0.9
        assert model.fitness.verifier == 0.6


class TestLoadPipelineConfig:
    def test_loads_real_defaults(self):
        config = load_pipeline_config(_CONFIG_DIR / "defaults.toml")
        assert config.arbiter_mode == ArbiterMode.BOOKEND
        assert config.reconciliation_enabled is False
        assert config.default_timeout == 120
        assert config.max_retries == 2
        assert config.reconciliation_retries == 1

    def test_missing_file_raises(self):
        with pytest.raises(FileNotFoundError):
            load_pipeline_config(Path("/nonexistent/defaults.toml"))

    def test_empty_toml_uses_defaults(self, tmp_path):
        empty = tmp_path / "defaults.toml"
        empty.write_text("")
        config = load_pipeline_config(empty)
        # Should fall back to PipelineConfig defaults
        assert config.arbiter_mode == ArbiterMode.BOOKEND
        assert config.default_timeout == 120

    def test_custom_config(self, tmp_path):
        toml_content = """
[pipeline]
arbiter_mode = "full"
reconciliation_enabled = true
default_timeout = 240
max_retries = 3
reconciliation_retries = 2
arbiter_model = "claude-opus"
reconcile_model = "grok-4"
"""
        custom = tmp_path / "defaults.toml"
        custom.write_text(toml_content)

        config = load_pipeline_config(custom)
        assert config.arbiter_mode == ArbiterMode.FULL
        assert config.reconciliation_enabled is True
        assert config.default_timeout == 240
        assert config.max_retries == 3
        assert config.arbiter_model == "claude-opus"

    def test_stage_overrides(self, tmp_path):
        toml_content = """
[pipeline]
arbiter_mode = "bookend"

[pipeline.stages.architect]
model = "gemini-pro"
timeout = 300
max_retries = 1
"""
        custom = tmp_path / "defaults.toml"
        custom.write_text(toml_content)

        config = load_pipeline_config(custom)
        assert PipelineStage.ARCHITECT in config.stages
        stage = config.stages[PipelineStage.ARCHITECT]
        assert stage.model == "gemini-pro"
        assert stage.timeout == 300
        assert stage.max_retries == 1


class TestGetBestModelForRole:
    def test_best_architect(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        best = get_best_model_for_role(registry, PipelineStage.ARCHITECT)
        # Gemini Pro has architect=0.90, highest in the registry
        assert best == "gemini-pro"

    def test_best_refactorer(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        best = get_best_model_for_role(registry, PipelineStage.REFACTOR)
        # Claude Opus has refactorer=0.95, highest in the registry
        assert best == "claude-opus"

    def test_best_implementer(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        best = get_best_model_for_role(registry, PipelineStage.IMPLEMENT)
        # GPT-4o and Claude Sonnet both have implementer=0.85
        # max() returns the first one found at that value
        best_model = registry[best]
        assert best_model.fitness.implementer == 0.85

    def test_best_verifier(self):
        registry = load_models(_CONFIG_DIR / "models.toml")
        best = get_best_model_for_role(registry, PipelineStage.VERIFY)
        # Grok 4 has verifier=0.92, highest
        assert best == "grok-4"

    def test_empty_registry(self):
        result = get_best_model_for_role({}, PipelineStage.ARCHITECT)
        assert result is None

    def test_single_model_registry(self):
        registry = {
            "only": ModelConfig(
                provider="test",
                model="test",
                display_name="Test",
                api_key_env="TEST",
                context_window=1000,
                cost_input=0.0,
                cost_output=0.0,
                fitness=RoleFitness(architect=0.5),
            ),
        }
        assert get_best_model_for_role(registry, PipelineStage.ARCHITECT) == "only"

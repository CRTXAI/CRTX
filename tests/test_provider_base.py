"""Tests for triad.providers.base — ModelProvider ABC."""

import pytest

from triad.providers.base import ModelProvider
from triad.schemas.messages import AgentMessage
from triad.schemas.pipeline import ModelConfig, RoleFitness


def _make_config(**overrides) -> ModelConfig:
    """Helper to create a ModelConfig with sensible defaults."""
    defaults = {
        "provider": "test",
        "model": "test-model-v1",
        "display_name": "Test Model",
        "api_key_env": "TEST_API_KEY",
        "context_window": 128000,
        "supports_tools": True,
        "supports_structured": True,
        "supports_vision": False,
        "supports_thinking": False,
        "cost_input": 3.00,
        "cost_output": 15.00,
        "fitness": RoleFitness(architect=0.8, implementer=0.9, refactorer=0.7, verifier=0.85),
    }
    defaults.update(overrides)
    return ModelConfig(**defaults)


class ConcreteProvider(ModelProvider):
    """Minimal concrete implementation for testing the ABC."""

    async def complete(self, messages, system, *, output_schema=None, timeout=120):
        return AgentMessage(
            from_agent="architect",
            to_agent="implement",
            msg_type="proposal",
            content="test",
            confidence=0.5,
        )


class TestModelProviderProperties:
    def test_identity_properties(self):
        config = _make_config()
        provider = ConcreteProvider(config)
        assert provider.provider_id == "test"
        assert provider.model_id == "test-model-v1"
        assert provider.display_name == "Test Model"

    def test_capability_properties(self):
        config = _make_config()
        provider = ConcreteProvider(config)
        assert provider.context_window == 128000
        assert provider.supports_tools is True
        assert provider.supports_structured is True
        assert provider.supports_vision is False
        assert provider.supports_thinking is False

    def test_cost_properties(self):
        config = _make_config(cost_input=2.50, cost_output=10.00)
        provider = ConcreteProvider(config)
        assert provider.cost_per_1m_input == 2.50
        assert provider.cost_per_1m_output == 10.00

    def test_fitness_property(self):
        config = _make_config()
        provider = ConcreteProvider(config)
        assert provider.fitness.architect == 0.8
        assert provider.fitness.implementer == 0.9

    def test_config_property(self):
        config = _make_config()
        provider = ConcreteProvider(config)
        assert provider.config is config


class TestCalculateCost:
    def test_basic_cost_calculation(self):
        config = _make_config(cost_input=3.00, cost_output=15.00)
        provider = ConcreteProvider(config)
        # 1000 prompt tokens = 1000/1M * 3.00 = 0.003
        # 500 completion tokens = 500/1M * 15.00 = 0.0075
        cost = provider.calculate_cost(1000, 500)
        assert abs(cost - 0.0105) < 1e-10

    def test_zero_tokens(self):
        config = _make_config(cost_input=3.00, cost_output=15.00)
        provider = ConcreteProvider(config)
        assert provider.calculate_cost(0, 0) == 0.0

    def test_free_model(self):
        config = _make_config(cost_input=0.0, cost_output=0.0)
        provider = ConcreteProvider(config)
        assert provider.calculate_cost(100000, 50000) == 0.0

    def test_large_token_count(self):
        config = _make_config(cost_input=15.00, cost_output=75.00)
        provider = ConcreteProvider(config)
        # 1M input = $15, 1M output = $75
        cost = provider.calculate_cost(1_000_000, 1_000_000)
        assert abs(cost - 90.0) < 1e-10


class TestAbstractEnforcement:
    def test_cannot_instantiate_abc_directly(self):
        config = _make_config()
        with pytest.raises(TypeError):
            ModelProvider(config)

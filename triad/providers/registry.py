"""Model registry and TOML configuration loader.

Loads model definitions from models.toml and pipeline defaults from
defaults.toml. Provides lookup methods for model discovery and
role-based assignment.
"""

from __future__ import annotations

import tomllib
from pathlib import Path

from triad.schemas.pipeline import (
    ArbiterMode,
    ModelConfig,
    PipelineConfig,
    PipelineStage,
    RoleFitness,
    StageConfig,
)

# Default config directory relative to the triad package
_CONFIG_DIR = Path(__file__).parent.parent / "config"


def load_models(config_path: Path | None = None) -> dict[str, ModelConfig]:
    """Load the model registry from a TOML file.

    Args:
        config_path: Path to models.toml. Defaults to triad/config/models.toml.

    Returns:
        Dictionary mapping model keys to ModelConfig instances.

    Raises:
        FileNotFoundError: If the config file does not exist.
        ValueError: If the TOML structure is invalid.
    """
    path = config_path or _CONFIG_DIR / "models.toml"
    if not path.exists():
        raise FileNotFoundError(f"Model registry not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    models_section = raw.get("models")
    if not models_section or not isinstance(models_section, dict):
        raise ValueError(f"No [models] section found in {path}")

    registry: dict[str, ModelConfig] = {}
    for key, entry in models_section.items():
        if not isinstance(entry, dict):
            continue

        # Extract nested fitness scores if present
        fitness_data = entry.pop("fitness", {})
        fitness = RoleFitness(**fitness_data) if fitness_data else RoleFitness()

        registry[key] = ModelConfig(**entry, fitness=fitness)

    return registry


def load_pipeline_config(config_path: Path | None = None) -> PipelineConfig:
    """Load pipeline defaults from a TOML file.

    Args:
        config_path: Path to defaults.toml. Defaults to triad/config/defaults.toml.

    Returns:
        PipelineConfig with values from the TOML file.

    Raises:
        FileNotFoundError: If the config file does not exist.
    """
    path = config_path or _CONFIG_DIR / "defaults.toml"
    if not path.exists():
        raise FileNotFoundError(f"Pipeline config not found: {path}")

    with open(path, "rb") as f:
        raw = tomllib.load(f)

    pipeline_section = raw.get("pipeline", {})

    # Parse arbiter_mode from string
    arbiter_mode = ArbiterMode(pipeline_section.get("arbiter_mode", "bookend"))

    # Parse per-stage overrides if present
    stages: dict[PipelineStage, StageConfig] = {}
    raw_stages = pipeline_section.get("stages", {})
    for stage_name, stage_data in raw_stages.items():
        stage = PipelineStage(stage_name)
        stages[stage] = StageConfig(**stage_data)

    return PipelineConfig(
        arbiter_mode=arbiter_mode,
        reconciliation_enabled=pipeline_section.get("reconciliation_enabled", False),
        default_timeout=pipeline_section.get("default_timeout", 120),
        max_retries=pipeline_section.get("max_retries", 2),
        reconciliation_retries=pipeline_section.get("reconciliation_retries", 1),
        stages=stages,
        arbiter_model=pipeline_section.get("arbiter_model", ""),
        reconcile_model=pipeline_section.get("reconcile_model", ""),
    )


def get_best_model_for_role(
    registry: dict[str, ModelConfig], role: PipelineStage
) -> str | None:
    """Find the model with the highest fitness score for a given pipeline role.

    Args:
        registry: The loaded model registry.
        role: The pipeline stage to find the best model for.

    Returns:
        The registry key of the best-fit model, or None if registry is empty.
    """
    if not registry:
        return None

    # Map pipeline stages to RoleFitness field names
    role_field_map: dict[PipelineStage, str] = {
        PipelineStage.ARCHITECT: "architect",
        PipelineStage.IMPLEMENT: "implementer",
        PipelineStage.REFACTOR: "refactorer",
        PipelineStage.VERIFY: "verifier",
    }

    field = role_field_map[role]
    best_key = max(registry, key=lambda k: getattr(registry[k].fitness, field))
    return best_key

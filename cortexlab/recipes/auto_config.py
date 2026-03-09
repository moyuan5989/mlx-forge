"""Auto-configuration rules for recipes.

Given a recipe and hardware profile, produces a fully resolved
TrainingConfig with smart defaults.
"""

from __future__ import annotations

from typing import Optional

from cortexlab.models.memory import (
    HardwareProfile,
    auto_configure,
)
from cortexlab.recipes.registry import Recipe


def resolve_config(
    recipe: Recipe,
    model_id: str,
    train_path: str,
    valid_path: str,
    *,
    hardware: Optional[HardwareProfile] = None,
    dataset_samples: Optional[int] = None,
    overrides: Optional[dict] = None,
) -> dict:
    """Build a fully resolved training config from a recipe.

    Args:
        recipe: The recipe template.
        model_id: HuggingFace model ID.
        train_path: Path to training data file.
        valid_path: Path to validation data file.
        hardware: Hardware profile (auto-detected if None).
        dataset_samples: Number of dataset samples (for auto-config).
        overrides: User-specified overrides (applied last).

    Returns:
        Complete config dict ready for TrainingConfig.from_dict().
    """
    hw = hardware or HardwareProfile.detect()

    # Start from recipe template
    config = _deep_copy_dict(recipe.config_template)

    # Set model
    config.setdefault("model", {})
    config["model"]["path"] = model_id

    # Set data paths
    config.setdefault("data", {})
    config["data"]["train"] = train_path
    config["data"]["valid"] = valid_path

    # Set training type
    config.setdefault("training", {})
    config["training"]["training_type"] = recipe.training_type

    # Apply auto-configuration rules
    auto = auto_configure(
        model_id,
        system_memory_gb=hw.total_memory_gb,
        dataset_samples=dataset_samples,
    )

    for key, value in auto.items():
        _set_nested(config, key, value)

    # Apply user overrides last
    if overrides:
        for key, value in overrides.items():
            _set_nested(config, key, value)

    # Set schema version
    config["schema_version"] = 1

    return config


def _deep_copy_dict(d: dict) -> dict:
    """Deep copy a dict (simple implementation for YAML-like structures)."""
    result = {}
    for k, v in d.items():
        if isinstance(v, dict):
            result[k] = _deep_copy_dict(v)
        elif isinstance(v, list):
            result[k] = list(v)
        else:
            result[k] = v
    return result


def _set_nested(d: dict, key: str, value) -> None:
    """Set a nested dict value using dot-separated key.

    Example: _set_nested(d, "model.quantization.bits", 4)
    """
    parts = key.split(".")
    for part in parts[:-1]:
        if part not in d:
            d[part] = {}
        d = d[part]
    d[parts[-1]] = value

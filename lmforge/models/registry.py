"""Model registry for LMForge.

Maps model_type from config.json to architecture implementations.
Uses an explicit allowlist for supported models.
"""

from __future__ import annotations

import importlib
from typing import Tuple, Type

import mlx.nn as nn


# Remap model types to their architecture implementation
# e.g., "mistral" models use the same architecture as "llama"
MODEL_REMAPPING = {
    "mistral": "llama",
    # Add more remappings as needed
}

# Explicit allowlist of supported architectures
# Maps model_type -> module path under lmforge.models.architectures
SUPPORTED_ARCHITECTURES = {
    "llama": "lmforge.models.architectures.llama",
    "phi3": "lmforge.models.architectures.phi3",
    "qwen3": "lmforge.models.architectures.qwen3",
    "gemma": "lmforge.models.architectures.gemma",
    "gemma2": "lmforge.models.architectures.gemma",
    "gemma3": "lmforge.models.architectures.gemma",
}


def get_model_classes(config: dict) -> Tuple[Type[nn.Module], Type]:
    """
    Get Model and ModelArgs classes for a given config.

    Args:
        config: Model config dict (from config.json)

    Returns:
        (Model, ModelArgs) tuple

    Raises:
        ValueError: If model_type is not supported

    Example:
        >>> config = {"model_type": "qwen3", "hidden_size": 1024, ...}
        >>> Model, ModelArgs = get_model_classes(config)
        >>> args = ModelArgs.from_dict(config)
        >>> model = Model(args)
    """
    model_type = config.get("model_type")
    if model_type is None:
        raise ValueError(
            "config.json missing 'model_type' field. "
            "Cannot determine model architecture."
        )

    # Apply remapping (e.g., mistral -> llama)
    original_type = model_type
    model_type = MODEL_REMAPPING.get(model_type, model_type)

    if model_type not in SUPPORTED_ARCHITECTURES:
        supported = sorted(SUPPORTED_ARCHITECTURES.keys())
        remapped_note = ""
        if original_type != model_type:
            remapped_note = f" (remapped from '{original_type}')"

        raise ValueError(
            f"Model type '{model_type}'{remapped_note} is not supported.\n\n"
            f"Supported architectures: {supported}\n\n"
            f"If you need this architecture, please open an issue."
        )

    module_path = SUPPORTED_ARCHITECTURES[model_type]

    try:
        arch = importlib.import_module(module_path)
    except ImportError as e:
        raise ImportError(
            f"Failed to import architecture module '{module_path}': {e}"
        )

    # Verify the module exports required classes
    if not hasattr(arch, "Model"):
        raise AttributeError(
            f"Architecture module '{module_path}' missing required 'Model' class"
        )
    if not hasattr(arch, "ModelArgs"):
        raise AttributeError(
            f"Architecture module '{module_path}' missing required 'ModelArgs' class"
        )

    return arch.Model, arch.ModelArgs


def list_supported_architectures() -> list[str]:
    """Return list of supported model architectures."""
    return sorted(SUPPORTED_ARCHITECTURES.keys())


def is_supported(model_type: str) -> bool:
    """Check if a model type is supported."""
    model_type = MODEL_REMAPPING.get(model_type, model_type)
    return model_type in SUPPORTED_ARCHITECTURES

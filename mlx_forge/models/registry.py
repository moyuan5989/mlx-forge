"""Model registry for MLX Forge.

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
    "llama3": "llama",
    "gemma2": "gemma",
    "gemma3": "gemma",
    "falcon_mamba": "mamba",
    "qwen2_5": "qwen2",
    "deepseek": "llama",
}

# Explicit allowlist of supported architectures
# Maps model_type -> module path under mlx_forge.models.architectures
SUPPORTED_ARCHITECTURES = {
    "llama": "mlx_forge.models.architectures.llama",
    "phi3": "mlx_forge.models.architectures.phi3",
    "phi4": "mlx_forge.models.architectures.phi4",
    "qwen2": "mlx_forge.models.architectures.qwen2",
    "qwen3": "mlx_forge.models.architectures.qwen3",
    "qwen3_5": "mlx_forge.models.architectures.qwen3_5",
    "gemma": "mlx_forge.models.architectures.gemma",
    "mixtral": "mlx_forge.models.architectures.mixtral",
    "deepseek_v2": "mlx_forge.models.architectures.deepseek_v2",
    "deepseek_v3": "mlx_forge.models.architectures.deepseek_v3",
    "cohere": "mlx_forge.models.architectures.cohere",
    "cohere2": "mlx_forge.models.architectures.cohere2",
    "llama4": "mlx_forge.models.architectures.llama4",
    "mamba": "mlx_forge.models.architectures.mamba",
    "mamba2": "mlx_forge.models.architectures.mamba2",
    "jamba": "mlx_forge.models.architectures.jamba",
    "falcon_h1": "mlx_forge.models.architectures.falcon_h1",
    "olmo2": "mlx_forge.models.architectures.olmo2",
    "internlm2": "mlx_forge.models.architectures.internlm2",
    "starcoder2": "mlx_forge.models.architectures.starcoder2",
    "glm4": "mlx_forge.models.architectures.glm4",
    "granite": "mlx_forge.models.architectures.granite",
    "stablelm": "mlx_forge.models.architectures.stablelm",
    "openelm": "mlx_forge.models.architectures.openelm",
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

"""Model and tokenizer loading for LMForge v0.

Self-contained model loading without external dependencies on mlx-lm.
"""

from __future__ import annotations

import glob
import json
from pathlib import Path
from typing import Optional, Tuple

import mlx.core as mx
import mlx.nn as nn
from transformers import AutoTokenizer

from .registry import get_model_classes


def load_config(model_path: Path) -> dict:
    """
    Load model configuration from config.json.

    Args:
        model_path: Path to model directory

    Returns:
        Configuration dictionary

    Raises:
        FileNotFoundError: If config.json is missing
    """
    config_path = model_path / "config.json"
    if not config_path.exists():
        raise FileNotFoundError(
            f"config.json not found in {model_path}\n\n"
            f"Ensure the model directory contains:\n"
            f"  - config.json (model configuration)\n"
            f"  - model*.safetensors (model weights)"
        )

    with open(config_path) as f:
        return json.load(f)


def load_weights(model_path: Path) -> dict[str, mx.array]:
    """
    Load all safetensors weight files from model directory.

    Args:
        model_path: Path to model directory

    Returns:
        Dictionary mapping weight names to arrays

    Raises:
        FileNotFoundError: If no safetensors files found
    """
    weight_files = sorted(glob.glob(str(model_path / "model*.safetensors")))

    if not weight_files:
        # List what files are present for debugging
        found_files = list(model_path.glob("*"))
        found_names = [f.name for f in found_files[:10]]
        raise FileNotFoundError(
            f"No safetensors weight files found in {model_path}\n\n"
            f"Expected files matching 'model*.safetensors'.\n"
            f"Found files: {found_names}\n\n"
            f"The model may not have been fully downloaded. Try:\n"
            f"  huggingface-cli download <model-id> --local-dir {model_path}"
        )

    weights = {}
    for wf in weight_files:
        weights.update(mx.load(wf))

    return weights


def load_model(
    model_path: str,
    tokenizer_path: Optional[str] = None,
    trust_remote_code: bool = False,
) -> Tuple[nn.Module, "AutoTokenizer"]:
    """
    Load a model and tokenizer from a local path.

    This function provides self-contained model loading without requiring
    mlx-lm. It supports the architectures defined in the LMForge registry.

    Args:
        model_path: Local path to model directory (post-resolution)
        tokenizer_path: Optional separate tokenizer path (post-resolution)
        trust_remote_code: Whether to trust remote code in tokenizer

    Returns:
        (model, tokenizer) tuple

    Raises:
        FileNotFoundError: If config.json or weights are missing
        ValueError: If model architecture is not supported

    Note:
        This function expects a local path. HF resolution should happen
        before calling this function (see lmforge.models.resolve).

    Example:
        >>> from lmforge.models.loader import load_model
        >>> model, tokenizer = load_model("/path/to/model")
        >>> logits = model(mx.array([[1, 2, 3]]))
    """
    model_path = Path(model_path)
    tok_path = tokenizer_path if tokenizer_path is not None else str(model_path)

    # Load tokenizer using transformers
    tokenizer = AutoTokenizer.from_pretrained(
        tok_path,
        trust_remote_code=trust_remote_code,
    )

    # Load config and resolve model class
    config = load_config(model_path)
    model_class, model_args_class = get_model_classes(config)

    # Instantiate model with config
    model_args = model_args_class.from_dict(config)
    model = model_class(model_args)

    # Load weights
    weights = load_weights(model_path)

    # Apply model-specific weight sanitization
    if hasattr(model, "sanitize"):
        weights = model.sanitize(weights)

    # Set to eval mode and load weights
    model.eval()
    model.load_weights(list(weights.items()), strict=True)

    # Force evaluation to ensure weights are loaded into memory
    mx.eval(model.parameters())

    return model, tokenizer

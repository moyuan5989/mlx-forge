"""Model quantization for QLoRA training.

Quantizes base model weights to 4-bit or 8-bit while keeping LoRA adapters
in float32. This reduces memory ~67% for 4-bit, enabling 7B+ models on 32GB.
"""

from __future__ import annotations

import mlx.nn as nn

from lmforge.config import QuantizationConfig


def quantize_model(model: nn.Module, config: QuantizationConfig) -> nn.Module:
    """Quantize model linear layers in-place.

    Uses MLX's built-in nn.quantize which replaces nn.Linear modules with
    nn.QuantizedLinear. Only targets Linear layers; embeddings and norms
    are left in full precision.

    Args:
        model: The model to quantize (modified in-place).
        config: Quantization configuration (bits, group_size).

    Returns:
        The quantized model (same object, modified in-place).
    """
    nn.quantize(
        model,
        bits=config.bits,
        group_size=config.group_size,
        class_predicate=lambda path, m: isinstance(m, nn.Linear) and "lm_head" not in path,
    )
    # Freeze all parameters after quantization. LoRA's from_base() will
    # create new unfrozen LoRA params on top, so only LoRA params get gradients.
    # Without this, value_and_grad would try to differentiate through quantized
    # weights in non-LoRA layers and fail.
    model.freeze()
    return model

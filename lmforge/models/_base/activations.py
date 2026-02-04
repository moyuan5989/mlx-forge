"""Activation functions for model architectures."""

from __future__ import annotations

from functools import partial

import mlx.core as mx
import mlx.nn as nn


@partial(mx.compile, shapeless=True)
def swiglu(gate: mx.array, x: mx.array) -> mx.array:
    """
    SwiGLU activation function.

    Computes SiLU(gate) * x, used in modern LLM MLPs.

    Args:
        gate: Gate tensor from gate_proj
        x: Input tensor from up_proj

    Returns:
        Activated tensor
    """
    return nn.silu(gate) * x

"""DoRA (Weight-Decomposed Low-Rank Adaptation) module for MLX Forge.

DoRA decomposes weight updates into magnitude and direction:
  W' = m * (W + ΔW) / ||W + ΔW||_col
where m is a learned magnitude vector and ΔW = (scale/r) * B @ A
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.adapters.lora import LoRALinear


class DoRALinear(LoRALinear):
    """DoRA wrapper for nn.Linear — LoRA with learned magnitude.

    Adds a trainable magnitude vector (out_features,) on top of LoRALinear.
    The forward pass normalizes the combined weight by column norms,
    then scales by the learned magnitude.
    """

    def __init__(
        self,
        in_features: int,
        out_features: int,
        r: int = 8,
        scale: float = 20.0,
        dropout: float = 0.0,
        bias: bool = False,
    ):
        super().__init__(
            in_features=in_features,
            out_features=out_features,
            r=r,
            scale=scale,
            dropout=dropout,
            bias=bias,
        )
        # Magnitude vector initialized later in from_base()
        self.magnitude = mx.ones((out_features,))

    def __call__(self, x):
        """Forward: y = magnitude * (x @ normalized(W + ΔW).T)"""
        # Get base weight
        if isinstance(self.base_layer, nn.QuantizedLinear):
            # Dequantize for DoRA computation
            base_weight = mx.dequantize(
                self.base_layer.weight,
                self.base_layer.scales,
                self.base_layer.biases,
                self.base_layer.group_size,
                self.base_layer.bits,
            )
        else:
            base_weight = self.base_layer.weight

        # Combined weight: W + (scale/r) * B @ A
        lora_weight = self.lora_b @ self.lora_a  # (out, in)
        combined = base_weight + lora_weight * (self.scale / self.r)

        # Column norms: ||combined||_col  — norm over in_features (axis=1)
        col_norms = mx.sqrt((combined * combined).sum(axis=1, keepdims=True))
        col_norms = mx.maximum(col_norms, 1e-8)  # avoid division by zero

        # Normalized direction
        direction = combined / col_norms  # (out, in)

        # Output: magnitude * (x @ direction.T)
        out = x @ direction.T
        out = out * self.magnitude  # broadcast (out_features,)

        # Add bias if present
        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            out = out + self.base_layer.bias

        return out

    @classmethod
    def from_base(
        cls, base_linear, *, r: int, scale: float, dropout: float = 0.0
    ) -> "DoRALinear":
        """Create DoRALinear from an existing Linear or QuantizedLinear.

        Initializes magnitude from column norms of the base weight.
        """
        # Get dimensions
        if isinstance(base_linear, nn.QuantizedLinear):
            out_features = base_linear.weight.shape[0]
            in_features = base_linear.weight.shape[1] * (32 // base_linear.bits)
            bias = hasattr(base_linear, "bias") and base_linear.bias is not None
        elif isinstance(base_linear, nn.Linear):
            out_features, in_features = base_linear.weight.shape
            bias = hasattr(base_linear, "bias") and base_linear.bias is not None
        elif hasattr(base_linear, "weight"):
            out_features, in_features = base_linear.weight.shape
            bias = hasattr(base_linear, "bias") and base_linear.bias is not None
        else:
            raise ValueError(
                f"Cannot create DoRALinear from {type(base_linear).__name__}. "
                "Expected nn.Linear or a module with a 'weight' attribute."
            )

        # Create module
        dora = cls(
            in_features=in_features,
            out_features=out_features,
            r=r,
            scale=scale,
            dropout=dropout,
            bias=bias,
        )

        # Store base layer
        dora.base_layer = base_linear
        base_linear.freeze()

        # Initialize magnitude from base weight column norms
        if isinstance(base_linear, nn.QuantizedLinear):
            base_weight = mx.dequantize(
                base_linear.weight,
                base_linear.scales,
                base_linear.biases,
                base_linear.group_size,
                base_linear.bits,
            )
        else:
            base_weight = base_linear.weight

        col_norms = mx.sqrt((base_weight * base_weight).sum(axis=1))
        dora.magnitude = col_norms

        return dora

    def fuse(self):
        """Merge DoRA weights into plain Linear.

        W_fused = magnitude * (W + scale/r * B @ A) / ||W + scale/r * B @ A||_col
        """
        if not hasattr(self, "base_layer"):
            raise ValueError(
                "Cannot fuse: this DoRALinear was not created from a base layer."
            )

        if isinstance(self.base_layer, nn.QuantizedLinear):
            base_weight = mx.dequantize(
                self.base_layer.weight,
                self.base_layer.scales,
                self.base_layer.biases,
                self.base_layer.group_size,
                self.base_layer.bits,
            )
        else:
            base_weight = self.base_layer.weight

        # Combined weight
        lora_weight = self.lora_b @ self.lora_a
        combined = base_weight + lora_weight * (self.scale / self.r)

        # Normalize and scale by magnitude
        col_norms = mx.sqrt((combined * combined).sum(axis=1, keepdims=True))
        col_norms = mx.maximum(col_norms, 1e-8)
        fused_weight = self.magnitude[:, None] * (combined / col_norms)

        # Create plain Linear
        fused_linear = nn.Linear(self.in_features, self.out_features)
        fused_linear.weight = fused_weight

        if hasattr(self.base_layer, "bias") and self.base_layer.bias is not None:
            fused_linear.bias = self.base_layer.bias

        return fused_linear

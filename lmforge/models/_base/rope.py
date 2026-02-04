"""Rotary Position Embedding (RoPE) utilities."""

from __future__ import annotations

import math
from typing import Dict, List, Optional, Union

import mlx.core as mx
import mlx.nn as nn


class Llama3RoPE(nn.Module):
    """
    Llama 3 style RoPE with frequency scaling.

    Uses smooth interpolation between low and high frequency factors.
    """

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_config: dict = None,
    ):
        super().__init__()
        self.dims = dims
        self.max_position_embeddings = max_position_embeddings
        self.traditional = traditional

        factor = scaling_config["factor"]
        low_freq_factor = scaling_config.get("low_freq_factor", 1.0)
        high_freq_factor = scaling_config.get("high_freq_factor", 4.0)
        old_context_len = scaling_config.get(
            "original_max_position_embeddings",
            8192,
        )

        low_freq_wavelen = old_context_len / low_freq_factor
        high_freq_wavelen = old_context_len / high_freq_factor

        freqs = base ** (mx.arange(0, dims, 2) / dims)
        wavelens = 2 * mx.pi * freqs

        freqs = mx.where(wavelens > low_freq_wavelen, freqs * factor, freqs)
        is_medium_freq = (wavelens > high_freq_wavelen) & (wavelens < low_freq_wavelen)
        smooth_factors = (old_context_len / wavelens - low_freq_factor) / (
            high_freq_factor - low_freq_factor
        )
        smooth_freqs = freqs / ((1 - smooth_factors) / factor + smooth_factors)
        self._freqs = mx.where(is_medium_freq, smooth_freqs, freqs)

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class SuScaledRoPE(nn.Module):
    """
    Su-Scaled RoPE (longrope) for extended context lengths.

    Used by Phi-3 and other models with long context support.
    """

    def __init__(
        self,
        dims: int,
        base: float = 10000.0,
        max_position_embeddings: int = 131072,
        original_max_position_embeddings: int = 4096,
        short_factor: Union[List[float], float] = 1.0,
        long_factor: Union[List[float], float] = 1.0,
        short_mscale: float = None,
        long_mscale: float = None,
    ):
        super().__init__()
        self.original_max_position_embeddings = original_max_position_embeddings
        self.dim = dims

        freqs = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        self._freqs = mx.array(long_factor, dtype=mx.float32) * freqs

        def default_scale(factor):
            return math.sqrt(
                1 + math.log(factor) / math.log(original_max_position_embeddings)
            )

        factor = max_position_embeddings / original_max_position_embeddings
        self._scale = long_mscale or (1.0 if factor <= 1.0 else default_scale(factor))

    def __call__(self, x: mx.array, offset: Union[int, mx.array] = 0) -> mx.array:
        x[..., : self.dim] = self._scale * x[..., : self.dim]
        return mx.fast.rope(
            x,
            self.dim,
            traditional=False,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


class YarnRoPE(nn.Module):
    """
    YaRN (Yet another RoPE extensioN) for extended context.

    Provides smooth interpolation with attention scaling.
    """

    def __init__(
        self,
        dims: int,
        max_position_embeddings: int = 2048,
        traditional: bool = False,
        base: float = 10000,
        scaling_factor: float = 1.0,
        original_max_position_embeddings: int = 4096,
        beta_fast: int = 32,
        beta_slow: int = 1,
        mscale: float = 1,
        mscale_all_dim: float = 0,
    ):
        super().__init__()

        def yarn_find_correction_dim(num_rotations):
            return (
                dims
                * math.log(
                    original_max_position_embeddings / (num_rotations * 2 * math.pi)
                )
            ) / (2 * math.log(base))

        def yarn_find_correction_range():
            low = math.floor(yarn_find_correction_dim(beta_fast))
            high = math.ceil(yarn_find_correction_dim(beta_slow))
            return max(low, 0), min(high, dims - 1)

        def yarn_get_mscale(scale=1, mscale=1):
            if scale <= 1:
                return 1.0
            return 0.1 * mscale * math.log(scale) + 1.0

        def yarn_linear_ramp_mask(min_val, max_val, dim):
            if min_val == max_val:
                max_val += 0.001

            linear_func = (mx.arange(dim, dtype=mx.float32) - min_val) / (
                max_val - min_val
            )
            return mx.clip(linear_func, 0, 1)

        self.mscale = yarn_get_mscale(scaling_factor, mscale) / yarn_get_mscale(
            scaling_factor, mscale_all_dim
        )
        freq_extra = base ** (mx.arange(0, dims, 2, dtype=mx.float32) / dims)
        freq_inter = scaling_factor * freq_extra
        low, high = yarn_find_correction_range()
        freq_mask = 1.0 - yarn_linear_ramp_mask(low, high, dims // 2)
        self._freqs = (freq_inter * freq_extra) / (
            freq_inter * freq_mask + freq_extra * (1 - freq_mask)
        )
        self.dims = dims
        self.traditional = traditional

    def __call__(self, x: mx.array, offset: int = 0) -> mx.array:
        if self.mscale != 1.0:
            x[..., : self.dims] = self.mscale * x[..., : self.dims]
        return mx.fast.rope(
            x,
            self.dims,
            traditional=self.traditional,
            base=None,
            scale=1.0,
            offset=offset,
            freqs=self._freqs,
        )


def initialize_rope(
    dims: int,
    base: float,
    traditional: bool,
    scaling_config: Optional[Dict] = None,
    max_position_embeddings: Optional[int] = None,
) -> nn.Module:
    """
    Initialize the appropriate RoPE implementation based on config.

    Args:
        dims: Rotary embedding dimensions
        base: Base for exponential frequency computation
        traditional: Use traditional RoPE formulation
        scaling_config: Optional scaling configuration dict
        max_position_embeddings: Maximum sequence length

    Returns:
        Initialized RoPE module

    Supported rope_type values:
        - "default": Standard RoPE
        - "linear": Linear frequency scaling
        - "llama3": Llama 3 smooth interpolation
        - "longrope" / "su": Su-scaled for long context
        - "yarn": YaRN interpolation
    """
    if scaling_config is None:
        return nn.RoPE(dims, traditional=traditional, base=base)

    rope_type = scaling_config.get("type") or scaling_config.get("rope_type", "default")

    if rope_type in ["default", "linear"]:
        scale = 1 / scaling_config["factor"] if rope_type == "linear" else 1.0
        return nn.RoPE(dims, traditional=traditional, base=base, scale=scale)

    elif rope_type == "llama3":
        return Llama3RoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            base=base,
            scaling_config=scaling_config,
        )

    elif rope_type in ["longrope", "su"]:
        return SuScaledRoPE(
            dims=dims,
            base=base,
            max_position_embeddings=max_position_embeddings,
            original_max_position_embeddings=scaling_config[
                "original_max_position_embeddings"
            ],
            short_factor=scaling_config.get("short_factor", 1.0),
            long_factor=scaling_config.get("long_factor", 1.0),
        )

    elif rope_type in ["yarn", "deepseek_yarn"]:
        rope_kwargs = {
            key: scaling_config[key]
            for key in [
                "original_max_position_embeddings",
                "beta_fast",
                "beta_slow",
                "mscale",
                "mscale_all_dim",
            ]
            if key in scaling_config
        }
        return YarnRoPE(
            dims=dims,
            max_position_embeddings=max_position_embeddings,
            traditional=traditional,
            scaling_factor=scaling_config["factor"],
            base=base,
            **rope_kwargs,
        )

    else:
        raise ValueError(f"Unsupported RoPE type: {rope_type}")

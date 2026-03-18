"""Mamba-2 architecture with Structured State Space Duality (SSD) for MLX Forge.

Supports:
- Mamba-2 models

Key features:
- Structured state-space duality (SSD)
- Multi-head SSM with headdim parameter
- Uses RecurrentCache for inference

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.inference.cache import RecurrentCache

from .._base import BaseModelArgs


@dataclass
class ModelArgs(BaseModelArgs):
    """Mamba-2 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    d_state: int = 128
    d_conv: int = 4
    expand: int = 2
    intermediate_size: Optional[int] = None
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    headdim: int = 64
    n_groups: int = 1
    chunk_size: int = 256
    use_bias: bool = False
    use_conv_bias: bool = True
    time_step_rank: Optional[int] = None
    num_heads: Optional[int] = None

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.expand
        if self.num_heads is None:
            self.num_heads = self.intermediate_size // self.headdim
        if self.time_step_rank is None:
            self.time_step_rank = max(1, self.hidden_size // 16)


class DepthwiseConv1d(nn.Module):
    """Causal depthwise 1D convolution."""

    def __init__(self, channels: int, kernel_size: int, bias: bool = True):
        super().__init__()
        self.channels = channels
        self.kernel_size = kernel_size
        self.weight = mx.zeros((channels, kernel_size, 1))
        if bias:
            self.bias = mx.zeros((channels,))
        else:
            self.bias = None

    def __call__(self, x: mx.array) -> mx.array:
        B, L, C = x.shape
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])

        out = mx.zeros((B, L, C))
        for i in range(self.kernel_size):
            offset = self.kernel_size - 1 - i
            out = out + x[:, offset : offset + L, :] * self.weight[:, i, :].reshape(1, 1, C)

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, C)
        return out


class Mamba2Block(nn.Module):
    """Mamba-2 block with multi-head SSD."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.d_model = args.hidden_size
        self.d_inner = args.intermediate_size
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.headdim = args.headdim
        self.n_heads = args.num_heads
        self.n_groups = args.n_groups

        # Input projection: x, z, B, C, dt
        d_in_proj = 2 * self.d_inner + 2 * self.n_groups * self.d_state + self.n_heads
        self.in_proj = nn.Linear(self.d_model, d_in_proj, bias=args.use_bias)

        # Causal convolution on x branch
        self.conv1d = DepthwiseConv1d(
            self.d_inner, self.d_conv, bias=args.use_conv_bias
        )

        # dt bias
        self.dt_bias = mx.zeros((self.n_heads,))

        # Learnable A (log space)
        A = mx.repeat(
            mx.arange(1.0, self.d_state + 1).reshape(1, -1), self.n_heads, axis=0
        )
        self.A_log = mx.log(A)
        self.D = mx.ones((self.n_heads,))

        # Norm and output
        self.norm = nn.RMSNorm(self.d_inner, eps=args.rms_norm_eps)
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)

        self.layer_norm = nn.RMSNorm(self.d_model, eps=args.rms_norm_eps)

    def _ssd_scan(self, x, dt, B_mat, C_mat):
        """Multi-head SSD scan."""
        B_size, L, n_heads, headdim = x.shape
        d_state = B_mat.shape[-1]
        A = -mx.exp(self.A_log)  # (n_heads, d_state)

        dt = nn.softplus(dt + self.dt_bias[None, None, :])  # (B, L, n_heads)

        outputs = []
        h = mx.zeros((B_size, n_heads, headdim, d_state))

        for t in range(L):
            x_t = x[:, t, :, :]  # (B, n_heads, headdim)
            dt_t = dt[:, t, :]  # (B, n_heads)
            B_t = B_mat[:, t, :, :]  # (B, n_groups, d_state)
            C_t = C_mat[:, t, :, :]  # (B, n_groups, d_state)

            # Repeat B, C for groups
            heads_per_group = n_heads // self.n_groups
            B_t = mx.repeat(B_t, heads_per_group, axis=1)  # (B, n_heads, d_state)
            C_t = mx.repeat(C_t, heads_per_group, axis=1)

            dA = mx.exp(dt_t[:, :, None, None] * A[None, :, None, :])  # (B, n_heads, 1, d_state)
            dB = dt_t[:, :, None, None] * B_t[:, :, None, :]  # (B, n_heads, 1, d_state)

            h = h * dA + x_t[:, :, :, None] * dB
            y_t = mx.sum(h * C_t[:, :, None, :], axis=-1)  # (B, n_heads, headdim)
            outputs.append(y_t)

        return mx.stack(outputs, axis=1)  # (B, L, n_heads, headdim)

    def __call__(self, x: mx.array, cache: Optional[RecurrentCache] = None) -> mx.array:
        B, L, D = x.shape
        residual = x
        x = self.layer_norm(x)

        # Project
        proj = self.in_proj(x)

        # Split into components
        x_in = proj[:, :, : self.d_inner]
        z = proj[:, :, self.d_inner : 2 * self.d_inner]
        B_mat = proj[
            :, :, 2 * self.d_inner : 2 * self.d_inner + self.n_groups * self.d_state
        ].reshape(B, L, self.n_groups, self.d_state)
        C_mat = proj[
            :,
            :,
            2 * self.d_inner + self.n_groups * self.d_state : 2 * self.d_inner + 2 * self.n_groups * self.d_state,
        ].reshape(B, L, self.n_groups, self.d_state)
        dt = proj[:, :, -self.n_heads :]  # (B, L, n_heads)

        # Convolution with cache
        if cache is not None and cache.conv_state is not None:
            x_conv_in = mx.concatenate([cache.conv_state, x_in], axis=1)
            x_in = self.conv1d(x_conv_in)[:, -L:, :]
            cache.conv_state = x_conv_in[:, -(self.d_conv - 1) :, :]
        else:
            x_in = self.conv1d(x_in)
            if cache is not None:
                cache.conv_state = x_in[:, -(self.d_conv - 1) :, :]

        x_in = nn.silu(x_in)

        # Reshape for multi-head
        x_heads = x_in.reshape(B, L, self.n_heads, self.headdim)

        # SSD scan
        y = self._ssd_scan(x_heads, dt, B_mat, C_mat)

        # Add D term
        y = y + x_heads * self.D[None, None, :, None]

        # Reshape back
        y = y.reshape(B, L, self.d_inner)

        # Norm + gate + output
        y = self.norm(y)
        y = y * nn.silu(z)
        out = self.out_proj(y)

        if cache is not None:
            cache.offset += L

        return residual + out


class Mamba2Backbone(nn.Module):
    """Mamba-2 backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [Mamba2Block(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c)

        return self.norm_f(h)


class Model(nn.Module):
    """Mamba-2 model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = Mamba2Backbone(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.backbone(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.backbone.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def make_cache(self):
        return [RecurrentCache() for _ in range(self.args.num_hidden_layers)]

    def sanitize(self, weights: dict) -> dict:
        new_weights = {}
        for k, v in weights.items():
            if k.startswith("model."):
                k = k.replace("model.", "backbone.", 1)
            new_weights[k] = v
        if self.args.tie_word_embeddings:
            new_weights.pop("lm_head.weight", None)
        return new_weights

    @property
    def layers(self):
        return self.backbone.layers

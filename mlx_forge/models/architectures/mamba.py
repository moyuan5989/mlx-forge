"""Mamba (pure SSM) architecture for MLX Forge.

Supports:
- Mamba-1 (state-spaces/mamba-*)
- Falcon-Mamba (via model_type remapping)

Key features:
- No attention — uses selective state space model (SSM)
- Selective scan operation
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
    """Mamba model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    vocab_size: int
    d_state: int = 16
    d_conv: int = 4
    expand: int = 2
    intermediate_size: Optional[int] = None
    rms_norm_eps: float = 1e-5
    tie_word_embeddings: bool = True
    time_step_rank: Optional[int] = None
    use_bias: bool = False
    use_conv_bias: bool = True

    def __post_init__(self):
        if self.intermediate_size is None:
            self.intermediate_size = self.hidden_size * self.expand
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
        # x: (B, L, C) -> causal conv
        B, L, C = x.shape

        # Pad left for causal convolution
        x = mx.pad(x, [(0, 0), (self.kernel_size - 1, 0), (0, 0)])

        # Depthwise conv via sliding window sum
        out = mx.zeros((B, L, C))
        for i in range(self.kernel_size):
            offset = self.kernel_size - 1 - i
            out = out + x[:, offset : offset + L, :] * self.weight[:, i, :].reshape(1, 1, C)

        if self.bias is not None:
            out = out + self.bias.reshape(1, 1, C)

        return out


class MambaBlock(nn.Module):
    """Single Mamba SSM block with selective scan."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.d_model = args.hidden_size
        self.d_inner = args.intermediate_size
        self.d_state = args.d_state
        self.d_conv = args.d_conv
        self.dt_rank = args.time_step_rank

        # Input projection: projects to 2 * d_inner (for gate and input)
        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=args.use_bias)

        # Causal convolution
        self.conv1d = DepthwiseConv1d(
            self.d_inner, self.d_conv, bias=args.use_conv_bias
        )

        # SSM parameters
        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        # S4D real initialization for A
        A = mx.repeat(mx.arange(1.0, self.d_state + 1).reshape(1, -1), self.d_inner, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones((self.d_inner,))

        # Output projection
        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=args.use_bias)

        self.norm = nn.RMSNorm(self.d_model, eps=args.rms_norm_eps)

    def _ssm_scan(self, x: mx.array, dt: mx.array, B_mat: mx.array, C_mat: mx.array) -> mx.array:
        """Selective scan (sequential SSM)."""
        batch_size, seq_len, d_inner = x.shape
        d_state = self.d_state

        A = -mx.exp(self.A_log)  # (d_inner, d_state)

        # Discretize
        dt = nn.softplus(dt)  # (B, L, d_inner)

        outputs = []
        h = mx.zeros((batch_size, d_inner, d_state))

        for t in range(seq_len):
            x_t = x[:, t, :]  # (B, d_inner)
            dt_t = dt[:, t, :]  # (B, d_inner)
            B_t = B_mat[:, t, :]  # (B, d_state)
            C_t = C_mat[:, t, :]  # (B, d_state)

            # Discretize A and B for this timestep
            dA = mx.exp(dt_t[:, :, None] * A[None, :, :])  # (B, d_inner, d_state)
            dB = dt_t[:, :, None] * B_t[:, None, :]  # (B, d_inner, d_state)

            # SSM step
            h = h * dA + x_t[:, :, None] * dB
            y_t = mx.sum(h * C_t[:, None, :], axis=-1)  # (B, d_inner)
            outputs.append(y_t)

        return mx.stack(outputs, axis=1)  # (B, L, d_inner)

    def _ssm_step(self, x: mx.array, dt: mx.array, B_mat: mx.array, C_mat: mx.array, ssm_state: mx.array):
        """Single SSM step for inference with cache."""
        A = -mx.exp(self.A_log)

        dt = nn.softplus(dt)  # (B, 1, d_inner)
        dt = dt.squeeze(1)  # (B, d_inner)
        B_mat = B_mat.squeeze(1)  # (B, d_state)
        C_mat = C_mat.squeeze(1)  # (B, d_state)
        x = x.squeeze(1)  # (B, d_inner)

        dA = mx.exp(dt[:, :, None] * A[None, :, :])
        dB = dt[:, :, None] * B_mat[:, None, :]

        ssm_state = ssm_state * dA + x[:, :, None] * dB
        y = mx.sum(ssm_state * C_mat[:, None, :], axis=-1)

        return y[:, None, :], ssm_state  # (B, 1, d_inner), (B, d_inner, d_state)

    def __call__(self, x: mx.array, cache: Optional[RecurrentCache] = None) -> mx.array:
        B, L, D = x.shape
        residual = x
        x = self.norm(x)

        # Input projection
        xz = self.in_proj(x)
        x_in, z = mx.split(xz, 2, axis=-1)

        # Convolution with cache
        if cache is not None and cache.conv_state is not None:
            # Prepend cached conv state
            x_conv_in = mx.concatenate([cache.conv_state, x_in], axis=1)
            x_in = self.conv1d(x_conv_in)[:, -L:, :]
            # Update conv state
            cache.conv_state = x_conv_in[:, -(self.d_conv - 1):, :]
        else:
            x_in = self.conv1d(x_in)
            if cache is not None:
                cache.conv_state = x_in[:, -(self.d_conv - 1):, :]

        x_in = nn.silu(x_in)

        # SSM parameters
        x_dbl = self.x_proj(x_in)  # (B, L, dt_rank + 2*d_state)
        dt, B_mat, C_mat = mx.split(
            x_dbl, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        dt = self.dt_proj(dt)  # (B, L, d_inner)

        # SSM
        if cache is not None and L == 1 and cache.ssm_state is not None:
            y, ssm_state = self._ssm_step(x_in, dt, B_mat, C_mat, cache.ssm_state)
            cache.ssm_state = ssm_state
        else:
            y = self._ssm_scan(x_in, dt, B_mat, C_mat)
            if cache is not None:
                # Store final SSM state for next step
                # Re-run last step to get state
                A = -mx.exp(self.A_log)
                dt_all = nn.softplus(dt)
                h = mx.zeros((B, self.d_inner, self.d_state))
                for t in range(L):
                    dA = mx.exp(dt_all[:, t, :][:, :, None] * A[None, :, :])
                    dB = dt_all[:, t, :][:, :, None] * B_mat[:, t, :][:, None, :]
                    h = h * dA + x_in[:, t, :][:, :, None] * dB
                cache.ssm_state = h

        # Gate and output
        y = y + x_in * self.D[None, None, :]
        y = y * nn.silu(z)
        out = self.out_proj(y)

        if cache is not None:
            cache.offset += L

        return residual + out


class MambaBackbone(nn.Module):
    """Mamba backbone (embeddings + layers + norm)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [MambaBlock(args) for _ in range(args.num_hidden_layers)]
        self.norm_f = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        for layer, c in zip(self.layers, cache):
            h = layer(h, cache=c)

        return self.norm_f(h)


class Model(nn.Module):
    """Mamba model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.backbone = MambaBackbone(args)
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
        """Create recurrent caches for all layers."""
        return [RecurrentCache() for _ in range(self.args.num_hidden_layers)]

    def sanitize(self, weights: dict) -> dict:
        # Remap HF weight names if needed
        new_weights = {}
        for k, v in weights.items():
            # Map "model." prefix to "backbone."
            if k.startswith("model."):
                k = k.replace("model.", "backbone.", 1)
            new_weights[k] = v

        if self.args.tie_word_embeddings:
            new_weights.pop("lm_head.weight", None)
        return new_weights

    @property
    def layers(self):
        return self.backbone.layers

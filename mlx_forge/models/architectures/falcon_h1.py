"""Falcon H1 hybrid SSM+Attention architecture for MLX Forge.

Supports:
- Falcon H1 (TII)

Key features:
- Mix of Mamba SSM and standard attention layers
- ssm_cfg controls SSM parameters
- Standard transformer attention on alternating layers

Based on mlx-lm implementation (MIT License).
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Dict, Optional, Union

import mlx.core as mx
import mlx.nn as nn

from mlx_forge.inference.cache import KVCache, RecurrentCache

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.activations import swiglu
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Falcon H1 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    vocab_size: int
    rms_norm_eps: float = 1e-5
    num_key_value_heads: Optional[int] = None
    head_dim: Optional[int] = None
    max_position_embeddings: int = 8192
    rope_theta: float = 10000
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    tie_word_embeddings: bool = False
    attention_bias: bool = False

    # SSM parameters
    ssm_d_state: int = 16
    ssm_d_conv: int = 4
    ssm_expand: int = 2
    ssm_intermediate_size: Optional[int] = None
    ssm_time_step_rank: Optional[int] = None

    # Hybrid pattern: attention every N layers
    attn_layer_period: int = 6
    attn_layer_offset: int = 0

    def __post_init__(self):
        if self.num_key_value_heads is None:
            self.num_key_value_heads = self.num_attention_heads
        if self.head_dim is None:
            self.head_dim = self.hidden_size // self.num_attention_heads
        if self.ssm_intermediate_size is None:
            self.ssm_intermediate_size = self.hidden_size * self.ssm_expand
        if self.ssm_time_step_rank is None:
            self.ssm_time_step_rank = max(1, self.hidden_size // 16)


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


class MambaLayer(nn.Module):
    """Mamba SSM layer for Falcon H1."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.d_model = args.hidden_size
        self.d_inner = args.ssm_intermediate_size
        self.d_state = args.ssm_d_state
        self.d_conv = args.ssm_d_conv
        self.dt_rank = args.ssm_time_step_rank

        self.in_proj = nn.Linear(self.d_model, 2 * self.d_inner, bias=False)
        self.conv1d = DepthwiseConv1d(self.d_inner, self.d_conv, bias=True)

        self.x_proj = nn.Linear(self.d_inner, self.dt_rank + 2 * self.d_state, bias=False)
        self.dt_proj = nn.Linear(self.dt_rank, self.d_inner, bias=True)

        A = mx.repeat(mx.arange(1.0, self.d_state + 1).reshape(1, -1), self.d_inner, axis=0)
        self.A_log = mx.log(A)
        self.D = mx.ones((self.d_inner,))

        self.out_proj = nn.Linear(self.d_inner, self.d_model, bias=False)

    def _ssm_scan(self, x, dt, B_mat, C_mat):
        B_size, L, d_inner = x.shape
        A = -mx.exp(self.A_log)
        dt = nn.softplus(dt)

        outputs = []
        h = mx.zeros((B_size, d_inner, self.d_state))

        for t in range(L):
            x_t = x[:, t, :]
            dt_t = dt[:, t, :]
            B_t = B_mat[:, t, :]
            C_t = C_mat[:, t, :]

            dA = mx.exp(dt_t[:, :, None] * A[None, :, :])
            dB = dt_t[:, :, None] * B_t[:, None, :]

            h = h * dA + x_t[:, :, None] * dB
            y_t = mx.sum(h * C_t[:, None, :], axis=-1)
            outputs.append(y_t)

        return mx.stack(outputs, axis=1)

    def __call__(self, x: mx.array, cache: Optional[RecurrentCache] = None) -> mx.array:
        B, L, D = x.shape

        xz = self.in_proj(x)
        x_in, z = mx.split(xz, 2, axis=-1)

        if cache is not None and cache.conv_state is not None:
            x_conv_in = mx.concatenate([cache.conv_state, x_in], axis=1)
            x_in = self.conv1d(x_conv_in)[:, -L:, :]
            cache.conv_state = x_conv_in[:, -(self.d_conv - 1):, :]
        else:
            x_in = self.conv1d(x_in)
            if cache is not None:
                cache.conv_state = x_in[:, -(self.d_conv - 1):, :]

        x_in = nn.silu(x_in)

        x_dbl = self.x_proj(x_in)
        dt, B_mat, C_mat = mx.split(
            x_dbl, [self.dt_rank, self.dt_rank + self.d_state], axis=-1
        )
        dt = self.dt_proj(dt)

        y = self._ssm_scan(x_in, dt, B_mat, C_mat)
        y = y + x_in * self.D[None, None, :]
        y = y * nn.silu(z)
        out = self.out_proj(y)

        if cache is not None:
            cache.offset += L

        return out


class Attention(nn.Module):
    """Standard multi-head attention for Falcon H1."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = args.num_attention_heads
        self.n_kv_heads = args.num_key_value_heads
        self.head_dim = args.head_dim
        self.scale = self.head_dim**-0.5

        self.q_proj = nn.Linear(dim, self.n_heads * self.head_dim, bias=args.attention_bias)
        self.k_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.v_proj = nn.Linear(dim, self.n_kv_heads * self.head_dim, bias=args.attention_bias)
        self.o_proj = nn.Linear(self.n_heads * self.head_dim, dim, bias=args.attention_bias)

        self.rope = initialize_rope(
            self.head_dim,
            args.rope_theta,
            False,
            args.rope_scaling,
            args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[KVCache] = None,
    ) -> mx.array:
        B, L, D = x.shape

        queries, keys, values = self.q_proj(x), self.k_proj(x), self.v_proj(x)

        queries = queries.reshape(B, L, self.n_heads, -1).transpose(0, 2, 1, 3)
        keys = keys.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        if cache is not None:
            queries = self.rope(queries, offset=cache.offset)
            keys = self.rope(keys, offset=cache.offset)
            keys, values = cache.update_and_fetch(keys, values)
        else:
            queries = self.rope(queries)
            keys = self.rope(keys)

        output = scaled_dot_product_attention(
            queries, keys, values, scale=self.scale, mask=mask
        )
        output = output.transpose(0, 2, 1, 3).reshape(B, L, -1)
        return self.o_proj(output)


class MLP(nn.Module):
    """SwiGLU MLP."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class FalconH1Block(nn.Module):
    """Falcon H1 block: either SSM or attention + MLP."""

    def __init__(self, args: ModelArgs, layer_idx: int = 0):
        super().__init__()

        self.use_attention = (
            (layer_idx - args.attn_layer_offset) % args.attn_layer_period == 0
        )

        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

        if self.use_attention:
            self.self_attn = Attention(args)
        else:
            self.ssm = MambaLayer(args)

        self.mlp = MLP(args.hidden_size, args.intermediate_size)

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        h = self.input_layernorm(x)

        if self.use_attention:
            h = self.self_attn(h, mask, cache)
        else:
            h = self.ssm(h, cache)

        x = x + h
        r = self.mlp(self.post_attention_layernorm(x))
        return x + r


class FalconH1Model(nn.Module):
    """Falcon H1 transformer backbone."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.vocab_size = args.vocab_size
        assert self.vocab_size > 0

        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            FalconH1Block(args, layer_idx=i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        mask = create_attention_mask(h, None)

        for layer, c in zip(self.layers, cache):
            h = layer(h, mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Falcon H1 hybrid model for MLX Forge."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = FalconH1Model(args)
        if not args.tie_word_embeddings:
            self.lm_head = nn.Linear(args.hidden_size, args.vocab_size, bias=False)

    def __call__(self, inputs: mx.array, cache=None):
        out = self.model(inputs, cache)
        if self.args.tie_word_embeddings:
            out = self.model.embed_tokens.as_linear(out)
        else:
            out = self.lm_head(out)
        return out

    def make_cache(self):
        """Create mixed cache types: KVCache for attention, RecurrentCache for SSM."""
        caches = []
        for i in range(self.args.num_hidden_layers):
            is_attn = (
                (i - self.args.attn_layer_offset) % self.args.attn_layer_period == 0
            )
            if is_attn:
                caches.append(KVCache())
            else:
                caches.append(RecurrentCache())
        return caches

    def sanitize(self, weights: dict) -> dict:
        weights = {k: v for k, v in weights.items() if "rotary_emb" not in k}
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)
        return weights

    @property
    def layers(self):
        return self.model.layers

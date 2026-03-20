"""Qwen3.5 model architecture for MLX Forge.

Supports:
- Qwen3.5-0.8B, 2B (hybrid architecture)

Key features:
- Hybrid architecture: 75% Gated DeltaNet (linear recurrent) + 25% standard softmax attention
- 3:1 alternating pattern (every 4th layer is full attention)
- Partial RoPE (25% of head dims rotated) on full attention layers
- Gated output on full attention: o_proj(attn * sigmoid(gate_from_q_proj))
- Gated DeltaNet with learnable A_log, dt_bias, and SwiGLU output norm
- 1D causal depthwise convolution with SiLU activation on DeltaNet layers

Based on mlx-lm qwen3_next implementation (MIT License).
"""

from __future__ import annotations

import inspect
from dataclasses import dataclass
from typing import Any, Dict, List, Optional, Tuple, Union

import mlx.core as mx
import mlx.nn as nn

from .._base import BaseModelArgs, create_attention_mask, scaled_dot_product_attention
from .._base.activations import swiglu
from .._base.rope import initialize_rope


@dataclass
class ModelArgs(BaseModelArgs):
    """Qwen3.5 model configuration arguments."""

    model_type: str
    hidden_size: int
    num_hidden_layers: int
    intermediate_size: int
    num_attention_heads: int
    num_key_value_heads: int
    head_dim: int
    rms_norm_eps: float
    vocab_size: int
    rope_theta: float = 10000000.0
    max_position_embeddings: int = 262144
    partial_rotary_factor: float = 0.25
    tie_word_embeddings: bool = True
    attention_bias: bool = False
    rope_scaling: Optional[Dict[str, Union[float, str]]] = None
    # DeltaNet config
    linear_num_key_heads: int = 16
    linear_num_value_heads: int = 16
    linear_key_head_dim: int = 128
    linear_value_head_dim: int = 128
    linear_conv_kernel_dim: int = 4
    # Hybrid config
    full_attention_interval: int = 4
    layer_types: Optional[List] = None

    @classmethod
    def from_dict(cls, params: dict) -> "ModelArgs":
        """Create ModelArgs from config dict, handling VLM nested structure.

        HF Qwen3.5 config.json nests text model params under 'text_config',
        with rope params further nested under 'rope_parameters'. This method
        flattens everything before creating the dataclass.
        """
        # If VLM config, extract text_config and merge top-level fields
        if "text_config" in params:
            flat = dict(params["text_config"])
            # Preserve top-level model_type ("qwen3_5") over text_config's ("qwen3_5_text")
            flat["model_type"] = params.get("model_type", flat.get("model_type"))
            # Top-level tie_word_embeddings takes precedence if present
            if "tie_word_embeddings" in params:
                flat["tie_word_embeddings"] = params["tie_word_embeddings"]
        else:
            flat = dict(params)

        # Extract rope params from nested rope_parameters if present
        rope_params = flat.pop("rope_parameters", None)
        if rope_params:
            if "rope_theta" in rope_params and "rope_theta" not in flat:
                flat["rope_theta"] = rope_params["rope_theta"]
            if "partial_rotary_factor" in rope_params and "partial_rotary_factor" not in flat:
                flat["partial_rotary_factor"] = rope_params["partial_rotary_factor"]
            # Map rope_type to rope_scaling if non-default
            rope_type = rope_params.get("rope_type", "default")
            if rope_type != "default" and "rope_scaling" not in flat:
                flat["rope_scaling"] = rope_params

        # Filter to only fields defined in the dataclass
        return cls(
            **{
                k: v
                for k, v in flat.items()
                if k in inspect.signature(cls).parameters
            }
        )


# ---------------------------------------------------------------------------
# Gated Delta Rule Recurrence
# ---------------------------------------------------------------------------


def _compute_decay(
    A_log: mx.array, a: mx.array, dt_bias: mx.array
) -> mx.array:
    """Compute decay gate: exp(-exp(A_log) * softplus(a + dt_bias)).

    All computation in float32 for numerical stability.
    """
    return mx.exp(
        -mx.exp(A_log.astype(mx.float32))
        * nn.softplus(a.astype(mx.float32) + dt_bias.astype(mx.float32))
    )


def _gated_delta_step(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    g: mx.array,
    beta: mx.array,
    state: mx.array,
) -> Tuple[mx.array, mx.array]:
    """Single step of gated delta rule recurrence.

    All computation done in float32 for numerical stability.

    Args:
        q, k: (B, H, Dk) — float32
        v: (B, H, Dv) — float32
        g: (B, H) — float32, decay gate
        beta: (B, H) — float32, update strength
        state: (B, H, Dv, Dk) — float32, recurrent state

    Returns:
        y: (B, H, Dv) float32, new_state: (B, H, Dv, Dk) float32
    """
    # Decay existing state
    state = state * g[..., None, None]  # (B, H, 1, 1) broadcast

    # Retrieve current memory for each value dimension
    kv_mem = (state * k[..., None, :]).sum(axis=-1)  # (B, H, Dv)

    # Delta update: correct towards target value
    delta = (v - kv_mem) * beta[..., None]  # (B, H, Dv)
    state = state + k[..., None, :] * delta[..., None]  # (B, H, Dv, Dk)

    # Readout with queries
    y = (state * q[..., None, :]).sum(axis=-1)  # (B, H, Dv)

    return y, state


def gated_delta_recurrence(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
) -> Tuple[mx.array, mx.array]:
    """Sequential gated delta recurrence.

    All internal computation is done in float32 for numerical stability.
    Output is cast back to the input dtype.

    Args:
        q, k: (B, T, Hk, Dk) — queries and keys
        v: (B, T, Hv, Dv) — values
        a: (B, T, Hv) — raw decay input
        b: (B, T, Hv) — raw update input
        A_log: (Hv,) — learnable log-scale parameter
        dt_bias: (Hv,) — learnable time-step bias
        state: (B, Hv, Dv, Dk) or None — recurrent state

    Returns:
        output: (B, T, Hv, Dv), new_state: (B, Hv, Dv, Dk)
    """
    orig_dtype = q.dtype
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]

    # Upcast everything to float32 for numerical stability
    q = q.astype(mx.float32)
    k = k.astype(mx.float32)
    v = v.astype(mx.float32)

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)

    # Compute gates (already float32 internally)
    beta = mx.sigmoid(b.astype(mx.float32))  # (B, T, Hv)
    g = _compute_decay(A_log, a, dt_bias)  # (B, T, Hv) — float32

    # Repeat Q, K if value heads > key heads (GQA-style grouping)
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=-2)
        k = mx.repeat(k, repeat_factor, axis=-2)

    outputs = []
    for t in range(T):
        y, state = _gated_delta_step(
            q[:, t], k[:, t], v[:, t],
            g[:, t], beta[:, t], state,
        )
        outputs.append(y)

    output = mx.stack(outputs, axis=1)  # (B, T, Hv, Dv)
    return output.astype(orig_dtype), state


def gated_delta_chunkwise(
    q: mx.array,
    k: mx.array,
    v: mx.array,
    a: mx.array,
    b: mx.array,
    A_log: mx.array,
    dt_bias: mx.array,
    state: Optional[mx.array] = None,
    chunk_size: int = 64,
) -> Tuple[mx.array, mx.array]:
    """Chunked gated delta recurrence — O(T/C) Python iterations instead of O(T).

    Uses intra-chunk parallel matmuls and inter-chunk state propagation.
    The WY-representation factors Householder products for efficient updates.

    Args:
        q, k: (B, T, Hk, Dk) — queries and keys
        v: (B, T, Hv, Dv) — values
        a: (B, T, Hv) — raw decay input
        b: (B, T, Hv) — raw update input
        A_log: (Hv,) — learnable log-scale parameter
        dt_bias: (Hv,) — learnable time-step bias
        state: (B, Hv, Dv, Dk) or None — recurrent state
        chunk_size: Tokens per chunk (default 64)

    Returns:
        output: (B, T, Hv, Dv), new_state: (B, Hv, Dv, Dk)
    """
    orig_dtype = q.dtype
    B, T, Hk, Dk = q.shape
    Hv, Dv = v.shape[-2:]
    C = chunk_size

    # Pad T to multiple of chunk_size
    pad_len = (C - T % C) % C
    if pad_len > 0:
        q = mx.pad(q, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        k = mx.pad(k, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        v = mx.pad(v, [(0, 0), (0, pad_len), (0, 0), (0, 0)])
        a = mx.pad(a, [(0, 0), (0, pad_len), (0, 0)])
        b = mx.pad(b, [(0, 0), (0, pad_len), (0, 0)])

    T_padded = q.shape[1]
    n_chunks = T_padded // C

    # NOTE: Do NOT upcast the full (B, T, H, D) tensors to float32 here.
    # That would allocate ~384 MB per DeltaNet layer × 21 layers = 8 GB.
    # Instead, upcast per-chunk inside the loop to keep peak memory low.

    if state is None:
        state = mx.zeros((B, Hv, Dv, Dk), dtype=mx.float32)
    else:
        state = state.astype(mx.float32)

    # Compute gates in float32 (these are small: B × T × Hv)
    beta = mx.sigmoid(b.astype(mx.float32))  # (B, T_padded, Hv)
    g = _compute_decay(A_log, a, dt_bias)     # (B, T_padded, Hv)

    # GQA: repeat Q, K to match value heads (still in original dtype)
    repeat_factor = Hv // Hk
    if repeat_factor > 1:
        q = mx.repeat(q, repeat_factor, axis=-2)
        k = mx.repeat(k, repeat_factor, axis=-2)

    # Pre-compute causal mask: (C, C) lower triangular
    causal_mask = mx.tril(mx.ones((C, C), dtype=mx.float32))

    # Transpose once before loop: (B, T, H, D) -> (B, H, T, D)
    # Keep in original dtype to save memory
    q = q.transpose(0, 2, 1, 3)       # (B, Hv, T_padded, Dk)
    k = k.transpose(0, 2, 1, 3)       # (B, Hv, T_padded, Dk)
    v = v.transpose(0, 2, 1, 3)       # (B, Hv, T_padded, Dv)
    g = g.transpose(0, 2, 1)          # (B, Hv, T_padded)
    beta = beta.transpose(0, 2, 1)    # (B, Hv, T_padded)

    outputs = []
    for ci in range(n_chunks):
        s = ci * C
        e = s + C

        # Slice and upcast per-chunk (keeps only C tokens in float32 at a time)
        q_h = q[:, :, s:e, :].astype(mx.float32)    # (B, Hv, C, Dk)
        k_h = k[:, :, s:e, :].astype(mx.float32)    # (B, Hv, C, Dk)
        v_h = v[:, :, s:e, :].astype(mx.float32)    # (B, Hv, C, Dv)
        g_h = g[:, :, s:e]       # (B, Hv, C) — already float32
        beta_h = beta[:, :, s:e] # (B, Hv, C) — already float32

        # --- 1. Intra-chunk attention (parallel matmul) ---
        # Q @ K^T -> (B, Hv, C, C), then apply causal mask
        attn = q_h @ k_h.transpose(0, 1, 3, 2)  # (B, Hv, C, C)

        # Apply decay mask: for positions (i, j), scale by product of g from j+1..i
        # Approximate with cumulative product ratio
        log_g = mx.log(mx.clip(g_h, a_min=1e-6, a_max=None))  # (B, Hv, C)
        cum_log_g = mx.cumsum(log_g, axis=-1)  # (B, Hv, C)

        # decay_mask[i, j] = exp(cum_log_g[i] - cum_log_g[j]) for i >= j
        # Apply causal mask BEFORE exp to avoid inf*0=nan:
        # set upper-triangle positions to -inf so exp(-inf)=0
        log_decay = cum_log_g[:, :, :, None] - cum_log_g[:, :, None, :]
        log_decay = mx.where(causal_mask, log_decay, -float("inf"))
        decay_mask = mx.exp(log_decay)  # (B, Hv, C, C)

        # Scale attention by decay and beta
        attn = attn * decay_mask * beta_h[:, :, None, :]  # (B, Hv, C, C)
        intra = attn @ v_h  # (B, Hv, C, Dv)

        # --- 2. Inter-chunk: contribution from previous state ---
        # Q @ state^T: (B, Hv, C, Dk) @ (B, Hv, Dk, Dv) -> (B, Hv, C, Dv)
        # state is (B, Hv, Dv, Dk) -> need (B, Hv, Dk, Dv) for matmul
        state_t = state.transpose(0, 1, 3, 2)  # (B, Hv, Dk, Dv)

        # Apply per-position decay to state contribution
        # Position i sees state decayed by product(g[0..i])
        # Clamp to avoid overflow in exp (values < -87 already give ~0 in float32)
        pos_decay = mx.exp(mx.clip(cum_log_g, a_min=-80.0, a_max=0.0))  # (B, Hv, C)
        inter = (q_h @ state_t) * pos_decay[:, :, :, None]  # (B, Hv, C, Dv)

        # Combine — already in (B, Hv, C, Dv)
        O_c = inter + intra
        outputs.append(O_c)

        # --- 3. Update state for next chunk ---
        # Decay state by full chunk's cumulative decay
        # Clamp to avoid underflow (exp(-304) ≈ 0 in float32 anyway)
        chunk_total_decay = mx.exp(mx.clip(cum_log_g[:, :, -1], a_min=-80.0, a_max=0.0))  # (B, Hv)
        state = state * chunk_total_decay[:, :, None, None]

        # Add contribution from this chunk's key-value pairs
        # Each timestep contributes: beta * v outer k, decayed to end of chunk
        # Decay from position t to end: exp(cum_log_g[-1] - cum_log_g[t])
        # This diff is always <= 0 (later positions have more cumulative decay)
        end_decay = mx.exp(
            cum_log_g[:, :, -1:] - cum_log_g
        )  # (B, Hv, C)

        weighted_k = k_h * (beta_h * end_decay)[:, :, :, None]  # (B, Hv, C, Dk)
        # v_h^T @ weighted_k -> (B, Hv, Dv, Dk)
        state = state + v_h.transpose(0, 1, 3, 2) @ weighted_k

    # Concatenate all chunks: (B, Hv, T_padded, Dv) -> (B, T_padded, Hv, Dv)
    output = mx.concatenate(outputs, axis=2)  # (B, Hv, T_padded, Dv)
    output = output.transpose(0, 2, 1, 3)     # (B, T_padded, Hv, Dv)
    if pad_len > 0:
        output = output[:, :T, :, :]

    return output.astype(orig_dtype), state


# Default chunk size for chunked recurrence
DEFAULT_CHUNK_SIZE = 64


# ---------------------------------------------------------------------------
# Modules
# ---------------------------------------------------------------------------


class Qwen3_5RMSNormGated(nn.Module):
    """RMSNorm with optional SwiGLU gating for DeltaNet output."""

    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = mx.ones(hidden_size)

    def __call__(
        self, hidden_states: mx.array, gate: Optional[mx.array] = None
    ) -> mx.array:
        x = mx.fast.rms_norm(hidden_states, self.weight, self.eps)
        if gate is not None:
            x = swiglu(gate, x)
        return x


class GatedDeltaNet(nn.Module):
    """Gated DeltaNet linear attention layer."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        self.hidden_size = args.hidden_size
        self.num_k_heads = args.linear_num_key_heads
        self.num_v_heads = args.linear_num_value_heads
        self.key_head_dim = args.linear_key_head_dim
        self.value_head_dim = args.linear_value_head_dim

        self.key_dim = self.num_k_heads * self.key_head_dim
        self.value_dim = self.num_v_heads * self.value_head_dim
        self.conv_dim = self.key_dim * 2 + self.value_dim
        self.conv_kernel_size = args.linear_conv_kernel_dim

        # Projections (names match HF weight keys)
        self.in_proj_qkv = nn.Linear(args.hidden_size, self.conv_dim, bias=False)
        self.in_proj_z = nn.Linear(args.hidden_size, self.value_dim, bias=False)
        self.in_proj_a = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.in_proj_b = nn.Linear(args.hidden_size, self.num_v_heads, bias=False)
        self.out_proj = nn.Linear(self.value_dim, args.hidden_size, bias=False)

        # Learnable decay and bias parameters
        self.dt_bias = mx.ones(self.num_v_heads)
        self.A_log = mx.log(
            mx.random.uniform(low=0.001, high=16, shape=(self.num_v_heads,))
        )

        # Output norm with SwiGLU gating
        self.norm = Qwen3_5RMSNormGated(self.value_head_dim, eps=args.rms_norm_eps)

        # 1D causal depthwise convolution
        self.conv1d = nn.Conv1d(
            in_channels=self.conv_dim,
            out_channels=self.conv_dim,
            kernel_size=self.conv_kernel_size,
            groups=self.conv_dim,
            bias=False,
            padding=0,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, T, _ = x.shape

        # Project QKV and gates
        qkv = self.in_proj_qkv(x)       # (B, T, conv_dim)
        z = self.in_proj_z(x)            # (B, T, value_dim) — output gate
        a_logit = self.in_proj_a(x)      # (B, T, num_v_heads) — decay
        b_logit = self.in_proj_b(x)      # (B, T, num_v_heads) — update

        # Prepare conv state
        if cache is not None and cache.conv_state is not None:
            conv_state = cache.conv_state
        else:
            conv_state = mx.zeros(
                (B, self.conv_kernel_size - 1, self.conv_dim), dtype=x.dtype
            )

        conv_input = mx.concatenate([conv_state, qkv], axis=1)

        # Update conv cache
        if cache is not None:
            n_keep = self.conv_kernel_size - 1
            cache.conv_state = conv_input[:, -n_keep:, :]

        # Conv + SiLU activation
        conv_out = nn.silu(self.conv1d(conv_input))

        # Split into Q, K, V and reshape to heads
        q, k, v = [
            t.reshape(B, T, h, d)
            for t, h, d in zip(
                mx.split(conv_out, [self.key_dim, 2 * self.key_dim], axis=-1),
                [self.num_k_heads, self.num_k_heads, self.num_v_heads],
                [self.key_head_dim, self.key_head_dim, self.value_head_dim],
            )
        ]

        # QK-norm with inverse scaling (no learnable weights)
        inv_scale = self.key_head_dim**-0.5
        q = (inv_scale**2) * mx.fast.rms_norm(q, None, 1e-6)
        k = inv_scale * mx.fast.rms_norm(k, None, 1e-6)

        # Gated delta recurrence — use chunked path for training (long sequences)
        ssm_state = cache.ssm_state if cache is not None else None
        if cache is not None or T <= DEFAULT_CHUNK_SIZE:
            # Inference (T=1 per step) or short sequences: sequential is optimal
            output, new_state = gated_delta_recurrence(
                q, k, v, a_logit, b_logit, self.A_log, self.dt_bias, ssm_state
            )
        else:
            # Training: chunked parallel path for long sequences
            output, new_state = gated_delta_chunkwise(
                q, k, v, a_logit, b_logit, self.A_log, self.dt_bias, ssm_state,
                chunk_size=DEFAULT_CHUNK_SIZE,
            )

        # Update SSM cache
        if cache is not None:
            cache.ssm_state = new_state
            cache.offset += T

        # Output norm with SwiGLU gating by z
        z = z.reshape(B, T, self.num_v_heads, self.value_head_dim)
        output = self.norm(output, z)

        return self.out_proj(output.reshape(B, T, -1))


class Attention(nn.Module):
    """Full softmax attention with QK-norm, partial RoPE, and fused output gate."""

    def __init__(self, args: ModelArgs):
        super().__init__()

        dim = args.hidden_size
        self.n_heads = n_heads = args.num_attention_heads
        self.n_kv_heads = n_kv_heads = args.num_key_value_heads
        self.head_dim = head_dim = args.head_dim
        self.scale = head_dim**-0.5

        # Q proj outputs 2x head_dim (queries + gate fused)
        self.q_proj = nn.Linear(
            dim, n_heads * head_dim * 2, bias=args.attention_bias
        )
        self.k_proj = nn.Linear(
            dim, n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.v_proj = nn.Linear(
            dim, n_kv_heads * head_dim, bias=args.attention_bias
        )
        self.o_proj = nn.Linear(
            n_heads * head_dim, dim, bias=args.attention_bias
        )

        # QK-norm
        self.q_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)
        self.k_norm = nn.RMSNorm(head_dim, eps=args.rms_norm_eps)

        # Partial RoPE: only rotate a fraction of head dims
        rope_dim = int(head_dim * args.partial_rotary_factor)
        self.rope = initialize_rope(
            rope_dim,
            base=args.rope_theta,
            traditional=False,
            scaling_config=args.rope_scaling,
            max_position_embeddings=args.max_position_embeddings,
        )

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        B, L, D = x.shape

        # Q projection with fused gate
        q_proj_output = self.q_proj(x)
        queries, gate = mx.split(
            q_proj_output.reshape(B, L, self.n_heads, -1), 2, axis=-1
        )
        gate = gate.reshape(B, L, -1)

        keys, values = self.k_proj(x), self.v_proj(x)

        # QK-norm and reshape
        queries = self.q_norm(queries).transpose(0, 2, 1, 3)
        keys = self.k_norm(
            keys.reshape(B, L, self.n_kv_heads, -1)
        ).transpose(0, 2, 1, 3)
        values = values.reshape(B, L, self.n_kv_heads, -1).transpose(0, 2, 1, 3)

        # Partial RoPE + KV cache
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

        # Apply gate (sigmoid) before output projection
        return self.o_proj(output * mx.sigmoid(gate))


class MLP(nn.Module):
    """Feed-forward network with SwiGLU activation."""

    def __init__(self, dim: int, hidden_dim: int):
        super().__init__()
        self.gate_proj = nn.Linear(dim, hidden_dim, bias=False)
        self.down_proj = nn.Linear(hidden_dim, dim, bias=False)
        self.up_proj = nn.Linear(dim, hidden_dim, bias=False)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))


class DecoderLayer(nn.Module):
    """Hybrid decoder layer — dispatches to DeltaNet or full attention."""

    def __init__(self, args: ModelArgs, layer_idx: int):
        super().__init__()
        self.is_linear = self._is_linear_layer(args, layer_idx)

        if self.is_linear:
            self.linear_attn = GatedDeltaNet(args)
        else:
            self.self_attn = Attention(args)

        self.mlp = MLP(args.hidden_size, args.intermediate_size)
        self.input_layernorm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self.post_attention_layernorm = nn.RMSNorm(
            args.hidden_size, eps=args.rms_norm_eps
        )

    @staticmethod
    def _is_linear_layer(args: ModelArgs, layer_idx: int) -> bool:
        """Determine if a layer is DeltaNet (linear) or full attention."""
        if args.layer_types is not None:
            layer_type = args.layer_types[layer_idx]
            return layer_type == 0 or layer_type == "linear_attention"
        return (layer_idx + 1) % args.full_attention_interval != 0

    def __call__(
        self,
        x: mx.array,
        mask: Optional[mx.array] = None,
        cache: Optional[Any] = None,
    ) -> mx.array:
        if self.is_linear:
            r = self.linear_attn(self.input_layernorm(x), mask, cache)
        else:
            r = self.self_attn(self.input_layernorm(x), mask, cache)
        h = x + r
        r = self.mlp(self.post_attention_layernorm(h))
        return h + r


class Qwen3_5Model(nn.Module):
    """Qwen3.5 transformer backbone (embeddings + hybrid layers + norm)."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.embed_tokens = nn.Embedding(args.vocab_size, args.hidden_size)
        self.layers = [
            DecoderLayer(args, i) for i in range(args.num_hidden_layers)
        ]
        self.norm = nn.RMSNorm(args.hidden_size, eps=args.rms_norm_eps)
        self._fa_idx = args.full_attention_interval - 1

    def __call__(self, inputs: mx.array, cache=None):
        h = self.embed_tokens(inputs)

        if cache is None:
            cache = [None] * len(self.layers)

        # Attention mask only for full-attention layers
        mask = create_attention_mask(h, cache[self._fa_idx])

        for layer, c in zip(self.layers, cache):
            if layer.is_linear:
                h = layer(h, mask=None, cache=c)
            else:
                h = layer(h, mask=mask, cache=c)

        return self.norm(h)


class Model(nn.Module):
    """Qwen3.5 model for MLX Forge training and inference."""

    def __init__(self, args: ModelArgs):
        super().__init__()
        self.args = args
        self.model_type = args.model_type
        self.model = Qwen3_5Model(args)
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
        """Create hybrid cache: RecurrentCache for DeltaNet, KVCache for attention."""
        from mlx_forge.inference.cache import KVCache, RecurrentCache

        caches = []
        for layer in self.model.layers:
            if layer.is_linear:
                caches.append(RecurrentCache())
            else:
                caches.append(KVCache())
        return caches

    def sanitize(self, weights: dict) -> dict:
        """Clean up weight dict before loading.

        - Remap VLM weight prefix: model.language_model.* -> model.*
        - Remove vision encoder weights (model.visual.*)
        - Remove MTP (multi-token prediction) weights
        - Transpose conv1d weights from HF to MLX format
        - Add 1.0 to norm weights stored in (weight - 1) convention
        - Remove lm_head if using tied embeddings
        """
        # Remap VLM weight prefix and remove non-text weights
        remapped = {}
        for k, v in weights.items():
            if k.startswith("model.visual."):
                continue  # Drop vision encoder
            if "mtp." in k:
                continue  # Drop multi-token prediction
            if k.startswith("model.language_model."):
                k = "model." + k[len("model.language_model."):]
            remapped[k] = v
        weights = remapped

        # Handle tied embeddings
        if self.args.tie_word_embeddings:
            weights.pop("lm_head.weight", None)

        # Fix conv1d weights and norm weights
        # Qwen3.5 stores standard RMSNorm weights in (weight - 1) convention
        # The DeltaNet output norm (linear_attn.norm) is NOT shifted
        norm_suffixes = (
            ".input_layernorm.weight",
            ".post_attention_layernorm.weight",
            "model.norm.weight",
            ".q_norm.weight",
            ".k_norm.weight",
        )
        for k, v in list(weights.items()):
            if "conv1d.weight" in k and v.shape[-1] != 1:
                weights[k] = v.moveaxis(2, 1)
            if any(k.endswith(sfx) for sfx in norm_suffixes):
                if v.ndim == 1:
                    weights[k] = v + 1.0

        return weights

    @property
    def layers(self):
        """Access transformer layers for LoRA targeting."""
        return self.model.layers

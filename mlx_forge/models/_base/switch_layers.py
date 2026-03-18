"""Mixture-of-Experts (MoE) routing layers.

Used by Mixtral, DeepSeek, Llama 4, and other MoE architectures.
"""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from .activations import swiglu


class SwitchGLU(nn.Module):
    """Sparse MoE layer with top-k expert routing.

    gate(x) → top-k indices → softmax weights →
    dispatch to expert MLPs → weighted sum

    Args:
        hidden_size: Input/output dimension
        intermediate_size: Expert hidden dimension
        num_experts: Total number of experts
        num_experts_per_tok: Number of active experts per token (top-k)
    """

    def __init__(
        self,
        hidden_size: int,
        intermediate_size: int,
        num_experts: int,
        num_experts_per_tok: int,
        bias: bool = False,
    ):
        super().__init__()
        self.num_experts = num_experts
        self.num_experts_per_tok = num_experts_per_tok

        # Router gate
        self.gate = nn.Linear(hidden_size, num_experts, bias=False)

        # Expert MLPs (each is a SwiGLU MLP)
        self.experts = [
            ExpertMLP(hidden_size, intermediate_size, bias=bias)
            for _ in range(num_experts)
        ]

    def __call__(self, x: mx.array) -> mx.array:
        """Route tokens to top-k experts and combine outputs.

        Args:
            x: Input tensor of shape (..., hidden_size)

        Returns:
            Output tensor of same shape as input
        """
        orig_shape = x.shape
        x_flat = x.reshape(-1, orig_shape[-1])  # (N, D)
        N, D = x_flat.shape

        # Compute routing logits and select top-k experts
        router_logits = self.gate(x_flat)  # (N, num_experts)
        top_k_indices = mx.argpartition(-router_logits, kth=self.num_experts_per_tok - 1, axis=-1)[
            :, :self.num_experts_per_tok
        ]  # (N, K)

        # Softmax weights for selected experts
        top_k_logits = mx.take_along_axis(router_logits, top_k_indices, axis=-1)
        top_k_weights = mx.softmax(top_k_logits, axis=-1)  # (N, K)

        # Dispatch to experts and combine
        output = mx.zeros_like(x_flat)
        for k in range(self.num_experts_per_tok):
            expert_indices = top_k_indices[:, k]  # (N,)
            weights = top_k_weights[:, k:k + 1]   # (N, 1)

            for e in range(self.num_experts):
                mask = expert_indices == e  # (N,)
                if not mx.any(mask).item():
                    continue
                expert_input = x_flat * mask[:, None]
                expert_output = self.experts[e](expert_input)
                output = output + expert_output * weights * mask[:, None]

        return output.reshape(orig_shape)


class ExpertMLP(nn.Module):
    """Single expert MLP with SwiGLU activation."""

    def __init__(self, hidden_size: int, intermediate_size: int, bias: bool = False):
        super().__init__()
        self.gate_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.up_proj = nn.Linear(hidden_size, intermediate_size, bias=bias)
        self.down_proj = nn.Linear(intermediate_size, hidden_size, bias=bias)

    def __call__(self, x: mx.array) -> mx.array:
        return self.down_proj(swiglu(self.gate_proj(x), self.up_proj(x)))

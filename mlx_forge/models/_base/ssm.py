"""Selective State Space Model (SSM) utilities for Mamba architectures.

Implements selective scan operations for both training (full sequence)
and inference (single-step with state caching).
"""

from __future__ import annotations

from typing import Optional

import mlx.core as mx
import mlx.nn as nn


def ssm_step(
    x: mx.array,
    conv_state: Optional[mx.array],
    ssm_state: Optional[mx.array],
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    delta: mx.array,
) -> tuple[mx.array, mx.array, mx.array]:
    """Single-step selective scan (for inference).

    Args:
        x: Input tensor (B, D)
        conv_state: Previous conv state (B, D, d_conv-1) or None
        ssm_state: Previous SSM state (B, D, d_state) or None
        A: State transition matrix (D, d_state)
        B: Input projection (B, d_state)
        C: Output projection (B, d_state)
        D: Skip connection (D,)
        delta: Time step (B, D)

    Returns:
        (output, new_conv_state, new_ssm_state)
    """
    B_batch, D_inner = x.shape

    # Discretize: A_bar = exp(delta * A), B_bar = delta * B
    delta_A = mx.exp(delta[:, :, None] * A[None, :, :])  # (B, D, d_state)
    delta_B = delta[:, :, None] * B[:, None, :]           # (B, D, d_state)

    # Update SSM state
    if ssm_state is None:
        ssm_state = mx.zeros((B_batch, D_inner, A.shape[-1]))

    new_ssm_state = delta_A * ssm_state + delta_B * x[:, :, None]  # (B, D, d_state)

    # Output
    y = (new_ssm_state * C[:, None, :]).sum(axis=-1)  # (B, D)
    y = y + D[None, :] * x

    return y, conv_state, new_ssm_state


def ssm_scan(
    x: mx.array,
    A: mx.array,
    B: mx.array,
    C: mx.array,
    D: mx.array,
    delta: mx.array,
) -> mx.array:
    """Full sequence selective scan (for training prefill).

    Args:
        x: Input tensor (B, T, D)
        A: State transition matrix (D, d_state)
        B: Input projection (B, T, d_state)
        C: Output projection (B, T, d_state)
        D: Skip connection (D,)
        delta: Time step (B, T, D)

    Returns:
        Output tensor (B, T, D)
    """
    B_batch, T, D_inner = x.shape
    d_state = A.shape[-1]

    # Sequential scan (can be parallelized with associative scan in future)
    state = mx.zeros((B_batch, D_inner, d_state))
    outputs = []

    for t in range(T):
        x_t = x[:, t, :]            # (B, D)
        B_t = B[:, t, :]            # (B, d_state)
        C_t = C[:, t, :]            # (B, d_state)
        delta_t = delta[:, t, :]    # (B, D)

        # Discretize
        delta_A = mx.exp(delta_t[:, :, None] * A[None, :, :])  # (B, D, d_state)
        delta_B = delta_t[:, :, None] * B_t[:, None, :]         # (B, D, d_state)

        # Update state
        state = delta_A * state + delta_B * x_t[:, :, None]

        # Output
        y_t = (state * C_t[:, None, :]).sum(axis=-1)  # (B, D)
        y_t = y_t + D[None, :] * x_t
        outputs.append(y_t)

    return mx.stack(outputs, axis=1)  # (B, T, D)


class MambaBlock(nn.Module):
    """Mamba selective state space block.

    Implements the core Mamba architecture:
    1. Linear projection to expanded dimension
    2. Causal convolution
    3. Selective SSM with input-dependent parameters
    4. Output projection
    """

    def __init__(
        self,
        d_model: int,
        d_state: int = 16,
        d_conv: int = 4,
        expand: int = 2,
    ):
        super().__init__()
        self.d_model = d_model
        self.d_state = d_state
        self.d_conv = d_conv
        d_inner = d_model * expand

        # Input projection (2x for gate)
        self.in_proj = nn.Linear(d_model, d_inner * 2, bias=False)

        # Causal conv1d
        self.conv1d = nn.Conv1d(
            d_inner, d_inner, kernel_size=d_conv,
            bias=True, padding=d_conv - 1,
        )

        # SSM parameters
        self.x_proj = nn.Linear(d_inner, d_state * 2 + 1, bias=False)  # B, C, delta
        self.dt_proj = nn.Linear(1, d_inner, bias=True)

        # Learnable SSM parameters
        self.A_log = mx.zeros((d_inner, d_state))
        self.D = mx.ones((d_inner,))

        # Output projection
        self.out_proj = nn.Linear(d_inner, d_model, bias=False)

    def __call__(self, x: mx.array, cache=None) -> mx.array:
        """Forward pass.

        Args:
            x: Input tensor (B, T, d_model)
            cache: Optional RecurrentCache

        Returns:
            Output tensor (B, T, d_model)
        """
        B, T, _ = x.shape

        # Project input
        xz = self.in_proj(x)  # (B, T, 2*d_inner)
        d_inner = xz.shape[-1] // 2
        x_proj, z = xz[:, :, :d_inner], xz[:, :, d_inner:]

        # Causal conv1d
        x_conv = self.conv1d(x_proj.transpose(0, 2, 1))  # (B, d_inner, T+padding)
        x_conv = x_conv[:, :, :T].transpose(0, 2, 1)     # (B, T, d_inner)
        x_conv = nn.silu(x_conv)

        # SSM parameters from input
        ssm_params = self.x_proj(x_conv)  # (B, T, d_state*2+1)
        B_ssm = ssm_params[:, :, :self.d_state]
        C_ssm = ssm_params[:, :, self.d_state:2 * self.d_state]
        delta_raw = ssm_params[:, :, -1:]  # (B, T, 1)
        delta = nn.softplus(self.dt_proj(delta_raw))  # (B, T, d_inner)

        # State matrix
        A = -mx.exp(self.A_log)  # (d_inner, d_state)

        # Run SSM
        y = ssm_scan(x_conv, A, B_ssm, C_ssm, self.D, delta)

        # Gate and output
        y = y * nn.silu(z)
        return self.out_proj(y)

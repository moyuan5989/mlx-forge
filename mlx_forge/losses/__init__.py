"""Pluggable loss functions for MLX Forge V2."""

from mlx_forge.losses.dpo import DPOLoss
from mlx_forge.losses.grpo import GRPOLoss
from mlx_forge.losses.sft import SFTLoss

__all__ = ["SFTLoss", "DPOLoss", "GRPOLoss"]

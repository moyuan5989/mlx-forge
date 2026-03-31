"""Pluggable loss functions for MLX Forge V2."""

from mlx_forge.losses.dpo import DPOLoss
from mlx_forge.losses.grpo import GRPOLoss
from mlx_forge.losses.mlm import MLMLoss
from mlx_forge.losses.preference import (
    compute_sequence_log_probs,
    kto_loss,
    orpo_loss,
    simpo_loss,
)
from mlx_forge.losses.seq2seq import Seq2SeqLoss
from mlx_forge.losses.sft import SFTLoss

__all__ = [
    "SFTLoss", "DPOLoss", "GRPOLoss", "MLMLoss", "Seq2SeqLoss",
    "orpo_loss", "kto_loss", "simpo_loss",
    "compute_sequence_log_probs",
]

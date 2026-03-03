"""Pluggable loss functions for LMForge V2."""

from lmforge.losses.sft import SFTLoss
from lmforge.losses.dpo import DPOLoss

__all__ = ["SFTLoss", "DPOLoss"]

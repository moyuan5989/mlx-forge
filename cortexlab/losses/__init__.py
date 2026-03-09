"""Pluggable loss functions for CortexLab V2."""

from cortexlab.losses.dpo import DPOLoss
from cortexlab.losses.sft import SFTLoss

__all__ = ["SFTLoss", "DPOLoss"]

"""Base utilities for model architectures."""

from .activations import swiglu
from .args import BaseModelArgs
from .attention import create_attention_mask, scaled_dot_product_attention

__all__ = [
    "BaseModelArgs",
    "create_attention_mask",
    "scaled_dot_product_attention",
    "swiglu",
]

# Lazy imports for optional base utilities
# from .switch_layers import SwitchGLU, ExpertMLP  # MoE
# from .ssm import MambaBlock, ssm_step, ssm_scan  # SSM
# from .mla import MultiLatentAttention             # MLA

"""Vision model support for MLX Forge.

Provides VLM inference and LoRA fine-tuning via mlx-vlm integration.
"""

from __future__ import annotations


def _check_mlx_vlm():
    """Check that mlx-vlm is installed."""
    try:
        import mlx_vlm  # noqa: F401
        return True
    except ImportError:
        raise ImportError(
            "Vision support requires mlx-vlm: pip install mlx-vlm\n"
            "Install with: pip install 'mlx-forge[vision]'"
        )

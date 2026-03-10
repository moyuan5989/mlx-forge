"""Memory profiler and hardware detection for MLX Forge V2.

Estimates GPU memory usage before training starts,
provides hardware tier detection, and model compatibility matrix.
"""

from __future__ import annotations

import math
import os
from dataclasses import dataclass
from typing import Optional


@dataclass
class HardwareProfile:
    """System hardware profile."""
    total_memory_gb: float
    chip_name: str = "Unknown"
    training_budget_gb: float = 0.0  # ~75% of total for training

    @classmethod
    def detect(cls) -> HardwareProfile:
        """Detect the current system's hardware profile."""
        total_bytes = _get_system_memory()
        total_gb = total_bytes / (1024 ** 3)

        chip = _detect_chip_name()

        # Training budget: ~75% of total (leave room for OS + apps)
        budget = total_gb * 0.75

        return cls(
            total_memory_gb=round(total_gb, 1),
            chip_name=chip,
            training_budget_gb=round(budget, 1),
        )


@dataclass
class MemoryEstimate:
    """Detailed memory breakdown for a training configuration."""
    base_weights_gb: float
    lora_overhead_gb: float
    optimizer_state_gb: float
    peak_activations_gb: float
    mlx_overhead_gb: float = 0.5
    total_gb: float = 0.0
    fits: bool = True
    budget_gb: float = 0.0

    def __post_init__(self):
        self.total_gb = (
            self.base_weights_gb
            + self.lora_overhead_gb
            + self.optimizer_state_gb
            + self.peak_activations_gb
            + self.mlx_overhead_gb
        )

    def bar_segments(self) -> list[dict]:
        """Return stacked bar segments for UI visualization."""
        return [
            {"label": "Base Weights", "gb": self.base_weights_gb, "color": "green"},
            {"label": "LoRA + Optimizer", "gb": self.lora_overhead_gb + self.optimizer_state_gb, "color": "blue"},
            {"label": "Activations", "gb": self.peak_activations_gb, "color": "orange"},
            {"label": "MLX Overhead", "gb": self.mlx_overhead_gb, "color": "gray"},
        ]


@dataclass
class ModelProfile:
    """Pre-computed profile for a known model."""
    model_id: str
    display_name: str
    num_params: float  # in billions
    hidden_dim: int
    num_layers: int
    num_heads: int
    vocab_size: int
    default_max_seq_length: int = 2048


# Pre-computed profiles for popular models
MODEL_PROFILES: dict[str, ModelProfile] = {
    "Qwen/Qwen3-0.6B": ModelProfile(
        model_id="Qwen/Qwen3-0.6B", display_name="Qwen3 0.6B",
        num_params=0.6, hidden_dim=1024, num_layers=28,
        num_heads=16, vocab_size=151936,
    ),
    "Qwen/Qwen3-1.7B": ModelProfile(
        model_id="Qwen/Qwen3-1.7B", display_name="Qwen3 1.7B",
        num_params=1.7, hidden_dim=2048, num_layers=28,
        num_heads=16, vocab_size=151936,
    ),
    "Qwen/Qwen3-4B": ModelProfile(
        model_id="Qwen/Qwen3-4B", display_name="Qwen3 4B",
        num_params=4.0, hidden_dim=2560, num_layers=36,
        num_heads=32, vocab_size=151936,
    ),
    "Qwen/Qwen3-8B": ModelProfile(
        model_id="Qwen/Qwen3-8B", display_name="Qwen3 8B",
        num_params=8.0, hidden_dim=4096, num_layers=36,
        num_heads=32, vocab_size=151936,
    ),
    "Qwen/Qwen3.5-0.8B": ModelProfile(
        model_id="Qwen/Qwen3.5-0.8B", display_name="Qwen3.5 0.8B",
        num_params=0.8, hidden_dim=1024, num_layers=28,
        num_heads=16, vocab_size=151936,
    ),
    "Qwen/Qwen3.5-3B": ModelProfile(
        model_id="Qwen/Qwen3.5-3B", display_name="Qwen3.5 3B",
        num_params=3.0, hidden_dim=2048, num_layers=36,
        num_heads=32, vocab_size=151936,
    ),
    "Qwen/Qwen2.5-0.5B": ModelProfile(
        model_id="Qwen/Qwen2.5-0.5B", display_name="Qwen2.5 0.5B",
        num_params=0.5, hidden_dim=896, num_layers=24,
        num_heads=14, vocab_size=151936,
    ),
    "Qwen/Qwen2.5-1.5B": ModelProfile(
        model_id="Qwen/Qwen2.5-1.5B", display_name="Qwen2.5 1.5B",
        num_params=1.5, hidden_dim=1536, num_layers=28,
        num_heads=12, vocab_size=151936,
    ),
    "Qwen/Qwen2.5-3B": ModelProfile(
        model_id="Qwen/Qwen2.5-3B", display_name="Qwen2.5 3B",
        num_params=3.0, hidden_dim=2048, num_layers=36,
        num_heads=16, vocab_size=151936,
    ),
    "Qwen/Qwen2.5-7B": ModelProfile(
        model_id="Qwen/Qwen2.5-7B", display_name="Qwen2.5 7B",
        num_params=7.0, hidden_dim=3584, num_layers=28,
        num_heads=28, vocab_size=152064,
    ),
    "meta-llama/Llama-3.1-8B": ModelProfile(
        model_id="meta-llama/Llama-3.1-8B", display_name="Llama 3.1 8B",
        num_params=8.0, hidden_dim=4096, num_layers=32,
        num_heads=32, vocab_size=128256,
    ),
    "google/gemma-2-2b": ModelProfile(
        model_id="google/gemma-2-2b", display_name="Gemma 2 2B",
        num_params=2.0, hidden_dim=2304, num_layers=26,
        num_heads=8, vocab_size=256000,
    ),
    "google/gemma-2-9b": ModelProfile(
        model_id="google/gemma-2-9b", display_name="Gemma 2 9B",
        num_params=9.0, hidden_dim=3584, num_layers=42,
        num_heads=16, vocab_size=256000,
    ),
    "google/gemma-3-1b-pt": ModelProfile(
        model_id="google/gemma-3-1b-pt", display_name="Gemma 3 1B",
        num_params=1.0, hidden_dim=1152, num_layers=26,
        num_heads=4, vocab_size=262144,
    ),
    "microsoft/phi-3-mini-4k-instruct": ModelProfile(
        model_id="microsoft/phi-3-mini-4k-instruct", display_name="Phi-3 Mini (3.8B)",
        num_params=3.8, hidden_dim=3072, num_layers=32,
        num_heads=32, vocab_size=32064,
    ),
    "microsoft/Phi-4-mini-instruct": ModelProfile(
        model_id="microsoft/Phi-4-mini-instruct", display_name="Phi-4 Mini (3.8B)",
        num_params=3.8, hidden_dim=3072, num_layers=32,
        num_heads=32, vocab_size=100352,
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B": ModelProfile(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-1.5B",
        display_name="DeepSeek-R1-Distill 1.5B",
        num_params=1.5, hidden_dim=1536, num_layers=28,
        num_heads=12, vocab_size=151936,
    ),
    "deepseek-ai/DeepSeek-R1-Distill-Qwen-7B": ModelProfile(
        model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-7B",
        display_name="DeepSeek-R1-Distill 7B",
        num_params=7.0, hidden_dim=3584, num_layers=28,
        num_heads=28, vocab_size=152064,
    ),
}


def estimate_memory(
    model_id: str,
    *,
    quantization_bits: Optional[int] = None,
    lora_rank: int = 16,
    lora_targets: int = 4,  # number of LoRA target modules per layer
    batch_size: int = 4,
    max_seq_length: int = 2048,
    gradient_checkpointing: bool = False,
    hardware: Optional[HardwareProfile] = None,
) -> MemoryEstimate:
    """Estimate training memory usage for a model configuration.

    Args:
        model_id: HuggingFace model ID (must be in MODEL_PROFILES).
        quantization_bits: None for fp16, 4 for QLoRA 4-bit, 8 for 8-bit.
        lora_rank: LoRA adapter rank.
        lora_targets: Number of LoRA target modules per layer.
        batch_size: Training batch size.
        max_seq_length: Maximum sequence length.
        gradient_checkpointing: Whether gradient checkpointing is enabled.
        hardware: Hardware profile (auto-detected if None).

    Returns:
        MemoryEstimate with detailed breakdown.
    """
    hw = hardware or HardwareProfile.detect()

    profile = MODEL_PROFILES.get(model_id)
    if profile is None:
        # Try to estimate from model name
        profile = _estimate_profile_from_id(model_id)
    if profile is None:
        raise ValueError(
            f"Unknown model '{model_id}'. Known models: {sorted(MODEL_PROFILES.keys())}"
        )

    # 1. Base weights
    bits_per_param = quantization_bits if quantization_bits else 16
    base_weights_gb = (profile.num_params * 1e9 * bits_per_param / 8) / (1024 ** 3)

    # 2. LoRA overhead: rank * (in_dim + out_dim) * 4 bytes (fp32)
    #    For attention QV: 2 modules per layer, each hidden_dim -> hidden_dim
    per_module_params = lora_rank * (profile.hidden_dim + profile.hidden_dim)
    total_lora_params = per_module_params * lora_targets * profile.num_layers
    lora_overhead_gb = (total_lora_params * 4) / (1024 ** 3)

    # 3. Optimizer state: Adam has 2 states (m, v) per trainable param
    optimizer_state_gb = 2 * lora_overhead_gb

    # 4. Peak activations (forward + backward)
    # Per-layer activation memory: QKV projections, attention output, MLP intermediates,
    # plus backward pass storage for gradient computation.
    # Uses 8x multiplier on (B * T * D * bytes) per layer to account for:
    #   - Multiple intermediate tensors (Q, K, V, attn_out, MLP up/gate/down)
    #   - Backward pass storing activations for gradient computation
    #   - Float32 upcast in some architectures (e.g., DeltaNet in Qwen3.5)
    bytes_per_element = 2  # fp16

    if gradient_checkpointing:
        # With checkpointing: ~sqrt(layers) layers stored; rest recomputed
        effective_layers = math.sqrt(profile.num_layers)
    else:
        effective_layers = profile.num_layers

    peak_activations_gb = (
        batch_size * max_seq_length * profile.hidden_dim
        * bytes_per_element * effective_layers * 8
    ) / (1024 ** 3)

    estimate = MemoryEstimate(
        base_weights_gb=round(base_weights_gb, 2),
        lora_overhead_gb=round(lora_overhead_gb, 2),
        optimizer_state_gb=round(optimizer_state_gb, 2),
        peak_activations_gb=round(peak_activations_gb, 2),
        budget_gb=hw.training_budget_gb,
    )
    estimate.fits = estimate.total_gb <= hw.training_budget_gb

    return estimate


def get_compatible_models(hardware: Optional[HardwareProfile] = None) -> list[dict]:
    """Return models compatible with current hardware, sorted by size.

    Each entry includes memory estimate and recommendation status.
    """
    hw = hardware or HardwareProfile.detect()
    results = []

    for model_id, profile in sorted(MODEL_PROFILES.items(), key=lambda x: x[1].num_params):
        # Try fp16 first, then QLoRA 4-bit
        fp16_est = estimate_memory(model_id, hardware=hw)
        qlora_est = estimate_memory(model_id, quantization_bits=4, hardware=hw)

        entry = {
            "model_id": model_id,
            "display_name": profile.display_name,
            "num_params_b": profile.num_params,
            "fp16": {
                "total_gb": fp16_est.total_gb,
                "fits": fp16_est.fits,
            },
            "qlora_4bit": {
                "total_gb": qlora_est.total_gb,
                "fits": qlora_est.fits,
            },
        }

        budget = hw.training_budget_gb
        qlora_ratio = qlora_est.total_gb / budget if budget > 0 else 1.0

        if qlora_est.fits and qlora_ratio < 0.5:
            entry["fit_level"] = "comfortable"
        elif qlora_est.fits and qlora_ratio < 0.8:
            entry["fit_level"] = "tight"
        else:
            entry["fit_level"] = "unlikely"

        results.append(entry)

    return results


def auto_configure(
    model_id: str,
    system_memory_gb: Optional[float] = None,
    dataset_samples: Optional[int] = None,
) -> dict:
    """Auto-configure training parameters based on hardware, model, and dataset.

    Uses memory estimation to find the largest batch_size that fits.

    Returns a dict of recommended config overrides.
    """
    if system_memory_gb is None:
        hw = HardwareProfile.detect()
        system_memory_gb = hw.total_memory_gb

    hw = HardwareProfile(
        total_memory_gb=system_memory_gb,
        training_budget_gb=system_memory_gb * 0.75,
    )

    overrides = {}

    # Step 1: Determine quantization based on memory tier
    use_qlora = system_memory_gb < 32
    if use_qlora:
        overrides["model.quantization"] = {"bits": 4, "group_size": 64}

    # Step 2: Choose batch_size and enable gradient checkpointing
    # Always enable gradient checkpointing (~10-15% slower but prevents OOM,
    # especially critical for hybrid architectures like Qwen3.5 DeltaNet).
    # Cap batch_size at 2 for safety — MLX compilation and graph overhead
    # make batch_size=4 unreliable on most hardware.
    if system_memory_gb < 16:
        chosen_batch = 1
    elif system_memory_gb < 48:
        chosen_batch = 2
    else:
        chosen_batch = 2  # even on large systems, batch=2 is safer

    overrides["training.batch_size"] = chosen_batch
    overrides["training.gradient_checkpointing"] = True
    overrides["training.grad_accumulation_steps"] = 4 // chosen_batch

    # Dataset-based rules
    if dataset_samples is not None:
        if dataset_samples < 500:
            overrides["training.num_iters"] = 500
            overrides["training.steps_per_eval"] = 50
            overrides["training.steps_per_save"] = 50
        elif dataset_samples > 10000:
            overrides["data.packing"] = True

    return overrides


def _get_system_memory() -> int:
    """Get total system memory in bytes."""
    try:
        return os.sysconf("SC_PAGE_SIZE") * os.sysconf("SC_PHYS_PAGES")
    except (ValueError, OSError):
        # Fallback: try sysctl on macOS
        try:
            import subprocess
            result = subprocess.run(
                ["sysctl", "-n", "hw.memsize"],
                capture_output=True, text=True, timeout=5
            )
            return int(result.stdout.strip())
        except Exception:
            return 16 * (1024 ** 3)  # Default to 16GB


def _detect_chip_name() -> str:
    """Detect Apple Silicon chip name."""
    try:
        import subprocess
        result = subprocess.run(
            ["sysctl", "-n", "machdep.cpu.brand_string"],
            capture_output=True, text=True, timeout=5
        )
        return result.stdout.strip()
    except Exception:
        return "Unknown"


def _estimate_profile_from_id(model_id: str) -> Optional[ModelProfile]:
    """Try to guess model profile from the model ID string."""
    # Check if any known profile is a prefix match
    lower_id = model_id.lower()
    for known_id, profile in MODEL_PROFILES.items():
        if known_id.lower() in lower_id or lower_id in known_id.lower():
            return profile
    return None

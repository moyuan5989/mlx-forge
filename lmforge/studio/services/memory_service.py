"""Memory Service — hardware detection and memory estimation for Studio."""

from __future__ import annotations

from typing import Optional

from lmforge.models.memory import (
    HardwareProfile,
    MemoryEstimate,
    estimate_memory,
    get_compatible_models,
)


class MemoryService:
    """Provides memory estimation and hardware info for the Studio API."""

    def __init__(self):
        self._hardware: Optional[HardwareProfile] = None

    @property
    def hardware(self) -> HardwareProfile:
        if self._hardware is None:
            self._hardware = HardwareProfile.detect()
        return self._hardware

    def get_hardware_info(self) -> dict:
        """Return current hardware profile."""
        hw = self.hardware
        return {
            "total_memory_gb": hw.total_memory_gb,
            "training_budget_gb": hw.training_budget_gb,
            "chip_name": hw.chip_name,
        }

    def estimate(
        self,
        model_id: str,
        *,
        quantization_bits: Optional[int] = None,
        lora_rank: int = 16,
        lora_targets: int = 4,
        batch_size: int = 4,
        max_seq_length: int = 2048,
        gradient_checkpointing: bool = False,
    ) -> dict:
        """Estimate memory for a training configuration."""
        est = estimate_memory(
            model_id,
            quantization_bits=quantization_bits,
            lora_rank=lora_rank,
            lora_targets=lora_targets,
            batch_size=batch_size,
            max_seq_length=max_seq_length,
            gradient_checkpointing=gradient_checkpointing,
            hardware=self.hardware,
        )
        return {
            "base_weights_gb": est.base_weights_gb,
            "lora_overhead_gb": est.lora_overhead_gb,
            "optimizer_state_gb": est.optimizer_state_gb,
            "peak_activations_gb": est.peak_activations_gb,
            "mlx_overhead_gb": est.mlx_overhead_gb,
            "total_gb": round(est.total_gb, 2),
            "budget_gb": est.budget_gb,
            "fits": est.fits,
            "bar_segments": est.bar_segments(),
        }

    def get_compatible_models(self) -> list[dict]:
        """Return models compatible with current hardware."""
        return get_compatible_models(self.hardware)

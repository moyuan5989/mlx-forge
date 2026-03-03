"""Model Library service — curated catalog of known models with memory estimates."""

from __future__ import annotations

from lmforge.models.memory import MODEL_PROFILES, HardwareProfile, estimate_memory
from lmforge.studio.services.model_service import ModelService


# Map model ID prefixes to architecture families
_ARCH_MAP = {
    "Qwen/Qwen3": "qwen3",
    "Qwen/Qwen2": "qwen2",
    "meta-llama": "llama",
    "google/gemma-3": "gemma3",
    "google/gemma-2": "gemma2",
    "microsoft/phi-3": "phi3",
    "microsoft/Phi-4": "phi4",
    "deepseek-ai": "qwen2",  # Distill models use Qwen2 arch
}


def _infer_architecture(model_id: str) -> str:
    """Infer architecture family from model ID string."""
    for prefix, arch in _ARCH_MAP.items():
        if model_id.startswith(prefix):
            return arch
    return "unknown"


class ModelLibraryService:
    """Combines MODEL_PROFILES with HF cache download checks."""

    def __init__(self):
        self._model_service = ModelService()

    def list_library(self) -> list[dict]:
        """Return curated model catalog sorted by param count.

        Each entry includes memory estimates, download status, and recommendation.
        """
        hw = HardwareProfile.detect()
        results = []

        for model_id, profile in sorted(
            MODEL_PROFILES.items(), key=lambda x: x[1].num_params
        ):
            fp16_est = estimate_memory(model_id, hardware=hw)
            qlora_est = estimate_memory(model_id, quantization_bits=4, hardware=hw)

            downloaded = self._model_service.get_model(model_id) is not None

            recommended = (
                qlora_est.fits
                and qlora_est.total_gb < hw.training_budget_gb * 0.8
            )

            results.append({
                "model_id": model_id,
                "display_name": profile.display_name,
                "num_params_b": profile.num_params,
                "architecture": _infer_architecture(model_id),
                "hidden_dim": profile.hidden_dim,
                "num_layers": profile.num_layers,
                "vocab_size": profile.vocab_size,
                "downloaded": downloaded,
                "fp16": {
                    "total_gb": round(fp16_est.total_gb, 1),
                    "fits": fp16_est.fits,
                },
                "qlora_4bit": {
                    "total_gb": round(qlora_est.total_gb, 1),
                    "fits": qlora_est.fits,
                },
                "recommended": recommended,
            })

        return results

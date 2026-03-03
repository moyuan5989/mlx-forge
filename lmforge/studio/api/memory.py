"""Memory API — hardware detection and memory estimation."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lmforge.studio.services.memory_service import MemoryService

router = APIRouter(prefix="/api/v2/memory", tags=["memory"])

_memory_service = MemoryService()


def get_memory_service() -> MemoryService:
    return _memory_service


@router.get("/hardware")
def get_hardware():
    """Get current hardware profile."""
    return get_memory_service().get_hardware_info()


@router.post("/estimate")
def estimate_memory(body: dict):
    """Estimate memory for a training configuration.

    Body:
    - model_id: HuggingFace model ID
    - quantization_bits: (optional) 4 or 8
    - lora_rank: (optional, default 16)
    - lora_targets: (optional, default 4)
    - batch_size: (optional, default 4)
    - max_seq_length: (optional, default 2048)
    - gradient_checkpointing: (optional, default false)
    """
    model_id = body.get("model_id")
    if not model_id:
        raise HTTPException(status_code=400, detail="'model_id' is required")

    try:
        return get_memory_service().estimate(
            model_id,
            quantization_bits=body.get("quantization_bits"),
            lora_rank=body.get("lora_rank", 16),
            lora_targets=body.get("lora_targets", 4),
            batch_size=body.get("batch_size", 4),
            max_seq_length=body.get("max_seq_length", 2048),
            gradient_checkpointing=body.get("gradient_checkpointing", False),
        )
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.get("/compatible-models")
def get_compatible_models():
    """List models compatible with current hardware."""
    return get_memory_service().get_compatible_models()

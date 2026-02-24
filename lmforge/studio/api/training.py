"""Training API — start, stop, and monitor training runs."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lmforge.studio.services.training_service import TrainingService

router = APIRouter(prefix="/api/v1/training", tags=["training"])

_training_service = TrainingService()


def get_training_service() -> TrainingService:
    return _training_service


def set_training_service(service: TrainingService):
    global _training_service
    _training_service = service


@router.post("/start")
async def start_training(config: dict):
    """Start a new training run as a subprocess.

    Accepts a full training config dict (same format as config.yaml).
    """
    try:
        result = await get_training_service().start_training(config)
        return result
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{track_id}/stop")
async def stop_training(track_id: str):
    """Stop a running training process."""
    result = await get_training_service().stop_training(track_id)
    if result["status"] == "not_found":
        raise HTTPException(status_code=404, detail=f"Training '{track_id}' not found")
    return result


@router.get("/active")
def list_active():
    """List all active training processes."""
    return get_training_service().list_active()

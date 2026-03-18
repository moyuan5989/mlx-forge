"""Runs API — training run discovery and management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel as PydanticBaseModel

from mlx_forge.studio.security import validate_safe_name
from mlx_forge.studio.services.run_service import RunService

router = APIRouter(prefix="/api/v1/runs", tags=["runs"])

_run_service = RunService()


def get_run_service() -> RunService:
    return _run_service


def set_run_service(service: RunService):
    global _run_service
    _run_service = service


@router.get("")
def list_runs():
    """List all training runs with summary info."""
    return get_run_service().list_runs()


@router.get("/adapters")
def list_adapters():
    """List available LoRA adapters from training runs."""
    return get_run_service().list_adapters()


@router.get("/{run_id}")
def get_run(run_id: str):
    """Get full details for a training run."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    run = get_run_service().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@router.get("/{run_id}/metrics")
def get_metrics(run_id: str):
    """Get train and eval metrics for a run."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    service = get_run_service()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return service.get_metrics(run_id)


@router.get("/{run_id}/config")
def get_config(run_id: str):
    """Get the training config for a run."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    config = get_run_service().get_config(run_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or config missing")
    return config


@router.get("/{run_id}/checkpoints")
def get_checkpoints(run_id: str):
    """List checkpoints for a run."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    service = get_run_service()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return service.get_checkpoints(run_id)


@router.post("/{run_id}/export")
def export_run(run_id: str, body: dict | None = None):
    """Export a fused (merged) model for a run."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    checkpoint = (body or {}).get("checkpoint")
    try:
        result = get_run_service().export_run(run_id, checkpoint=checkpoint)
    except RuntimeError as e:
        raise HTTPException(status_code=500, detail=str(e))
    if result is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"status": "exported", "run_id": run_id, "output_dir": str(result)}


class PushToHubRequest(PydanticBaseModel):
    repo_id: str
    adapter_only: bool = False
    private: bool = False


@router.post("/{run_id}/push-to-hub")
def push_to_hub_endpoint(run_id: str, body: PushToHubRequest):
    """Push a run's model or adapters to HuggingFace Hub."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")

    service = get_run_service()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")

    try:
        # Export first if not already exported
        from pathlib import Path
        exports_dir = Path("~/.mlxforge/exports").expanduser() / run_id
        if not exports_dir.exists():
            service.export_run(run_id)

        from mlx_forge.hub.upload import push_to_hub
        url = push_to_hub(
            exports_dir,
            body.repo_id,
            adapter_only=body.adapter_only,
            private=body.private,
        )
        return {"status": "uploaded", "url": url, "repo_id": body.repo_id}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.delete("/{run_id}")
def delete_run(run_id: str):
    """Delete a training run and all its artifacts."""
    try:
        validate_safe_name(run_id, "run_id")
    except ValueError:
        raise HTTPException(status_code=400, detail=f"Invalid run_id: {run_id!r}")
    deleted = get_run_service().delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"status": "deleted", "run_id": run_id}

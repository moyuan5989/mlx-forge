"""Runs API — training run discovery and management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lmforge.studio.services.run_service import RunService

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


@router.get("/{run_id}")
def get_run(run_id: str):
    """Get full details for a training run."""
    run = get_run_service().get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return run


@router.get("/{run_id}/metrics")
def get_metrics(run_id: str):
    """Get train and eval metrics for a run."""
    service = get_run_service()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return service.get_metrics(run_id)


@router.get("/{run_id}/config")
def get_config(run_id: str):
    """Get the training config for a run."""
    config = get_run_service().get_config(run_id)
    if config is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found or config missing")
    return config


@router.get("/{run_id}/checkpoints")
def get_checkpoints(run_id: str):
    """List checkpoints for a run."""
    service = get_run_service()
    run = service.get_run(run_id)
    if run is None:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return service.get_checkpoints(run_id)


@router.delete("/{run_id}")
def delete_run(run_id: str):
    """Delete a training run and all its artifacts."""
    deleted = get_run_service().delete_run(run_id)
    if not deleted:
        raise HTTPException(status_code=404, detail=f"Run '{run_id}' not found")
    return {"status": "deleted", "run_id": run_id}

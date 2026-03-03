"""Job Queue API — FIFO training job management."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

from lmforge.studio.services.queue_service import QueueService

router = APIRouter(prefix="/api/v2/queue", tags=["queue"])

_queue_service = QueueService()


def get_queue_service() -> QueueService:
    return _queue_service


def set_queue_service(service: QueueService):
    global _queue_service
    _queue_service = service


@router.get("")
def list_jobs():
    """List all jobs in the queue."""
    return get_queue_service().list_jobs()


@router.post("/submit")
async def submit_job(config: dict):
    """Submit a new training job to the queue.

    The job will be queued and started when resources are available.
    """
    try:
        job = await get_queue_service().submit(config)
        return job
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.post("/{job_id}/cancel")
async def cancel_job(job_id: str):
    """Cancel a queued or running job."""
    result = await get_queue_service().cancel(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return result


@router.post("/{job_id}/promote")
def promote_job(job_id: str):
    """Move a queued job to the front of the queue."""
    result = get_queue_service().promote(job_id)
    if result is None:
        raise HTTPException(status_code=404, detail=f"Job '{job_id}' not found")
    return result


@router.get("/stats")
def queue_stats():
    """Get queue statistics."""
    return get_queue_service().stats()

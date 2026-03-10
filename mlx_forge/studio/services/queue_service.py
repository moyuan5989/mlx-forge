"""Job Queue Service — FIFO training job queue.

Single-job concurrency by default (unified memory means two
training jobs will OOM). In-memory, no persistence.
"""

from __future__ import annotations

import asyncio
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from typing import Optional


class JobStatus(str, Enum):
    QUEUED = "queued"
    RUNNING = "running"
    COMPLETED = "completed"
    FAILED = "failed"
    CANCELLED = "cancelled"


@dataclass
class Job:
    id: str
    config: dict
    status: JobStatus = JobStatus.QUEUED
    created_at: float = 0.0
    started_at: Optional[float] = None
    completed_at: Optional[float] = None
    run_id: Optional[str] = None
    error: Optional[str] = None
    position: int = 0

    def to_dict(self) -> dict:
        return {
            "id": self.id,
            "config": self.config,
            "status": self.status.value,
            "created_at": self.created_at,
            "started_at": self.started_at,
            "completed_at": self.completed_at,
            "run_id": self.run_id,
            "error": self.error,
            "position": self.position,
        }


class QueueService:
    """FIFO job queue with single-job concurrency.

    Args:
        max_concurrent: Maximum concurrent training jobs.
                       Default 1 (safe for unified memory).
    """

    def __init__(self, max_concurrent: int = 1):
        self.max_concurrent = max_concurrent
        self._queue: deque[Job] = deque()
        self._running: dict[str, Job] = {}
        self._completed: list[Job] = []
        self._lock = asyncio.Lock()

    async def submit(self, config: dict) -> dict:
        """Submit a training job to the queue."""
        job = Job(
            id=str(uuid.uuid4())[:8],
            config=config,
            created_at=time.time(),
            position=len(self._queue),
        )
        self._queue.append(job)

        # Try to start if there's capacity
        await self._try_start_next()

        return job.to_dict()

    async def cancel(self, job_id: str) -> Optional[dict]:
        """Cancel a queued or running job."""
        # Check queue
        for job in self._queue:
            if job.id == job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                self._queue.remove(job)
                self._completed.append(job)
                self._update_positions()
                return job.to_dict()

        # Check running
        if job_id in self._running:
            job = self._running[job_id]
            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            del self._running[job_id]
            self._completed.append(job)
            await self._try_start_next()
            return job.to_dict()

        return None

    def promote(self, job_id: str) -> Optional[dict]:
        """Move a queued job to the front."""
        for i, job in enumerate(self._queue):
            if job.id == job_id:
                if i == 0:
                    return job.to_dict()
                self._queue.remove(job)
                self._queue.appendleft(job)
                self._update_positions()
                return job.to_dict()
        return None

    def list_jobs(self) -> list[dict]:
        """List all jobs (queued, running, recent completed)."""
        jobs = []
        for job in self._queue:
            jobs.append(job.to_dict())
        for job in self._running.values():
            jobs.append(job.to_dict())
        # Show last 20 completed
        for job in self._completed[-20:]:
            jobs.append(job.to_dict())
        return jobs

    def stats(self) -> dict:
        """Get queue statistics."""
        return {
            "queued": len(self._queue),
            "running": len(self._running),
            "completed": len([j for j in self._completed if j.status == JobStatus.COMPLETED]),
            "failed": len([j for j in self._completed if j.status == JobStatus.FAILED]),
            "cancelled": len([j for j in self._completed if j.status == JobStatus.CANCELLED]),
            "max_concurrent": self.max_concurrent,
        }

    async def _try_start_next(self):
        """Start the next queued job if there's capacity."""
        if len(self._running) >= self.max_concurrent:
            return
        if not self._queue:
            return

        job = self._queue.popleft()
        job.status = JobStatus.RUNNING
        job.started_at = time.time()
        self._running[job.id] = job
        self._update_positions()

        # Start training in background
        asyncio.create_task(self._run_job(job))

    async def _run_job(self, job: Job):
        """Execute a training job and wait for the subprocess to finish."""
        try:
            from mlx_forge.studio.services.training_service import TrainingService
            service = TrainingService()
            result = await service.start_training(job.config)
            job.run_id = result.get("track_id")

            # Wait for the training subprocess to actually finish
            proc = result.get("_process")
            if proc is not None:
                returncode, stderr = await service.wait_for_completion(proc)
                if returncode != 0:
                    job.status = JobStatus.FAILED
                    # Capture last few lines of stderr for error context
                    error_lines = stderr.strip().split("\n")[-5:]
                    job.error = "\n".join(error_lines) if error_lines else f"Process exited with code {returncode}"
                else:
                    job.status = JobStatus.COMPLETED
            else:
                job.status = JobStatus.COMPLETED
        except Exception as e:
            job.status = JobStatus.FAILED
            job.error = str(e)
        finally:
            job.completed_at = time.time()
            if job.id in self._running:
                del self._running[job.id]
            self._completed.append(job)
            await self._try_start_next()

    def _update_positions(self):
        """Update position numbers for queued jobs."""
        for i, job in enumerate(self._queue):
            job.position = i

"""Job Queue Service — FIFO training job queue with disk persistence.

Single-job concurrency by default (unified memory means two
training jobs will OOM). Queue state persists to ~/.mlxforge/queue.json.
"""

from __future__ import annotations

import asyncio
import json
import time
import uuid
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
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
    track_id: Optional[str] = None
    pid: Optional[int] = None
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
            "track_id": self.track_id,
            "pid": self.pid,
            "error": self.error,
            "position": self.position,
        }

    @classmethod
    def from_dict(cls, d: dict) -> Job:
        return cls(
            id=d["id"],
            config=d.get("config", {}),
            status=JobStatus(d.get("status", "queued")),
            created_at=d.get("created_at", 0.0),
            started_at=d.get("started_at"),
            completed_at=d.get("completed_at"),
            run_id=d.get("run_id"),
            track_id=d.get("track_id"),
            pid=d.get("pid"),
            error=d.get("error"),
            position=d.get("position", 0),
        )


class QueueService:
    """FIFO job queue with single-job concurrency and disk persistence.

    Args:
        max_concurrent: Maximum concurrent training jobs.
                       Default 1 (safe for unified memory).
        queue_path: Path to queue state file. None disables persistence.
    """

    def __init__(
        self,
        max_concurrent: int = 1,
        queue_path: str | Path | None = "~/.mlxforge/queue.json",
    ):
        self.max_concurrent = max_concurrent
        self._queue: deque[Job] = deque()
        self._running: dict[str, Job] = {}
        self._completed: list[Job] = []
        self._lock = asyncio.Lock()
        self._queue_path: Path | None = (
            Path(queue_path).expanduser() if queue_path else None
        )
        self._load_from_disk()

    async def submit(self, config: dict) -> dict:
        """Submit a training job to the queue."""
        job = Job(
            id=str(uuid.uuid4())[:8],
            config=config,
            created_at=time.time(),
            position=len(self._queue),
        )
        self._queue.append(job)
        self._save_to_disk()

        # Try to start if there's capacity
        await self._try_start_next()

        return job.to_dict()

    async def cancel(self, job_id: str) -> Optional[dict]:
        """Cancel a queued or running job.

        For running jobs, sends SIGINT to the training subprocess
        before updating the job status.
        """
        # Check queue (not yet started — just remove)
        for job in self._queue:
            if job.id == job_id:
                job.status = JobStatus.CANCELLED
                job.completed_at = time.time()
                self._queue.remove(job)
                self._completed.append(job)
                self._update_positions()
                self._save_to_disk()
                return job.to_dict()

        # Check running — must kill the subprocess first
        if job_id in self._running:
            job = self._running[job_id]

            # Kill the actual training subprocess via TrainingService
            if job.track_id:
                try:
                    from mlx_forge.studio.api.training import get_training_service
                    await get_training_service().stop_training(job.track_id)
                except Exception:
                    pass  # Subprocess may have already exited

            # If we have a PID but no track_id, kill directly
            if job.pid and not job.track_id:
                try:
                    import os
                    import signal
                    os.kill(job.pid, signal.SIGINT)
                except (OSError, ProcessLookupError):
                    pass

            job.status = JobStatus.CANCELLED
            job.completed_at = time.time()
            del self._running[job_id]
            self._completed.append(job)
            self._save_to_disk()
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
                self._save_to_disk()
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
        self._save_to_disk()

        # Start training in background
        asyncio.create_task(self._run_job(job))

    async def _run_job(self, job: Job):
        """Execute a training job and wait for the subprocess to finish."""
        try:
            # Use the shared TrainingService so list_active() works consistently
            from mlx_forge.studio.api.training import get_training_service
            service = get_training_service()
            result = await service.start_training(job.config)
            job.track_id = result.get("track_id")
            job.pid = result.get("pid")
            # run_id is not yet known — will be captured from subprocess stdout
            job.run_id = None
            self._save_to_disk()

            # Wait for the training subprocess to actually finish
            proc = result.get("_process")
            if proc is not None:
                def _on_run_id(rid):
                    """Called as soon as subprocess prints the run directory."""
                    job.run_id = rid
                    self._save_to_disk()

                returncode, stderr, run_id = await service.wait_for_completion(
                    proc, on_run_id=_on_run_id,
                )
                # Fallback: use the real experiment run_id if captured from stdout
                if run_id and job.run_id != run_id:
                    job.run_id = run_id
                    self._save_to_disk()
                if returncode != 0:
                    job.status = JobStatus.FAILED
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
            self._save_to_disk()
            await self._try_start_next()

    def _update_positions(self):
        """Update position numbers for queued jobs."""
        for i, job in enumerate(self._queue):
            job.position = i

    def _save_to_disk(self):
        """Persist queue state to disk."""
        if self._queue_path is None:
            return
        try:
            self._queue_path.parent.mkdir(parents=True, exist_ok=True)
            state = {
                "queue": [j.to_dict() for j in self._queue],
                "running": [j.to_dict() for j in self._running.values()],
                "completed": [j.to_dict() for j in self._completed[-20:]],
            }
            tmp_path = self._queue_path.with_suffix(".tmp")
            with open(tmp_path, "w") as f:
                json.dump(state, f)
            tmp_path.replace(self._queue_path)
        except Exception:
            pass  # Best-effort persistence

    def _load_from_disk(self):
        """Load queue state from disk on init. Mark stale running jobs as failed."""
        if self._queue_path is None or not self._queue_path.exists():
            return
        try:
            with open(self._queue_path) as f:
                state = json.load(f)
        except (json.JSONDecodeError, OSError):
            return

        # Restore queued jobs
        for d in state.get("queue", []):
            self._queue.append(Job.from_dict(d))

        # Previously-running jobs are stale (process no longer exists)
        for d in state.get("running", []):
            job = Job.from_dict(d)
            job.status = JobStatus.FAILED
            job.error = "Server restarted while job was running"
            job.completed_at = time.time()
            self._completed.append(job)

        # Restore completed history
        for d in state.get("completed", []):
            self._completed.append(Job.from_dict(d))

        self._update_positions()

"""M16 — Queue persistence tests."""

from __future__ import annotations

import asyncio
import json

import pytest

from mlx_forge.studio.services.queue_service import Job, JobStatus, QueueService


@pytest.fixture
def queue_file(tmp_path):
    return tmp_path / "queue.json"


def run_async(coro):
    """Run an async coroutine in a new event loop."""
    loop = asyncio.new_event_loop()
    try:
        return loop.run_until_complete(coro)
    finally:
        loop.close()


class TestQueuePersistence:
    def test_submit_creates_file(self, queue_file):
        svc = QueueService(queue_path=queue_file)
        run_async(svc.submit({"model": {"path": "test"}}))
        assert queue_file.exists()
        data = json.loads(queue_file.read_text())
        # Job should be in queue or running (if auto-started)
        total = len(data["queue"]) + len(data["running"])
        assert total >= 1

    def test_state_survives_reinstantiation(self, queue_file):
        # Submit a job with persistence
        svc1 = QueueService(queue_path=queue_file)
        # Write a manual queued job to avoid starting subprocess
        job = Job(
            id="test-01",
            config={"model": {"path": "test"}},
            status=JobStatus.QUEUED,
            created_at=1000.0,
        )
        svc1._queue.append(job)
        svc1._save_to_disk()

        # Re-instantiate and check
        svc2 = QueueService(queue_path=queue_file)
        jobs = svc2.list_jobs()
        queued_ids = [j["id"] for j in jobs if j["status"] == "queued"]
        assert "test-01" in queued_ids

    def test_running_jobs_marked_failed_on_restart(self, queue_file):
        # Simulate a running job in the persisted state
        state = {
            "queue": [],
            "running": [
                {
                    "id": "run-01",
                    "config": {},
                    "status": "running",
                    "created_at": 1000.0,
                    "started_at": 1001.0,
                    "completed_at": None,
                    "run_id": None,
                    "track_id": "model_123",
                    "error": None,
                    "position": 0,
                }
            ],
            "completed": [],
        }
        queue_file.write_text(json.dumps(state))

        svc = QueueService(queue_path=queue_file)
        jobs = svc.list_jobs()
        failed = [j for j in jobs if j["status"] == "failed"]
        assert len(failed) == 1
        assert failed[0]["id"] == "run-01"
        assert "restarted" in failed[0]["error"].lower()

    def test_completed_jobs_restored(self, queue_file):
        state = {
            "queue": [],
            "running": [],
            "completed": [
                {
                    "id": "done-01",
                    "config": {},
                    "status": "completed",
                    "created_at": 1000.0,
                    "started_at": 1001.0,
                    "completed_at": 1002.0,
                    "run_id": "real-run-id",
                    "track_id": None,
                    "error": None,
                    "position": 0,
                }
            ],
        }
        queue_file.write_text(json.dumps(state))

        svc = QueueService(queue_path=queue_file)
        jobs = svc.list_jobs()
        completed = [j for j in jobs if j["status"] == "completed"]
        assert len(completed) == 1
        assert completed[0]["run_id"] == "real-run-id"

    def test_no_persistence_when_disabled(self, tmp_path):
        svc = QueueService(queue_path=None)
        run_async(svc.submit({"model": {"path": "test"}}))
        # No file should be written
        assert not list(tmp_path.glob("*.json"))

    def test_cancel_persists(self, queue_file):
        svc = QueueService(queue_path=queue_file)
        job = Job(
            id="cancel-me",
            config={},
            status=JobStatus.QUEUED,
            created_at=1000.0,
        )
        svc._queue.append(job)
        svc._save_to_disk()

        result = run_async(svc.cancel("cancel-me"))
        assert result is not None
        assert result["status"] == "cancelled"

        # Verify on disk
        data = json.loads(queue_file.read_text())
        assert len(data["queue"]) == 0

    def test_promote_persists(self, queue_file):
        svc = QueueService(queue_path=queue_file)
        for i in range(3):
            job = Job(id=f"job-{i}", config={}, created_at=1000.0 + i, position=i)
            svc._queue.append(job)
        svc._save_to_disk()

        result = svc.promote("job-2")
        assert result is not None
        assert result["id"] == "job-2"

        data = json.loads(queue_file.read_text())
        queue_ids = [j["id"] for j in data["queue"]]
        assert queue_ids[0] == "job-2"

    def test_job_dataclass_roundtrip(self):
        job = Job(
            id="rt-01",
            config={"key": "val"},
            status=JobStatus.RUNNING,
            created_at=1000.0,
            started_at=1001.0,
            track_id="model_456",
        )
        d = job.to_dict()
        restored = Job.from_dict(d)
        assert restored.id == "rt-01"
        assert restored.status == JobStatus.RUNNING
        assert restored.track_id == "model_456"

    def test_corrupt_file_handled(self, queue_file):
        queue_file.write_text("not valid json{{{")
        svc = QueueService(queue_path=queue_file)
        # Should start with empty state, not crash
        assert svc.list_jobs() == []

    def test_stats_correct(self, queue_file):
        state = {
            "queue": [
                {"id": "q1", "config": {}, "status": "queued", "created_at": 0},
            ],
            "running": [],
            "completed": [
                {"id": "c1", "config": {}, "status": "completed", "created_at": 0},
                {"id": "f1", "config": {}, "status": "failed", "created_at": 0},
            ],
        }
        queue_file.write_text(json.dumps(state))
        svc = QueueService(queue_path=queue_file)
        s = svc.stats()
        assert s["queued"] == 1
        assert s["completed"] == 1
        assert s["failed"] == 1

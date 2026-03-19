"""Training subprocess management.

Spawns `mlx_forge train` as a child process and tracks active runs.
"""

from __future__ import annotations

import asyncio
import signal
import sys
import tempfile
from pathlib import Path

import yaml


class TrainingService:
    """Manages training subprocesses."""

    def __init__(self):
        self._processes: dict[str, asyncio.subprocess.Process] = {}
        self._configs: dict[str, dict] = {}

    async def start_training(self, config_dict: dict) -> dict:
        """Start a training run as a subprocess and return immediately.

        Args:
            config_dict: Full training config as a dict.

        Returns:
            Dict with run info including the subprocess for callers to await.
        """
        # Write config to a temp file
        config_dir = Path(tempfile.mkdtemp(prefix="mlx_forge_studio_"))
        config_path = config_dir / "config.yaml"
        with open(config_path, "w") as f:
            yaml.dump(config_dict, f, default_flow_style=False)

        # Spawn subprocess
        python = sys.executable
        proc = await asyncio.create_subprocess_exec(
            python, "-m", "mlx_forge.cli.main", "train", "--config", str(config_path),
            stdout=asyncio.subprocess.PIPE,
            stderr=asyncio.subprocess.PIPE,
        )

        # Generate a tracking ID from the config
        model_name = config_dict.get("model", {}).get("path", "unknown")
        track_id = f"{model_name}_{proc.pid}"

        self._processes[track_id] = proc
        self._configs[track_id] = config_dict

        return {
            "status": "started",
            "track_id": track_id,
            "pid": proc.pid,
            "config_path": str(config_path),
            "_process": proc,
        }

    async def wait_for_completion(
        self, proc: asyncio.subprocess.Process,
        on_run_id: callable | None = None,
    ) -> tuple[int, str, str | None]:
        """Wait for a training subprocess to finish.

        Args:
            proc: The subprocess to wait on.
            on_run_id: Optional callback invoked as soon as the run_id is
                       captured from stdout (before the process finishes).

        Returns:
            Tuple of (return_code, stderr_tail, run_id).
            run_id is extracted from the "Run directory:" stdout line.
        """
        stderr_tail: list[str] = []
        run_id: list[str | None] = [None]  # mutable container for closure

        async def _drain_stdout(stream):
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                text = line.decode(errors="replace").rstrip()
                # Parse "Run directory: /path/to/runs/{run_id}"
                if text.startswith("Run directory:") and run_id[0] is None:
                    path = text.split(":", 1)[1].strip()
                    run_id[0] = Path(path).name
                    # Notify caller immediately so UI can show the run_id
                    if on_run_id is not None:
                        try:
                            on_run_id(run_id[0])
                        except Exception:
                            pass

        async def _drain_stderr(stream):
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                stderr_tail.append(line.decode(errors="replace").rstrip())
                if len(stderr_tail) > 20:
                    stderr_tail.pop(0)

        await asyncio.gather(_drain_stdout(proc.stdout), _drain_stderr(proc.stderr))
        await proc.wait()
        return proc.returncode, "\n".join(stderr_tail), run_id[0]

    async def stop_training(self, track_id: str) -> dict:
        """Stop a running training process via SIGINT.

        Args:
            track_id: The tracking ID returned from start_training.

        Returns:
            Dict with status info.
        """
        proc = self._processes.get(track_id)
        if proc is None:
            return {"status": "not_found", "track_id": track_id}

        if proc.returncode is not None:
            # Already finished
            self._cleanup(track_id)
            return {"status": "already_finished", "track_id": track_id, "returncode": proc.returncode}

        # Send SIGINT for cooperative shutdown
        proc.send_signal(signal.SIGINT)
        try:
            await asyncio.wait_for(proc.wait(), timeout=30)
        except asyncio.TimeoutError:
            proc.kill()
            await proc.wait()

        self._cleanup(track_id)
        return {"status": "stopped", "track_id": track_id, "returncode": proc.returncode}

    def list_active(self) -> list[dict]:
        """Return list of active training processes.

        Enriches results with run_id from QueueService when available.
        """
        active = []
        finished = []
        for track_id, proc in self._processes.items():
            if proc.returncode is not None:
                finished.append(track_id)
                continue
            entry = {
                "track_id": track_id,
                "pid": proc.pid,
                "config": self._configs.get(track_id),
                "run_id": None,
            }
            active.append(entry)

        # Clean up finished processes
        for track_id in finished:
            self._cleanup(track_id)

        # Enrich with run_id from queue service (if jobs were submitted via queue)
        try:
            from mlx_forge.studio.api.queue import get_queue_service
            queue = get_queue_service()
            for job in queue._running.values():
                for entry in active:
                    if entry["track_id"] == job.track_id:
                        entry["run_id"] = job.run_id
                        break
        except Exception:
            pass

        return active

    def _cleanup(self, track_id: str):
        """Remove a process from tracking."""
        self._processes.pop(track_id, None)
        self._configs.pop(track_id, None)

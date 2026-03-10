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

    async def wait_for_completion(self, proc: asyncio.subprocess.Process) -> tuple[int, str]:
        """Wait for a training subprocess to finish.

        Returns:
            Tuple of (return_code, stderr_tail).
        """
        # Stream stdout/stderr to avoid memory buildup on long runs
        stderr_tail: list[str] = []

        async def _drain(stream, collector: list[str] | None):
            if stream is None:
                return
            while True:
                line = await stream.readline()
                if not line:
                    break
                if collector is not None:
                    collector.append(line.decode(errors="replace").rstrip())
                    if len(collector) > 20:
                        collector.pop(0)

        await asyncio.gather(
            _drain(proc.stdout, None),
            _drain(proc.stderr, stderr_tail),
        )
        await proc.wait()
        return proc.returncode, "\n".join(stderr_tail)

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
        """Return list of active training processes."""
        active = []
        finished = []
        for track_id, proc in self._processes.items():
            if proc.returncode is not None:
                finished.append(track_id)
                continue
            active.append({
                "track_id": track_id,
                "pid": proc.pid,
                "config": self._configs.get(track_id),
            })

        # Clean up finished processes
        for track_id in finished:
            self._cleanup(track_id)

        return active

    def _cleanup(self, track_id: str):
        """Remove a process from tracking."""
        self._processes.pop(track_id, None)
        self._configs.pop(track_id, None)

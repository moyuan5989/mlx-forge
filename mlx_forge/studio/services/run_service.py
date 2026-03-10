"""Run discovery and management from ~/.mlxforge/runs/.

Scans the filesystem directly — no database.
"""

from __future__ import annotations

import json
import math
import shutil
import time
from pathlib import Path
from typing import Optional

import yaml


def _sanitize_for_json(obj):
    """Replace inf/nan floats with None for JSON serialization."""
    if isinstance(obj, float) and (math.isinf(obj) or math.isnan(obj)):
        return None
    if isinstance(obj, dict):
        return {k: _sanitize_for_json(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_sanitize_for_json(v) for v in obj]
    return obj


class RunService:
    """Discovers and manages training runs from the filesystem."""

    def __init__(self, runs_dir: str = "~/.mlxforge/runs"):
        self.runs_dir = Path(runs_dir).expanduser()

    def list_runs(self) -> list[dict]:
        """List all runs with summary info.

        Returns list of dicts with: id, status, model, created_at,
        current_step, num_iters, latest_train_loss, latest_val_loss.
        """
        if not self.runs_dir.exists():
            return []

        runs = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            try:
                summary = self._build_summary(run_dir)
                runs.append(summary)
            except Exception:
                # Skip malformed run directories
                continue
        return runs

    def get_run(self, run_id: str) -> Optional[dict]:
        """Get full details for a run.

        Returns config, manifest, state from last checkpoint, metrics summary.
        """
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return None

        result = self._build_summary(run_dir)

        # Add config
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            with open(config_path) as f:
                result["config"] = yaml.safe_load(f)

        # Add manifest
        manifest_path = run_dir / "manifest.json"
        if manifest_path.exists():
            with open(manifest_path) as f:
                result["manifest"] = json.load(f)

        # Add environment
        env_path = run_dir / "environment.json"
        if env_path.exists():
            with open(env_path) as f:
                result["environment"] = json.load(f)

        return result

    def get_metrics(self, run_id: str) -> dict:
        """Parse metrics.jsonl into train and eval arrays.

        Returns {"train": [...], "eval": [...]}.
        """
        metrics_path = self.runs_dir / run_id / "logs" / "metrics.jsonl"
        train_metrics = []
        eval_metrics = []

        if metrics_path.exists():
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        event = entry.get("event", "train")
                        if event == "eval":
                            eval_metrics.append(entry)
                        else:
                            train_metrics.append(entry)
                    except json.JSONDecodeError:
                        continue

        return _sanitize_for_json({"train": train_metrics, "eval": eval_metrics})

    def get_config(self, run_id: str) -> Optional[dict]:
        """Read config.yaml for a run."""
        config_path = self.runs_dir / run_id / "config.yaml"
        if not config_path.exists():
            return None
        with open(config_path) as f:
            return yaml.safe_load(f)

    def get_checkpoints(self, run_id: str) -> list[dict]:
        """List checkpoint directories with their state info."""
        ckpt_dir = self.runs_dir / run_id / "checkpoints"
        if not ckpt_dir.exists():
            return []

        checkpoints = []
        # Find best symlink target
        best_link = ckpt_dir / "best"
        best_name = None
        if best_link.is_symlink():
            best_name = best_link.readlink().name if best_link.readlink().is_absolute() else str(best_link.readlink())

        for entry in sorted(ckpt_dir.iterdir()):
            if not entry.is_dir() or entry.is_symlink() or entry.name.endswith(".tmp"):
                continue
            state_file = entry / "state.json"
            ckpt_info = {"name": entry.name, "path": str(entry), "is_best": entry.name == best_name}
            if state_file.exists():
                try:
                    with open(state_file) as f:
                        ckpt_info["state"] = json.load(f)
                except json.JSONDecodeError:
                    pass
            checkpoints.append(ckpt_info)

        return _sanitize_for_json(checkpoints)

    def list_adapters(self) -> list[dict]:
        """List available adapters from completed runs (best checkpoint per run)."""
        if not self.runs_dir.exists():
            return []

        adapters = []
        for run_dir in sorted(self.runs_dir.iterdir(), reverse=True):
            if not run_dir.is_dir() or run_dir.name.startswith("."):
                continue
            try:
                summary = self._build_summary(run_dir)
            except Exception:
                continue

            # Find best checkpoint (or latest if no best symlink)
            ckpt_dir = run_dir / "checkpoints"
            if not ckpt_dir.exists():
                continue

            best_link = ckpt_dir / "best"
            if best_link.exists():
                ckpt_path = best_link.resolve()
            else:
                # Fall back to last checkpoint directory
                ckpts = sorted(
                    [d for d in ckpt_dir.iterdir() if d.is_dir() and not d.is_symlink()],
                )
                if not ckpts:
                    continue
                ckpt_path = ckpts[-1]

            if not (ckpt_path / "adapters.safetensors").exists():
                continue

            adapters.append({
                "run_id": summary["id"],
                "model": summary.get("model", "unknown"),
                "status": summary.get("status", "unknown"),
                "checkpoint": ckpt_path.name,
                "path": str(ckpt_path),
                "label": f"{summary['id']} ({summary.get('model', '?').split('/')[-1]})",
            })

        return adapters

    def delete_run(self, run_id: str) -> bool:
        """Delete a run directory. Returns True if deleted."""
        run_dir = self.runs_dir / run_id
        if not run_dir.exists():
            return False
        shutil.rmtree(run_dir)
        return True

    def _build_summary(self, run_dir: Path) -> dict:
        """Build a summary dict for a run directory."""
        run_id = run_dir.name
        summary = {
            "id": run_id,
            "path": str(run_dir),
            "status": "unknown",
            "model": None,
            "current_step": 0,
            "num_iters": 0,
            "latest_train_loss": None,
            "latest_val_loss": None,
        }

        # Read config for model name and num_iters
        config_path = run_dir / "config.yaml"
        if config_path.exists():
            try:
                with open(config_path) as f:
                    config = yaml.safe_load(f)
                summary["model"] = config.get("model", {}).get("path")
                summary["num_iters"] = config.get("training", {}).get("num_iters", 0)
            except Exception:
                pass

        # Read latest metrics
        metrics_path = run_dir / "logs" / "metrics.jsonl"
        if metrics_path.exists():
            last_train, last_eval = self._read_last_metrics(metrics_path)
            if last_train:
                summary["current_step"] = last_train.get("step", 0)
                summary["latest_train_loss"] = last_train.get("train_loss")
            if last_eval:
                summary["latest_val_loss"] = last_eval.get("val_loss")

        # Infer status
        summary["status"] = self._infer_status(run_dir, summary)

        return _sanitize_for_json(summary)

    def _read_last_metrics(self, metrics_path: Path) -> tuple:
        """Read last train and eval metrics from a JSONL file."""
        last_train = None
        last_eval = None
        try:
            with open(metrics_path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        entry = json.loads(line)
                        event = entry.get("event", "train")
                        if event == "eval":
                            last_eval = entry
                        else:
                            last_train = entry
                    except json.JSONDecodeError:
                        continue
        except Exception:
            pass
        return last_train, last_eval

    def _infer_status(self, run_dir: Path, summary: dict) -> str:
        """Infer run status from filesystem state.

        Returns: "completed", "running", "stopped", "unknown".
        """
        current_step = summary.get("current_step", 0)
        num_iters = summary.get("num_iters", 0)

        if num_iters > 0 and current_step >= num_iters:
            return "completed"

        # Check if metrics file was recently modified (within 60s)
        metrics_path = run_dir / "logs" / "metrics.jsonl"
        if metrics_path.exists():
            mtime = metrics_path.stat().st_mtime
            if time.time() - mtime < 60:
                return "running"

        if current_step > 0:
            return "stopped"

        return "unknown"

"""Checkpoint management for LMForge v0.

Each checkpoint contains exactly three files per V0_DESIGN_FREEZE.md §2.3:
- adapters.safetensors
- optimizer.safetensors
- state.json
"""

from __future__ import annotations

import json
import os
import shutil
from pathlib import Path

import mlx.core as mx
from mlx.utils import tree_flatten, tree_unflatten

from lmforge.trainer.state import TrainState


class CheckpointManager:
    """Handles atomic checkpoint save/load and retention policy."""

    def __init__(self, config):
        self.config = config
        self.run_dir = Path(config.runtime.run_dir).expanduser() / self._generate_run_id(config)
        self.checkpoint_dir = self.run_dir / "checkpoints"
        self.checkpoint_dir.mkdir(parents=True, exist_ok=True)

        self._best_val_loss = float("inf")
        self.last_checkpoint_dir = None

    def save(self, state: TrainState, model, optimizer) -> Path:
        """Save a checkpoint atomically (tmp dir -> rename).

        Returns the checkpoint directory path.
        """
        # Create checkpoint directory name
        ckpt_name = f"step-{state.step:07d}"
        ckpt_dir = self.checkpoint_dir / ckpt_name
        tmp_dir = self.checkpoint_dir / f"{ckpt_name}.tmp"

        # Create temp directory
        tmp_dir.mkdir(parents=True, exist_ok=True)

        try:
            # 1. Save adapter weights (trainable parameters only)
            adapter_weights = dict(tree_flatten(model.trainable_parameters()))
            mx.save_safetensors(str(tmp_dir / "adapters.safetensors"), adapter_weights)

            # 2. Save optimizer state
            opt_state = dict(tree_flatten(optimizer.state))
            mx.save_safetensors(str(tmp_dir / "optimizer.safetensors"), opt_state)

            # 3. Save training state metadata
            state_dict = {
                "schema_version": 1,
                "step": state.step,
                "epoch": state.epoch,
                "trained_tokens": state.trained_tokens,
                "best_val_loss": state.best_val_loss,
                "learning_rate": float(optimizer.learning_rate),
                "rng_seed": state.rng_seed,
            }
            (tmp_dir / "state.json").write_text(json.dumps(state_dict, indent=2))

            # Atomic rename
            tmp_dir.rename(ckpt_dir)
            self.last_checkpoint_dir = ckpt_dir

        except Exception as e:
            # Clean up temp dir on failure
            if tmp_dir.exists():
                shutil.rmtree(tmp_dir)
            raise RuntimeError(f"Failed to save checkpoint: {e}") from e

        # Update best symlink if this is the best checkpoint
        if state.best_val_loss < self._best_val_loss:
            self._best_val_loss = state.best_val_loss
            best_link = self.checkpoint_dir / "best"
            if best_link.is_symlink() or best_link.exists():
                best_link.unlink()
            best_link.symlink_to(ckpt_name)

        # Enforce retention policy
        self._enforce_retention()

        return ckpt_dir

    def load(self, ckpt_dir: Path, model, optimizer) -> TrainState:
        """Load a checkpoint and restore model/optimizer state.

        Returns the restored TrainState.
        """
        ckpt_dir = Path(ckpt_dir)

        # Validate checkpoint integrity
        required_files = ["adapters.safetensors", "optimizer.safetensors", "state.json"]
        for filename in required_files:
            if not (ckpt_dir / filename).exists():
                raise FileNotFoundError(
                    f"Checkpoint missing {filename} in {ckpt_dir}. "
                    f"Expected files: {required_files}"
                )

        # Load state metadata
        state_dict = json.loads((ckpt_dir / "state.json").read_text())

        # Check schema version
        schema_version = state_dict.get("schema_version", 1)
        if schema_version > 1:
            raise ValueError(
                f"Unsupported checkpoint schema version: {schema_version}. "
                f"This version of lmforge only supports schema version 1."
            )

        # Load adapter weights
        adapter_weights = mx.load(str(ckpt_dir / "adapters.safetensors"))
        model.load_weights(list(adapter_weights.items()), strict=False)

        # Load optimizer state
        opt_weights = mx.load(str(ckpt_dir / "optimizer.safetensors"))
        optimizer.state = tree_unflatten(list(opt_weights.items()))

        # Restore RNG seed (Tier-1: re-seed, not exact state restoration)
        rng_seed = state_dict.get("rng_seed", 42)
        step = state_dict.get("step", 0)
        mx.random.seed(rng_seed + step)

        # Create TrainState from saved metadata
        train_state = TrainState(
            step=state_dict.get("step", 0),
            epoch=state_dict.get("epoch", 0),
            trained_tokens=state_dict.get("trained_tokens", 0),
            best_val_loss=state_dict.get("best_val_loss", float("inf")),
            rng_seed=rng_seed,
        )

        return train_state

    def _enforce_retention(self):
        """Enforce retention policy: keep last N checkpoints + best."""
        keep_last_n = self.config.training.keep_last_n_checkpoints

        # Get all checkpoint directories (excluding .tmp and symlinks)
        checkpoints = [
            d for d in self.checkpoint_dir.iterdir()
            if d.is_dir() and not d.name.endswith(".tmp") and not d.is_symlink()
        ]

        # Sort by step number (extracted from name)
        checkpoints.sort(key=lambda d: int(d.name.split("-")[1]))

        # Identify best checkpoint (via symlink)
        best_link = self.checkpoint_dir / "best"
        best_ckpt = None
        if best_link.is_symlink():
            best_ckpt = self.checkpoint_dir / os.readlink(best_link)

        # Keep last N + best
        to_keep = set(checkpoints[-keep_last_n:])
        if best_ckpt is not None:
            to_keep.add(best_ckpt)

        # Delete old checkpoints
        for ckpt in checkpoints:
            if ckpt not in to_keep:
                shutil.rmtree(ckpt)

    def _generate_run_id(self, config) -> str:
        """Generate a run ID: YYYYMMDD-HHMMSS-sft-{model_short}-{hash4}"""
        import hashlib
        from datetime import datetime

        # Get timestamp
        timestamp = datetime.now().strftime("%Y%m%d-%H%M%S")

        # Get model short name (last component, truncated to 20 chars)
        model_path = config.model.path
        model_short = model_path.split("/")[-1][:20]

        # Compute config hash
        config_str = json.dumps(config.model_dump(), sort_keys=True)
        config_hash = hashlib.sha256(config_str.encode()).hexdigest()[:4]

        return f"{timestamp}-sft-{model_short}-{config_hash}"

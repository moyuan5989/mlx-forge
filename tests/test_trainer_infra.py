"""Tests for trainer infrastructure (M4)."""

from __future__ import annotations

import json

import mlx.nn as nn

from cortexlab.config import (
    AdapterConfig,
    DataConfig,
    LRScheduleConfig,
    ModelConfig,
    RuntimeConfig,
    TrainingConfig,
    TrainingParams,
)
from cortexlab.logging.metrics import write_metrics_line
from cortexlab.trainer.checkpoint import CheckpointManager
from cortexlab.trainer.optimizer import build_optimizer, build_scheduler
from cortexlab.trainer.state import TrainState


class MockModel(nn.Module):
    """Simple model for testing."""

    def __init__(self, dim: int = 32):
        super().__init__()
        self.linear = nn.Linear(dim, dim)

    def __call__(self, x):
        return self.linear(x)


class TestOptimizerFactory:
    def test_build_adam_optimizer(self):
        """Test that build_optimizer creates Adam optimizer with correct LR."""
        model = MockModel()
        training_params = TrainingParams(
            optimizer="adam",
            learning_rate=1e-4,
            num_iters=100,
            batch_size=4,
        )

        optimizer = build_optimizer(training_params, model)

        # Check optimizer type
        assert optimizer.__class__.__name__ == "Adam"

        # Check learning rate (constant schedule)
        lr_value = float(optimizer.learning_rate)
        assert abs(lr_value - 1e-4) < 1e-7

    def test_lr_schedule_changes_over_steps(self):
        """Test that LR schedule is a function of step number."""
        training_params = TrainingParams(
            optimizer="adam",
            learning_rate=1e-3,
            num_iters=1000,
            batch_size=4,
            lr_schedule=LRScheduleConfig(
                name="linear_schedule",
                arguments=[1e-3, 1e-5, 1000],
                warmup=0,
                warmup_init=0.0,
            ),
        )

        model = MockModel()
        build_optimizer(training_params, model)

        # Build the schedule directly to test it
        lr_schedule = build_scheduler(training_params)

        # Check that LR changes over steps
        lr_at_0 = float(lr_schedule(0))
        lr_at_500 = float(lr_schedule(500))
        lr_at_999 = float(lr_schedule(999))

        # LR should decrease from init to end
        assert lr_at_0 > lr_at_500 > lr_at_999
        assert abs(lr_at_0 - 1e-3) < 1e-6
        assert abs(lr_at_999 - 1e-5) < 1e-6


class TestCheckpointManager:
    def test_save_produces_three_files(self, tmp_path):
        """Test that checkpoint contains exactly 3 files."""
        # Create minimal config
        config = TrainingConfig(
            schema_version=1,
            model=ModelConfig(path="test-model"),
            adapter=AdapterConfig(preset="attention-qv", rank=8),
            data=DataConfig(train="train.jsonl", valid="val.jsonl"),
            training=TrainingParams(
                optimizer="adam",
                learning_rate=1e-4,
                num_iters=100,
                batch_size=4,
            ),
            runtime=RuntimeConfig(run_dir=str(tmp_path)),
        )

        manager = CheckpointManager(config)
        model = MockModel()
        optimizer = build_optimizer(config.training, model)

        state = TrainState(
            step=10,
            epoch=0,
            trained_tokens=1000,
            best_val_loss=2.5,
            rng_seed=42,
        )

        # Save checkpoint
        ckpt_dir = manager.save(state, model, optimizer)

        # Verify exactly 3 files
        files = sorted([f.name for f in ckpt_dir.iterdir()])
        assert files == ["adapters.safetensors", "optimizer.safetensors", "state.json"]

    def test_load_restores_state(self, tmp_path):
        """Test that load restores model, optimizer, and training state."""
        config = TrainingConfig(
            schema_version=1,
            model=ModelConfig(path="test-model"),
            adapter=AdapterConfig(preset="attention-qv", rank=8),
            data=DataConfig(train="train.jsonl", valid="val.jsonl"),
            training=TrainingParams(
                optimizer="adam",
                learning_rate=1e-4,
                num_iters=100,
                batch_size=4,
            ),
            runtime=RuntimeConfig(run_dir=str(tmp_path)),
        )

        manager = CheckpointManager(config)
        model = MockModel()
        optimizer = build_optimizer(config.training, model)

        original_state = TrainState(
            step=50,
            epoch=1,
            trained_tokens=5000,
            best_val_loss=1.8,
            rng_seed=42,
        )

        # Save checkpoint
        ckpt_dir = manager.save(original_state, model, optimizer)

        # Create new model and optimizer
        new_model = MockModel()
        new_optimizer = build_optimizer(config.training, new_model)

        # Load checkpoint
        loaded_state = manager.load(ckpt_dir, new_model, new_optimizer)

        # Verify state matches
        assert loaded_state.step == original_state.step
        assert loaded_state.epoch == original_state.epoch
        assert loaded_state.trained_tokens == original_state.trained_tokens
        assert loaded_state.best_val_loss == original_state.best_val_loss
        assert loaded_state.rng_seed == original_state.rng_seed

    def test_atomic_write_uses_tmp_dir(self, tmp_path):
        """Test that save uses temp dir then atomic rename."""
        config = TrainingConfig(
            schema_version=1,
            model=ModelConfig(path="test-model"),
            adapter=AdapterConfig(preset="attention-qv", rank=8),
            data=DataConfig(train="train.jsonl", valid="val.jsonl"),
            training=TrainingParams(
                optimizer="adam",
                learning_rate=1e-4,
                num_iters=100,
                batch_size=4,
            ),
            runtime=RuntimeConfig(run_dir=str(tmp_path)),
        )

        manager = CheckpointManager(config)
        model = MockModel()
        optimizer = build_optimizer(config.training, model)

        state = TrainState(
            step=10,
            epoch=0,
            trained_tokens=1000,
            best_val_loss=2.5,
            rng_seed=42,
        )

        # Save checkpoint
        ckpt_dir = manager.save(state, model, optimizer)

        # Verify no .tmp directories remain
        checkpoint_dir = manager.checkpoint_dir
        tmp_dirs = [d for d in checkpoint_dir.iterdir() if d.name.endswith(".tmp")]
        assert len(tmp_dirs) == 0

        # Verify checkpoint directory exists and is not a tmp
        assert ckpt_dir.exists()
        assert not ckpt_dir.name.endswith(".tmp")

    def test_retention_keeps_last_n_plus_best(self, tmp_path):
        """Test that retention policy keeps last N checkpoints + best."""
        config = TrainingConfig(
            schema_version=1,
            model=ModelConfig(path="test-model"),
            adapter=AdapterConfig(preset="attention-qv", rank=8),
            data=DataConfig(train="train.jsonl", valid="val.jsonl"),
            training=TrainingParams(
                optimizer="adam",
                learning_rate=1e-4,
                num_iters=100,
                batch_size=4,
                keep_last_n_checkpoints=2,
            ),
            runtime=RuntimeConfig(run_dir=str(tmp_path)),
        )

        manager = CheckpointManager(config)
        model = MockModel()
        optimizer = build_optimizer(config.training, model)

        # Save 5 checkpoints with varying val loss
        val_losses = [3.0, 2.5, 1.8, 2.0, 2.2]  # index 2 (step 30) has best loss of 1.8
        for i, val_loss in enumerate(val_losses):
            state = TrainState(
                step=(i + 1) * 10,
                epoch=0,
                trained_tokens=(i + 1) * 1000,
                best_val_loss=min(val_losses[: i + 1]),
                rng_seed=42,
            )
            manager.save(state, model, optimizer)

        # Should keep last 2 (step-0000040, step-0000050) + best (step-0000030)
        checkpoint_dirs = [
            d.name
            for d in manager.checkpoint_dir.iterdir()
            if d.is_dir() and not d.is_symlink()
        ]

        assert len(checkpoint_dirs) == 3
        assert "step-0000030" in checkpoint_dirs  # best (val_loss=1.8)
        assert "step-0000040" in checkpoint_dirs  # last 2
        assert "step-0000050" in checkpoint_dirs  # last 2
        assert "step-0000010" not in checkpoint_dirs  # deleted
        assert "step-0000020" not in checkpoint_dirs  # deleted


class TestMetricsLogger:
    def test_jsonl_output_format(self, tmp_path):
        """Test that metrics are written as JSONL with correct format."""
        log_path = tmp_path / "metrics.jsonl"

        # Write training metric
        train_metrics = {
            "step": 100,
            "train_loss": 2.345,
            "learning_rate": 1e-5,
            "tokens_per_second": 15234.5,
        }
        write_metrics_line(log_path, train_metrics)

        # Write eval metric
        eval_metrics = {
            "step": 200,
            "event": "eval",
            "val_loss": 1.987,
        }
        write_metrics_line(log_path, eval_metrics)

        # Read and verify
        lines = log_path.read_text().strip().split("\n")
        assert len(lines) == 2

        # Parse first line
        line1 = json.loads(lines[0])
        assert line1["step"] == 100
        assert line1["train_loss"] == 2.345
        assert "timestamp" in line1

        # Parse second line
        line2 = json.loads(lines[1])
        assert line2["step"] == 200
        assert line2["event"] == "eval"
        assert line2["val_loss"] == 1.987
        assert "timestamp" in line2

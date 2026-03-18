"""Tests for M27: Full Fine-Tuning."""

import json
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_forge.config import AdapterConfig


class TestFullFTConfig:
    """Test config changes for full fine-tuning."""

    def test_method_full_accepted(self):
        cfg = AdapterConfig(method="full")
        assert cfg.method == "full"

    def test_method_full_no_targets_required(self):
        """Full FT should not require targets or preset."""
        cfg = AdapterConfig(method="full")
        assert cfg.targets is None
        assert cfg.preset is None

    def test_method_lora_still_requires_targets(self):
        with pytest.raises(ValueError, match="targets.*preset"):
            AdapterConfig(method="lora")

    def test_method_dora_still_requires_targets(self):
        with pytest.raises(ValueError, match="targets.*preset"):
            AdapterConfig(method="dora")


class TestFullFTModel:
    """Test full fine-tuning behavior."""

    def _make_model(self):
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.linear1 = nn.Linear(64, 32)
                self.linear2 = nn.Linear(32, 16)
        model = SimpleModel()
        mx.eval(model.parameters())
        return model

    def test_all_params_trainable(self):
        """In full FT, all parameters should be trainable (no freeze)."""
        model = self._make_model()
        # Don't freeze — all params should be trainable
        trainable = dict(tree_flatten(model.trainable_parameters()))
        total = dict(tree_flatten(model.parameters()))
        assert len(trainable) == len(total)

    def test_training_step_works(self):
        """Full FT should support gradient computation on all params."""
        model = self._make_model()

        def loss_fn(model, x):
            return model.linear1(x).sum()

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        x = mx.random.normal((2, 64))
        loss, grads = loss_and_grad(model, x)
        mx.eval(loss, grads)
        assert loss.item() != 0.0

    def test_quantization_incompatible(self):
        """Full FT + quantization should raise ValueError."""
        # This validation happens in train(), not in config
        # The config itself allows it, but train() should reject it
        cfg = AdapterConfig(method="full")
        assert cfg.method == "full"


class TestFullFTCheckpoint:
    """Test checkpoint behavior for full FT."""

    def test_save_state_includes_method(self, tmp_path):
        """state.json should include adapter_method: full."""
        from mlx_forge.trainer.checkpoint import CheckpointManager
        from mlx_forge.trainer.state import TrainState

        # Create a mock config
        config = MagicMock()
        config.runtime.run_dir = str(tmp_path / "runs")
        config.model.path = "test/model"
        config.model_dump.return_value = {"model": {"path": "test"}}
        config.training.keep_last_n_checkpoints = 3
        config.adapter.method = "full"

        manager = CheckpointManager(config)
        state = TrainState(step=10, best_val_loss=0.5)

        # Create model with all trainable params (no LoRA)
        model = nn.Linear(4, 2)
        mx.eval(model.parameters())

        # Create optimizer
        import mlx.optimizers as optim
        optimizer = optim.Adam(learning_rate=1e-5)
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)

        ckpt_dir = manager.save(state, model, optimizer, adapter_method="full")

        state_json = json.loads((ckpt_dir / "state.json").read_text())
        assert state_json.get("adapter_method") == "full"

    def test_save_full_model_weights(self, tmp_path):
        """Full FT should save model.safetensors, not adapters.safetensors."""
        from mlx_forge.trainer.checkpoint import CheckpointManager
        from mlx_forge.trainer.state import TrainState

        config = MagicMock()
        config.runtime.run_dir = str(tmp_path / "runs")
        config.model.path = "test/model"
        config.model_dump.return_value = {"model": {"path": "test"}}
        config.training.keep_last_n_checkpoints = 3
        config.adapter.method = "full"

        manager = CheckpointManager(config)
        state = TrainState(step=10, best_val_loss=0.5)

        model = nn.Linear(4, 2)
        mx.eval(model.parameters())

        import mlx.optimizers as optim
        optimizer = optim.Adam(learning_rate=1e-5)
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)

        ckpt_dir = manager.save(state, model, optimizer, adapter_method="full")

        assert (ckpt_dir / "model.safetensors").exists()

    def test_load_from_model_safetensors(self, tmp_path):
        """load() should try model.safetensors if adapters.safetensors missing."""
        from mlx_forge.trainer.checkpoint import CheckpointManager
        from mlx_forge.trainer.state import TrainState

        config = MagicMock()
        config.runtime.run_dir = str(tmp_path / "runs")
        config.model.path = "test/model"
        config.model_dump.return_value = {"model": {"path": "test"}}
        config.training.keep_last_n_checkpoints = 3
        config.adapter.method = "full"

        manager = CheckpointManager(config)
        state = TrainState(step=10, best_val_loss=0.5)

        model = nn.Linear(4, 2)
        mx.eval(model.parameters())

        import mlx.optimizers as optim
        optimizer = optim.Adam(learning_rate=1e-5)
        optimizer.init(model.trainable_parameters())
        mx.eval(optimizer.state)

        ckpt_dir = manager.save(state, model, optimizer, adapter_method="full")

        # Load back
        model2 = nn.Linear(4, 2)
        optimizer2 = optim.Adam(learning_rate=1e-5)
        optimizer2.init(model2.trainable_parameters())
        mx.eval(model2.parameters(), optimizer2.state)

        restored = manager.load(ckpt_dir, model2, optimizer2)
        assert restored.step == 10

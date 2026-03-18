"""Tests for M23: DoRA (Weight-Decomposed Low-Rank Adaptation)."""

import mlx.core as mx
import mlx.nn as nn
import pytest
from mlx.utils import tree_flatten

from mlx_forge.adapters.dora import DoRALinear
from mlx_forge.adapters.fuse import fuse_model
from mlx_forge.adapters.lora import LoRALinear, apply_lora
from mlx_forge.config import AdapterConfig


class TestDoRALinear:
    """Test DoRALinear module."""

    def test_from_base_creates_module(self):
        base = nn.Linear(64, 32)
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        assert isinstance(dora, DoRALinear)
        assert isinstance(dora, LoRALinear)  # inherits from LoRALinear

    def test_forward_shape(self):
        base = nn.Linear(64, 32)
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        x = mx.random.normal((2, 10, 64))
        out = dora(x)
        assert out.shape == (2, 10, 32)

    def test_magnitude_shape(self):
        base = nn.Linear(64, 32)
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        assert dora.magnitude.shape == (32,)

    def test_magnitude_initialized_from_base_norms(self):
        base = nn.Linear(64, 32)
        mx.eval(base.parameters())
        expected_norms = mx.sqrt((base.weight * base.weight).sum(axis=1))
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        mx.eval(dora.parameters())
        diff = mx.abs(dora.magnitude - expected_norms).max().item()
        assert diff < 1e-5

    def test_magnitude_is_trainable(self):
        base = nn.Linear(64, 32)
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        trainable = dict(tree_flatten(dora.trainable_parameters()))
        assert "magnitude" in trainable
        assert "lora_a" in trainable
        assert "lora_b" in trainable

    def test_base_layer_is_frozen(self):
        base = nn.Linear(64, 32)
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        trainable = dict(tree_flatten(dora.trainable_parameters()))
        assert "base_layer.weight" not in trainable

    def test_dora_output_differs_from_lora(self):
        """DoRA and LoRA should produce different outputs for same input."""
        mx.random.seed(42)
        base = nn.Linear(64, 32)
        mx.eval(base.parameters())

        # Create LoRA and DoRA with same params
        lora = LoRALinear.from_base(base, r=4, scale=1.0)
        # Need fresh base since from_base freezes in-place
        base2 = nn.Linear(64, 32)
        base2.weight = base.weight  # same weights
        mx.eval(base2.parameters())
        dora = DoRALinear.from_base(base2, r=4, scale=1.0)

        # Set same LoRA weights
        dora.lora_a = lora.lora_a
        dora.lora_b = mx.ones_like(lora.lora_b) * 0.1  # non-zero B
        lora.lora_b = mx.ones_like(lora.lora_b) * 0.1
        mx.eval(lora.parameters(), dora.parameters())

        x = mx.random.normal((1, 5, 64))
        lora_out = lora(x)
        dora_out = dora(x)
        mx.eval(lora_out, dora_out)

        # They should differ because DoRA normalizes by direction
        diff = mx.abs(lora_out - dora_out).max().item()
        assert diff > 1e-6, "DoRA and LoRA should produce different outputs"

    def test_fuse_returns_linear(self):
        base = nn.Linear(64, 32)
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        fused = dora.fuse()
        assert isinstance(fused, nn.Linear)
        assert fused.weight.shape == (32, 64)

    def test_fuse_preserves_output(self):
        """Fused module should produce same output as unfused DoRA."""
        mx.random.seed(42)
        base = nn.Linear(64, 32)
        mx.eval(base.parameters())
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        mx.eval(dora.parameters())

        x = mx.random.normal((2, 5, 64))
        original_out = dora(x)
        mx.eval(original_out)

        fused = dora.fuse()
        mx.eval(fused.parameters())
        fused_out = fused(x)
        mx.eval(fused_out)

        diff = mx.abs(original_out - fused_out).max().item()
        assert diff < 1e-4, f"Fuse changed output by {diff}"

    def test_fuse_with_bias(self):
        base = nn.Linear(64, 32, bias=True)
        mx.eval(base.parameters())
        dora = DoRALinear.from_base(base, r=4, scale=1.0)
        fused = dora.fuse()
        assert hasattr(fused, "bias")
        assert fused.bias is not None


class TestDoRAConfig:
    """Test DoRA config integration."""

    def test_config_accepts_dora_method(self):
        cfg = AdapterConfig(method="dora", preset="attention-qv")
        assert cfg.method == "dora"

    def test_config_rejects_invalid_method(self):
        with pytest.raises(Exception):
            AdapterConfig(method="invalid", preset="attention-qv")

    def test_config_default_is_lora(self):
        cfg = AdapterConfig(preset="attention-qv")
        assert cfg.method == "lora"


class TestDoRAApply:
    """Test DoRA application via apply_lora."""

    def _make_model(self):
        """Create a simple model with linear layers."""
        class SimpleModel(nn.Module):
            def __init__(self):
                super().__init__()
                self.layer1 = nn.Linear(64, 32)
                self.layer2 = nn.Linear(32, 16)
        model = SimpleModel()
        mx.eval(model.parameters())
        return model

    def test_apply_dora_creates_dora_modules(self):
        model = self._make_model()
        model.freeze()
        cfg = AdapterConfig(method="dora", targets=["layer1", "layer2"])
        targets = [("layer1", model.layer1), ("layer2", model.layer2)]
        apply_lora(model, targets, cfg)
        assert isinstance(model.layer1, DoRALinear)
        assert isinstance(model.layer2, DoRALinear)

    def test_apply_lora_creates_lora_modules(self):
        """Regular LoRA should still create LoRALinear, not DoRA."""
        model = self._make_model()
        model.freeze()
        cfg = AdapterConfig(method="lora", targets=["layer1"])
        targets = [("layer1", model.layer1)]
        apply_lora(model, targets, cfg)
        assert isinstance(model.layer1, LoRALinear)
        assert not isinstance(model.layer1, DoRALinear)

    def test_dora_training_step(self):
        """DoRA modules should support gradient computation."""
        model = self._make_model()
        model.freeze()
        cfg = AdapterConfig(method="dora", targets=["layer1"])
        targets = [("layer1", model.layer1)]
        apply_lora(model, targets, cfg)

        def loss_fn(model, x):
            return model.layer1(x).sum()

        loss_and_grad = nn.value_and_grad(model, loss_fn)
        x = mx.random.normal((2, 64))
        loss, grads = loss_and_grad(model, x)
        mx.eval(loss, grads)
        assert loss.item() != 0.0

    def test_fuse_model_handles_dora(self):
        """fuse_model() should handle DoRALinear modules."""
        model = self._make_model()
        model.freeze()
        cfg = AdapterConfig(method="dora", targets=["layer1"])
        targets = [("layer1", model.layer1)]
        apply_lora(model, targets, cfg)
        assert isinstance(model.layer1, DoRALinear)

        fuse_model(model)
        assert isinstance(model.layer1, nn.Linear)
        assert not isinstance(model.layer1, DoRALinear)


class TestDoRAQuantized:
    """Test DoRA with quantized base layers."""

    def test_from_quantized_linear(self):
        """DoRA should work with QuantizedLinear base."""
        base = nn.Linear(64, 32)
        mx.eval(base.parameters())
        # Quantize
        qlinear = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        mx.eval(qlinear.parameters())

        dora = DoRALinear.from_base(qlinear, r=4, scale=1.0)
        x = mx.random.normal((1, 5, 64))
        out = dora(x)
        mx.eval(out)
        assert out.shape == (1, 5, 32)

    def test_fuse_quantized_returns_linear(self):
        base = nn.Linear(64, 32)
        mx.eval(base.parameters())
        qlinear = nn.QuantizedLinear.from_linear(base, bits=4, group_size=64)
        mx.eval(qlinear.parameters())
        dora = DoRALinear.from_base(qlinear, r=4, scale=1.0)
        fused = dora.fuse()
        assert isinstance(fused, nn.Linear)

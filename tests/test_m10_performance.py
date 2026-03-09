"""Tests for M10: QLoRA, Gradient Checkpointing, Sequence Packing.

Tests cover:
- QLoRA config validation, quantization, forward/backward pass
- Gradient checkpointing correctness
- Sequence packing algorithm, packed batching, packed loss
- Backward compatibility (all defaults = v0 behavior)
"""

from __future__ import annotations

from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import numpy as np
import pytest

# =============================================================================
# Helper: tiny Llama model for testing
# =============================================================================

def _make_tiny_model():
    """Create a tiny Llama model for testing."""
    from cortexlab.models.architectures.llama import Model, ModelArgs

    args = ModelArgs(
        model_type="llama",
        hidden_size=64,
        num_hidden_layers=2,
        intermediate_size=128,
        num_attention_heads=4,
        rms_norm_eps=1e-5,
        vocab_size=100,
        head_dim=16,
        num_key_value_heads=2,
    )
    model = Model(args)
    mx.eval(model.parameters())
    return model


# =============================================================================
# QLoRA Config tests
# =============================================================================

class TestQLoRAConfig:
    """Tests for QLoRA configuration."""

    def test_quantization_config_defaults(self):
        """Default quantization config should be 4-bit, group_size=64."""
        from cortexlab.config import QuantizationConfig

        config = QuantizationConfig()
        assert config.bits == 4
        assert config.group_size == 64

    def test_quantization_config_4bit(self):
        """4-bit quantization should be accepted."""
        from cortexlab.config import QuantizationConfig

        config = QuantizationConfig(bits=4, group_size=64)
        assert config.bits == 4

    def test_quantization_config_8bit(self):
        """8-bit quantization should be accepted."""
        from cortexlab.config import QuantizationConfig

        config = QuantizationConfig(bits=8, group_size=32)
        assert config.bits == 8

    def test_quantization_config_invalid_bits(self):
        """Non-4/8 bits should be rejected."""
        from cortexlab.config import QuantizationConfig

        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantizationConfig(bits=3)

        with pytest.raises(ValueError, match="bits must be 4 or 8"):
            QuantizationConfig(bits=16)

    def test_quantization_config_invalid_group_size(self):
        """Non-32/64/128 group_size should be rejected."""
        from cortexlab.config import QuantizationConfig

        with pytest.raises(ValueError, match="group_size must be 32, 64, or 128"):
            QuantizationConfig(group_size=48)

    def test_quantization_config_valid_group_sizes(self):
        """All valid group sizes should be accepted."""
        from cortexlab.config import QuantizationConfig

        for gs in (32, 64, 128):
            config = QuantizationConfig(group_size=gs)
            assert config.group_size == gs

    def test_model_config_quantization_none_default(self):
        """ModelConfig should default to quantization=None."""
        from cortexlab.config import ModelConfig

        config = ModelConfig(path="test/model")
        assert config.quantization is None

    def test_model_config_with_quantization(self):
        """ModelConfig should accept quantization sub-config."""
        from cortexlab.config import ModelConfig

        config = ModelConfig(
            path="test/model",
            quantization={"bits": 4, "group_size": 64},
        )
        assert config.quantization is not None
        assert config.quantization.bits == 4

    def test_training_config_backward_compat(self):
        """Config without quantization should still work (v0 compat)."""
        from cortexlab.config import TrainingConfig

        config = TrainingConfig(
            model={"path": "test/model"},
            adapter={"preset": "attention-qv"},
            data={"train": "train.jsonl", "valid": "val.jsonl"},
            training={"num_iters": 10},
        )
        assert config.model.quantization is None
        assert config.training.gradient_checkpointing is False
        assert config.data.packing is False


# =============================================================================
# QLoRA Quantization tests
# =============================================================================

class TestQLoRAQuantization:
    """Tests for model quantization."""

    def test_quantize_converts_linear_to_quantized(self):
        """quantize_model should convert Linear to QuantizedLinear."""
        from cortexlab.config import QuantizationConfig
        from cortexlab.models.quantize import quantize_model

        model = _make_tiny_model()

        # Check a layer is Linear before quantization
        q_proj = model.model.layers[0].self_attn.q_proj
        assert isinstance(q_proj, nn.Linear)

        config = QuantizationConfig(bits=4, group_size=32)
        quantize_model(model, config)

        # After quantization, should be QuantizedLinear
        q_proj = model.model.layers[0].self_attn.q_proj
        assert isinstance(q_proj, nn.QuantizedLinear)

    def test_quantize_preserves_forward_pass(self):
        """Quantized model should still produce logits of correct shape."""
        from cortexlab.config import QuantizationConfig
        from cortexlab.models.quantize import quantize_model

        model = _make_tiny_model()
        config = QuantizationConfig(bits=4, group_size=32)
        quantize_model(model, config)

        tokens = mx.array([[1, 2, 3, 4]])
        logits = model(tokens)
        mx.eval(logits)

        assert logits.shape == (1, 4, 100)

    def test_qlora_quantize_then_lora(self):
        """QLoRA: quantize first, then apply LoRA. Forward pass should work."""
        from cortexlab.adapters.lora import apply_lora
        from cortexlab.adapters.targeting import resolve_targets
        from cortexlab.config import AdapterConfig, QuantizationConfig
        from cortexlab.models.quantize import quantize_model

        model = _make_tiny_model()

        # Step 1: Quantize
        q_config = QuantizationConfig(bits=4, group_size=32)
        quantize_model(model, q_config)

        # Step 2: Apply LoRA
        adapter_config = AdapterConfig(preset="attention-qv", rank=4, scale=10.0)
        patterns = ["*.self_attn.q_proj", "*.self_attn.v_proj"]
        targets = resolve_targets(model, patterns, num_layers=None)
        apply_lora(model, targets, adapter_config)

        # Step 3: Forward pass
        tokens = mx.array([[1, 2, 3]])
        logits = model(tokens)
        mx.eval(logits)

        assert logits.shape == (1, 3, 100)

    def test_qlora_gradients_flow(self):
        """QLoRA gradients should flow through LoRA params only."""
        from cortexlab.adapters.lora import apply_lora
        from cortexlab.adapters.targeting import resolve_targets
        from cortexlab.config import AdapterConfig, QuantizationConfig
        from cortexlab.models.quantize import quantize_model

        model = _make_tiny_model()

        q_config = QuantizationConfig(bits=4, group_size=32)
        quantize_model(model, q_config)

        adapter_config = AdapterConfig(preset="attention-qv", rank=4, scale=10.0)
        patterns = ["*.self_attn.q_proj", "*.self_attn.v_proj"]
        targets = resolve_targets(model, patterns, num_layers=None)
        apply_lora(model, targets, adapter_config)

        # Compute loss and gradients using nn.value_and_grad
        # (required for QLoRA: only differentiates trainable params)
        def simple_loss(model, x):
            logits = model(x)
            return logits.sum()

        loss_and_grad = nn.value_and_grad(model, simple_loss)
        tokens = mx.array([[1, 2, 3]])
        loss, grads = loss_and_grad(model, tokens)
        mx.eval(loss, grads)

        # Should have computed loss
        assert loss.item() != 0.0

    def test_quantize_8bit(self):
        """8-bit quantization should also work."""
        from cortexlab.config import QuantizationConfig
        from cortexlab.models.quantize import quantize_model

        model = _make_tiny_model()
        config = QuantizationConfig(bits=8, group_size=32)
        quantize_model(model, config)

        tokens = mx.array([[1, 2, 3]])
        logits = model(tokens)
        mx.eval(logits)

        assert logits.shape == (1, 3, 100)


# =============================================================================
# Gradient Checkpointing tests
# =============================================================================

class TestGradientCheckpointing:
    """Tests for gradient checkpointing."""

    def test_config_default_false(self):
        """gradient_checkpointing should default to False."""
        from cortexlab.config import TrainingParams

        params = TrainingParams()
        assert params.gradient_checkpointing is False

    def test_config_accepts_true(self):
        """gradient_checkpointing=True should be accepted."""
        from cortexlab.config import TrainingParams

        params = TrainingParams(gradient_checkpointing=True)
        assert params.gradient_checkpointing is True

    def test_checkpointing_same_output(self):
        """Model with checkpointing should produce same logits as without."""
        from cortexlab import _enable_gradient_checkpointing

        # Model without checkpointing
        model1 = _make_tiny_model()
        tokens = mx.array([[1, 2, 3, 4]])
        logits1 = model1(tokens)
        mx.eval(logits1)

        # Model with checkpointing (same weights)
        model2 = _make_tiny_model()
        # Copy weights from model1
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(model1.parameters()))
        model2.load_weights(list(weights.items()))
        _enable_gradient_checkpointing(model2)

        logits2 = model2(tokens)
        mx.eval(logits2)

        assert mx.allclose(logits1, logits2, atol=1e-5).item()

    def test_checkpointing_same_gradients(self):
        """Gradients with checkpointing should match gradients without."""
        from cortexlab import _enable_gradient_checkpointing

        def simple_loss(model, x):
            return model(x).sum()

        # Without checkpointing
        model1 = _make_tiny_model()
        tokens = mx.array([[1, 2, 3]])
        _, grads1 = mx.value_and_grad(simple_loss)(model1, tokens)
        mx.eval(grads1)

        # With checkpointing (same weights)
        model2 = _make_tiny_model()
        from mlx.utils import tree_flatten
        weights = dict(tree_flatten(model1.parameters()))
        model2.load_weights(list(weights.items()))
        _enable_gradient_checkpointing(model2)
        _, grads2 = mx.value_and_grad(simple_loss)(model2, tokens)
        mx.eval(grads2)

        # Compare gradients
        flat1 = tree_flatten(grads1)
        flat2 = tree_flatten(grads2)
        for (n1, g1), (n2, g2) in zip(flat1, flat2):
            assert n1 == n2
            assert mx.allclose(g1, g2, atol=1e-4).item(), f"Gradient mismatch at {n1}"

    def test_enable_gradient_checkpointing_function(self):
        """_enable_gradient_checkpointing should wrap layer __call__."""
        from cortexlab import _enable_gradient_checkpointing

        model = _make_tiny_model()
        original_call = model.model.layers[0].__call__

        _enable_gradient_checkpointing(model)

        # After enabling, the __call__ should be different (wrapped)
        new_call = model.model.layers[0].__call__
        assert new_call is not original_call


# =============================================================================
# Sequence Packing tests
# =============================================================================

class TestSequencePacking:
    """Tests for sequence packing algorithm (V2: input_ids + labels)."""

    def test_pack_single_sequence(self):
        """Single sequence should create one bin."""
        from cortexlab.data.packing import pack_sequences

        dataset = [{"input_ids": [1, 2, 3, 4, 5], "labels": [-100, -100, 3, 4, 5]}]
        packed = pack_sequences(dataset, max_seq_length=32)

        assert len(packed) == 1
        assert packed[0].input_ids == [1, 2, 3, 4, 5]
        assert packed[0].labels == [-100, -100, 3, 4, 5]
        assert packed[0].segment_ids == [0, 0, 0, 0, 0]

    def test_pack_multiple_short_sequences(self):
        """Short sequences should be packed into fewer bins."""
        from cortexlab.data.packing import pack_sequences

        dataset = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5, 6], "labels": [-100, 5, 6]},
            {"input_ids": [7, 8, 9], "labels": [-100, 8, 9]},
        ]
        packed = pack_sequences(dataset, max_seq_length=32)

        # All 3 sequences (total 9 tokens) should fit in 1 bin (max=32)
        assert len(packed) == 1
        assert len(packed[0].input_ids) == 9
        assert len(packed[0].labels) == 9

    def test_pack_respects_max_seq_length(self):
        """No bin should exceed max_seq_length."""
        from cortexlab.data.packing import pack_sequences

        dataset = [
            {"input_ids": list(range(10)), "labels": list(range(10))},
            {"input_ids": list(range(10)), "labels": list(range(10))},
            {"input_ids": list(range(10)), "labels": list(range(10))},
        ]
        packed = pack_sequences(dataset, max_seq_length=15)

        for ps in packed:
            assert len(ps.input_ids) <= 15

    def test_pack_segment_ids_correct(self):
        """Segment IDs should identify which sequence each token belongs to."""
        from cortexlab.data.packing import pack_sequences

        dataset = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]
        packed = pack_sequences(dataset, max_seq_length=32)

        assert len(packed) == 1
        ps = packed[0]

        seg_counts = {}
        for sid in ps.segment_ids:
            seg_counts[sid] = seg_counts.get(sid, 0) + 1

        assert len(seg_counts) == 2
        assert sum(seg_counts.values()) == 5

    def test_pack_truncates_long_sequences(self):
        """Sequences longer than max_seq_length should be truncated."""
        from cortexlab.data.packing import pack_sequences

        dataset = [{"input_ids": list(range(100)), "labels": list(range(100))}]
        packed = pack_sequences(dataset, max_seq_length=32)

        assert len(packed) == 1
        assert len(packed[0].input_ids) == 32
        assert len(packed[0].labels) == 32

    def test_pack_labels_preserved(self):
        """Labels should be preserved through packing."""
        from cortexlab.data.packing import pack_sequences

        dataset = [
            {"input_ids": [10, 20, 30, 40, 50], "labels": [-100, -100, 30, 40, 50]},
        ]
        packed = pack_sequences(dataset, max_seq_length=32)

        assert len(packed) == 1
        ps = packed[0]
        assert ps.labels == [-100, -100, 30, 40, 50]

    def test_pack_empty_dataset(self):
        """Empty dataset should return no packed sequences."""
        from cortexlab.data.packing import pack_sequences

        packed = pack_sequences([], max_seq_length=32)
        assert len(packed) == 0


# =============================================================================
# Packed Batching tests
# =============================================================================

class TestPackedBatching:
    """Tests for packed batch iteration (V2: input_ids + labels + segment_ids)."""

    def test_iterate_packed_batches_shapes(self):
        """Packed batches should have correct shapes: (B,T) x3."""
        from cortexlab.data.batching import iterate_packed_batches

        dataset = [
            {"input_ids": list(range(10)), "labels": list(range(10))}
            for _ in range(8)
        ]

        config = MagicMock()
        config.training.batch_size = 2
        config.data.max_seq_length = 64

        batches = list(iterate_packed_batches(dataset, config))
        assert len(batches) > 0

        for input_ids, labels, segment_ids in batches:
            B = input_ids.shape[0]
            T = input_ids.shape[1]
            assert B == 2
            assert labels.shape == (B, T)
            assert segment_ids.shape == (B, T)

    def test_iterate_packed_batches_padding(self):
        """Padding positions should have segment_id = -1 and labels = -100."""
        from cortexlab.data.batching import iterate_packed_batches

        dataset = [
            {"input_ids": [1, 2, 3], "labels": [-100, 2, 3]},
            {"input_ids": [4, 5], "labels": [-100, 5]},
        ]

        config = MagicMock()
        config.training.batch_size = 1
        config.data.max_seq_length = 64

        batches = list(iterate_packed_batches(dataset, config))

        if batches:
            _, labels, seg_ids = batches[0]
            mx.eval(seg_ids, labels)
            seg_np = np.array(seg_ids)
            labels_np = np.array(labels)
            # Padding positions should be -1 for segment_ids
            assert (seg_np == -1).any()
            # Padding positions should be -100 for labels
            assert (labels_np == -100).any()

    def test_packing_config_default_false(self):
        """DataConfig.packing should default to False."""
        from cortexlab.config import DataConfig

        config = DataConfig(train="t.jsonl", valid="v.jsonl")
        assert config.packing is False

    def test_packing_config_accepts_true(self):
        """DataConfig.packing=True should be accepted."""
        from cortexlab.config import DataConfig

        config = DataConfig(train="t.jsonl", valid="v.jsonl", packing=True)
        assert config.packing is True


# =============================================================================
# Packed Loss tests
# =============================================================================

class TestPackedLoss:
    """Tests for packed sequence loss computation (V2: labels-based)."""

    def test_packed_loss_basic(self):
        """Packed loss should compute without errors."""
        from cortexlab.trainer.trainer import loss_fn_packed

        model = _make_tiny_model()

        # Two segments packed together, padding at end
        input_ids = mx.array([[1, 2, 3, 4, 5, 6, 0, 0]], dtype=mx.int32)
        # Labels: -100 for prompt tokens + padding, real values for completion tokens
        labels = mx.array([[-100, 2, 3, -100, 5, 6, -100, -100]], dtype=mx.int32)
        segment_ids = mx.array([[0, 0, 0, 1, 1, 1, -1, -1]], dtype=mx.int32)

        loss, ntoks = loss_fn_packed(model, input_ids, labels, segment_ids)
        mx.eval(loss, ntoks)

        assert loss.item() > 0
        assert ntoks.item() > 0

    def test_packed_loss_no_cross_segment(self):
        """Loss should not be computed across segment boundaries."""
        from cortexlab.trainer.trainer import loss_fn_packed

        model = _make_tiny_model()

        input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)  # All trainable
        segment_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)

        loss, ntoks = loss_fn_packed(model, input_ids, labels, segment_ids)
        mx.eval(loss, ntoks)

        # inputs = input_ids[:, :-1], targets = labels[:, 1:]
        # seg_in = [0,0,0,1,1], seg_out = [0,0,1,1,1]
        # same_seg = [T,T,F,T,T] (position 2: seg_in=0 != seg_out=1)
        # targets != -100 = all True
        # mask = same_seg & (targets != -100) & (seg_out >= 0) = [T,T,F,T,T]
        # ntoks should be 4
        assert ntoks.item() == 4

    def test_packed_loss_excludes_padding(self):
        """Loss should not be computed on padding tokens."""
        from cortexlab.trainer.trainer import loss_fn_packed

        model = _make_tiny_model()

        input_ids = mx.array([[1, 2, 3, 0, 0, 0]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, -100, -100, -100]], dtype=mx.int32)
        segment_ids = mx.array([[0, 0, 0, -1, -1, -1]], dtype=mx.int32)

        loss, ntoks = loss_fn_packed(model, input_ids, labels, segment_ids)
        mx.eval(loss, ntoks)

        # Only 2 tokens: targets are labels[:, 1:] = [2, 3, -100, -100, -100]
        # seg_in = [0, 0, 0, -1, -1], seg_out = [0, 0, -1, -1, -1]
        # same_seg & seg_out>=0 = [T,T,F,F,F], mask with labels: [T,T,F,F,F]
        assert ntoks.item() == 2

    def test_packed_loss_gradients(self):
        """Packed loss should produce valid gradients."""
        from cortexlab.trainer.trainer import loss_fn_packed

        model = _make_tiny_model()

        input_ids = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
        labels = mx.array([[1, 2, 3, 4, 5, 6]], dtype=mx.int32)
        segment_ids = mx.array([[0, 0, 0, 1, 1, 1]], dtype=mx.int32)

        loss_and_grad = mx.value_and_grad(loss_fn_packed)
        (loss, ntoks), grads = loss_and_grad(model, input_ids, labels, segment_ids)
        mx.eval(loss, ntoks, grads)

        assert loss.item() > 0


# =============================================================================
# Combined Feature tests
# =============================================================================

class TestCombinedFeatures:
    """Tests for using multiple M10 features together."""

    def test_qlora_with_checkpointing(self):
        """QLoRA + gradient checkpointing should work together."""
        from cortexlab import _enable_gradient_checkpointing
        from cortexlab.adapters.lora import apply_lora
        from cortexlab.adapters.targeting import resolve_targets
        from cortexlab.config import AdapterConfig, QuantizationConfig
        from cortexlab.models.quantize import quantize_model

        model = _make_tiny_model()

        # Quantize
        q_config = QuantizationConfig(bits=4, group_size=32)
        quantize_model(model, q_config)

        # LoRA
        adapter_config = AdapterConfig(preset="attention-qv", rank=4, scale=10.0)
        patterns = ["*.self_attn.q_proj", "*.self_attn.v_proj"]
        targets = resolve_targets(model, patterns, num_layers=None)
        apply_lora(model, targets, adapter_config)

        # Gradient checkpointing
        _enable_gradient_checkpointing(model)

        # Forward + backward (nn.value_and_grad for QLoRA compat)
        def simple_loss(model, x):
            return model(x).sum()

        tokens = mx.array([[1, 2, 3]])
        loss, grads = nn.value_and_grad(model, simple_loss)(model, tokens)
        mx.eval(loss, grads)

        assert loss.item() != 0.0

    def test_all_defaults_v0_compat(self):
        """All new config fields should default to v0 behavior."""
        from cortexlab.config import TrainingConfig

        config = TrainingConfig(
            model={"path": "test/model"},
            adapter={"preset": "attention-qv"},
            data={"train": "t.jsonl", "valid": "v.jsonl"},
            training={"num_iters": 10},
        )

        # All new features disabled by default
        assert config.model.quantization is None
        assert config.training.gradient_checkpointing is False
        assert config.data.packing is False

    def test_config_from_yaml_with_m10_fields(self, tmp_path):
        """YAML config with M10 fields should parse correctly."""
        from cortexlab.config import TrainingConfig

        yaml_content = """
schema_version: 1
model:
  path: test/model
  quantization:
    bits: 4
    group_size: 64
adapter:
  preset: attention-qv
data:
  train: train.jsonl
  valid: val.jsonl
  packing: true
training:
  num_iters: 100
  gradient_checkpointing: true
"""
        config_path = tmp_path / "config.yaml"
        config_path.write_text(yaml_content)

        config = TrainingConfig.from_yaml(str(config_path))

        assert config.model.quantization.bits == 4
        assert config.model.quantization.group_size == 64
        assert config.training.gradient_checkpointing is True
        assert config.data.packing is True

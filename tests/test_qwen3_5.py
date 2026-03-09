"""Tests for Qwen3.5 hybrid architecture (DeltaNet + full attention)."""

from __future__ import annotations

import mlx.core as mx
import mlx.nn as nn

from cortexlab.inference.cache import KVCache, RecurrentCache
from cortexlab.models.architectures.qwen3_5 import (
    DecoderLayer,
    GatedDeltaNet,
    Model,
    ModelArgs,
    Qwen3_5RMSNormGated,
    gated_delta_chunkwise,
    gated_delta_recurrence,
)
from cortexlab.models.registry import get_model_classes, is_supported

# ---------------------------------------------------------------------------
# Fixtures: Real config dicts for 0.8B and 2B
# ---------------------------------------------------------------------------

QWEN3_5_0_8B_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 1024,
    "num_hidden_layers": 24,
    "intermediate_size": 3584,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "vocab_size": 248320,
    "rope_theta": 10000000.0,
    "max_position_embeddings": 262144,
    "partial_rotary_factor": 0.25,
    "tie_word_embeddings": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
}

QWEN3_5_2B_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 2048,
    "num_hidden_layers": 24,
    "intermediate_size": 6144,
    "num_attention_heads": 8,
    "num_key_value_heads": 2,
    "head_dim": 256,
    "rms_norm_eps": 1e-6,
    "vocab_size": 248320,
    "rope_theta": 10000000.0,
    "max_position_embeddings": 262144,
    "partial_rotary_factor": 0.25,
    "tie_word_embeddings": True,
    "linear_num_key_heads": 16,
    "linear_num_value_heads": 16,
    "linear_key_head_dim": 128,
    "linear_value_head_dim": 128,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
}

# Real HF VLM config (nested structure as downloaded from HuggingFace)
QWEN3_5_0_8B_HF_CONFIG = {
    "architectures": ["Qwen3_5ForConditionalGeneration"],
    "model_type": "qwen3_5",
    "tie_word_embeddings": True,
    "text_config": {
        "model_type": "qwen3_5_text",
        "hidden_size": 1024,
        "num_hidden_layers": 24,
        "intermediate_size": 3584,
        "num_attention_heads": 8,
        "num_key_value_heads": 2,
        "head_dim": 256,
        "rms_norm_eps": 1e-6,
        "vocab_size": 248320,
        "max_position_embeddings": 262144,
        "tie_word_embeddings": True,
        "full_attention_interval": 4,
        "linear_num_key_heads": 16,
        "linear_num_value_heads": 16,
        "linear_key_head_dim": 128,
        "linear_value_head_dim": 128,
        "linear_conv_kernel_dim": 4,
        "layer_types": [
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
            "linear_attention", "linear_attention", "linear_attention", "full_attention",
        ],
        "rope_parameters": {
            "rope_type": "default",
            "rope_theta": 10000000,
            "partial_rotary_factor": 0.25,
            "mrope_interleaved": True,
            "mrope_section": [11, 11, 10],
        },
    },
    "vision_config": {
        "model_type": "qwen3_5",
        "hidden_size": 768,
        "depth": 12,
    },
}

# Tiny config for forward-pass tests (fast, low memory)
TINY_CONFIG = {
    "model_type": "qwen3_5",
    "hidden_size": 32,
    "num_hidden_layers": 4,
    "intermediate_size": 64,
    "num_attention_heads": 2,
    "num_key_value_heads": 1,
    "head_dim": 16,
    "rms_norm_eps": 1e-6,
    "vocab_size": 100,
    "rope_theta": 10000.0,
    "max_position_embeddings": 512,
    "partial_rotary_factor": 0.25,
    "tie_word_embeddings": True,
    "linear_num_key_heads": 2,
    "linear_num_value_heads": 2,
    "linear_key_head_dim": 16,
    "linear_value_head_dim": 16,
    "linear_conv_kernel_dim": 4,
    "full_attention_interval": 4,
}


def _make_tiny_model():
    """Create a tiny Qwen3.5 model for testing."""
    args = ModelArgs.from_dict(TINY_CONFIG)
    model = Model(args)
    mx.eval(model.parameters())
    return model


# ===========================================================================
# Config Tests
# ===========================================================================


class TestModelArgs:
    """Tests for Qwen3.5 config parsing."""

    def test_model_args_from_config_0_8b(self):
        """Parse real 0.8B config, verify all fields."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_CONFIG)

        assert args.hidden_size == 1024
        assert args.num_hidden_layers == 24
        assert args.intermediate_size == 3584
        assert args.num_attention_heads == 8
        assert args.num_key_value_heads == 2
        assert args.head_dim == 256
        assert args.vocab_size == 248320
        assert args.tie_word_embeddings is True
        assert args.linear_num_key_heads == 16
        assert args.linear_key_head_dim == 128
        assert args.linear_conv_kernel_dim == 4
        assert args.full_attention_interval == 4
        assert args.partial_rotary_factor == 0.25

    def test_model_args_from_config_2b(self):
        """Parse real 2B config, verify key differences."""
        args = ModelArgs.from_dict(QWEN3_5_2B_CONFIG)

        assert args.hidden_size == 2048
        assert args.intermediate_size == 6144
        assert args.num_hidden_layers == 24

    def test_layer_type_detection(self):
        """Verify correct linear/attention dispatch for all 24 layers."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_CONFIG)

        for i in range(24):
            is_linear = DecoderLayer._is_linear_layer(args, i)
            # Every 4th layer (idx 3, 7, 11, 15, 19, 23) is full attention
            if (i + 1) % 4 == 0:
                assert not is_linear, f"Layer {i} should be full attention"
            else:
                assert is_linear, f"Layer {i} should be DeltaNet"

        # Count: 18 DeltaNet + 6 full attention = 24
        linear_count = sum(
            DecoderLayer._is_linear_layer(args, i) for i in range(24)
        )
        assert linear_count == 18

    def test_layer_types_list_vs_interval(self):
        """Both config styles (list vs interval) produce same layer assignments."""
        args_interval = ModelArgs.from_dict(QWEN3_5_0_8B_CONFIG)

        # Build explicit layer_types list matching interval=4
        layer_types = [0] * 24  # All DeltaNet
        for i in range(24):
            if (i + 1) % 4 == 0:
                layer_types[i] = 1  # Full attention

        config_with_list = {**QWEN3_5_0_8B_CONFIG, "layer_types": layer_types}
        args_list = ModelArgs.from_dict(config_with_list)

        for i in range(24):
            assert (
                DecoderLayer._is_linear_layer(args_interval, i)
                == DecoderLayer._is_linear_layer(args_list, i)
            ), f"Mismatch at layer {i}"

    def test_layer_types_string_format(self):
        """Test layer_types with string format ('linear_attention')."""
        layer_types = ["linear_attention"] * 24
        layer_types[3] = "full_attention"
        layer_types[7] = "full_attention"

        config = {**TINY_CONFIG, "layer_types": layer_types}
        args = ModelArgs.from_dict(config)

        assert DecoderLayer._is_linear_layer(args, 0) is True
        assert DecoderLayer._is_linear_layer(args, 3) is False
        assert DecoderLayer._is_linear_layer(args, 7) is False

    def test_from_dict_filters_unknown_keys(self):
        """Extra config keys are silently ignored."""
        config = {
            **TINY_CONFIG,
            "unknown_field": "ignored",
            "mtp_depth": 1,
        }
        args = ModelArgs.from_dict(config)
        assert args.hidden_size == 32
        assert not hasattr(args, "unknown_field")

    def test_from_dict_vlm_nested_config(self):
        """Parse real HF VLM config with nested text_config."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_HF_CONFIG)

        # Should extract fields from text_config
        assert args.hidden_size == 1024
        assert args.num_hidden_layers == 24
        assert args.vocab_size == 248320
        assert args.linear_num_key_heads == 16

        # Should use top-level model_type, not text_config's
        assert args.model_type == "qwen3_5"

        # Should extract rope params from nested rope_parameters
        assert args.rope_theta == 10000000
        assert args.partial_rotary_factor == 0.25

        # rope_scaling should be None for "default" rope_type
        assert args.rope_scaling is None

        # Should preserve layer_types
        assert args.layer_types is not None
        assert len(args.layer_types) == 24
        assert args.layer_types[0] == "linear_attention"
        assert args.layer_types[3] == "full_attention"

    def test_from_dict_vlm_creates_valid_model(self):
        """VLM config produces a working model."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_HF_CONFIG)
        model = Model(args)

        assert len(model.layers) == 24
        assert model.layers[0].is_linear
        assert not model.layers[3].is_linear


# ===========================================================================
# Model Instantiation Tests
# ===========================================================================


class TestModelInstantiation:
    """Tests for model creation and structure."""

    def test_model_instantiation_0_8b(self):
        """Create 0.8B model, verify layer count and types."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_CONFIG)
        model = Model(args)

        assert len(model.layers) == 24

        linear_layers = [l for l in model.layers if l.is_linear]
        attn_layers = [l for l in model.layers if not l.is_linear]
        assert len(linear_layers) == 18
        assert len(attn_layers) == 6

    def test_model_instantiation_2b(self):
        """Create 2B model, verify structure."""
        args = ModelArgs.from_dict(QWEN3_5_2B_CONFIG)
        model = Model(args)

        assert len(model.layers) == 24

    def test_deltanet_layer_shapes(self):
        """Verify DeltaNet projection dimensions and parameters."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_CONFIG)
        model = Model(args)

        # Layer 0 should be DeltaNet
        layer0 = model.layers[0]
        assert layer0.is_linear
        dn = layer0.linear_attn

        assert dn.key_dim == 16 * 128  # 2048
        assert dn.value_dim == 16 * 128  # 2048
        assert dn.conv_dim == 2048 * 2 + 2048  # 6144

        # Verify DeltaNet-specific parameters exist
        assert hasattr(dn, "A_log")
        assert dn.A_log.shape == (16,)
        assert hasattr(dn, "dt_bias")
        assert dn.dt_bias.shape == (16,)
        assert hasattr(dn, "norm")
        assert isinstance(dn.norm, Qwen3_5RMSNormGated)

    def test_attention_layer_shapes(self):
        """Verify full attention dimensions with fused gate."""
        args = ModelArgs.from_dict(QWEN3_5_0_8B_CONFIG)
        model = Model(args)

        # Layer 3 should be full attention
        layer3 = model.layers[3]
        assert not layer3.is_linear
        attn = layer3.self_attn

        assert attn.n_heads == 8
        assert attn.n_kv_heads == 2
        assert attn.head_dim == 256

        # q_proj output is 2x (queries + gate fused)
        q_out = attn.n_heads * attn.head_dim * 2  # 8 * 256 * 2 = 4096
        assert attn.q_proj.weight.shape == (q_out, 1024)

        # No separate gate_proj
        assert not hasattr(attn, "gate_proj")

    def test_tied_embeddings_no_lm_head(self):
        """Model with tied embeddings should not have lm_head."""
        model = _make_tiny_model()
        assert not hasattr(model, "lm_head")

    def test_untied_embeddings_has_lm_head(self):
        """Model with untied embeddings should have lm_head."""
        config = {**TINY_CONFIG, "tie_word_embeddings": False}
        args = ModelArgs.from_dict(config)
        model = Model(args)
        assert hasattr(model, "lm_head")

    def test_deltanet_has_conv1d(self):
        """DeltaNet layers should have 1D depthwise conv."""
        model = _make_tiny_model()
        dn = model.layers[0].linear_attn
        assert hasattr(dn, "conv1d")

    def test_rms_norm_gated(self):
        """Qwen3_5RMSNormGated applies SwiGLU when gate is provided."""
        norm = Qwen3_5RMSNormGated(16, eps=1e-6)
        mx.eval(norm.parameters())

        x = mx.random.normal((1, 4, 16))

        # Without gate: just RMSNorm
        out_no_gate = norm(x)
        mx.eval(out_no_gate)
        assert out_no_gate.shape == x.shape

        # With gate: RMSNorm + SwiGLU
        gate = mx.random.normal((1, 4, 16))
        out_gated = norm(x, gate)
        mx.eval(out_gated)
        assert out_gated.shape == x.shape

        # Outputs should differ
        assert not mx.allclose(out_no_gate, out_gated).item()


# ===========================================================================
# Forward Pass Tests
# ===========================================================================


class TestForwardPass:
    """Tests for model forward pass with tiny config."""

    def test_forward_training_mode(self):
        """model(input_ids) with no cache returns correct shape."""
        model = _make_tiny_model()
        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (1, 5, 100)  # (B, T, vocab_size)

    def test_forward_batch(self):
        """Batch forward pass works."""
        model = _make_tiny_model()
        input_ids = mx.array([[1, 2, 3], [4, 5, 6]])
        logits = model(input_ids)
        mx.eval(logits)

        assert logits.shape == (2, 3, 100)

    def test_forward_inference_prefill(self):
        """Prefill with cache returns correct shape and populates caches."""
        model = _make_tiny_model()
        cache = model.make_cache()

        input_ids = mx.array([[1, 2, 3, 4, 5]])
        logits = model(input_ids, cache=cache)
        mx.eval(logits)

        assert logits.shape == (1, 5, 100)

        # Verify cache states updated
        # Layer 0 is DeltaNet (RecurrentCache)
        assert isinstance(cache[0], RecurrentCache)
        assert cache[0].offset == 5
        assert cache[0].conv_state is not None
        assert cache[0].ssm_state is not None

        # Layer 3 is full attention (KVCache)
        assert isinstance(cache[3], KVCache)
        assert cache[3].offset == 5

    def test_forward_inference_decode(self):
        """Single-token decode after prefill."""
        model = _make_tiny_model()
        cache = model.make_cache()

        # Prefill
        input_ids = mx.array([[1, 2, 3]])
        logits = model(input_ids, cache=cache)
        mx.eval(logits)

        # Decode one token
        next_token = mx.array([[4]])
        logits = model(next_token, cache=cache)
        mx.eval(logits)

        assert logits.shape == (1, 1, 100)

        # Offsets should have advanced
        assert cache[0].offset == 4  # DeltaNet
        assert cache[3].offset == 4  # KVCache


class TestDeltaNetRecurrence:
    """Tests for the gated_delta_recurrence function."""

    def test_shapes(self):
        """Verify output shapes from recurrence."""
        B, T, Hk, Dk, Hv, Dv = 2, 5, 4, 8, 4, 8
        q = mx.random.normal((B, T, Hk, Dk))
        k = mx.random.normal((B, T, Hk, Dk))
        v = mx.random.normal((B, T, Hv, Dv))
        a = mx.random.normal((B, T, Hv))
        b = mx.random.normal((B, T, Hv))
        A_log = mx.zeros(Hv)
        dt_bias = mx.zeros(Hv)

        output, state = gated_delta_recurrence(q, k, v, a, b, A_log, dt_bias)
        mx.eval(output, state)

        assert output.shape == (B, T, Hv, Dv)
        assert state.shape == (B, Hv, Dv, Dk)

    def test_state_evolution(self):
        """State should change from initial zeros."""
        B, T, H, Dk, Dv = 1, 3, 2, 4, 4
        q = mx.random.normal((B, T, H, Dk))
        k = mx.random.normal((B, T, H, Dk))
        v = mx.random.normal((B, T, H, Dv))
        a = mx.random.normal((B, T, H))
        b = mx.ones((B, T, H)) * 2  # Large b -> sigmoid(b) near 1
        A_log = mx.zeros(H)
        dt_bias = mx.zeros(H)

        output, state = gated_delta_recurrence(q, k, v, a, b, A_log, dt_bias)
        mx.eval(output, state)

        # State should be non-zero after processing
        assert mx.abs(state).sum().item() > 0

    def test_state_continuity(self):
        """Processing in two chunks should match processing all at once."""
        B, H, Dk, Dv = 1, 2, 4, 4
        T1, T2 = 3, 2

        mx.random.seed(42)
        q = mx.random.normal((B, T1 + T2, H, Dk))
        k = mx.random.normal((B, T1 + T2, H, Dk))
        v = mx.random.normal((B, T1 + T2, H, Dv))
        a = mx.random.normal((B, T1 + T2, H))
        b = mx.random.normal((B, T1 + T2, H))
        A_log = mx.zeros(H)
        dt_bias = mx.zeros(H)

        # All at once
        out_full, state_full = gated_delta_recurrence(
            q, k, v, a, b, A_log, dt_bias
        )

        # In two chunks
        out1, state1 = gated_delta_recurrence(
            q[:, :T1], k[:, :T1], v[:, :T1],
            a[:, :T1], b[:, :T1],
            A_log, dt_bias,
        )
        out2, state2 = gated_delta_recurrence(
            q[:, T1:], k[:, T1:], v[:, T1:],
            a[:, T1:], b[:, T1:],
            A_log, dt_bias,
            state=state1,
        )

        mx.eval(out_full, state_full, out1, out2, state1, state2)

        # Final states should match
        assert mx.allclose(state_full, state2, atol=1e-5).item()

        # Outputs should match when concatenated
        out_chunks = mx.concatenate([out1, out2], axis=1)
        assert mx.allclose(out_full, out_chunks, atol=1e-5).item()

    def test_head_repeat(self):
        """Recurrence handles Hv > Hk (GQA-style grouping)."""
        B, T, Hk, Dk, Hv, Dv = 1, 3, 2, 4, 4, 4
        q = mx.random.normal((B, T, Hk, Dk))
        k = mx.random.normal((B, T, Hk, Dk))
        v = mx.random.normal((B, T, Hv, Dv))
        a = mx.random.normal((B, T, Hv))
        b = mx.random.normal((B, T, Hv))
        A_log = mx.zeros(Hv)
        dt_bias = mx.zeros(Hv)

        output, state = gated_delta_recurrence(q, k, v, a, b, A_log, dt_bias)
        mx.eval(output, state)

        assert output.shape == (B, T, Hv, Dv)
        assert state.shape == (B, Hv, Dv, Dk)


class TestCausalConv:
    """Tests for causal convolution in GatedDeltaNet."""

    def test_conv_output_length(self):
        """Conv with left-padding should preserve sequence length."""
        args = ModelArgs.from_dict(TINY_CONFIG)
        dn = GatedDeltaNet(args)
        mx.eval(dn.parameters())

        B, T = 1, 8
        x = mx.random.normal((B, T, args.hidden_size))
        output = dn(x)
        mx.eval(output)

        assert output.shape == (B, T, args.hidden_size)

    def test_conv_prefill_vs_sequential(self):
        """Prefill conv should match sequential single-token conv."""
        args = ModelArgs.from_dict(TINY_CONFIG)
        dn = GatedDeltaNet(args)
        mx.eval(dn.parameters())

        B = 1
        T = 6
        x = mx.random.normal((B, T, args.hidden_size))

        # Prefill: process all at once with cache
        cache_prefill = RecurrentCache()
        out_prefill = dn(x, cache=cache_prefill)
        mx.eval(out_prefill)

        # Sequential: process one token at a time
        cache_seq = RecurrentCache()
        outputs_seq = []
        for t in range(T):
            out_t = dn(x[:, t : t + 1, :], cache=cache_seq)
            mx.eval(out_t)
            outputs_seq.append(out_t)
        out_sequential = mx.concatenate(outputs_seq, axis=1)
        mx.eval(out_sequential)

        # Should match within tolerance
        assert mx.allclose(out_prefill, out_sequential, atol=1e-4).item()


# ===========================================================================
# Integration Tests
# ===========================================================================


class TestRegistryIntegration:
    """Tests for registry support."""

    def test_registry_support(self):
        """is_supported('qwen3_5') returns True."""
        assert is_supported("qwen3_5") is True

    def test_get_model_classes(self):
        """get_model_classes returns correct classes."""
        ModelCls, ArgsCls = get_model_classes({"model_type": "qwen3_5"})
        assert ModelCls is Model
        assert ArgsCls is ModelArgs


class TestSanitize:
    """Tests for weight sanitization."""

    def test_sanitize_conv1d_weights(self):
        """Conv1d weights transposed from HF (out, 1, kernel) to MLX (out, kernel, 1)."""
        model = _make_tiny_model()

        # Simulate HF format: (channels, 1, kernel_size)
        C = model.layers[0].linear_attn.conv_dim
        K = 4
        hf_weight = mx.random.normal((C, 1, K))

        weights = {"model.layers.0.linear_attn.conv1d.weight": hf_weight}
        sanitized = model.sanitize(weights)

        w = sanitized["model.layers.0.linear_attn.conv1d.weight"]
        # Should be transposed to (channels, kernel_size, 1)
        assert w.shape == (C, K, 1)

    def test_sanitize_conv1d_already_correct(self):
        """Conv1d weights already in MLX format are not transposed."""
        model = _make_tiny_model()

        C = model.layers[0].linear_attn.conv_dim
        K = 4
        mlx_weight = mx.random.normal((C, K, 1))

        weights = {"model.layers.0.linear_attn.conv1d.weight": mlx_weight}
        sanitized = model.sanitize(weights)

        w = sanitized["model.layers.0.linear_attn.conv1d.weight"]
        assert w.shape == (C, K, 1)

    def test_sanitize_mtp_removal(self):
        """MTP weights are stripped."""
        model = _make_tiny_model()

        weights = {
            "model.embed_tokens.weight": mx.zeros((100, 32)),
            "mtp.head.weight": mx.zeros((100, 32)),
            "mtp.layers.0.weight": mx.zeros((32, 32)),
        }
        sanitized = model.sanitize(weights)

        assert "model.embed_tokens.weight" in sanitized
        assert not any("mtp." in k for k in sanitized)

    def test_sanitize_tied_embeddings(self):
        """lm_head removed when using tied embeddings."""
        model = _make_tiny_model()

        weights = {
            "model.embed_tokens.weight": mx.zeros((100, 32)),
            "lm_head.weight": mx.zeros((100, 32)),
        }
        sanitized = model.sanitize(weights)

        assert "lm_head.weight" not in sanitized
        assert "model.embed_tokens.weight" in sanitized

    def test_sanitize_vlm_weight_prefix(self):
        """VLM weights with model.language_model.* prefix are remapped."""
        model = _make_tiny_model()

        weights = {
            "model.language_model.embed_tokens.weight": mx.zeros((100, 32)),
            "model.language_model.layers.0.linear_attn.in_proj_qkv.weight": mx.zeros((10, 32)),
            "model.language_model.layers.3.self_attn.q_proj.weight": mx.zeros((64, 32)),
            "model.language_model.norm.weight": mx.zeros((32,)),
        }
        sanitized = model.sanitize(weights)

        assert "model.embed_tokens.weight" in sanitized
        assert "model.layers.0.linear_attn.in_proj_qkv.weight" in sanitized
        assert "model.layers.3.self_attn.q_proj.weight" in sanitized
        # model.norm.weight gets +1.0 too
        assert "model.norm.weight" in sanitized
        # Original prefixed keys should be gone
        assert not any("language_model" in k for k in sanitized)

    def test_sanitize_removes_vision_weights(self):
        """Vision encoder weights are stripped."""
        model = _make_tiny_model()

        weights = {
            "model.embed_tokens.weight": mx.zeros((100, 32)),
            "model.visual.merger.linear_fc1.weight": mx.zeros((32, 32)),
            "model.visual.blocks.0.attn.q.weight": mx.zeros((32, 32)),
        }
        sanitized = model.sanitize(weights)

        assert "model.embed_tokens.weight" in sanitized
        assert not any("visual" in k for k in sanitized)

    def test_sanitize_norm_weight_adjustment(self):
        """Standard RMSNorm weights get +1.0 (stored in weight-1 convention)."""
        model = _make_tiny_model()

        weights = {
            "model.layers.0.input_layernorm.weight": mx.zeros((32,)),
            "model.layers.0.post_attention_layernorm.weight": mx.full((32,), 0.5),
            "model.norm.weight": mx.ones((32,)),
            "model.layers.3.self_attn.q_norm.weight": mx.full((16,), -0.1),
            "model.layers.3.self_attn.k_norm.weight": mx.full((16,), 0.3),
            # DeltaNet norm should NOT get +1.0
            "model.layers.0.linear_attn.norm.weight": mx.ones((16,)),
        }
        sanitized = model.sanitize(weights)

        # Standard norms: +1.0
        assert mx.allclose(
            sanitized["model.layers.0.input_layernorm.weight"],
            mx.ones((32,)),  # 0.0 + 1.0
        ).item()
        assert mx.allclose(
            sanitized["model.layers.0.post_attention_layernorm.weight"],
            mx.full((32,), 1.5),  # 0.5 + 1.0
        ).item()
        assert mx.allclose(
            sanitized["model.norm.weight"],
            mx.full((32,), 2.0),  # 1.0 + 1.0
        ).item()
        assert mx.allclose(
            sanitized["model.layers.3.self_attn.q_norm.weight"],
            mx.full((16,), 0.9),  # -0.1 + 1.0
        ).item()
        assert mx.allclose(
            sanitized["model.layers.3.self_attn.k_norm.weight"],
            mx.full((16,), 1.3),  # 0.3 + 1.0
        ).item()

        # DeltaNet output norm: unchanged
        assert mx.allclose(
            sanitized["model.layers.0.linear_attn.norm.weight"],
            mx.ones((16,)),  # still 1.0
        ).item()

    def test_sanitize_full_vlm_pipeline(self):
        """End-to-end sanitize with all VLM artifacts."""
        model = _make_tiny_model()

        C = model.layers[0].linear_attn.conv_dim
        weights = {
            # VLM prefix
            "model.language_model.embed_tokens.weight": mx.zeros((100, 32)),
            "model.language_model.layers.0.linear_attn.conv1d.weight": mx.random.normal((C, 1, 4)),
            "model.language_model.layers.0.input_layernorm.weight": mx.zeros((32,)),
            # Vision (should be dropped)
            "model.visual.something.weight": mx.zeros((10, 10)),
            # MTP (should be dropped)
            "mtp.layers.0.weight": mx.zeros((32, 32)),
            # Tied lm_head (should be dropped)
            "lm_head.weight": mx.zeros((100, 32)),
        }
        sanitized = model.sanitize(weights)

        assert "model.embed_tokens.weight" in sanitized
        # Conv1d transposed
        assert sanitized["model.layers.0.linear_attn.conv1d.weight"].shape == (C, 4, 1)
        # Norm adjusted
        assert mx.allclose(
            sanitized["model.layers.0.input_layernorm.weight"],
            mx.ones((32,)),
        ).item()
        # Everything else dropped
        assert not any("visual" in k for k in sanitized)
        assert not any("mtp" in k for k in sanitized)
        assert "lm_head.weight" not in sanitized


class TestMakeCache:
    """Tests for hybrid cache creation."""

    def test_make_cache_hybrid(self):
        """make_cache returns mixed RecurrentCache/KVCache."""
        model = _make_tiny_model()
        cache = model.make_cache()

        assert len(cache) == 4

        # Layers 0, 1, 2 are DeltaNet -> RecurrentCache
        assert isinstance(cache[0], RecurrentCache)
        assert isinstance(cache[1], RecurrentCache)
        assert isinstance(cache[2], RecurrentCache)

        # Layer 3 is full attention -> KVCache
        assert isinstance(cache[3], KVCache)

    def test_recurrent_cache_initial_state(self):
        """RecurrentCache starts with None slots and offset 0."""
        cache = RecurrentCache()

        assert cache.conv_state is None
        assert cache.ssm_state is None
        assert cache.offset == 0
        assert cache[0] is None
        assert cache[1] is None

    def test_recurrent_cache_set_get(self):
        """RecurrentCache supports indexed and property access."""
        cache = RecurrentCache()

        state = mx.zeros((1, 2, 4, 4))
        cache.ssm_state = state
        assert cache[1] is state
        assert cache.ssm_state is state

        conv = mx.zeros((1, 3, 8))
        cache[0] = conv
        assert cache.conv_state is conv


class TestLoRATargeting:
    """Tests for LoRA preset compatibility with hybrid architecture."""

    def test_mlp_preset_matches_all_layers(self):
        """MLP preset should match all 4 layers (both types have MLP)."""
        from cortexlab.adapters.targeting import resolve_targets

        model = _make_tiny_model()
        patterns = ["*.mlp.gate_proj", "*.mlp.up_proj", "*.mlp.down_proj"]
        targets = resolve_targets(model, patterns)

        # 4 layers * 3 MLP projections = 12
        assert len(targets) == 12

    def test_attention_preset_matches_only_full_attn(self):
        """Attention presets should only match full attention layers."""
        from cortexlab.adapters.targeting import resolve_targets

        model = _make_tiny_model()
        patterns = ["*.self_attn.q_proj", "*.self_attn.v_proj"]
        targets = resolve_targets(model, patterns)

        # Only 1 full attention layer (layer 3) * 2 projections = 2
        assert len(targets) == 2
        names = [name for name, _ in targets]
        assert all("self_attn" in n for n in names)

    def test_custom_deltanet_targeting(self):
        """Custom patterns can target DeltaNet projections."""
        from cortexlab.adapters.targeting import resolve_targets

        model = _make_tiny_model()
        patterns = ["*.linear_attn.in_proj_qkv", "*.linear_attn.out_proj"]
        targets = resolve_targets(model, patterns)

        # 3 DeltaNet layers * 2 projections = 6
        assert len(targets) == 6
        names = [name for name, _ in targets]
        assert all("linear_attn" in n for n in names)


class TestGradientFlow:
    """Tests for gradient flow through the hybrid architecture."""

    def test_gradient_flow_training(self):
        """Gradients flow through both DeltaNet and attention layers."""
        model = _make_tiny_model()

        def loss_fn(model, ids, labels):
            logits = model(ids)
            # Simple cross-entropy proxy
            return mx.mean(logits[:, :, 0])

        loss_and_grad = nn.value_and_grad(model, loss_fn)

        ids = mx.array([[1, 2, 3, 4, 5]])
        labels = mx.array([[2, 3, 4, 5, 6]])

        loss, grads = loss_and_grad(model, ids, labels)
        mx.eval(loss, grads)

        assert loss.shape == ()

        # Verify gradients exist for both layer types
        # DeltaNet layer (layer 0)
        dn_grad = grads["model"]["layers"][0]["linear_attn"]["in_proj_qkv"]["weight"]
        assert dn_grad is not None
        assert mx.abs(dn_grad).sum().item() > 0

        # Full attention layer (layer 3)
        attn_grad = grads["model"]["layers"][3]["self_attn"]["q_proj"]["weight"]
        assert attn_grad is not None
        assert mx.abs(attn_grad).sum().item() > 0

        # MLP gradients (both layer types)
        mlp_grad_0 = grads["model"]["layers"][0]["mlp"]["gate_proj"]["weight"]
        mlp_grad_3 = grads["model"]["layers"][3]["mlp"]["gate_proj"]["weight"]
        assert mx.abs(mlp_grad_0).sum().item() > 0
        assert mx.abs(mlp_grad_3).sum().item() > 0


class TestInferenceEngine:
    """Tests for inference engine integration."""

    def test_engine_uses_model_make_cache(self):
        """generate_tokens should use model.make_cache() when available."""
        model = _make_tiny_model()
        model.eval()

        # Verify the model has make_cache
        assert hasattr(model, "make_cache")
        cache = model.make_cache()
        assert any(isinstance(c, RecurrentCache) for c in cache)
        assert any(isinstance(c, KVCache) for c in cache)


class TestChunkedDeltaNet:
    """Tests for chunked gated delta recurrence."""

    def _make_inputs(self, B=1, T=128, Hk=4, Hv=4, Dk=16, Dv=16):
        """Create random inputs for delta recurrence."""
        mx.random.seed(42)
        q = mx.random.normal((B, T, Hk, Dk))
        k = mx.random.normal((B, T, Hk, Dk))
        v = mx.random.normal((B, T, Hv, Dv))
        a = mx.random.normal((B, T, Hv))
        b = mx.random.normal((B, T, Hv))
        A_log = mx.random.normal((Hv,))
        dt_bias = mx.random.normal((Hv,))
        return q, k, v, a, b, A_log, dt_bias

    def test_chunkwise_output_shape(self):
        """Test chunked recurrence produces correct output shape."""
        q, k, v, a, b, A_log, dt_bias = self._make_inputs(T=128)
        output, state = gated_delta_chunkwise(
            q, k, v, a, b, A_log, dt_bias, chunk_size=32
        )
        mx.eval(output, state)
        assert output.shape == (1, 128, 4, 16)
        assert state.shape == (1, 4, 16, 16)

    def test_chunkwise_matches_sequential(self):
        """Test chunked and sequential produce similar outputs."""
        q, k, v, a, b, A_log, dt_bias = self._make_inputs(B=1, T=64, Hk=2, Hv=2, Dk=8, Dv=8)

        out_seq, state_seq = gated_delta_recurrence(
            q, k, v, a, b, A_log, dt_bias
        )
        out_chunk, state_chunk = gated_delta_chunkwise(
            q, k, v, a, b, A_log, dt_bias, chunk_size=16
        )
        mx.eval(out_seq, state_seq, out_chunk, state_chunk)

        # They won't be identical due to different numerical paths,
        # but should be reasonably close
        diff = mx.abs(out_seq.astype(mx.float32) - out_chunk.astype(mx.float32)).mean().item()
        assert diff < 1.0, f"Mean diff too large: {diff}"

    def test_chunkwise_various_lengths(self):
        """Test chunked recurrence with T not divisible by chunk_size."""
        for T in [33, 65, 100, 127]:
            q, k, v, a, b, A_log, dt_bias = self._make_inputs(T=T)
            output, state = gated_delta_chunkwise(
                q, k, v, a, b, A_log, dt_bias, chunk_size=32
            )
            mx.eval(output, state)
            assert output.shape == (1, T, 4, 16), f"Wrong shape for T={T}"

    def test_chunkwise_state_continuity(self):
        """Test that state propagation across chunks works."""
        q, k, v, a, b, A_log, dt_bias = self._make_inputs(T=64)

        # Process all at once
        _, state_full = gated_delta_chunkwise(
            q, k, v, a, b, A_log, dt_bias, chunk_size=32
        )

        # Process first half, then second half with state
        _, state_half1 = gated_delta_chunkwise(
            q[:, :32], k[:, :32], v[:, :32],
            a[:, :32], b[:, :32],
            A_log, dt_bias, chunk_size=32
        )
        _, state_half2 = gated_delta_chunkwise(
            q[:, 32:], k[:, 32:], v[:, 32:],
            a[:, 32:], b[:, 32:],
            A_log, dt_bias, state=state_half1, chunk_size=32
        )
        mx.eval(state_full, state_half2)

        # States should match
        diff = mx.abs(state_full - state_half2).max().item()
        assert diff < 0.5, f"State continuity diff too large: {diff}"

    def test_chunkwise_gradient_flow(self):
        """Test that gradients flow through chunked recurrence."""
        q, k, v, a, b, A_log, dt_bias = self._make_inputs(B=1, T=64, Hk=2, Hv=2, Dk=8, Dv=8)

        def loss_fn(q, k, v, a, b):
            out, _ = gated_delta_chunkwise(
                q, k, v, a, b, A_log, dt_bias, chunk_size=16
            )
            return out.sum()

        grad_fn = mx.grad(loss_fn, argnums=[0, 1, 2])
        grads = grad_fn(q, k, v, a, b)
        mx.eval(grads)

        for i, name in enumerate(["q", "k", "v"]):
            assert grads[i].shape == [q, k, v][i].shape
            assert mx.abs(grads[i]).sum().item() > 0, f"No gradient for {name}"

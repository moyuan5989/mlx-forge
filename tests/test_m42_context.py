"""M42 Tests: Context & Memory Management — rotating cache, inference memory, overflow."""

from __future__ import annotations

import mlx.core as mx
import pytest

# ─── Test RotatingKVCache ───


class TestRotatingKVCache:
    """Tests for rotating KV cache with sliding window eviction."""

    def test_basic_update(self):
        """Basic update within capacity works like KVCache."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=100, num_keep=0)
        keys = mx.zeros((1, 4, 10, 32))
        values = mx.zeros((1, 4, 10, 32))

        k, v = cache.update_and_fetch(keys, values)
        assert cache.offset == 10
        assert k.shape[2] == 10

    def test_rotation_at_max_size(self):
        """Cache rotates when max_size is exceeded."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=20, num_keep=5)

        # Fill to 15 tokens
        keys1 = mx.ones((1, 2, 15, 16))
        values1 = mx.ones((1, 2, 15, 16))
        cache.update_and_fetch(keys1, values1)
        assert cache.offset == 15

        # Add 10 more — exceeds max_size=20, triggers rotation
        keys2 = mx.ones((1, 2, 10, 16)) * 2
        values2 = mx.ones((1, 2, 10, 16)) * 2
        cache.update_and_fetch(keys2, values2)

        # Should have rotated: offset <= max_size
        assert cache.offset <= 20
        assert cache.has_rotated is True

    def test_num_keep_preserved(self):
        """First num_keep tokens are preserved after rotation."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=20, num_keep=5)

        # Fill with unique values to verify preservation
        keys1 = mx.ones((1, 1, 5, 8)) * 42  # prefix (should be kept)
        values1 = mx.ones((1, 1, 5, 8)) * 42
        cache.update_and_fetch(keys1, values1)

        # Fill rest
        keys2 = mx.ones((1, 1, 14, 8)) * 99
        values2 = mx.ones((1, 1, 14, 8)) * 99
        cache.update_and_fetch(keys2, values2)

        # Trigger rotation
        keys3 = mx.ones((1, 1, 5, 8)) * 77
        values3 = mx.ones((1, 1, 5, 8)) * 77
        k, v = cache.update_and_fetch(keys3, values3)

        # First 5 tokens should still be 42 (preserved prefix)
        prefix_vals = k[0, 0, :5, 0]
        mx.eval(prefix_vals)
        assert all(prefix_vals[i].item() == 42.0 for i in range(5))

    def test_recent_tokens_kept(self):
        """Most recent tokens are kept after rotation (not middle)."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=10, num_keep=0)

        # Add 8 tokens
        keys1 = mx.ones((1, 1, 8, 4))
        values1 = mx.ones((1, 1, 8, 4))
        cache.update_and_fetch(keys1, values1)

        # Add 5 more — exceeds 10, rotation happens
        keys2 = mx.ones((1, 1, 5, 4)) * 2
        values2 = mx.ones((1, 1, 5, 4)) * 2
        k, v = cache.update_and_fetch(keys2, values2)

        # Should have at most 10 tokens
        assert cache.offset <= 10

    def test_trim(self):
        """Trim removes tokens from end."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=50, num_keep=0)
        keys = mx.zeros((1, 2, 20, 16))
        values = mx.zeros((1, 2, 20, 16))
        cache.update_and_fetch(keys, values)

        cache.trim(5)
        assert cache.offset == 15

    def test_reset(self):
        """Reset clears everything."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=50, num_keep=0)
        keys = mx.zeros((1, 2, 10, 16))
        values = mx.zeros((1, 2, 10, 16))
        cache.update_and_fetch(keys, values)

        cache.reset()
        assert cache.offset == 0
        assert cache.keys is None
        assert cache.has_rotated is False

    def test_is_full(self):
        """is_full reports correctly."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=10, num_keep=0)
        assert cache.is_full is False

        keys = mx.zeros((1, 1, 10, 4))
        values = mx.zeros((1, 1, 10, 4))
        cache.update_and_fetch(keys, values)
        assert cache.is_full is True

    def test_small_max_size(self):
        """Small max_size doesn't crash."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=4, num_keep=1)
        keys = mx.zeros((1, 1, 3, 4))
        values = mx.zeros((1, 1, 3, 4))
        cache.update_and_fetch(keys, values)

        # Add more to trigger rotation
        keys2 = mx.zeros((1, 1, 3, 4))
        values2 = mx.zeros((1, 1, 3, 4))
        cache.update_and_fetch(keys2, values2)
        assert cache.offset <= 4

    def test_zero_num_keep(self):
        """num_keep=0 rotates without prefix preservation."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        cache = RotatingKVCache(max_size=5, num_keep=0)
        for _ in range(3):
            keys = mx.zeros((1, 1, 3, 4))
            values = mx.zeros((1, 1, 3, 4))
            cache.update_and_fetch(keys, values)
        assert cache.offset <= 5

    def test_invalid_params(self):
        """Invalid parameters raise errors."""
        from mlx_forge.inference.rotating_cache import RotatingKVCache

        with pytest.raises(ValueError):
            RotatingKVCache(max_size=0)
        with pytest.raises(ValueError):
            RotatingKVCache(max_size=10, num_keep=-1)
        with pytest.raises(ValueError):
            RotatingKVCache(max_size=10, num_keep=10)


# ─── Test make_rotating_cache ───


class TestMakeRotatingCache:
    """Tests for rotating cache factory."""

    def test_creates_correct_count(self):
        from mlx_forge.inference.rotating_cache import make_rotating_cache

        caches = make_rotating_cache(32, max_size=4096, num_keep=256)
        assert len(caches) == 32
        assert all(c.max_size == 4096 for c in caches)
        assert all(c.num_keep == 256 for c in caches)


# ─── Test Inference Memory Estimation ───


class TestInferenceMemory:
    """Tests for inference memory estimation."""

    def test_estimate_fp16(self):
        """fp16 model estimation gives reasonable values."""
        from mlx_forge.models.memory import HardwareProfile, estimate_inference_memory

        hw = HardwareProfile(total_memory_gb=16.0, training_budget_gb=12.0)
        est = estimate_inference_memory("Qwen/Qwen3-0.6B", hardware=hw)
        assert est.model_weights_gb > 0
        assert est.kv_cache_gb > 0
        assert est.total_gb > 0

    def test_estimate_4bit(self):
        """4-bit model uses less memory."""
        from mlx_forge.models.memory import HardwareProfile, estimate_inference_memory

        hw = HardwareProfile(total_memory_gb=16.0, training_budget_gb=12.0)
        fp16 = estimate_inference_memory("Qwen/Qwen3-0.6B", hardware=hw)
        q4 = estimate_inference_memory(
            "Qwen/Qwen3-0.6B", quantization_bits=4, hardware=hw
        )
        assert q4.model_weights_gb < fp16.model_weights_gb

    def test_max_context_computed(self):
        """max_context_that_fits is calculated."""
        from mlx_forge.models.memory import HardwareProfile, estimate_inference_memory

        hw = HardwareProfile(total_memory_gb=16.0, training_budget_gb=12.0)
        est = estimate_inference_memory("Qwen/Qwen3-0.6B", hardware=hw)
        assert est.max_context_that_fits > 0
        assert est.max_context_that_fits % 256 == 0  # rounded to 256

    def test_kv_scales_with_context(self):
        """KV cache memory scales linearly with context length."""
        from mlx_forge.models.memory import HardwareProfile, estimate_inference_memory

        hw = HardwareProfile(total_memory_gb=64.0, training_budget_gb=48.0)
        est_4k = estimate_inference_memory(
            "Qwen/Qwen3-0.6B", context_length=4096, hardware=hw
        )
        est_8k = estimate_inference_memory(
            "Qwen/Qwen3-0.6B", context_length=8192, hardware=hw
        )
        ratio = est_8k.kv_cache_gb / est_4k.kv_cache_gb
        assert 1.8 < ratio < 2.2  # should be ~2x

    def test_fits_check(self):
        """fits flag is computed correctly."""
        from mlx_forge.models.memory import HardwareProfile, estimate_inference_memory

        hw_small = HardwareProfile(total_memory_gb=4.0, training_budget_gb=3.0)
        est = estimate_inference_memory(
            "meta-llama/Llama-3.1-8B", context_length=32768, hardware=hw_small
        )
        assert est.fits is False

    def test_unknown_model(self):
        """Unknown model raises ValueError."""
        from mlx_forge.models.memory import estimate_inference_memory

        with pytest.raises(ValueError, match="Unknown model"):
            estimate_inference_memory("totally-unknown/model-xyz")

    def test_dataclass_fields(self):
        """InferenceMemoryEstimate has expected fields."""
        from mlx_forge.models.memory import InferenceMemoryEstimate

        est = InferenceMemoryEstimate(
            model_weights_gb=1.0, kv_cache_gb=0.5, budget_gb=12.0, max_context_that_fits=8192
        )
        assert est.total_gb == 1.0 + 0.5 + 0.3  # default overhead


# ─── Test Engine Integration ───


class TestEngineIntegration:
    """Tests for context params in engine."""

    def test_generate_steps_accepts_context_params(self):
        """generate_steps accepts context_length and num_keep."""
        import inspect

        from mlx_forge.inference.engine import generate_steps

        sig = inspect.signature(generate_steps)
        assert "context_length" in sig.parameters
        assert "num_keep" in sig.parameters

    def test_make_model_cache_with_num_keep(self):
        """_make_model_cache creates RotatingKVCache when num_keep > 0."""
        from unittest.mock import MagicMock

        from mlx_forge.inference.engine import _make_model_cache

        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(4)]
        del model.make_cache

        cache = _make_model_cache(model, max_size=1024, num_keep=64)

        from mlx_forge.inference.rotating_cache import RotatingKVCache

        assert all(isinstance(c, RotatingKVCache) for c in cache)
        assert all(c.max_size == 1024 for c in cache)
        assert all(c.num_keep == 64 for c in cache)

    def test_make_model_cache_without_num_keep(self):
        """_make_model_cache creates regular KVCache when num_keep=0."""
        from unittest.mock import MagicMock

        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.engine import _make_model_cache

        model = MagicMock()
        model.model = MagicMock()
        model.model.layers = [MagicMock() for _ in range(4)]
        del model.make_cache

        cache = _make_model_cache(model, max_size=1024, num_keep=0)
        assert all(isinstance(c, KVCache) for c in cache)


# ─── Test CLI & Routes Wiring ───


class TestWiring:
    """Tests for context params wired through CLI and routes."""

    def test_serve_cli_has_context_flags(self):
        """Serve command has --context-length and --num-keep flags."""
        from mlx_forge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["serve", "--context-length", "8192", "--num-keep", "256"])
        assert args.context_length == 8192
        assert args.num_keep == 256

    def test_app_factory_accepts_context_params(self):
        """create_serving_app accepts context_length and num_keep."""
        import inspect

        from mlx_forge.serving.app import create_serving_app

        sig = inspect.signature(create_serving_app)
        assert "context_length" in sig.parameters
        assert "num_keep" in sig.parameters

    def test_request_has_num_ctx(self):
        """ChatCompletionRequest has num_ctx and num_keep fields."""
        from mlx_forge.serving.openai_types import ChatCompletionRequest

        req = ChatCompletionRequest(
            model="test", messages=[], num_ctx=8192, num_keep=256
        )
        assert req.num_ctx == 8192
        assert req.num_keep == 256

    def test_context_defaults_set(self):
        """set_context_defaults stores values."""
        from mlx_forge.serving.routes import set_context_defaults

        set_context_defaults(context_length=4096, num_keep=128)
        from mlx_forge.serving import routes

        assert routes._default_context_length == 4096
        assert routes._default_num_keep == 128

        # Reset
        set_context_defaults(0, 0)

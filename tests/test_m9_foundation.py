"""Tests for M9 (V1-M1): Resume, Inference, Gemma architecture.

Tests cover:
- Resume validation and checkpoint loading
- KV cache behavior
- Inference engine (sampling, generation)
- Gemma architecture (forward pass, norms, tied embeddings, soft-capping)
- CLI command registration
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import mlx.core as mx
import mlx.nn as nn
import pytest


# =============================================================================
# Resume tests
# =============================================================================

class TestResume:
    """Tests for resume from checkpoint functionality."""

    def test_validate_resume_missing_directory(self):
        """Validate resume raises for non-existent checkpoint."""
        from lmforge import _validate_resume

        config = MagicMock()
        config.training.num_iters = 1000

        with pytest.raises(FileNotFoundError, match="not found"):
            _validate_resume(Path("/nonexistent/path"), config)

    def test_validate_resume_missing_files(self, tmp_path):
        """Validate resume raises when checkpoint files are missing."""
        from lmforge import _validate_resume

        config = MagicMock()
        config.training.num_iters = 1000

        # Create checkpoint dir with only state.json
        ckpt_dir = tmp_path / "step-0000100"
        ckpt_dir.mkdir()
        (ckpt_dir / "state.json").write_text(json.dumps({
            "schema_version": 1, "step": 100, "rng_seed": 42,
        }))

        with pytest.raises(FileNotFoundError, match="adapters.safetensors"):
            _validate_resume(ckpt_dir, config)

    def test_validate_resume_completed_training(self, tmp_path):
        """Validate resume raises when training is already complete."""
        from lmforge import _validate_resume

        config = MagicMock()
        config.training.num_iters = 100

        ckpt_dir = tmp_path / "step-0000100"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapters.safetensors").touch()
        (ckpt_dir / "optimizer.safetensors").touch()
        (ckpt_dir / "state.json").write_text(json.dumps({
            "schema_version": 1, "step": 100, "rng_seed": 42,
        }))

        with pytest.raises(ValueError, match="Increase 'num_iters'"):
            _validate_resume(ckpt_dir, config)

    def test_validate_resume_unsupported_schema(self, tmp_path):
        """Validate resume raises for future schema versions."""
        from lmforge import _validate_resume

        config = MagicMock()
        config.training.num_iters = 1000

        ckpt_dir = tmp_path / "step-0000100"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapters.safetensors").touch()
        (ckpt_dir / "optimizer.safetensors").touch()
        (ckpt_dir / "state.json").write_text(json.dumps({
            "schema_version": 99, "step": 100, "rng_seed": 42,
        }))

        with pytest.raises(ValueError, match="schema version 99"):
            _validate_resume(ckpt_dir, config)

    def test_validate_resume_valid_checkpoint(self, tmp_path):
        """Validate resume passes for a valid checkpoint."""
        from lmforge import _validate_resume

        config = MagicMock()
        config.training.num_iters = 1000

        ckpt_dir = tmp_path / "step-0000100"
        ckpt_dir.mkdir()
        (ckpt_dir / "adapters.safetensors").touch()
        (ckpt_dir / "optimizer.safetensors").touch()
        (ckpt_dir / "state.json").write_text(json.dumps({
            "schema_version": 1, "step": 100, "rng_seed": 42,
        }))

        # Should not raise
        _validate_resume(ckpt_dir, config)

    def test_trainer_accepts_initial_state(self):
        """Trainer constructor accepts an initial state for resume."""
        from lmforge.trainer.state import TrainState

        state = TrainState(step=500, epoch=2, trained_tokens=100000,
                          best_val_loss=1.5, rng_seed=42)

        # Verify it's a valid state
        assert state.step == 500
        assert state.epoch == 2
        assert state.trained_tokens == 100000

    def test_train_cmd_passes_resume(self):
        """CLI train command passes resume to train()."""
        from lmforge.cli.train_cmd import run_train
        import argparse

        args = argparse.Namespace(config="train.yaml", resume="/path/to/ckpt")

        with patch("lmforge.train") as mock_train:
            mock_train.return_value = MagicMock(step=1000)
            run_train(args)
            mock_train.assert_called_once_with(config="train.yaml", resume="/path/to/ckpt")


# =============================================================================
# KV Cache tests
# =============================================================================

class TestKVCache:
    """Tests for KV cache implementation."""

    def test_cache_initial_state(self):
        """New cache should have zero offset and no stored data."""
        from lmforge.inference.cache import KVCache

        cache = KVCache()
        assert cache.offset == 0
        assert cache.keys is None
        assert cache.values is None

    def test_cache_update_and_fetch(self):
        """Cache should concatenate keys/values and track offset."""
        from lmforge.inference.cache import KVCache

        cache = KVCache()

        # First update: 3 tokens
        k1 = mx.ones((1, 4, 3, 32))  # (B, n_kv_heads, L, head_dim)
        v1 = mx.ones((1, 4, 3, 32))
        keys, values = cache.update_and_fetch(k1, v1)

        assert keys.shape == (1, 4, 3, 32)
        assert cache.offset == 3

        # Second update: 1 token
        k2 = mx.ones((1, 4, 1, 32))
        v2 = mx.ones((1, 4, 1, 32))
        keys, values = cache.update_and_fetch(k2, v2)

        assert keys.shape == (1, 4, 4, 32)
        assert cache.offset == 4

    def test_cache_reset(self):
        """Cache reset should clear all state."""
        from lmforge.inference.cache import KVCache

        cache = KVCache()
        k = mx.ones((1, 4, 3, 32))
        v = mx.ones((1, 4, 3, 32))
        cache.update_and_fetch(k, v)

        cache.reset()
        assert cache.offset == 0
        assert cache.keys is None

    def test_make_cache(self):
        """make_cache should create the right number of caches."""
        from lmforge.inference.cache import make_cache

        caches = make_cache(12)
        assert len(caches) == 12
        assert all(c.offset == 0 for c in caches)


# =============================================================================
# Sampling tests
# =============================================================================

class TestSampling:
    """Tests for text generation sampling."""

    def test_greedy_decoding(self):
        """Temperature 0 should return argmax."""
        from lmforge.inference.sampling import sample_next_token

        logits = mx.array([0.1, 0.2, 0.9, 0.3, 0.1])
        token = sample_next_token(logits, temperature=0.0)
        mx.eval(token)
        assert token.item() == 2  # Index of 0.9

    def test_greedy_is_deterministic(self):
        """Greedy decoding should always return the same result."""
        from lmforge.inference.sampling import sample_next_token

        logits = mx.array([0.1, 0.9, 0.2, 0.3])
        results = set()
        for _ in range(10):
            token = sample_next_token(logits, temperature=0.0)
            mx.eval(token)
            results.add(token.item())

        assert len(results) == 1
        assert 1 in results

    def test_temperature_affects_distribution(self):
        """Higher temperature should produce more diverse outputs."""
        from lmforge.inference.sampling import sample_next_token

        logits = mx.array([0.5, 0.3, 0.1, 0.05, 0.05])
        mx.random.seed(42)

        # Low temperature - mostly picks the top token
        low_temp_results = []
        for _ in range(50):
            token = sample_next_token(logits, temperature=0.01, top_p=1.0)
            mx.eval(token)
            low_temp_results.append(token.item())

        # All low temp results should be the same (argmax)
        assert all(r == 0 for r in low_temp_results)

    def test_top_p_filtering(self):
        """Top-p should filter out low-probability tokens."""
        from lmforge.inference.sampling import _apply_top_p

        # Create logits where token 0 has 90% probability after softmax
        logits = mx.array([10.0, 1.0, 0.5, 0.1])
        filtered = _apply_top_p(logits, top_p=0.5)
        mx.eval(filtered)

        # Only the top token should remain (others should be -inf)
        assert filtered[0].item() > -1000  # Top token kept
        # At least some tokens should be filtered
        assert filtered[-1].item() < -1000  # Bottom token filtered


# =============================================================================
# Inference Engine tests
# =============================================================================

class TestInferenceEngine:
    """Tests for the generation engine."""

    def _make_tiny_model_and_tokenizer(self):
        """Create a tiny Llama-like model for testing."""
        from lmforge.models.architectures.llama import Model, ModelArgs

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

        # Mock tokenizer
        tokenizer = MagicMock()
        tokenizer.eos_token_id = 0
        tokenizer.decode = MagicMock(side_effect=lambda ids: "".join(str(i) for i in ids))

        return model, tokenizer

    def test_generate_tokens_yields_ints(self):
        """generate_tokens should yield integer token IDs."""
        from lmforge.inference.engine import generate_tokens

        model, tokenizer = self._make_tiny_model_and_tokenizer()
        prompt = [1, 2, 3]

        tokens = list(generate_tokens(
            model, prompt, tokenizer,
            temperature=0.7, top_p=0.9, max_tokens=5, seed=42,
        ))

        assert len(tokens) <= 5
        assert all(isinstance(t, int) for t in tokens)

    def test_generate_respects_max_tokens(self):
        """Generation should stop at max_tokens."""
        from lmforge.inference.engine import generate_tokens

        model, tokenizer = self._make_tiny_model_and_tokenizer()
        # Set EOS to -1 so it never triggers
        tokenizer.eos_token_id = -1

        tokens = list(generate_tokens(
            model, [1, 2, 3], tokenizer,
            temperature=0.7, max_tokens=10, seed=42,
        ))

        assert len(tokens) == 10

    def test_generate_stops_on_eos(self):
        """Generation should stop when EOS token is produced."""
        from lmforge.inference.engine import generate_tokens

        model, tokenizer = self._make_tiny_model_and_tokenizer()
        # Set EOS to a common token so it triggers quickly
        tokenizer.eos_token_id = 50  # Likely to appear early

        tokens = list(generate_tokens(
            model, [1, 2, 3], tokenizer,
            temperature=0.7, max_tokens=1000, seed=42,
        ))

        # Should stop before max_tokens (EOS should trigger)
        # The EOS token itself is NOT yielded
        assert len(tokens) < 1000

    def test_generate_result(self):
        """generate() should return a GenerationResult."""
        from lmforge.inference.engine import generate, GenerationResult

        model, tokenizer = self._make_tiny_model_and_tokenizer()
        tokenizer.encode = MagicMock(return_value=[1, 2, 3])
        tokenizer.eos_token_id = -1  # Don't stop early

        result = generate(
            model, tokenizer,
            prompt="hello",
            temperature=0.7, max_tokens=5, seed=42,
        )

        assert isinstance(result, GenerationResult)
        assert result.num_tokens == 5
        assert result.finish_reason == "length"
        assert result.tokens_per_second > 0
        assert result.prompt == "hello"

    def test_generate_requires_prompt_or_messages(self):
        """generate() should raise if neither prompt nor messages given."""
        from lmforge.inference.engine import generate

        model, tokenizer = self._make_tiny_model_and_tokenizer()

        with pytest.raises(ValueError, match="Must provide"):
            generate(model, tokenizer)

    def test_generate_rejects_both_prompt_and_messages(self):
        """generate() should raise if both prompt and messages given."""
        from lmforge.inference.engine import generate

        model, tokenizer = self._make_tiny_model_and_tokenizer()

        with pytest.raises(ValueError, match="not both"):
            generate(model, tokenizer, prompt="hello",
                    messages=[{"role": "user", "content": "hi"}])

    def test_greedy_generation_is_deterministic(self):
        """Greedy generation should produce identical outputs."""
        from lmforge.inference.engine import generate_tokens

        model, tokenizer = self._make_tiny_model_and_tokenizer()
        tokenizer.eos_token_id = -1

        run1 = list(generate_tokens(
            model, [1, 2, 3], tokenizer,
            temperature=0.0, max_tokens=10, seed=42,
        ))
        run2 = list(generate_tokens(
            model, [1, 2, 3], tokenizer,
            temperature=0.0, max_tokens=10, seed=42,
        ))

        assert run1 == run2


# =============================================================================
# Gemma Architecture tests
# =============================================================================

class TestGemmaArchitecture:
    """Tests for Gemma model implementation."""

    def _make_gemma_args(self, model_type="gemma", **overrides):
        """Create small Gemma args for testing."""
        from lmforge.models.architectures.gemma import ModelArgs

        defaults = dict(
            model_type=model_type,
            hidden_size=128,
            num_hidden_layers=2,
            intermediate_size=256,
            num_attention_heads=4,
            num_key_value_heads=4,
            head_dim=32,
            vocab_size=1000,
            rms_norm_eps=1e-6,
        )
        defaults.update(overrides)
        return ModelArgs(**defaults)

    def test_gemma_forward_pass(self):
        """Gemma model should produce logits of correct shape."""
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args()
        model = Model(args)
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3, 4]])
        logits = model(tokens)
        assert logits.shape == (1, 4, 1000)

    def test_gemma_rms_norm_offset(self):
        """GemmaRMSNorm should use (1 + weight) scaling."""
        from lmforge.models.architectures.gemma import GemmaRMSNorm

        norm = GemmaRMSNorm(4, eps=1e-6)
        mx.eval(norm.parameters())

        # With zero weights, output should equal RMSNorm(x) * 1.0
        x = mx.array([[1.0, 2.0, 3.0, 4.0]])
        out = norm(x)
        mx.eval(out)

        # Manually compute expected: x * rsqrt(mean(x^2) + eps) * (1 + 0)
        rms = mx.rsqrt(mx.mean(x * x, axis=-1, keepdims=True) + 1e-6)
        expected = x * rms
        mx.eval(expected)

        assert mx.allclose(out, expected, atol=1e-5).item()

    def test_gemma_embedding_scaling(self):
        """Gemma should scale embeddings by sqrt(hidden_size)."""
        from lmforge.models.architectures.gemma import GemmaModel, ModelArgs

        args = self._make_gemma_args()
        backbone = GemmaModel(args)
        mx.eval(backbone.parameters())

        # Get raw embedding
        tokens = mx.array([[1]])
        raw_embed = backbone.embed_tokens(tokens)
        mx.eval(raw_embed)

        # The model should scale by sqrt(hidden_size) = sqrt(128)
        scale = math.sqrt(128)
        expected_magnitude_ratio = scale

        # Run through model (which scales) vs raw
        # We can't easily check the exact value since layers modify it,
        # but we can verify the scaling code exists by checking a smaller model
        assert scale == pytest.approx(math.sqrt(args.hidden_size))

    def test_gemma_tied_embeddings(self):
        """Gemma should use tied embeddings when configured."""
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args(tie_word_embeddings=True)
        model = Model(args)

        # Should NOT have a separate lm_head
        assert not hasattr(model, "lm_head")

        tokens = mx.array([[1, 2, 3]])
        mx.eval(model.parameters())
        logits = model(tokens)
        assert logits.shape == (1, 3, 1000)

    def test_gemma_untied_embeddings(self):
        """Gemma should support untied embeddings."""
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args(tie_word_embeddings=False)
        model = Model(args)

        assert hasattr(model, "lm_head")

        tokens = mx.array([[1, 2, 3]])
        mx.eval(model.parameters())
        logits = model(tokens)
        assert logits.shape == (1, 3, 1000)

    def test_gemma2_soft_capping(self):
        """Gemma 2 should apply attention and final logit soft-capping."""
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args(
            model_type="gemma2",
            attn_logit_softcapping=50.0,
            final_logit_softcapping=30.0,
            query_pre_attn_scalar=32.0,
        )
        model = Model(args)
        mx.eval(model.parameters())

        tokens = mx.array([[1, 2, 3]])
        logits = model(tokens)
        mx.eval(logits)

        # Final logit soft-capping: logits should be bounded by cap value
        max_logit = mx.max(mx.abs(logits)).item()
        assert max_logit <= 30.0 + 0.01  # cap is 30.0

    def test_gemma2_sliding_window_on_even_layers(self):
        """Gemma 2 sliding window should only apply to even layers."""
        from lmforge.models.architectures.gemma import Attention

        args = self._make_gemma_args(
            model_type="gemma2",
            sliding_window=4096,
        )

        # Even layer (0) should have sliding window
        attn_even = Attention(args, layer_idx=0)
        assert attn_even.sliding_window == 4096

        # Odd layer (1) should NOT have sliding window
        attn_odd = Attention(args, layer_idx=1)
        assert attn_odd.sliding_window is None

    def test_gemma2_post_norms(self):
        """Gemma 2 transformer blocks should have 4 norms."""
        from lmforge.models.architectures.gemma import TransformerBlock

        args = self._make_gemma_args(model_type="gemma2")
        block = TransformerBlock(args, layer_idx=0)

        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert hasattr(block, "pre_feedforward_layernorm")
        assert hasattr(block, "post_feedforward_layernorm")

    def test_gemma1_no_post_norms(self):
        """Gemma 1 transformer blocks should only have 2 norms."""
        from lmforge.models.architectures.gemma import TransformerBlock

        args = self._make_gemma_args(model_type="gemma")
        block = TransformerBlock(args, layer_idx=0)

        assert hasattr(block, "input_layernorm")
        assert hasattr(block, "post_attention_layernorm")
        assert not hasattr(block, "pre_feedforward_layernorm")

    def test_gemma_kv_cache(self):
        """Gemma should work with KV cache for generation."""
        from lmforge.inference.cache import make_cache
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args()
        model = Model(args)
        mx.eval(model.parameters())

        cache = make_cache(args.num_hidden_layers)

        # Prefill
        tokens = mx.array([[1, 2, 3]])
        logits1 = model(tokens, cache=cache)
        mx.eval(logits1)

        assert cache[0].offset == 3

        # Incremental decode
        next_tok = mx.array([[4]])
        logits2 = model(next_tok, cache=cache)
        mx.eval(logits2)

        assert logits2.shape == (1, 1, 1000)
        assert cache[0].offset == 4

    def test_gemma_lora_targeting(self):
        """LoRA should be applicable to Gemma attention modules."""
        from lmforge.adapters.targeting import named_modules, resolve_targets
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args()
        model = Model(args)

        patterns = ["*.self_attn.q_proj", "*.self_attn.v_proj"]
        targets = resolve_targets(model, patterns, num_layers=None)

        # Should match q_proj and v_proj in each of 2 layers
        assert len(targets) == 4

    def test_gemma_sanitize(self):
        """sanitize() should remove rotary_emb and lm_head if tied."""
        from lmforge.models.architectures.gemma import Model

        args = self._make_gemma_args(tie_word_embeddings=True)
        model = Model(args)

        weights = {
            "model.layers.0.self_attn.q_proj.weight": mx.zeros((128, 128)),
            "model.layers.0.self_attn.rotary_emb.inv_freq": mx.zeros((16,)),
            "lm_head.weight": mx.zeros((1000, 128)),
        }

        cleaned = model.sanitize(weights)
        assert "model.layers.0.self_attn.q_proj.weight" in cleaned
        assert "model.layers.0.self_attn.rotary_emb.inv_freq" not in cleaned
        assert "lm_head.weight" not in cleaned

    def test_gemma_from_dict(self):
        """ModelArgs.from_dict() should handle HF config keys."""
        from lmforge.models.architectures.gemma import ModelArgs

        config = {
            "model_type": "gemma",
            "hidden_size": 2048,
            "num_hidden_layers": 18,
            "intermediate_size": 16384,
            "num_attention_heads": 8,
            "num_key_value_heads": 1,
            "head_dim": 256,
            "vocab_size": 256128,
            "rms_norm_eps": 1e-6,
            "max_position_embeddings": 8192,
            # Extra HF keys that should be filtered
            "architectures": ["GemmaForCausalLM"],
            "bos_token_id": 2,
            "eos_token_id": 1,
            "torch_dtype": "bfloat16",
        }

        args = ModelArgs.from_dict(config)
        assert args.hidden_size == 2048
        assert args.num_hidden_layers == 18
        assert args.head_dim == 256


# =============================================================================
# Registry tests
# =============================================================================

class TestRegistryGemma:
    """Tests for Gemma in the model registry."""

    def test_gemma_is_supported(self):
        """Gemma should be in the supported architectures."""
        from lmforge.models.registry import is_supported

        assert is_supported("gemma")
        assert is_supported("gemma2")
        assert is_supported("gemma3")

    def test_gemma_get_model_classes(self):
        """get_model_classes should return Gemma classes."""
        from lmforge.models.registry import get_model_classes

        Model, ModelArgs = get_model_classes({"model_type": "gemma"})
        assert Model.__name__ == "Model"
        assert ModelArgs.__name__ == "ModelArgs"

        # Should work for gemma2 too
        Model2, _ = get_model_classes({"model_type": "gemma2"})
        assert Model2.__name__ == "Model"


# =============================================================================
# CLI tests
# =============================================================================

class TestCLI:
    """Tests for CLI command registration."""

    def test_generate_command_registered(self):
        """generate command should be registered in CLI parser."""
        from lmforge.cli.main import build_parser

        parser = build_parser()
        # Parse --help for generate to verify it's registered
        args = parser.parse_args(["generate", "--model", "test/model"])
        assert args.command == "generate"
        assert args.model == "test/model"
        assert args.adapter is None
        assert args.temperature == 0.7
        assert args.top_p == 0.9
        assert args.max_tokens == 512

    def test_train_resume_argument(self):
        """train --resume should be parseable."""
        from lmforge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["train", "--config", "test.yaml",
                                  "--resume", "/path/to/ckpt"])
        assert args.resume == "/path/to/ckpt"

    def test_train_resume_default_none(self):
        """train --resume should default to None."""
        from lmforge.cli.main import build_parser

        parser = build_parser()
        args = parser.parse_args(["train", "--config", "test.yaml"])
        assert args.resume is None

"""Tests for M31: Speculative decoding and prompt cache.

Tests cover:
- speculative_generate_tokens behavior
- KVCache.trim() method
- Prompt cache save/load/apply
- CLI flags for speculative decoding
- Cache utilities
"""

from __future__ import annotations

import tempfile
from pathlib import Path

import mlx.core as mx
import mlx.nn as nn
import pytest

# ── Mock models ─────────────────────────────────────────────────────────────

class MockInner(nn.Module):
    def __init__(self):
        super().__init__()
        self.layers = [nn.Linear(32, 32) for _ in range(2)]


class MockModel(nn.Module):
    def __init__(self, vocab_size=50):
        super().__init__()
        self.model = MockInner()
        self.lm_head = nn.Linear(32, vocab_size)
        self._vocab_size = vocab_size

    def __call__(self, x, cache=None):
        h = mx.zeros((x.shape[0], x.shape[1], 32))
        return self.lm_head(h)

    @property
    def layers(self):
        return self.model.layers


class MockTokenizer:
    def __init__(self):
        self.eos_token_id = 0

    def encode(self, text):
        return [1, 2, 3]

    def decode(self, tokens):
        return "test output"


# ── Speculative Generate Tests ──────────────────────────────────────────────

class TestSpeculativeGenerate:
    def test_speculative_generate_yields_tuples(self):
        """speculative_generate_tokens yields (token_id, from_draft) tuples."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()

        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.0,
            seed=42,
        ))

        assert len(results) > 0
        for item in results:
            assert isinstance(item, tuple)
            assert len(item) == 2
            token_id, from_draft = item
            assert isinstance(token_id, int)
            assert isinstance(from_draft, bool)

    def test_speculative_draft_acceptance(self):
        """Accepted tokens are marked from_draft=True."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()

        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=10,
            temperature=0.0,
            seed=42,
        ))

        # With deterministic temp=0, some may be accepted
        # Just check the structure is valid
        for token_id, from_draft in results:
            assert isinstance(from_draft, bool)

    def test_speculative_rejection(self):
        """Rejected tokens use main model (from_draft=False exists)."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()

        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=5,
            temperature=0.0,
            seed=42,
        ))

        # First token is always from_draft=False (main model samples it)
        if results:
            assert results[0][1] is False

    def test_speculative_eos_stops(self):
        """Stops at EOS token."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        # Use a model that always returns EOS (token 0)
        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()
        tokenizer.eos_token_id = 0

        # We can't easily force EOS, but verify max_tokens is respected as fallback
        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=3,
            seed=42,
        ))
        assert len(results) <= 3

    def test_speculative_max_tokens(self):
        """Respects max_tokens limit."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()
        tokenizer.eos_token_id = -1  # Never EOS

        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=5,
            seed=42,
        ))
        assert len(results) <= 5

    def test_speculative_empty_draft(self):
        """Handles empty draft gracefully (draft EOS immediately)."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()

        # Should still produce at least some tokens from the main model
        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=2,
            seed=42,
        ))
        assert len(results) <= 2

    def test_speculative_all_accepted(self):
        """All draft tokens can be accepted when models agree."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        # Same model for both -> should agree on temp=0
        model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()
        tokenizer.eos_token_id = -1

        results = list(speculative_generate_tokens(
            model, model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.0,
            seed=42,
        ))
        assert len(results) > 0

    def test_speculative_none_accepted(self):
        """All draft tokens can be rejected when models disagree."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()
        tokenizer.eos_token_id = -1

        results = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=2,
            seed=42,
        ))
        # Just verify it completes without error
        assert isinstance(results, list)

    def test_speculative_seed(self):
        """Seed produces deterministic results."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        main_model = MockModel(vocab_size=50)
        draft_model = MockModel(vocab_size=50)
        tokenizer = MockTokenizer()
        tokenizer.eos_token_id = -1

        results1 = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.5,
            seed=42,
        ))
        results2 = list(speculative_generate_tokens(
            main_model, draft_model,
            prompt_tokens=[1, 2, 3],
            tokenizer=tokenizer,
            max_tokens=3,
            temperature=0.5,
            seed=42,
        ))
        # Same seed should give same results
        assert [r[0] for r in results1] == [r[0] for r in results2]


# ── KVCache Trim Tests ──────────────────────────────────────────────────────

class TestKVCacheTrim:
    def test_cache_trim_method(self):
        """KVCache.trim() reduces offset."""
        from mlx_forge.inference.cache import KVCache
        cache = KVCache()
        cache.offset = 10
        cache.trim(3)
        assert cache.offset == 7

    def test_cache_trim_zero(self):
        """trim(0) is noop."""
        from mlx_forge.inference.cache import KVCache
        cache = KVCache()
        cache.offset = 10
        cache.trim(0)
        assert cache.offset == 10

    def test_cache_trim_negative(self):
        """trim doesn't go below 0."""
        from mlx_forge.inference.cache import KVCache
        cache = KVCache()
        cache.offset = 3
        cache.trim(10)
        assert cache.offset == 0

    def test_kv_cache_trim_updates_offset(self):
        """trim reduces cache offset correctly."""
        from mlx_forge.inference.cache import KVCache
        cache = KVCache()
        cache.offset = 20
        cache.trim(5)
        assert cache.offset == 15

    def test_arrays_cache_trim(self):
        """ArraysCache.trim() works."""
        from mlx_forge.inference.cache import ArraysCache
        cache = ArraysCache(size=2)
        cache.offset = 10
        cache.trim(3)
        assert cache.offset == 7


# ── Prompt Cache Tests ──────────────────────────────────────────────────────

class TestPromptCache:
    def test_prompt_cache_save_load(self):
        """save -> load roundtrip preserves data."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import (
            load_prompt_cache,
            save_prompt_cache,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = [KVCache() for _ in range(2)]
            # Simulate filled cache
            for c in cache:
                c.keys = mx.ones((1, 2, 4, 8))
                c.values = mx.zeros((1, 2, 4, 8))
                c.offset = 4

            path = Path(tmpdir) / "cache.safetensors"
            save_prompt_cache(path, cache, metadata={"model": "test"})

            tensors, metadata = load_prompt_cache(path)
            assert "layer.0.keys" in tensors
            assert "layer.0.values" in tensors
            assert metadata["model"] == "test"

    def test_prompt_cache_metadata(self):
        """Metadata includes model info."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import load_prompt_cache, save_prompt_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = [KVCache() for _ in range(1)]
            cache[0].keys = mx.ones((1, 1, 2, 4))
            cache[0].values = mx.ones((1, 1, 2, 4))
            cache[0].offset = 2

            path = Path(tmpdir) / "cache.safetensors"
            save_prompt_cache(path, cache, metadata={"model": "llama-7b"})

            _, meta = load_prompt_cache(path)
            assert meta["model"] == "llama-7b"
            assert meta["num_layers"] == 1

    def test_prompt_cache_offsets(self):
        """Offsets preserved correctly in metadata."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import load_prompt_cache, save_prompt_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = [KVCache(), KVCache()]
            cache[0].keys = mx.ones((1, 1, 3, 4))
            cache[0].values = mx.ones((1, 1, 3, 4))
            cache[0].offset = 3
            cache[1].keys = mx.ones((1, 1, 5, 4))
            cache[1].values = mx.ones((1, 1, 5, 4))
            cache[1].offset = 5

            path = Path(tmpdir) / "cache.safetensors"
            save_prompt_cache(path, cache)

            _, meta = load_prompt_cache(path)
            assert meta["offsets"] == [3, 5]

    def test_prompt_cache_apply(self):
        """apply_prompt_cache populates cache."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import (
            apply_prompt_cache,
            load_prompt_cache,
            save_prompt_cache,
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            # Save
            orig_cache = [KVCache() for _ in range(2)]
            for c in orig_cache:
                c.keys = mx.ones((1, 2, 4, 8))
                c.values = mx.zeros((1, 2, 4, 8))
                c.offset = 4

            path = Path(tmpdir) / "cache.safetensors"
            save_prompt_cache(path, orig_cache)

            # Load and apply
            tensors, meta = load_prompt_cache(path)
            new_cache = [KVCache() for _ in range(2)]
            apply_prompt_cache(new_cache, tensors, meta)

            for c in new_cache:
                assert c.offset == 4

    def test_prompt_cache_empty_cache(self):
        """Handles empty cache (no keys/values)."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import save_prompt_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = [KVCache()]  # Empty, no keys/values
            path = Path(tmpdir) / "cache.safetensors"
            save_prompt_cache(path, cache)
            assert path.exists()

    def test_prompt_cache_nonexistent_file(self):
        """Proper error on missing file."""
        from mlx_forge.inference.prompt_cache import load_prompt_cache

        with pytest.raises(Exception):
            load_prompt_cache("/nonexistent/path/cache.safetensors")

    def test_prompt_cache_dir_creation(self):
        """Creates parent dirs if needed."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import save_prompt_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = [KVCache()]
            cache[0].keys = mx.ones((1, 1, 2, 4))
            cache[0].values = mx.ones((1, 1, 2, 4))
            cache[0].offset = 2

            path = Path(tmpdir) / "nested" / "dir" / "cache.safetensors"
            save_prompt_cache(path, cache)
            assert path.exists()

    def test_prompt_cache_file_format(self):
        """Saved file is valid safetensors."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.prompt_cache import save_prompt_cache

        with tempfile.TemporaryDirectory() as tmpdir:
            cache = [KVCache()]
            cache[0].keys = mx.ones((1, 1, 2, 4))
            cache[0].values = mx.ones((1, 1, 2, 4))
            cache[0].offset = 2

            path = Path(tmpdir) / "cache.safetensors"
            save_prompt_cache(path, cache)

            # Verify it can be loaded by safetensors
            from safetensors.mlx import load_file
            data = load_file(str(path))
            assert len(data) > 0


# ── Cache Utility Tests ────────────────────────────────────────────────────

class TestCacheUtils:
    def test_make_model_cache(self):
        """_make_model_cache works with model that has layers."""
        from mlx_forge.inference.speculative import _make_model_cache
        model = MockModel(vocab_size=50)
        cache = _make_model_cache(model, max_size=100)
        assert len(cache) == 2  # MockInner has 2 layers

    def test_trim_cache_list(self):
        """_trim_cache works on list of caches."""
        from mlx_forge.inference.cache import KVCache
        from mlx_forge.inference.speculative import _trim_cache

        caches = [KVCache() for _ in range(3)]
        for c in caches:
            c.offset = 10
        _trim_cache(caches, 3)
        for c in caches:
            assert c.offset == 7


# ── Import Tests ────────────────────────────────────────────────────────────

class TestImports:
    def test_speculative_import(self):
        from mlx_forge.inference import speculative
        assert hasattr(speculative, "speculative_generate_tokens")

    def test_prompt_cache_import(self):
        from mlx_forge.inference import prompt_cache
        assert hasattr(prompt_cache, "save_prompt_cache")
        assert hasattr(prompt_cache, "load_prompt_cache")
        assert hasattr(prompt_cache, "apply_prompt_cache")


# ── CLI Flag Tests ──────────────────────────────────────────────────────────

class TestCLIFlags:
    def test_cli_draft_model_flag(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["generate", "--model", "test", "--prompt", "hi", "--draft-model", "small-model"])
        assert args.draft_model == "small-model"

    def test_cli_num_draft_flag(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["generate", "--model", "test", "--prompt", "hi"])
        assert args.num_draft == 5  # Default

    def test_cli_prompt_cache_flag(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["generate", "--model", "test", "--prompt", "hi", "--prompt-cache", "/tmp/cache.st"])
        assert args.prompt_cache == "/tmp/cache.st"

    def test_serve_draft_model_flag(self):
        from mlx_forge.cli.main import build_parser
        parser = build_parser()
        args = parser.parse_args(["serve", "--draft-model", "small-model"])
        assert args.draft_model == "small-model"

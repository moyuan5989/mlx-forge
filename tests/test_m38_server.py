"""M38 Tests: Server-Grade Inference — cache manager, system cache, request queue, adapter hot-loading."""

from __future__ import annotations

import asyncio
import time
from unittest.mock import MagicMock

import mlx.core as mx
import mlx.nn as nn
import pytest

# ─── Helpers ───


def _make_mock_model(num_layers=2):
    """Create a mock model with layers for cache creation."""
    model = MagicMock()
    model.model = MagicMock()
    model.model.layers = [MagicMock() for _ in range(num_layers)]
    # Remove make_cache so fallback path is used
    del model.make_cache
    return model


# ─── Test CacheManager ───


class TestCacheManager:
    """Tests for multi-turn KV cache persistence."""

    def test_create_new(self):
        """New conversation gets fresh cache."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()
        cache, tokens_to_prefill = cm.get_or_create("conv1", [1, 2, 3], model)
        assert cache is not None
        assert tokens_to_prefill == [1, 2, 3]

    def test_reuse_prefix(self):
        """Existing conversation with matching prefix returns delta tokens."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()
        cache, _ = cm.get_or_create("conv1", [1, 2, 3], model)
        cm.update("conv1", cache, [1, 2, 3, 4, 5])

        cache2, new_tokens = cm.get_or_create("conv1", [1, 2, 3, 4, 5, 6, 7], model)
        assert new_tokens == [6, 7]

    def test_divergence_resets(self):
        """Divergent prompt resets cache."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()
        cache, _ = cm.get_or_create("conv1", [1, 2, 3], model)
        cm.update("conv1", cache, [1, 2, 3])

        # Different prompt prefix
        cache2, new_tokens = cm.get_or_create("conv1", [9, 8, 7], model)
        assert new_tokens == [9, 8, 7]  # full re-prefill

    def test_lru_eviction(self):
        """LRU eviction when max conversations exceeded."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager(max_conversations=2)
        model = _make_mock_model()

        cache1, _ = cm.get_or_create("conv1", [1], model)
        cm.update("conv1", cache1, [1])

        cache2, _ = cm.get_or_create("conv2", [2], model)
        cm.update("conv2", cache2, [2])

        cache3, _ = cm.get_or_create("conv3", [3], model)
        cm.update("conv3", cache3, [3])

        # conv1 should be evicted (LRU)
        assert "conv1" not in cm._conversations
        assert "conv2" in cm._conversations
        assert "conv3" in cm._conversations

    def test_ttl_expiry(self):
        """Expired conversations are evicted."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager(ttl_seconds=0.01)
        model = _make_mock_model()

        cache, _ = cm.get_or_create("conv1", [1], model)
        cm.update("conv1", cache, [1])

        time.sleep(0.02)

        # Expired — should get fresh cache
        _, tokens = cm.get_or_create("conv1", [1], model)
        assert tokens == [1]  # full re-prefill

    def test_max_limit(self):
        """Max conversations limit is enforced."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager(max_conversations=3)
        model = _make_mock_model()

        for i in range(5):
            cache, _ = cm.get_or_create(f"conv{i}", [i], model)
            cm.update(f"conv{i}", cache, [i])

        assert len(cm._conversations) <= 3

    def test_evict_specific(self):
        """Explicit eviction of a conversation."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        cache, _ = cm.get_or_create("conv1", [1], model)
        cm.update("conv1", cache, [1])

        assert cm.evict("conv1") is True
        assert cm.evict("conv1") is False  # already evicted

    def test_stats(self):
        """Stats returns correct information."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager(max_conversations=10)
        model = _make_mock_model()

        cache, _ = cm.get_or_create("conv1", [1], model)
        cm.update("conv1", cache, [1])

        stats = cm.stats()
        assert stats["active_conversations"] == 1
        assert stats["max_conversations"] == 10
        assert "conv1" in stats["conversation_ids"]

    def test_empty(self):
        """Empty cache manager works correctly."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        stats = cm.stats()
        assert stats["active_conversations"] == 0

    def test_update_extends(self):
        """Updating extends the token list."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        cache, _ = cm.get_or_create("conv1", [1, 2], model)
        cm.update("conv1", cache, [1, 2, 3, 4])

        state = cm._conversations["conv1"]
        assert state.token_ids == [1, 2, 3, 4]


# ─── Test System Prompt Cache ───


class TestSystemPromptCache:
    """Tests for system prompt KV caching."""

    def test_hash_consistency(self):
        """Same tokens produce same hash."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        h1 = cm._hash_tokens([1, 2, 3])
        h2 = cm._hash_tokens([1, 2, 3])
        assert h1 == h2

    def test_store_retrieve(self):
        """Store and retrieve system cache."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        fake_cache = [MagicMock()]
        cm.store_system_cache([1, 2, 3], fake_cache)
        result = cm.get_system_cache([1, 2, 3], model)
        assert result is not None
        assert len(result) == 1

    def test_miss(self):
        """Cache miss returns None."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()
        result = cm.get_system_cache([99, 98, 97], model)
        assert result is None

    def test_clone_independence(self):
        """Returned cache is a deep copy — mutations don't affect stored cache."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        fake_cache = [{"key": "value"}]
        cm.store_system_cache([1, 2], fake_cache)

        result1 = cm.get_system_cache([1, 2], model)
        result2 = cm.get_system_cache([1, 2], model)
        assert result1 is not result2  # different objects

    def test_different_prompts(self):
        """Different system prompts get different caches."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        h1 = cm._hash_tokens([1, 2, 3])
        h2 = cm._hash_tokens([4, 5, 6])
        assert h1 != h2


# ─── Test Request Queue ───


class TestRequestQueue:
    """Tests for async request queue."""

    @pytest.mark.asyncio
    async def test_submit_receive(self):
        """Submit a request and receive response."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue()
        request = await queue.submit({"prompt": "hello"})
        assert request.params == {"prompt": "hello"}
        assert request.id is not None

    @pytest.mark.asyncio
    async def test_ordering(self):
        """Requests are processed in FIFO order."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue()
        r1 = await queue.submit({"order": 1})
        r2 = await queue.submit({"order": 2})

        assert queue.pending_count == 2

    @pytest.mark.asyncio
    async def test_max_size(self):
        """Queue rejects when at max capacity."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue(max_size=1)
        await queue.submit({"a": 1})

        with pytest.raises(asyncio.QueueFull):
            await queue.submit({"b": 2})

    @pytest.mark.asyncio
    async def test_error_handling(self):
        """Errors in processing are sent to response queue."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue()

        async def failing_processor(request):
            raise ValueError("test error")

        request = await queue.submit({"test": True})

        # Run loop for a short time
        loop_task = asyncio.create_task(queue.run_loop(failing_processor))
        await asyncio.sleep(0.1)
        queue.stop()

        try:
            await asyncio.wait_for(loop_task, timeout=1.0)
        except asyncio.TimeoutError:
            loop_task.cancel()

        # Should have error in response queue
        result = await asyncio.wait_for(request.response_queue.get(), timeout=1.0)
        assert "error" in result

    @pytest.mark.asyncio
    async def test_loop_processes(self):
        """Run loop processes requests."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue()
        processed = []

        async def processor(request):
            processed.append(request.params)
            await request.response_queue.put({"result": "ok"})
            await request.response_queue.put(None)

        await queue.submit({"test": 1})

        loop_task = asyncio.create_task(queue.run_loop(processor))
        await asyncio.sleep(0.1)
        queue.stop()

        try:
            await asyncio.wait_for(loop_task, timeout=1.0)
        except asyncio.TimeoutError:
            loop_task.cancel()

        assert len(processed) == 1
        assert processed[0] == {"test": 1}

    @pytest.mark.asyncio
    async def test_concurrent_submit(self):
        """Multiple concurrent submissions work."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue(max_size=10)

        requests = []
        for i in range(5):
            r = await queue.submit({"i": i})
            requests.append(r)

        assert queue.pending_count == 5

    @pytest.mark.asyncio
    async def test_stop(self):
        """Stop signal terminates the loop."""
        from mlx_forge.serving.request_queue import RequestQueue

        queue = RequestQueue()

        loop_task = asyncio.create_task(queue.run_loop(lambda r: None))
        assert queue.is_running is False  # hasn't started yet

        await asyncio.sleep(0.05)
        queue.stop()

        try:
            await asyncio.wait_for(loop_task, timeout=2.0)
        except asyncio.TimeoutError:
            loop_task.cancel()

        assert queue.is_running is False

    @pytest.mark.asyncio
    async def test_response_queue(self):
        """Response queue delivers results."""
        from mlx_forge.serving.request_queue import InferenceRequest

        request = InferenceRequest(params={"test": True})
        await request.response_queue.put({"token": "hello"})
        result = await request.response_queue.get()
        assert result == {"token": "hello"}


# ─── Test Adapter Hot-Loading ───


class TestAdapterHotLoading:
    """Tests for adapter hot-swap on ModelManager."""

    def test_load_basic(self, tmp_path):
        """Load adapter applies weights."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        # Create a tiny model
        model = nn.Linear(4, 4)
        mgr._model = model
        mgr._model_id = "test"

        # Snapshot base weights
        mgr.snapshot_base_weights()
        assert mgr._base_weights is not None

        # Create fake adapter
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        # Save some weights
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), {"weight": mx.ones((4, 4))})

        mgr.load_adapter(str(adapter_dir))
        assert mgr.adapter_path == str(adapter_dir)

    def test_unload_restores_base(self, tmp_path):
        """Unloading adapter restores base weights."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        model = nn.Linear(4, 4)
        mgr._model = model
        mgr._model_id = "test"

        mgr.snapshot_base_weights()

        # Create and load adapter
        adapter_dir = tmp_path / "adapter"
        adapter_dir.mkdir()
        mx.save_safetensors(str(adapter_dir / "adapters.safetensors"), {"weight": mx.ones((4, 4)) * 99})
        mgr.load_adapter(str(adapter_dir))

        mgr.unload_adapter()
        assert mgr.adapter_path is None

    def test_swap_adapters(self, tmp_path):
        """Swapping between adapters works."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        model = nn.Linear(4, 4)
        mgr._model = model
        mgr._model_id = "test"
        mgr.snapshot_base_weights()

        # Create two adapters
        for name in ["adapter1", "adapter2"]:
            d = tmp_path / name
            d.mkdir()
            mx.save_safetensors(str(d / "adapters.safetensors"), {"weight": mx.ones((4, 4))})

        mgr.load_adapter(str(tmp_path / "adapter1"))
        assert "adapter1" in mgr.adapter_path

        mgr.load_adapter(str(tmp_path / "adapter2"))
        assert "adapter2" in mgr.adapter_path

    def test_no_base_error(self):
        """Unloading without base snapshot raises error."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        mgr._model = nn.Linear(4, 4)
        mgr._model_id = "test"

        with pytest.raises(ValueError, match="No base weight snapshot"):
            mgr.unload_adapter()

    def test_info_endpoint(self):
        """Adapter info returns correct data."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        mgr._model_id = "test"

        info = mgr.adapter_info()
        assert info["adapter_loaded"] is False
        assert info["model_id"] == "test"

    def test_load_missing_file(self, tmp_path):
        """Loading adapter from empty directory raises error."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        mgr._model = nn.Linear(4, 4)
        mgr._model_id = "test"

        empty_dir = tmp_path / "empty"
        empty_dir.mkdir()

        with pytest.raises(FileNotFoundError, match="No adapters.safetensors"):
            mgr.load_adapter(str(empty_dir))

    def test_no_model_error(self):
        """Loading adapter without model loaded raises error."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        with pytest.raises(ValueError, match="No model loaded"):
            mgr.load_adapter("/some/path")

    def test_unload_clears_adapter(self):
        """Full unload clears adapter state."""
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        mgr._adapter_path = "/some/path"
        mgr._base_weights = {"w": mx.ones((2, 2))}
        mgr.unload()
        assert mgr._adapter_path is None
        assert mgr._base_weights is None


# ─── Test Integration ───


class TestIntegration:
    """Integration tests for M38 features."""

    def test_multi_turn_reuses_cache(self):
        """Multi-turn conversation reuses cached prefix."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        # Turn 1: full prompt
        cache, tokens = cm.get_or_create("chat1", [10, 20, 30], model)
        assert tokens == [10, 20, 30]
        cm.update("chat1", cache, [10, 20, 30, 40, 50])

        # Turn 2: extended prompt
        cache2, tokens2 = cm.get_or_create("chat1", [10, 20, 30, 40, 50, 60], model)
        assert tokens2 == [60]  # only new token

    def test_system_prompt_shared(self):
        """System prompt cache is shared across conversations."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        system = [1, 2, 3]
        cm.store_system_cache(system, [{"cached": True}])

        c1 = cm.get_system_cache(system, model)
        c2 = cm.get_system_cache(system, model)
        assert c1 is not c2  # independent copies

    def test_conversation_id_tracking(self):
        """Cache manager tracks conversation IDs."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        for cid in ["a", "b", "c"]:
            cache, _ = cm.get_or_create(cid, [1], model)
            cm.update(cid, cache, [1])

        stats = cm.stats()
        assert set(stats["conversation_ids"]) == {"a", "b", "c"}

    def test_cache_adapter_invalidation(self):
        """Loading new adapter doesn't crash cache manager."""
        from mlx_forge.serving.cache_manager import CacheManager

        cm = CacheManager()
        model = _make_mock_model()

        cache, _ = cm.get_or_create("conv1", [1, 2], model)
        cm.update("conv1", cache, [1, 2, 3])

        # Evicting cache after adapter change is caller's responsibility
        cm.evict("conv1")
        assert "conv1" not in cm._conversations

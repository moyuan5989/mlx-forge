"""M41 Tests: Model Lifecycle — ModelPool, keep-alive, aliases, server endpoints."""

from __future__ import annotations

import json
import time
from unittest.mock import MagicMock, patch

import pytest

# ─── Helpers ───


def _mock_steps(*token_ids):
    from mlx_forge.inference.engine import StepResult

    return iter([StepResult(token_id=t) for t in token_ids])


def _make_mock_manager(model_id="test-model"):
    from mlx_forge.serving.model_manager import ModelManager

    mgr = ModelManager()
    mgr._model = MagicMock()
    mgr._tokenizer = MagicMock()
    mgr._model_id = model_id
    mgr._tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    mgr._tokenizer.encode.return_value = [1, 2, 3]
    mgr._tokenizer.eos_token_id = 0
    mgr._tokenizer.decode.return_value = "Hello world"
    return mgr


# ─── Test parse_keep_alive ───


class TestParseKeepAlive:
    """Tests for Ollama-style keep_alive parsing."""

    def test_parse_minutes(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive("5m", 300) == 300.0

    def test_parse_hours(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive("1h", 300) == 3600.0

    def test_parse_seconds(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive("30s", 300) == 30.0

    def test_parse_negative_one(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive(-1, 300) == float("inf")

    def test_parse_string_negative_one(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive("-1", 300) == float("inf")

    def test_parse_zero(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive(0, 300) == 0.0

    def test_parse_int(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive(120, 300) == 120.0

    def test_parse_none_returns_default(self):
        from mlx_forge.serving.model_pool import parse_keep_alive

        assert parse_keep_alive(None, 42.0) == 42.0


# ─── Test ModelPool ───


class TestModelPool:
    """Tests for multi-model pool with lifecycle management."""

    def test_load_on_demand(self):
        """Pool loads model on first get()."""
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool(max_models=2, default_keep_alive=300)
        mock_mgr = _make_mock_manager("model-a")

        with patch("mlx_forge.serving.model_pool.ModelManager") as MockMM:
            instance = MockMM.return_value
            instance.load = MagicMock()
            instance.snapshot_base_weights = MagicMock()
            instance._model = mock_mgr._model
            instance._tokenizer = mock_mgr._tokenizer
            instance._model_id = "model-a"
            instance.model_id = "model-a"
            mgr = pool.get("model-a")
            assert mgr is instance
            assert "model-a" in pool._models

    def test_reuse_loaded(self):
        """Pool returns already-loaded model without reloading."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool(max_models=2)
        mock_mgr = _make_mock_manager("model-a")
        pool._models["model-a"] = ManagedModel(
            manager=mock_mgr, model_id="model-a", keep_alive=300
        )

        result = pool.get("model-a")
        assert result is mock_mgr

    def test_lru_eviction(self):
        """Least-recently-used model is evicted when pool is full."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool(max_models=2)

        mgr_a = _make_mock_manager("a")
        mgr_b = _make_mock_manager("b")
        pool._models["a"] = ManagedModel(
            manager=mgr_a, model_id="a", keep_alive=300,
            last_access=time.time() - 100,  # older
        )
        pool._models["b"] = ManagedModel(
            manager=mgr_b, model_id="b", keep_alive=300,
            last_access=time.time(),  # newer
        )

        # Load a third model — should evict "a" (LRU)
        with patch("mlx_forge.serving.model_pool.ModelManager") as MockMM:
            instance = MockMM.return_value
            instance.load = MagicMock()
            instance.snapshot_base_weights = MagicMock()
            pool.get("c")

        assert "a" not in pool._models
        assert "b" in pool._models
        assert "c" in pool._models

    def test_max_models_enforced(self):
        """Pool never exceeds max_models."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool(max_models=1)
        mgr_a = _make_mock_manager("a")
        pool._models["a"] = ManagedModel(
            manager=mgr_a, model_id="a", keep_alive=300
        )

        with patch("mlx_forge.serving.model_pool.ModelManager") as MockMM:
            instance = MockMM.return_value
            instance.load = MagicMock()
            instance.snapshot_base_weights = MagicMock()
            pool.get("b")

        assert pool.loaded_count <= 1

    def test_unload_explicit(self):
        """Explicit unload removes model."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool()
        pool._models["a"] = ManagedModel(
            manager=_make_mock_manager("a"), model_id="a", keep_alive=300
        )

        assert pool.unload("a") is True
        assert "a" not in pool._models
        assert pool.unload("a") is False  # already gone

    def test_status_returns_loaded(self):
        """Status returns info about all loaded models."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool(max_models=3)
        pool._models["a"] = ManagedModel(
            manager=_make_mock_manager("a"), model_id="a", keep_alive=300
        )
        pool._models["b"] = ManagedModel(
            manager=_make_mock_manager("b"), model_id="b", keep_alive=float("inf")
        )

        status = pool.status()
        assert len(status) == 2
        ids = {s["model_id"] for s in status}
        assert ids == {"a", "b"}

        # Pinned model (inf) should have expires_at=None
        pinned = [s for s in status if s["model_id"] == "b"][0]
        assert pinned["expires_at"] is None

    def test_tick_evicts_expired(self):
        """tick() removes expired models."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool()
        pool._models["old"] = ManagedModel(
            manager=_make_mock_manager("old"),
            model_id="old",
            keep_alive=0.01,
            last_access=time.time() - 1,  # already expired
        )
        pool._models["new"] = ManagedModel(
            manager=_make_mock_manager("new"),
            model_id="new",
            keep_alive=3600,
        )

        evicted = pool.tick()
        assert "old" in evicted
        assert "old" not in pool._models
        assert "new" in pool._models

    def test_pinned_model_survives_tick(self):
        """Model with keep_alive=inf is never evicted by tick."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool()
        pool._models["pinned"] = ManagedModel(
            manager=_make_mock_manager("pinned"),
            model_id="pinned",
            keep_alive=float("inf"),
            last_access=time.time() - 99999,
        )

        evicted = pool.tick()
        assert evicted == []
        assert "pinned" in pool._models

    def test_reload_after_unload(self):
        """Can reload a model after unloading it."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool(max_models=2)
        mgr_a = _make_mock_manager("a")
        pool._models["a"] = ManagedModel(
            manager=mgr_a, model_id="a", keep_alive=300
        )

        pool.unload("a")
        assert "a" not in pool._models

        # Reload
        with patch("mlx_forge.serving.model_pool.ModelManager") as MockMM:
            instance = MockMM.return_value
            instance.load = MagicMock()
            instance.snapshot_base_weights = MagicMock()
            result = pool.get("a")
            assert result is instance

    def test_keep_alive_override_per_request(self):
        """Per-request keep_alive updates the model's TTL."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool(default_keep_alive=300)
        mgr = _make_mock_manager("x")
        pool._models["x"] = ManagedModel(
            manager=mgr, model_id="x", keep_alive=300
        )

        pool.get("x", keep_alive="1h")
        assert pool._models["x"].keep_alive == 3600.0

    def test_empty_pool_status(self):
        """Empty pool returns empty status."""
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool()
        assert pool.status() == []
        assert pool.loaded_count == 0


# ─── Test Aliases ───


class TestAliases:
    """Tests for model alias system."""

    def test_resolve_alias(self):
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool()
        pool.add_alias("chat", "Qwen/Qwen3-0.6B")
        assert pool.resolve_alias("chat") == "Qwen/Qwen3-0.6B"

    def test_unknown_alias_passthrough(self):
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool()
        assert pool.resolve_alias("Qwen/Qwen3-0.6B") == "Qwen/Qwen3-0.6B"

    def test_add_remove_alias(self):
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool()
        pool.add_alias("test", "some-model")
        assert pool.resolve_alias("test") == "some-model"
        assert pool.remove_alias("test") is True
        assert pool.resolve_alias("test") == "test"
        assert pool.remove_alias("test") is False

    def test_list_aliases(self):
        from mlx_forge.serving.model_pool import ModelPool

        pool = ModelPool()
        pool.add_alias("a", "model-a")
        pool.add_alias("b", "model-b")
        aliases = pool.list_aliases()
        assert aliases == {"a": "model-a", "b": "model-b"}

    def test_load_save_aliases(self, tmp_path):
        from mlx_forge.serving.model_pool import ModelPool

        path = tmp_path / "aliases.json"
        path.write_text(json.dumps({"chat": "Qwen/Qwen3-0.6B"}))

        pool = ModelPool()
        pool.load_aliases(path)
        assert pool.resolve_alias("chat") == "Qwen/Qwen3-0.6B"

        pool.add_alias("code", "deepseek")
        pool.save_aliases(path)

        pool2 = ModelPool()
        pool2.load_aliases(path)
        assert pool2.resolve_alias("code") == "deepseek"

    def test_alias_used_in_get(self):
        """Pool resolves alias in get()."""
        from mlx_forge.serving.model_pool import ManagedModel, ModelPool

        pool = ModelPool()
        pool.add_alias("chat", "resolved-model")
        mgr = _make_mock_manager("resolved-model")
        pool._models["resolved-model"] = ManagedModel(
            manager=mgr, model_id="resolved-model", keep_alive=300
        )

        result = pool.get("chat")
        assert result is mgr


# ─── Test Server Endpoints ───


@pytest.fixture
def mock_manager():
    return _make_mock_manager()


@pytest.fixture
def app_with_pool(mock_manager):
    from fastapi import FastAPI

    from mlx_forge.serving.cache_manager import CacheManager
    from mlx_forge.serving.model_pool import ManagedModel, ModelPool
    from mlx_forge.serving.routes import (
        router,
        set_cache_manager,
        set_manager,
        set_pool,
    )

    pool = ModelPool(max_models=3, default_keep_alive=300)
    pool._models["test-model"] = ManagedModel(
        manager=mock_manager, model_id="test-model", keep_alive=300
    )

    set_manager(mock_manager)
    set_pool(pool)
    set_cache_manager(CacheManager())

    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app_with_pool):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app_with_pool)
    return AsyncClient(transport=transport, base_url="http://test")


class TestServerEndpoints:
    """Tests for M41 server endpoints."""

    @pytest.mark.asyncio
    async def test_models_ps(self, client):
        """/v1/models/ps returns loaded models."""
        resp = await client.get("/v1/models/ps")
        assert resp.status_code == 200
        data = resp.json()
        assert "models" in data
        assert data["loaded_count"] >= 1
        assert any(m["model_id"] == "test-model" for m in data["models"])

    @pytest.mark.asyncio
    async def test_models_ps_has_timing(self, client):
        """/v1/models/ps includes timing metadata."""
        resp = await client.get("/v1/models/ps")
        data = resp.json()
        model = data["models"][0]
        assert "loaded_at" in model
        assert "last_access" in model
        assert "keep_alive" in model
        assert "idle_seconds" in model

    @pytest.mark.asyncio
    async def test_unload_model(self, client):
        """/v1/models/unload removes model."""
        resp = await client.post(
            "/v1/models/unload",
            json={"model": "test-model"},
        )
        assert resp.status_code == 200
        assert resp.json()["status"] == "ok"

    @pytest.mark.asyncio
    async def test_unload_unknown_404(self, client):
        """/v1/models/unload returns 404 for unknown model."""
        resp = await client.post(
            "/v1/models/unload",
            json={"model": "nonexistent"},
        )
        assert resp.status_code == 404

    @pytest.mark.asyncio
    async def test_keep_alive_in_request(self, client, mock_manager):
        """keep_alive in request updates model TTL."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                    "keep_alive": "1h",
                },
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_aliases_endpoint(self, client):
        """/v1/aliases returns aliases."""
        from mlx_forge.serving.routes import get_pool

        pool = get_pool()
        pool.add_alias("test-alias", "some-model")

        resp = await client.get("/v1/aliases")
        assert resp.status_code == 200
        data = resp.json()
        assert "test-alias" in data["aliases"]

    @pytest.mark.asyncio
    async def test_chat_uses_pool(self, client, mock_manager):
        """Chat endpoint uses ModelPool for model resolution."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200


# ─── Test CLI Alias Command ───


class TestAliasCLI:
    """Tests for the alias CLI command."""

    def test_save_and_load(self, tmp_path):
        """Aliases persist to disk."""
        from mlx_forge.cli.alias_cmd import _load_aliases, _save_aliases

        path = tmp_path / "aliases.json"

        import mlx_forge.cli.alias_cmd as alias_mod

        original_path = alias_mod.ALIASES_PATH
        alias_mod.ALIASES_PATH = path
        try:
            _save_aliases({"test": "model-id"})
            loaded = _load_aliases()
            assert loaded == {"test": "model-id"}
        finally:
            alias_mod.ALIASES_PATH = original_path

    def test_load_nonexistent(self, tmp_path):
        """Loading from nonexistent file returns empty dict."""
        import mlx_forge.cli.alias_cmd as alias_mod

        original_path = alias_mod.ALIASES_PATH
        alias_mod.ALIASES_PATH = tmp_path / "nonexistent.json"
        try:
            from mlx_forge.cli.alias_cmd import _load_aliases

            assert _load_aliases() == {}
        finally:
            alias_mod.ALIASES_PATH = original_path


# ─── Test App Factory ───


class TestAppFactoryM41:
    """Tests for app factory M41 additions."""

    def test_create_app_sets_pool(self):
        from mlx_forge.serving.app import create_serving_app
        from mlx_forge.serving.routes import get_pool

        create_serving_app()
        pool = get_pool()
        assert pool is not None

    def test_pool_has_correct_max_models(self):
        from mlx_forge.serving.app import create_serving_app
        from mlx_forge.serving.routes import get_pool

        create_serving_app(max_models=5)
        pool = get_pool()
        assert pool.max_models == 5

    def test_pool_has_correct_keep_alive(self):
        from mlx_forge.serving.app import create_serving_app
        from mlx_forge.serving.routes import get_pool

        create_serving_app(keep_alive="10m")
        pool = get_pool()
        assert pool._default_keep_alive == 600.0

    def test_aliases_loaded_from_file(self, tmp_path):
        path = tmp_path / "aliases.json"
        path.write_text(json.dumps({"mymodel": "Qwen/Qwen3-0.6B"}))

        from mlx_forge.serving.app import create_serving_app
        from mlx_forge.serving.routes import get_pool

        create_serving_app(aliases_path=str(path))
        pool = get_pool()
        assert pool.resolve_alias("mymodel") == "Qwen/Qwen3-0.6B"

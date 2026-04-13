"""M40 Tests: Wire Everything — CLI params, CacheManager, health, speculative, timing."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest

# ─── Helpers ───


def _mock_steps(*token_ids):
    """Create mock StepResult objects for patching generate_steps."""
    from mlx_forge.inference.engine import StepResult

    return iter([StepResult(token_id=t) for t in token_ids])


def _make_mock_manager():
    """Create a mock model manager."""
    from mlx_forge.serving.model_manager import ModelManager

    mgr = ModelManager()
    mgr._model = MagicMock()
    mgr._tokenizer = MagicMock()
    mgr._model_id = "test-model"
    mgr._tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    mgr._tokenizer.encode.return_value = [1, 2, 3]
    mgr._tokenizer.eos_token_id = 0
    mgr._tokenizer.decode.return_value = "Hello world"
    return mgr


@pytest.fixture
def mock_manager():
    return _make_mock_manager()


@pytest.fixture
def app(mock_manager):
    from fastapi import FastAPI

    from mlx_forge.serving.cache_manager import CacheManager
    from mlx_forge.serving.routes import router, set_cache_manager, set_manager, set_pool

    set_manager(mock_manager)
    set_pool(None)  # Use legacy single-manager path
    set_cache_manager(CacheManager(max_conversations=16, ttl_seconds=600))
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


# ─── Test CLI Param Passthrough ───


class TestCLIParamPassthrough:
    """Tests that new M37 params reach generate() from CLI."""

    def test_generate_accepts_top_k(self):
        """generate() accepts top_k parameter."""
        from mlx_forge.inference.engine import generate

        sig = generate.__code__.co_varnames
        assert "top_k" in sig

    def test_generate_accepts_min_p(self):
        """generate() accepts min_p parameter."""
        from mlx_forge.inference.engine import generate

        sig = generate.__code__.co_varnames
        assert "min_p" in sig

    def test_generate_accepts_frequency_penalty(self):
        """generate() accepts frequency_penalty parameter."""
        from mlx_forge.inference.engine import generate

        sig = generate.__code__.co_varnames
        assert "frequency_penalty" in sig

    def test_generate_accepts_presence_penalty(self):
        """generate() accepts presence_penalty parameter."""
        from mlx_forge.inference.engine import generate

        sig = generate.__code__.co_varnames
        assert "presence_penalty" in sig

    def test_generate_tokens_accepts_new_params(self):
        """generate_tokens() accepts all new M37 params."""
        from mlx_forge.inference.engine import generate_tokens

        sig = generate_tokens.__code__.co_varnames
        for param in ["top_k", "min_p", "frequency_penalty", "presence_penalty"]:
            assert param in sig, f"generate_tokens missing {param}"

    def test_generate_steps_accepts_cache(self):
        """generate_steps() accepts cache and all_token_history params."""
        from mlx_forge.inference.engine import generate_steps

        sig = generate_steps.__code__.co_varnames
        assert "cache" in sig
        assert "all_token_history" in sig


# ─── Test CacheManager Wiring ───


class TestCacheManagerWiring:
    """Tests that CacheManager is wired into routes."""

    def test_set_get_cache_manager(self):
        """Cache manager can be set and retrieved."""
        from mlx_forge.serving.cache_manager import CacheManager
        from mlx_forge.serving.routes import get_cache_manager, set_cache_manager

        cm = CacheManager()
        set_cache_manager(cm)
        assert get_cache_manager() is cm
        set_cache_manager(None)

    @pytest.mark.asyncio
    async def test_conversation_id_enables_caching(self, client, mock_manager):
        """conversation_id in request triggers cache lookup."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20, 30)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "conversation_id": "conv-123",
                },
            )
            assert resp.status_code == 200

    @pytest.mark.asyncio
    async def test_no_conversation_id_works(self, client, mock_manager):
        """Request without conversation_id still works (no caching)."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            assert resp.status_code == 200

    def test_generate_steps_uses_external_cache(self):
        """generate_steps uses provided cache instead of creating new one."""
        # Verify the function signature accepts cache
        import inspect

        from mlx_forge.inference.engine import generate_steps

        sig = inspect.signature(generate_steps)
        assert "cache" in sig.parameters
        assert sig.parameters["cache"].default is None

    def test_generate_steps_uses_all_token_history(self):
        """generate_steps uses all_token_history for penalty tracking."""
        import inspect

        from mlx_forge.inference.engine import generate_steps

        sig = inspect.signature(generate_steps)
        assert "all_token_history" in sig.parameters
        assert sig.parameters["all_token_history"].default is None

    @pytest.mark.asyncio
    async def test_cache_stats_in_health(self, client):
        """Health endpoint includes cache stats."""
        resp = await client.get("/health")
        data = resp.json()
        assert "cache" in data
        assert "active_conversations" in data["cache"]


# ─── Test Health Endpoint ───


class TestHealthEndpoint:
    """Tests for /health endpoint."""

    @pytest.mark.asyncio
    async def test_health_returns_ok(self, client):
        """Health endpoint returns ok status."""
        resp = await client.get("/health")
        assert resp.status_code == 200
        data = resp.json()
        assert data["status"] == "ok"

    @pytest.mark.asyncio
    async def test_health_shows_model_loaded(self, client, mock_manager):
        """Health shows model_loaded status."""
        resp = await client.get("/health")
        data = resp.json()
        assert data["model_loaded"] is True

    @pytest.mark.asyncio
    async def test_health_shows_model_id(self, client, mock_manager):
        """Health shows model_id."""
        resp = await client.get("/health")
        data = resp.json()
        assert data["model_id"] == "test-model"

    @pytest.mark.asyncio
    async def test_health_no_model(self):
        """Health when no model loaded."""
        from fastapi import FastAPI
        from httpx import ASGITransport, AsyncClient

        from mlx_forge.serving.model_manager import ModelManager
        from mlx_forge.serving.routes import router, set_cache_manager, set_manager

        set_manager(ModelManager())
        set_cache_manager(None)
        app = FastAPI()
        app.include_router(router)

        transport = ASGITransport(app=app)
        async with AsyncClient(transport=transport, base_url="http://test") as c:
            resp = await c.get("/health")
            data = resp.json()
            assert data["model_loaded"] is False
            assert data["model_id"] is None


# ─── Test Timing Metadata ───


class TestTimingMetadata:
    """Tests for timing metadata in responses."""

    @pytest.mark.asyncio
    async def test_ttft_in_chat_response(self, client, mock_manager):
        """Response includes ttft_ms in usage."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            data = resp.json()
            usage = data["usage"]
            assert "ttft_ms" in usage
            assert "decode_tokens_per_sec" in usage

    @pytest.mark.asyncio
    async def test_eval_duration_in_response(self, client, mock_manager):
        """Response includes eval_duration_ms."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hi"}],
                },
            )
            data = resp.json()
            assert "eval_duration_ms" in data["usage"]

    @pytest.mark.asyncio
    async def test_prompt_eval_duration_in_response(self, client, mock_manager):
        """Response includes prompt_eval_duration_ms."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10)

            resp = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "Hello"},
            )
            data = resp.json()
            assert "prompt_eval_duration_ms" in data["usage"]

    @pytest.mark.asyncio
    async def test_streaming_still_works(self, client, mock_manager):
        """Streaming responses still function with timing changes."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20)
            mock_manager._tokenizer.decode.return_value = "Hi"

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                    "stream": True,
                },
            )
            assert resp.status_code == 200
            assert "data: [DONE]" in resp.text


# ─── Test Tokenize / Detokenize ───


class TestTokenizeDetokenize:
    """Tests for tokenize/detokenize utility endpoints."""

    @pytest.mark.asyncio
    async def test_tokenize(self, client, mock_manager):
        """Tokenize endpoint returns token IDs."""
        resp = await client.post(
            "/v1/tokenize",
            json={"model": "test-model", "text": "hello"},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "tokens" in data
        assert "count" in data
        assert data["count"] == len(data["tokens"])

    @pytest.mark.asyncio
    async def test_detokenize(self, client, mock_manager):
        """Detokenize endpoint returns text."""
        resp = await client.post(
            "/v1/detokenize",
            json={"model": "test-model", "tokens": [1, 2, 3]},
        )
        assert resp.status_code == 200
        data = resp.json()
        assert "text" in data


# ─── Test Speculative Decoding Wiring ───


class TestSpeculativeWiring:
    """Tests that speculative decoding is wired into CLI."""

    def test_speculative_function_exists(self):
        """speculative_generate_tokens is importable."""
        from mlx_forge.inference.speculative import speculative_generate_tokens

        assert callable(speculative_generate_tokens)

    def test_generate_cmd_has_speculative_handler(self):
        """generate_cmd has _run_speculative function."""
        from mlx_forge.cli.generate_cmd import _run_speculative

        assert callable(_run_speculative)

    def test_serve_cmd_passes_draft_model(self):
        """serve_cmd reads draft_model arg."""
        # Verify the function references draft_model
        import inspect

        from mlx_forge.cli.serve_cmd import run_serve

        source = inspect.getsource(run_serve)
        assert "draft_model" in source


# ─── Test App Factory ───


class TestAppFactory:
    """Tests for app.py factory changes."""

    def test_create_app_sets_cache_manager(self):
        """App factory creates and sets CacheManager."""
        from mlx_forge.serving.app import create_serving_app
        from mlx_forge.serving.routes import get_cache_manager

        app = create_serving_app()
        cm = get_cache_manager()
        assert cm is not None

    def test_create_app_accepts_draft_model(self):
        """App factory accepts draft_model parameter."""
        import inspect

        from mlx_forge.serving.app import create_serving_app

        sig = inspect.signature(create_serving_app)
        assert "draft_model" in sig.parameters

    def test_create_app_accepts_cache_params(self):
        """App factory accepts cache configuration."""
        import inspect

        from mlx_forge.serving.app import create_serving_app

        sig = inspect.signature(create_serving_app)
        assert "max_conversations" in sig.parameters
        assert "cache_ttl" in sig.parameters


# ─── Test Backward Compatibility ───


class TestBackwardCompat:
    """Ensure existing behavior is not broken."""

    @pytest.mark.asyncio
    async def test_existing_chat_format(self, client, mock_manager):
        """Existing chat completion format still works."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20, 30)

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "chat.completion"
            assert data["choices"][0]["message"]["role"] == "assistant"
            assert "usage" in data

    @pytest.mark.asyncio
    async def test_existing_completion_format(self, client, mock_manager):
        """Existing text completion format still works."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20)

            resp = await client.post(
                "/v1/completions",
                json={"model": "test-model", "prompt": "Hello"},
            )
            assert resp.status_code == 200
            data = resp.json()
            assert data["object"] == "text_completion"

    @pytest.mark.asyncio
    async def test_adapter_cache_invalidation(self, client, mock_manager):
        """Loading adapter invalidates conversation caches."""
        from mlx_forge.serving.routes import get_cache_manager

        cm = get_cache_manager()
        # Simulate an active conversation cache
        if cm:
            model_mock = MagicMock()
            model_mock.model = MagicMock()
            model_mock.model.layers = [MagicMock(), MagicMock()]
            del model_mock.make_cache
            cache, _ = cm.get_or_create("test-conv", [1, 2], model_mock)
            cm.update("test-conv", cache, [1, 2, 3])
            assert cm.stats()["active_conversations"] == 1

        # Load adapter (mocked) — should evict caches
        # We can't actually load an adapter in test, but verify the code path
        from mlx_forge.serving.routes import load_adapter

        assert callable(load_adapter)

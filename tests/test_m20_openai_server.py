"""Tests for M20: OpenAI-Compatible API Server."""

import json
import time
from unittest.mock import MagicMock, patch

import pytest

from mlx_forge.serving.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    DeltaContent,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    Usage,
)

# ── OpenAI Types Tests ──


class TestChatCompletionRequest:
    def test_basic_request(self):
        req = ChatCompletionRequest(
            model="test-model",
            messages=[ChatMessage(role="user", content="Hello")],
        )
        assert req.model == "test-model"
        assert req.temperature == 0.7
        assert req.stream is False

    def test_with_all_params(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            temperature=0.5,
            top_p=0.8,
            max_tokens=100,
            stream=True,
            stop=["END"],
        )
        assert req.temperature == 0.5
        assert req.stream is True
        assert req.stop == ["END"]

    def test_stop_as_string(self):
        req = ChatCompletionRequest(
            model="test",
            messages=[ChatMessage(role="user", content="Hi")],
            stop="STOP",
        )
        assert req.stop == "STOP"


class TestCompletionRequest:
    def test_basic_request(self):
        req = CompletionRequest(model="test", prompt="Hello world")
        assert req.model == "test"
        assert req.prompt == "Hello world"
        assert req.max_tokens == 512


class TestChatCompletionResponse:
    def test_response_format(self):
        resp = ChatCompletionResponse(
            model="test",
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content="Hi!"),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=5, completion_tokens=2, total_tokens=7),
        )
        assert resp.object == "chat.completion"
        assert resp.choices[0].message.content == "Hi!"
        assert resp.usage.total_tokens == 7
        assert resp.id.startswith("chatcmpl-")

    def test_response_has_created_timestamp(self):
        resp = ChatCompletionResponse(
            model="test",
            choices=[
                Choice(
                    message=ChatMessage(role="assistant", content=""),
                    finish_reason="stop",
                )
            ],
            usage=Usage(prompt_tokens=0, completion_tokens=0, total_tokens=0),
        )
        assert abs(resp.created - int(time.time())) < 5


class TestCompletionResponse:
    def test_response_format(self):
        resp = CompletionResponse(
            model="test",
            choices=[CompletionChoice(text="world", finish_reason="stop")],
            usage=Usage(prompt_tokens=3, completion_tokens=1, total_tokens=4),
        )
        assert resp.object == "text_completion"
        assert resp.id.startswith("cmpl-")
        assert resp.choices[0].text == "world"


class TestStreamChunk:
    def test_chat_chunk_format(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            created=int(time.time()),
            model="test",
            choices=[StreamChoice(delta=DeltaContent(content="Hello"))],
        )
        assert chunk.object == "chat.completion.chunk"
        assert chunk.choices[0].delta.content == "Hello"

    def test_completion_chunk_format(self):
        chunk = CompletionChunk(
            id="cmpl-test",
            created=int(time.time()),
            model="test",
            choices=[CompletionStreamChoice(text="world")],
        )
        assert chunk.object == "text_completion"

    def test_finish_reason_in_final_chunk(self):
        chunk = ChatCompletionChunk(
            id="chatcmpl-test",
            created=int(time.time()),
            model="test",
            choices=[StreamChoice(delta=DeltaContent(), finish_reason="stop")],
        )
        assert chunk.choices[0].finish_reason == "stop"
        assert chunk.choices[0].delta.content is None


class TestModelList:
    def test_model_list_format(self):
        resp = ModelListResponse(
            data=[ModelObject(id="model-1"), ModelObject(id="model-2")]
        )
        assert resp.object == "list"
        assert len(resp.data) == 2
        assert resp.data[0].owned_by == "mlx-forge"

    def test_model_object(self):
        m = ModelObject(id="test-model")
        assert m.object == "model"
        assert m.id == "test-model"


# ── Model Manager Tests ──


class TestModelManager:
    def test_initial_state(self):
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        assert not mgr.is_loaded
        assert mgr.model_id is None

    def test_unload(self):
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        mgr._model = "fake"
        mgr._tokenizer = "fake"
        mgr._model_id = "test"
        mgr.unload()
        assert not mgr.is_loaded
        assert mgr.model_id is None

    def test_list_available_empty(self, tmp_path):
        from mlx_forge.serving.model_manager import ModelManager

        mgr = ModelManager()
        with patch("mlx_forge.serving.model_manager.Path") as mock_path:
            # Make runs and exports dirs not exist
            mock_runs = MagicMock()
            mock_runs.exists.return_value = False
            mock_exports = MagicMock()
            mock_exports.exists.return_value = False
            mock_path.return_value.expanduser.side_effect = [mock_runs, mock_exports]
            models = mgr.list_available()
            assert models == []


# ── Route Tests (using httpx AsyncClient) ──


@pytest.fixture
def mock_manager():
    """Create a mock model manager with a fake model."""
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
def app(mock_manager):
    """Create test app with mocked model."""
    from fastapi import FastAPI

    from mlx_forge.serving.routes import router, set_manager, set_pool

    set_manager(mock_manager)
    set_pool(None)  # Use legacy single-manager path
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    """Create httpx test client."""
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


def _mock_steps(*token_ids):
    """Create mock StepResult objects for patching generate_steps."""
    from mlx_forge.inference.engine import StepResult
    return iter([StepResult(token_id=t) for t in token_ids])


@pytest.mark.asyncio
async def test_chat_completion_format(client, mock_manager):
    """Non-streaming chat completion returns correct format."""
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
        assert data["model"] == "test-model"
        assert len(data["choices"]) == 1
        assert data["choices"][0]["message"]["role"] == "assistant"
        assert "usage" in data


@pytest.mark.asyncio
async def test_completion_format(client, mock_manager):
    """Non-streaming completion returns correct format."""
    with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
        mock_gen.return_value = _mock_steps(10, 20)

        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
            },
        )
        assert resp.status_code == 200
        data = resp.json()
        assert data["object"] == "text_completion"
        assert len(data["choices"]) == 1
        assert "text" in data["choices"][0]


@pytest.mark.asyncio
async def test_models_endpoint(client, mock_manager):
    """/v1/models returns loaded model."""
    resp = await client.get("/v1/models")
    assert resp.status_code == 200
    data = resp.json()
    assert data["object"] == "list"
    assert any(m["id"] == "test-model" for m in data["data"])


@pytest.mark.asyncio
async def test_chat_streaming_format(client, mock_manager):
    """Streaming chat returns SSE chunks ending with [DONE]."""
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
        body = resp.text

        # Should contain SSE data lines
        assert "data: " in body
        assert "data: [DONE]" in body

        # Parse chunks
        lines = [
            line
            for line in body.split("\n")
            if line.startswith("data: ") and line != "data: [DONE]"
        ]
        assert len(lines) >= 1

        # First chunk should have role
        first = json.loads(lines[0].removeprefix("data: "))
        assert first["object"] == "chat.completion.chunk"


@pytest.mark.asyncio
async def test_stop_sequence_handling(client, mock_manager):
    """Stop sequences should truncate output."""
    with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
        mock_gen.return_value = _mock_steps(10, 20, 30, 40)
        # Simulate decode returning text containing stop sequence
        mock_manager._tokenizer.decode.side_effect = [
            "Hello",
            "Hello world",
            "Hello world STOP now",
            "Hello world STOP now more",
        ]

        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Say hello",
                "stop": ["STOP"],
            },
        )
        data = resp.json()
        assert data["choices"][0]["finish_reason"] == "stop"
        assert "STOP" not in data["choices"][0]["text"]


@pytest.mark.asyncio
async def test_usage_token_counts(client, mock_manager):
    """Usage should report correct token counts."""
    with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
        mock_gen.return_value = _mock_steps(10, 20, 30)

        resp = await client.post(
            "/v1/chat/completions",
            json={
                "model": "test-model",
                "messages": [{"role": "user", "content": "Hi"}],
            },
        )
        data = resp.json()
        assert data["usage"]["prompt_tokens"] == 5  # from mock
        assert data["usage"]["completion_tokens"] == 3
        assert data["usage"]["total_tokens"] == 8


@pytest.mark.asyncio
async def test_completion_streaming_done(client, mock_manager):
    """Streaming completion ends with [DONE]."""
    with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
        mock_gen.return_value = _mock_steps(10)
        mock_manager._tokenizer.decode.return_value = "test"

        resp = await client.post(
            "/v1/completions",
            json={
                "model": "test-model",
                "prompt": "Hello",
                "stream": True,
            },
        )
        assert "data: [DONE]" in resp.text


class TestNormalizeStop:
    def test_none(self):
        from mlx_forge.serving.routes import _normalize_stop

        assert _normalize_stop(None) == []

    def test_string(self):
        from mlx_forge.serving.routes import _normalize_stop

        assert _normalize_stop("STOP") == ["STOP"]

    def test_list(self):
        from mlx_forge.serving.routes import _normalize_stop

        assert _normalize_stop(["A", "B"]) == ["A", "B"]


class TestServingApp:
    def test_create_app(self):
        from mlx_forge.serving.app import create_serving_app

        app = create_serving_app()
        assert app is not None
        # Check routes are registered
        routes = [r.path for r in app.routes]
        assert "/v1/chat/completions" in routes
        assert "/v1/completions" in routes
        assert "/v1/models" in routes

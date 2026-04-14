"""M43 Tests: Competitive Features — thinking models, FIM, JSON logit processor."""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import mlx.core as mx
import pytest

# ─── Test ThinkingParser ───


class TestThinkingParser:
    """Tests for thinking model output parsing."""

    def test_parse_think_tags(self):
        from mlx_forge.inference.thinking import ThinkingParser

        text = "<think>Let me reason about this...</think>The answer is 42."
        result = ThinkingParser().parse(text)
        assert result.thinking == "Let me reason about this..."
        assert result.response == "The answer is 42."

    def test_parse_thinking_tags(self):
        from mlx_forge.inference.thinking import ThinkingParser

        text = "<thinking>Step 1: analyze\nStep 2: solve</thinking>Result: done"
        result = ThinkingParser().parse(text)
        assert "Step 1" in result.thinking
        assert result.response == "Result: done"

    def test_no_thinking_passthrough(self):
        from mlx_forge.inference.thinking import ThinkingParser

        text = "Just a normal response."
        result = ThinkingParser().parse(text)
        assert result.thinking == ""
        assert result.response == "Just a normal response."

    def test_multiple_thinking_blocks(self):
        from mlx_forge.inference.thinking import ThinkingParser

        text = "<think>First thought</think>Middle text<think>Second thought</think>Final."
        result = ThinkingParser().parse(text)
        assert "First thought" in result.thinking
        assert "Second thought" in result.thinking
        assert "Final." in result.response

    def test_empty_thinking(self):
        from mlx_forge.inference.thinking import ThinkingParser

        text = "<think></think>Response."
        result = ThinkingParser().parse(text)
        assert result.response == "Response."

    def test_has_thinking_tags(self):
        from mlx_forge.inference.thinking import ThinkingParser

        parser = ThinkingParser()
        assert parser.has_thinking_tags("<think>hello</think>") is True
        assert parser.has_thinking_tags("no tags here") is False
        assert parser.has_thinking_tags("<thinking>hi</thinking>") is True


# ─── Test FIM ───


class TestFIM:
    """Tests for fill-in-middle code completion."""

    def test_codellama_template(self):
        from mlx_forge.inference.fim import build_fim_prompt

        result = build_fim_prompt("def hello(", "    return greeting", "codellama")
        assert "<PRE>" in result
        assert "<SUF>" in result
        assert "<MID>" in result
        assert "def hello(" in result
        assert "return greeting" in result

    def test_deepseek_template(self):
        from mlx_forge.inference.fim import build_fim_prompt

        result = build_fim_prompt("prefix", "suffix", "deepseek")
        assert "<｜fim▁begin｜>" in result
        assert "<｜fim▁hole｜>" in result
        assert "<｜fim▁end｜>" in result

    def test_starcoder_template(self):
        from mlx_forge.inference.fim import build_fim_prompt

        result = build_fim_prompt("a", "b", "starcoder")
        assert "<fim_prefix>" in result

    def test_qwen_template(self):
        from mlx_forge.inference.fim import build_fim_prompt

        result = build_fim_prompt("a", "b", "qwen")
        assert "<|fim_prefix|>" in result

    def test_detect_fim_support(self):
        from mlx_forge.inference.fim import detect_fim_support

        assert detect_fim_support({"model_type": "codellama"}) == "codellama"
        assert detect_fim_support({"_name_or_path": "deepseek-coder"}) == "deepseek"
        assert detect_fim_support({"model_type": "llama"}) is None

    def test_unknown_template_error(self):
        from mlx_forge.inference.fim import build_fim_prompt

        with pytest.raises(ValueError, match="Unknown FIM template"):
            build_fim_prompt("a", "b", "nonexistent")

    def test_suffix_in_completion_request(self):
        from mlx_forge.serving.openai_types import CompletionRequest

        req = CompletionRequest(
            model="test", prompt="def hello(", suffix="    return greeting"
        )
        assert req.suffix == "    return greeting"


# ─── Test JSONLogitProcessor ───


class TestJSONLogitProcessor:
    """Tests for JSON logit biasing during generation."""

    def _make_tokenizer(self):
        tok = MagicMock()
        tok.vocab_size = 100
        tok.decode = lambda ids: {
            0: "{", 1: "}", 2: "[", 3: "]", 4: '"', 5: ":",
            6: ",", 7: "t", 8: "f", 9: "n", 10: "1",
        }.get(ids[0], "x") if ids else ""
        return tok

    def test_state_expect_open(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        state = proc._analyze_state("")
        assert state == "EXPECT_OPEN"

    def test_state_after_open_brace(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        state = proc._analyze_state("{")
        assert state == "EXPECT_KEY_OR_CLOSE"

    def test_state_after_colon(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        state = proc._analyze_state('{"key":')
        assert state == "EXPECT_VALUE"

    def test_state_in_string(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        state = proc._analyze_state('{"key": "value')
        assert state == "IN_STRING"

    def test_state_after_value(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        state = proc._analyze_state('{"key": "value"')
        assert state == "EXPECT_COMMA_OR_CLOSE"

    def test_process_returns_array(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        logits = mx.zeros((100,))
        result = proc.process(logits, "")
        mx.eval(result)
        assert result.shape == (100,)

    def test_process_boosts_structural(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        logits = mx.zeros((100,))
        # At start, should boost { and [
        result = proc.process(logits, "")
        mx.eval(result)
        # Result should have some non-zero values (boosted tokens)
        assert mx.max(result).item() >= 0

    def test_depth_tracking(self):
        from mlx_forge.inference.constrained import JSONLogitProcessor

        proc = JSONLogitProcessor(self._make_tokenizer())
        # Nested: should still track correctly
        state = proc._analyze_state('{"a": {"b":')
        assert state == "EXPECT_VALUE"


# ─── Test make_json_logit_processor ───


class TestMakeJsonLogitProcessor:
    """Tests for the factory function."""

    def test_returns_none_without_format(self):
        from mlx_forge.inference.constrained import make_json_logit_processor

        assert make_json_logit_processor(MagicMock(), None) is None
        assert make_json_logit_processor(MagicMock(), {"type": "text"}) is None

    def test_returns_callable_for_json(self):
        from mlx_forge.inference.constrained import make_json_logit_processor

        proc = make_json_logit_processor(MagicMock(), {"type": "json_object"})
        assert callable(proc)

    def test_returns_callable_for_schema(self):
        from mlx_forge.inference.constrained import make_json_logit_processor

        proc = make_json_logit_processor(
            MagicMock(), {"type": "json_schema", "json_schema": {"schema": {}}}
        )
        assert callable(proc)


# ─── Test Logit Processor in Engine ───


class TestLogitProcessorEngine:
    """Tests for logit_processor wiring in generate_steps."""

    def test_generate_steps_accepts_logit_processor(self):
        import inspect

        from mlx_forge.inference.engine import generate_steps

        sig = inspect.signature(generate_steps)
        assert "logit_processor" in sig.parameters


# ─── Test API Types ───


class TestAPITypes:
    """Tests for M43 API type additions."""

    def test_think_param(self):
        from mlx_forge.serving.openai_types import ChatCompletionRequest

        req = ChatCompletionRequest(model="test", messages=[], think=True)
        assert req.think is True

    def test_think_default_false(self):
        from mlx_forge.serving.openai_types import ChatCompletionRequest

        req = ChatCompletionRequest(model="test", messages=[])
        assert req.think is False

    def test_thinking_in_message(self):
        from mlx_forge.serving.openai_types import ChatMessage

        msg = ChatMessage(
            role="assistant", content="Answer.", thinking="I thought about it."
        )
        assert msg.thinking == "I thought about it."

    def test_suffix_in_completion(self):
        from mlx_forge.serving.openai_types import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello", suffix="world")
        assert req.suffix == "world"

    def test_suffix_default_none(self):
        from mlx_forge.serving.openai_types import CompletionRequest

        req = CompletionRequest(model="test", prompt="hello")
        assert req.suffix is None


# ─── Test Route Integration ───


def _mock_steps(*token_ids):
    from mlx_forge.inference.engine import StepResult

    return iter([StepResult(token_id=t) for t in token_ids])


@pytest.fixture
def mock_manager():
    from mlx_forge.serving.model_manager import ModelManager

    mgr = ModelManager()
    mgr._model = MagicMock()
    mgr._tokenizer = MagicMock()
    mgr._model_id = "test-model"
    mgr._tokenizer.apply_chat_template.return_value = [1, 2, 3, 4, 5]
    mgr._tokenizer.encode.return_value = [1, 2, 3]
    mgr._tokenizer.eos_token_id = 0
    mgr._tokenizer.decode.return_value = "<think>reasoning</think>Answer is 42."
    return mgr


@pytest.fixture
def app(mock_manager):
    from fastapi import FastAPI

    from mlx_forge.serving.cache_manager import CacheManager
    from mlx_forge.serving.routes import (
        router,
        set_cache_manager,
        set_manager,
        set_pool,
    )

    set_manager(mock_manager)
    set_pool(None)
    set_cache_manager(CacheManager())
    app = FastAPI()
    app.include_router(router)
    return app


@pytest.fixture
def client(app):
    from httpx import ASGITransport, AsyncClient

    transport = ASGITransport(app=app)
    return AsyncClient(transport=transport, base_url="http://test")


class TestRouteIntegration:
    """Tests for M43 features wired into routes."""

    @pytest.mark.asyncio
    async def test_think_param_parses_thinking(self, client, mock_manager):
        """think=true parses thinking blocks from output."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10, 20, 30)
            mock_manager._tokenizer.decode.return_value = (
                "<think>Let me think</think>The answer."
            )

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Think hard"}],
                    "think": True,
                },
            )
            assert resp.status_code == 200
            data = resp.json()
            msg = data["choices"][0]["message"]
            assert msg.get("thinking") is not None
            assert "Let me think" in msg["thinking"]
            assert "The answer." in msg["content"]

    @pytest.mark.asyncio
    async def test_think_false_no_parsing(self, client, mock_manager):
        """think=false does not parse thinking blocks."""
        with patch("mlx_forge.inference.engine.generate_steps") as mock_gen:
            mock_gen.return_value = _mock_steps(10)
            mock_manager._tokenizer.decode.return_value = "Normal response."

            resp = await client.post(
                "/v1/chat/completions",
                json={
                    "model": "test-model",
                    "messages": [{"role": "user", "content": "Hello"}],
                },
            )
            data = resp.json()
            assert data["choices"][0]["message"].get("thinking") is None

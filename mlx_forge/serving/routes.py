"""OpenAI-compatible API routes."""

from __future__ import annotations

import json
import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse
from pydantic import BaseModel

from mlx_forge.inference.stop_conditions import StopChecker
from mlx_forge.serving.cache_manager import CacheManager
from mlx_forge.serving.model_manager import ModelManager
from mlx_forge.serving.model_pool import ModelPool
from mlx_forge.serving.openai_types import (
    ChatCompletionChunk,
    ChatCompletionRequest,
    ChatCompletionResponse,
    ChatMessage,
    Choice,
    ChoiceLogprobs,
    CompletionChoice,
    CompletionChunk,
    CompletionRequest,
    CompletionResponse,
    CompletionStreamChoice,
    DeltaContent,
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    LogprobContent,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    ToolCallFunction,
    ToolCallMessage,
    TopLogprob,
    Usage,
)

router = APIRouter()
_manager = ModelManager()
_cache_manager: CacheManager | None = None
_pool: ModelPool | None = None
_default_context_length: int = 0
_default_num_keep: int = 0


def get_manager() -> ModelManager:
    return _manager


def set_manager(manager: ModelManager) -> None:
    global _manager
    _manager = manager


def get_cache_manager() -> CacheManager | None:
    return _cache_manager


def set_cache_manager(cache_mgr: CacheManager | None) -> None:
    global _cache_manager
    _cache_manager = cache_mgr


def get_pool() -> ModelPool | None:
    return _pool


def set_pool(pool: ModelPool | None) -> None:
    global _pool
    _pool = pool


def set_context_defaults(context_length: int = 0, num_keep: int = 0) -> None:
    global _default_context_length, _default_num_keep
    _default_context_length = context_length
    _default_num_keep = num_keep


def _get_model_manager(
    request_model: str, keep_alive: str | int | float | None = None
) -> ModelManager:
    """Get a ModelManager for the requested model.

    Uses ModelPool if available (multi-model with lifecycle), otherwise
    falls back to single global manager.
    """
    pool = get_pool()
    if pool is not None:
        try:
            return pool.get(request_model, keep_alive=keep_alive)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model '{request_model}': {e}"
            )

    # Legacy single-manager path
    mgr = get_manager()
    if not mgr.is_loaded or mgr.model_id != request_model:
        try:
            mgr.load(request_model)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model '{request_model}': {e}"
            )
    return mgr


def _normalize_stop(stop) -> list[str]:
    """Normalize stop sequences to a list."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


def _build_stop_checker(request, tokenizer) -> StopChecker:
    """Build a StopChecker from request params."""
    stop_strings = _normalize_stop(getattr(request, "stop", None))
    stop_token_ids = getattr(request, "stop_token_ids", None) or []
    return StopChecker(
        stop_strings=stop_strings,
        stop_token_ids=stop_token_ids,
        eos_token_id=tokenizer.eos_token_id,
    )


def _get_context_params(request) -> tuple[int, int]:
    """Get context_length and num_keep from request or server defaults."""
    ctx = getattr(request, "num_ctx", None) or _default_context_length
    keep = getattr(request, "num_keep", None) or _default_num_keep
    return ctx, keep


def _build_logprobs_response(logprobs_list):
    """Build ChoiceLogprobs from a list of TokenLogprobResult."""
    if not logprobs_list:
        return None
    content = []
    for lp in logprobs_list:
        content.append(
            LogprobContent(
                token=lp.token,
                token_id=lp.token_id,
                logprob=lp.logprob,
                top_logprobs=[
                    TopLogprob(token=t.token, token_id=t.token_id, logprob=t.logprob)
                    for t in lp.top_logprobs
                ],
            )
        )
    return ChoiceLogprobs(content=content)


def _build_usage(prompt_tokens_count, generated_ids_count, metrics) -> Usage:
    """Build Usage with timing metadata."""
    return Usage(
        prompt_tokens=prompt_tokens_count,
        completion_tokens=generated_ids_count,
        total_tokens=prompt_tokens_count + generated_ids_count,
        ttft_ms=round(metrics.ttft_ms, 2) if metrics.ttft_ms else None,
        prompt_eval_duration_ms=(
            round(metrics.ttft_ms, 2) if metrics.ttft_ms else None
        ),
        eval_duration_ms=(
            round(metrics.total_time_ms - (metrics.ttft_ms or 0), 2)
            if metrics.total_time_ms
            else None
        ),
        decode_tokens_per_sec=(
            round(metrics.decode_tokens_per_sec, 1)
            if metrics.decode_tokens_per_sec
            else None
        ),
    )


def _inject_tools_into_messages(messages: list[dict], request) -> list[dict]:
    """If tools are provided, inject tool descriptions into system prompt."""
    tools = getattr(request, "tools", None)
    if not tools:
        return messages

    from mlx_forge.serving.tool_parser import ToolCallParser

    parser = ToolCallParser()
    tool_defs = [t.model_dump() for t in tools]
    tool_prompt = parser.format_tools_for_prompt(tool_defs)

    messages = list(messages)
    if messages and messages[0].get("role") == "system":
        messages[0] = dict(messages[0])
        messages[0]["content"] = tool_prompt + "\n\n" + (messages[0].get("content") or "")
    else:
        messages.insert(0, {"role": "system", "content": tool_prompt})

    return messages


def _parse_tool_calls_from_text(text: str):
    """Try to parse tool calls from generated text."""
    from mlx_forge.serving.tool_parser import ToolCallParser

    parser = ToolCallParser()
    calls = parser.parse(text)
    if not calls:
        return None

    return [
        ToolCallMessage(
            function=ToolCallFunction(
                name=call.name,
                arguments=json.dumps(call.arguments),
            )
        )
        for call in calls
    ]


def _apply_json_constraint(text: str, response_format: dict | None) -> str:
    """Apply JSON constraint if response_format requests it."""
    if not response_format:
        return text

    fmt_type = response_format.get("type")
    if fmt_type not in ("json_object", "json_schema"):
        return text

    from mlx_forge.inference.constrained import JSONConstraint, JSONSchemaConstraint

    if fmt_type == "json_schema":
        schema = response_format.get("json_schema", {}).get("schema", {})
        constraint = JSONSchemaConstraint(schema)
        text = constraint.validate_and_repair(text)
        constraint.validate_against_schema(text)
        return text
    else:
        constraint = JSONConstraint()
        return constraint.validate_and_repair(text)


# ─── Health ───


@router.get("/health")
async def health():
    """Health check endpoint."""
    mgr = get_manager()
    cache_mgr = get_cache_manager()
    result = {
        "status": "ok",
        "model_loaded": mgr.is_loaded,
        "model_id": mgr.model_id,
    }
    if cache_mgr:
        result["cache"] = cache_mgr.stats()
    return result


# ─── Chat Completions ───


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    keep_alive = getattr(request, "keep_alive", None)
    mgr = _get_model_manager(request.model, keep_alive=keep_alive)

    stop_checker = _build_stop_checker(request, mgr.tokenizer)

    # Build messages for chat template
    messages = [{"role": m.role, "content": m.content or ""} for m in request.messages]
    messages = _inject_tools_into_messages(messages, request)

    # Tokenize with chat template
    prompt_tokens = mgr.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if not isinstance(prompt_tokens, list):
        prompt_tokens = list(prompt_tokens)

    # Multi-turn cache lookup
    cache_mgr = get_cache_manager()
    conv_id = request.conversation_id
    kv_cache = None
    tokens_to_prefill = prompt_tokens
    all_token_history = None

    if cache_mgr and conv_id:
        kv_cache, tokens_to_prefill = cache_mgr.get_or_create(
            conv_id, prompt_tokens, mgr.model
        )
        all_token_history = list(prompt_tokens)

    if request.stream:
        return StreamingResponse(
            _stream_chat(
                mgr, tokens_to_prefill, request, stop_checker,
                kv_cache=kv_cache, all_token_history=all_token_history,
                conv_id=conv_id, full_prompt_tokens=prompt_tokens,
            ),
            media_type="text/event-stream",
        )

    # Non-streaming
    from mlx_forge.inference.engine import generate_steps
    from mlx_forge.inference.metrics import MetricsTracker

    ctx_len, n_keep = _get_context_params(request)
    tracker = MetricsTracker(num_prompt_tokens=len(tokens_to_prefill))
    generated_ids = []
    generated_text = ""
    finish_reason = "length"
    logprobs_list = []
    first_token = True

    for step in generate_steps(
        mgr.model,
        tokens_to_prefill,
        mgr.tokenizer,
        cache=kv_cache,
        all_token_history=all_token_history,
        context_length=ctx_len,
        num_keep=n_keep,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs or 5,
    ):
        if first_token:
            tracker.mark_prefill_done()
            first_token = False
        tracker.mark_token()

        generated_ids.append(step.token_id)

        if stop_checker.check_token(step.token_id):
            finish_reason = "stop"
            break

        generated_text = mgr.tokenizer.decode(generated_ids)

        if step.logprob_result:
            logprobs_list.append(step.logprob_result)

        stopped, trimmed = stop_checker.check_text(generated_text)
        if stopped:
            generated_text = trimmed
            finish_reason = "stop"
            break

    if not generated_ids:
        finish_reason = "stop"
    elif finish_reason != "stop" and len(generated_ids) < request.max_tokens:
        finish_reason = "stop"
        generated_text = mgr.tokenizer.decode(generated_ids)

    generated_text = _apply_json_constraint(generated_text, request.response_format)

    # Update multi-turn cache
    if cache_mgr and conv_id and kv_cache is not None:
        all_tokens = list(prompt_tokens) + generated_ids
        cache_mgr.update(conv_id, kv_cache, all_tokens)

    tool_calls = None
    if request.tools:
        tool_calls = _parse_tool_calls_from_text(generated_text)
        if tool_calls:
            finish_reason = "tool_calls"

    metrics = tracker.finish()

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            Choice(
                message=ChatMessage(
                    role="assistant",
                    content=generated_text if not tool_calls else None,
                    tool_calls=tool_calls,
                ),
                finish_reason=finish_reason,
                logprobs=_build_logprobs_response(logprobs_list) if request.logprobs else None,
            )
        ],
        usage=_build_usage(len(prompt_tokens), len(generated_ids), metrics),
    )


async def _stream_chat(
    mgr, tokens_to_prefill, request, stop_checker, *,
    kv_cache=None, all_token_history=None, conv_id=None, full_prompt_tokens=None,
):
    """Generate SSE stream for chat completions."""
    from mlx_forge.inference.engine import generate_steps

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    initial_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaContent(role="assistant"))],
    )
    yield f"data: {initial_chunk.model_dump_json()}\n\n"

    generated_ids = []
    generated_text = ""
    buffer = []
    finish_reason = None

    ctx_len, n_keep = _get_context_params(request)
    for step in generate_steps(
        mgr.model,
        tokens_to_prefill,
        mgr.tokenizer,
        cache=kv_cache,
        all_token_history=all_token_history,
        context_length=ctx_len,
        num_keep=n_keep,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    ):
        if stop_checker.check_token(step.token_id):
            finish_reason = "stop"
            break

        generated_ids.append(step.token_id)
        buffer.append(step.token_id)
        text = mgr.tokenizer.decode(buffer)

        if text and "\ufffd" not in text:
            generated_text += text
            buffer.clear()

            stopped, trimmed = stop_checker.check_text(generated_text)
            if stopped:
                finish_reason = "stop"
                break

            chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=request.model,
                choices=[StreamChoice(delta=DeltaContent(content=text))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    if finish_reason is None:
        finish_reason = "stop" if len(generated_ids) < request.max_tokens else "length"

    # Update multi-turn cache
    cache_mgr = get_cache_manager()
    if cache_mgr and conv_id and kv_cache is not None:
        all_tokens = list(full_prompt_tokens or []) + generated_ids
        cache_mgr.update(conv_id, kv_cache, all_tokens)

    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaContent(), finish_reason=finish_reason)],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ─── Text Completions ───


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible text completions endpoint."""
    keep_alive = getattr(request, "keep_alive", None)
    mgr = _get_model_manager(request.model, keep_alive=keep_alive)

    stop_checker = _build_stop_checker(request, mgr.tokenizer)
    prompt_tokens = mgr.tokenizer.encode(request.prompt)

    if request.stream:
        return StreamingResponse(
            _stream_completion(mgr, prompt_tokens, request, stop_checker),
            media_type="text/event-stream",
        )

    # Non-streaming
    from mlx_forge.inference.engine import generate_steps
    from mlx_forge.inference.metrics import MetricsTracker

    ctx_len, n_keep = _get_context_params(request)
    tracker = MetricsTracker(num_prompt_tokens=len(prompt_tokens))
    generated_ids = []
    generated_text = ""
    finish_reason = "length"
    logprobs_list = []
    first_token = True

    for step in generate_steps(
        mgr.model,
        prompt_tokens,
        mgr.tokenizer,
        context_length=ctx_len,
        num_keep=n_keep,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
        logprobs=request.logprobs,
        top_logprobs=request.top_logprobs or 5,
    ):
        if first_token:
            tracker.mark_prefill_done()
            first_token = False
        tracker.mark_token()

        generated_ids.append(step.token_id)

        if stop_checker.check_token(step.token_id):
            finish_reason = "stop"
            break

        generated_text = mgr.tokenizer.decode(generated_ids)

        if step.logprob_result:
            logprobs_list.append(step.logprob_result)

        stopped, trimmed = stop_checker.check_text(generated_text)
        if stopped:
            generated_text = trimmed
            finish_reason = "stop"
            break

    if not generated_ids:
        finish_reason = "stop"
    elif finish_reason != "stop" and len(generated_ids) < request.max_tokens:
        finish_reason = "stop"
        generated_text = mgr.tokenizer.decode(generated_ids)

    generated_text = _apply_json_constraint(generated_text, request.response_format)

    metrics = tracker.finish()

    return CompletionResponse(
        model=request.model,
        choices=[
            CompletionChoice(
                text=generated_text,
                finish_reason=finish_reason,
                logprobs=_build_logprobs_response(logprobs_list) if request.logprobs else None,
            )
        ],
        usage=_build_usage(len(prompt_tokens), len(generated_ids), metrics),
    )


async def _stream_completion(mgr, prompt_tokens, request, stop_checker):
    """Generate SSE stream for text completions."""
    from mlx_forge.inference.engine import generate_steps

    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    generated_ids = []
    generated_text = ""
    buffer = []
    finish_reason = None

    ctx_len, n_keep = _get_context_params(request)
    for step in generate_steps(
        mgr.model,
        prompt_tokens,
        mgr.tokenizer,
        context_length=ctx_len,
        num_keep=n_keep,
        temperature=request.temperature,
        top_p=request.top_p,
        top_k=request.top_k,
        min_p=request.min_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
        frequency_penalty=request.frequency_penalty,
        presence_penalty=request.presence_penalty,
    ):
        if stop_checker.check_token(step.token_id):
            finish_reason = "stop"
            break

        generated_ids.append(step.token_id)
        buffer.append(step.token_id)
        text = mgr.tokenizer.decode(buffer)

        if text and "\ufffd" not in text:
            generated_text += text
            buffer.clear()

            stopped, trimmed = stop_checker.check_text(generated_text)
            if stopped:
                finish_reason = "stop"
                break

            chunk = CompletionChunk(
                id=chunk_id,
                created=created,
                model=request.model,
                choices=[CompletionStreamChoice(text=text)],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    if finish_reason is None:
        finish_reason = "stop" if len(generated_ids) < request.max_tokens else "length"

    final_chunk = CompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[CompletionStreamChoice(text="", finish_reason=finish_reason)],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


# ─── Embeddings ───


@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint for encoder models."""
    mgr = _get_model_manager(request.model)

    model_cat = getattr(mgr.model, "model_category", "decoder")
    if model_cat != "encoder":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not an encoder model (got {model_cat}). "
            "Embeddings require BERT/RoBERTa/DeBERTa.",
        )

    from mlx_forge.inference.encoder import encode

    texts = request.input if isinstance(request.input, list) else [request.input]
    emb_list = encode(mgr.model, mgr.tokenizer, texts, pooling="cls", normalize=True)
    total_tokens = sum(len(mgr.tokenizer.encode(t)) for t in texts)

    data = [
        EmbeddingData(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(emb_list)
    ]

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(prompt_tokens=total_tokens, total_tokens=total_tokens),
    )


# ─── Adapter endpoints ───


class AdapterLoadRequest(BaseModel):
    adapter_path: str


@router.post("/v1/adapters/load")
async def load_adapter(request: AdapterLoadRequest):
    """Hot-load a LoRA adapter onto the current model."""
    mgr = get_manager()
    if not mgr.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        if mgr._base_weights is None:
            mgr.snapshot_base_weights()
        mgr.load_adapter(request.adapter_path)
    except (ValueError, FileNotFoundError) as e:
        raise HTTPException(status_code=400, detail=str(e))

    # Invalidate conversation caches (adapter changed base weights)
    cache_mgr = get_cache_manager()
    if cache_mgr:
        for cid in list(cache_mgr._conversations.keys()):
            cache_mgr.evict(cid)

    return {"status": "ok", "adapter_path": request.adapter_path}


@router.delete("/v1/adapters")
async def unload_adapter():
    """Unload the current LoRA adapter, restoring base weights."""
    mgr = get_manager()
    if not mgr.is_loaded:
        raise HTTPException(status_code=400, detail="No model loaded")

    try:
        mgr.unload_adapter()
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))

    return {"status": "ok", "message": "Adapter unloaded, base weights restored"}


@router.get("/v1/adapters")
async def get_adapter_info():
    """Get information about the currently loaded adapter."""
    mgr = get_manager()
    return mgr.adapter_info()


# ─── Tokenize / Detokenize ───


class TokenizeRequest(BaseModel):
    model: str
    text: str


class DetokenizeRequest(BaseModel):
    model: str
    tokens: list[int]


@router.post("/v1/tokenize")
async def tokenize(request: TokenizeRequest):
    """Tokenize text into token IDs."""
    mgr = _get_model_manager(request.model)
    tokens = mgr.tokenizer.encode(request.text)
    return {"tokens": tokens, "count": len(tokens)}


@router.post("/v1/detokenize")
async def detokenize(request: DetokenizeRequest):
    """Detokenize token IDs into text."""
    mgr = _get_model_manager(request.model)
    text = mgr.tokenizer.decode(request.tokens)
    return {"text": text}


# ─── Models ───


@router.get("/v1/models")
async def list_models():
    """List available models."""
    pool = get_pool()
    if pool is not None:
        # Include all loaded models + available on disk
        loaded_ids = {m["model_id"] for m in pool.status()}
        mgr = get_manager()
        available = mgr.list_available()
        models = [ModelObject(id=m["id"]) for m in available]
        for mid in loaded_ids:
            if not any(m.id == mid for m in models):
                models.insert(0, ModelObject(id=mid))
        return ModelListResponse(data=models)

    mgr = get_manager()
    available = mgr.list_available()
    models = [ModelObject(id=m["id"]) for m in available]

    if mgr.is_loaded and mgr.model_id:
        if not any(m.id == mgr.model_id for m in models):
            models.insert(0, ModelObject(id=mgr.model_id))

    return ModelListResponse(data=models)


# ─── Model lifecycle (M41) ───


@router.get("/v1/models/ps")
async def running_models():
    """List running models with lifecycle info (Ollama /api/ps equivalent)."""
    pool = get_pool()
    if pool is not None:
        return {
            "models": pool.status(),
            "max_models": pool.max_models,
            "loaded_count": pool.loaded_count,
        }
    # Legacy: single manager
    mgr = get_manager()
    if mgr.is_loaded:
        return {
            "models": [{"model_id": mgr.model_id, "adapter": mgr.adapter_path}],
            "max_models": 1,
            "loaded_count": 1,
        }
    return {"models": [], "max_models": 1, "loaded_count": 0}


class ModelLoadRequest(BaseModel):
    model: str
    keep_alive: str | int | None = None


@router.post("/v1/models/load")
async def preload_model(request: ModelLoadRequest):
    """Pre-warm a model without generating."""
    pool = get_pool()
    if pool is not None:
        try:
            pool.get(request.model, keep_alive=request.keep_alive)
        except Exception as e:
            raise HTTPException(status_code=400, detail=str(e))
        return {"status": "ok", "model": request.model}

    # Legacy
    mgr = get_manager()
    try:
        mgr.load(request.model)
    except Exception as e:
        raise HTTPException(status_code=400, detail=str(e))
    return {"status": "ok", "model": request.model}


class ModelUnloadRequest(BaseModel):
    model: str


@router.post("/v1/models/unload")
async def unload_model(request: ModelUnloadRequest):
    """Explicitly unload a model from memory."""
    pool = get_pool()
    if pool is not None:
        if pool.unload(request.model):
            return {"status": "ok", "model": request.model}
        raise HTTPException(
            status_code=404, detail=f"Model '{request.model}' is not loaded"
        )

    mgr = get_manager()
    if mgr.is_loaded and mgr.model_id == request.model:
        mgr.unload()
        return {"status": "ok", "model": request.model}
    raise HTTPException(
        status_code=404, detail=f"Model '{request.model}' is not loaded"
    )


@router.get("/v1/aliases")
async def list_aliases():
    """List model aliases."""
    pool = get_pool()
    if pool is not None:
        return {"aliases": pool.list_aliases()}
    return {"aliases": {}}

"""OpenAI-compatible API routes."""

from __future__ import annotations

import time
import uuid

from fastapi import APIRouter, HTTPException
from fastapi.responses import StreamingResponse

from mlx_forge.serving.model_manager import ModelManager
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
    EmbeddingData,
    EmbeddingRequest,
    EmbeddingResponse,
    EmbeddingUsage,
    ModelListResponse,
    ModelObject,
    StreamChoice,
    Usage,
)

router = APIRouter()
_manager = ModelManager()


def get_manager() -> ModelManager:
    return _manager


def set_manager(manager: ModelManager) -> None:
    global _manager
    _manager = manager


def _ensure_model_loaded(request_model: str) -> None:
    """Ensure the requested model is loaded."""
    mgr = get_manager()
    if not mgr.is_loaded or mgr.model_id != request_model:
        try:
            mgr.load(request_model)
        except Exception as e:
            raise HTTPException(
                status_code=400, detail=f"Failed to load model '{request_model}': {e}"
            )


def _normalize_stop(stop) -> list[str]:
    """Normalize stop sequences to a list."""
    if stop is None:
        return []
    if isinstance(stop, str):
        return [stop]
    return list(stop)


@router.post("/v1/chat/completions")
async def chat_completions(request: ChatCompletionRequest):
    """OpenAI-compatible chat completions endpoint."""
    _ensure_model_loaded(request.model)
    mgr = get_manager()

    stop_sequences = _normalize_stop(request.stop)

    # Build messages for chat template
    messages = [{"role": m.role, "content": m.content} for m in request.messages]

    # Tokenize with chat template
    prompt_tokens = mgr.tokenizer.apply_chat_template(messages, add_generation_prompt=True)
    if not isinstance(prompt_tokens, list):
        prompt_tokens = list(prompt_tokens)

    if request.stream:
        return StreamingResponse(
            _stream_chat(mgr, prompt_tokens, request, stop_sequences),
            media_type="text/event-stream",
        )

    # Non-streaming
    from mlx_forge.inference.engine import generate_tokens

    generated_ids = []
    generated_text = ""
    finish_reason = "length"

    for token_id in generate_tokens(
        mgr.model,
        prompt_tokens,
        mgr.tokenizer,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
    ):
        generated_ids.append(token_id)
        generated_text = mgr.tokenizer.decode(generated_ids)

        # Check stop sequences
        stopped = False
        for seq in stop_sequences:
            if seq in generated_text:
                generated_text = generated_text[: generated_text.index(seq)]
                finish_reason = "stop"
                stopped = True
                break
        if stopped:
            break

    if not generated_ids:
        finish_reason = "stop"
    elif finish_reason != "stop" and len(generated_ids) < request.max_tokens:
        finish_reason = "stop"  # EOS

    return ChatCompletionResponse(
        model=request.model,
        choices=[
            Choice(
                message=ChatMessage(role="assistant", content=generated_text),
                finish_reason=finish_reason,
            )
        ],
        usage=Usage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(generated_ids),
            total_tokens=len(prompt_tokens) + len(generated_ids),
        ),
    )


async def _stream_chat(mgr, prompt_tokens, request, stop_sequences):
    """Generate SSE stream for chat completions."""
    from mlx_forge.inference.engine import generate_tokens

    chunk_id = f"chatcmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    # Send initial chunk with role
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

    for token_id in generate_tokens(
        mgr.model,
        prompt_tokens,
        mgr.tokenizer,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
    ):
        generated_ids.append(token_id)
        buffer.append(token_id)
        text = mgr.tokenizer.decode(buffer)

        if text and "\ufffd" not in text:
            generated_text += text
            buffer.clear()

            # Check stop sequences
            stopped = False
            for seq in stop_sequences:
                if seq in generated_text:
                    finish_reason = "stop"
                    stopped = True
                    break

            if stopped:
                break

            chunk = ChatCompletionChunk(
                id=chunk_id,
                created=created,
                model=request.model,
                choices=[StreamChoice(delta=DeltaContent(content=text))],
            )
            yield f"data: {chunk.model_dump_json()}\n\n"

    # Final chunk
    if finish_reason is None:
        finish_reason = "stop" if len(generated_ids) < request.max_tokens else "length"

    final_chunk = ChatCompletionChunk(
        id=chunk_id,
        created=created,
        model=request.model,
        choices=[StreamChoice(delta=DeltaContent(), finish_reason=finish_reason)],
    )
    yield f"data: {final_chunk.model_dump_json()}\n\n"
    yield "data: [DONE]\n\n"


@router.post("/v1/completions")
async def completions(request: CompletionRequest):
    """OpenAI-compatible text completions endpoint."""
    _ensure_model_loaded(request.model)
    mgr = get_manager()

    stop_sequences = _normalize_stop(request.stop)
    prompt_tokens = mgr.tokenizer.encode(request.prompt)

    if request.stream:
        return StreamingResponse(
            _stream_completion(mgr, prompt_tokens, request, stop_sequences),
            media_type="text/event-stream",
        )

    # Non-streaming
    from mlx_forge.inference.engine import generate_tokens

    generated_ids = []
    generated_text = ""
    finish_reason = "length"

    for token_id in generate_tokens(
        mgr.model,
        prompt_tokens,
        mgr.tokenizer,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
    ):
        generated_ids.append(token_id)
        generated_text = mgr.tokenizer.decode(generated_ids)

        stopped = False
        for seq in stop_sequences:
            if seq in generated_text:
                generated_text = generated_text[: generated_text.index(seq)]
                finish_reason = "stop"
                stopped = True
                break
        if stopped:
            break

    if not generated_ids:
        finish_reason = "stop"
    elif finish_reason != "stop" and len(generated_ids) < request.max_tokens:
        finish_reason = "stop"

    return CompletionResponse(
        model=request.model,
        choices=[CompletionChoice(text=generated_text, finish_reason=finish_reason)],
        usage=Usage(
            prompt_tokens=len(prompt_tokens),
            completion_tokens=len(generated_ids),
            total_tokens=len(prompt_tokens) + len(generated_ids),
        ),
    )


async def _stream_completion(mgr, prompt_tokens, request, stop_sequences):
    """Generate SSE stream for text completions."""
    from mlx_forge.inference.engine import generate_tokens

    chunk_id = f"cmpl-{uuid.uuid4().hex[:8]}"
    created = int(time.time())

    generated_ids = []
    generated_text = ""
    buffer = []
    finish_reason = None

    for token_id in generate_tokens(
        mgr.model,
        prompt_tokens,
        mgr.tokenizer,
        temperature=request.temperature,
        top_p=request.top_p,
        max_tokens=request.max_tokens,
        repetition_penalty=request.repetition_penalty,
    ):
        generated_ids.append(token_id)
        buffer.append(token_id)
        text = mgr.tokenizer.decode(buffer)

        if text and "\ufffd" not in text:
            generated_text += text
            buffer.clear()

            stopped = False
            for seq in stop_sequences:
                if seq in generated_text:
                    finish_reason = "stop"
                    stopped = True
                    break

            if stopped:
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


@router.post("/v1/embeddings")
async def embeddings(request: EmbeddingRequest):
    """OpenAI-compatible embeddings endpoint for encoder models."""
    _ensure_model_loaded(request.model)
    mgr = get_manager()

    model_cat = getattr(mgr.model, "model_category", "decoder")
    if model_cat != "encoder":
        raise HTTPException(
            status_code=400,
            detail=f"Model '{request.model}' is not an encoder model (got {model_cat}). "
            "Embeddings require BERT/RoBERTa/DeBERTa.",
        )

    from mlx_forge.inference.encoder import encode

    texts = request.input if isinstance(request.input, list) else [request.input]

    emb_list = encode(
        mgr.model,
        mgr.tokenizer,
        texts,
        pooling="cls",
        normalize=True,
    )

    total_tokens = sum(len(mgr.tokenizer.encode(t)) for t in texts)

    data = [
        EmbeddingData(embedding=emb.tolist(), index=i)
        for i, emb in enumerate(emb_list)
    ]

    return EmbeddingResponse(
        data=data,
        model=request.model,
        usage=EmbeddingUsage(
            prompt_tokens=total_tokens,
            total_tokens=total_tokens,
        ),
    )


@router.get("/v1/models")
async def list_models():
    """List available models."""
    mgr = get_manager()
    available = mgr.list_available()

    models = [ModelObject(id=m["id"]) for m in available]

    # Include currently loaded model if not in list
    if mgr.is_loaded and mgr.model_id:
        if not any(m.id == mgr.model_id for m in models):
            models.insert(0, ModelObject(id=mgr.model_id))

    return ModelListResponse(data=models)

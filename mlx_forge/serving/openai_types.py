"""Pydantic models for OpenAI-compatible API requests and responses."""

from __future__ import annotations

import time
import uuid
from typing import Optional

from pydantic import BaseModel, Field

# --- Tool/Function calling types (M39) ---


class FunctionDef(BaseModel):
    name: str
    description: str = ""
    parameters: dict = Field(default_factory=dict)


class ToolDef(BaseModel):
    type: str = "function"
    function: FunctionDef


class ToolCallFunction(BaseModel):
    name: str
    arguments: str  # JSON string


class ToolCallMessage(BaseModel):
    id: str = Field(default_factory=lambda: f"call_{uuid.uuid4().hex[:8]}")
    type: str = "function"
    function: ToolCallFunction


# --- Chat messages ---


class ChatMessage(BaseModel):
    role: str
    content: str | None = None
    tool_calls: list[ToolCallMessage] | None = None


# --- Requests ---


class ChatCompletionRequest(BaseModel):
    model: str
    messages: list[ChatMessage]
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[list[str] | str] = None
    repetition_penalty: float = 1.0
    # M37: new sampling params
    top_k: int = 0
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int | None = None
    # M39: structured generation
    tools: list[ToolDef] | None = None
    tool_choice: str | dict | None = None
    response_format: dict | None = None
    stop_token_ids: list[int] | None = None
    # M40: multi-turn cache
    conversation_id: str | None = None
    # M41: model lifecycle
    keep_alive: str | int | None = None


class CompletionRequest(BaseModel):
    model: str
    prompt: str
    temperature: float = 0.7
    top_p: float = 0.9
    max_tokens: int = 512
    stream: bool = False
    stop: Optional[list[str] | str] = None
    repetition_penalty: float = 1.0
    # M37: new sampling params
    top_k: int = 0
    min_p: float = 0.0
    frequency_penalty: float = 0.0
    presence_penalty: float = 0.0
    logprobs: bool = False
    top_logprobs: int | None = None
    # M39
    response_format: dict | None = None
    stop_token_ids: list[int] | None = None
    # M41: model lifecycle
    keep_alive: str | int | None = None


# --- Logprobs response types ---


class TopLogprob(BaseModel):
    token: str
    token_id: int
    logprob: float


class LogprobContent(BaseModel):
    token: str
    token_id: int
    logprob: float
    top_logprobs: list[TopLogprob] = Field(default_factory=list)


class ChoiceLogprobs(BaseModel):
    content: list[LogprobContent] = Field(default_factory=list)


# --- Responses ---


class Choice(BaseModel):
    index: int = 0
    message: ChatMessage
    finish_reason: str
    logprobs: ChoiceLogprobs | None = None


class CompletionChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: str
    logprobs: ChoiceLogprobs | None = None


class Usage(BaseModel):
    prompt_tokens: int
    completion_tokens: int
    total_tokens: int
    # Timing extensions (Ollama-compatible)
    ttft_ms: float | None = None
    prompt_eval_duration_ms: float | None = None
    eval_duration_ms: float | None = None
    decode_tokens_per_sec: float | None = None


class ChatCompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"chatcmpl-{uuid.uuid4().hex[:8]}")
    object: str = "chat.completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[Choice]
    usage: Usage


class CompletionResponse(BaseModel):
    id: str = Field(default_factory=lambda: f"cmpl-{uuid.uuid4().hex[:8]}")
    object: str = "text_completion"
    created: int = Field(default_factory=lambda: int(time.time()))
    model: str
    choices: list[CompletionChoice]
    usage: Usage


class DeltaContent(BaseModel):
    role: Optional[str] = None
    content: Optional[str] = None
    tool_calls: list[ToolCallMessage] | None = None


class StreamChoice(BaseModel):
    index: int = 0
    delta: DeltaContent
    finish_reason: Optional[str] = None


class ChatCompletionChunk(BaseModel):
    id: str
    object: str = "chat.completion.chunk"
    created: int
    model: str
    choices: list[StreamChoice]


class CompletionStreamChoice(BaseModel):
    index: int = 0
    text: str
    finish_reason: Optional[str] = None


class CompletionChunk(BaseModel):
    id: str
    object: str = "text_completion"
    created: int
    model: str
    choices: list[CompletionStreamChoice]


class ModelObject(BaseModel):
    id: str
    object: str = "model"
    created: int = Field(default_factory=lambda: int(time.time()))
    owned_by: str = "mlx-forge"


class ModelListResponse(BaseModel):
    object: str = "list"
    data: list[ModelObject]


class EmbeddingRequest(BaseModel):
    model: str
    input: list[str] | str
    encoding_format: str = "float"


class EmbeddingData(BaseModel):
    object: str = "embedding"
    embedding: list[float]
    index: int


class EmbeddingUsage(BaseModel):
    prompt_tokens: int
    total_tokens: int


class EmbeddingResponse(BaseModel):
    object: str = "list"
    data: list[EmbeddingData]
    model: str
    usage: EmbeddingUsage

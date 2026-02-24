"""Inference API — text generation endpoint."""

from __future__ import annotations

from fastapi import APIRouter, HTTPException

router = APIRouter(prefix="/api/v1/inference", tags=["inference"])

# Track currently loaded model
_loaded_model_info: dict | None = None


@router.post("/generate")
def generate_text(request: dict):
    """Generate text using a model with optional LoRA adapter.

    Request body:
        model: HF model ID or local path (required)
        prompt: Text prompt (mutually exclusive with messages)
        messages: Chat messages list (mutually exclusive with prompt)
        adapter: Path to checkpoint directory
        temperature: float (default 0.7)
        top_p: float (default 0.9)
        max_tokens: int (default 512)
    """
    model_path = request.get("model")
    if not model_path:
        raise HTTPException(status_code=400, detail="'model' field is required")

    prompt = request.get("prompt")
    messages = request.get("messages")
    if prompt is None and messages is None:
        raise HTTPException(status_code=400, detail="Must provide 'prompt' or 'messages'")

    try:
        from lmforge import generate as lmforge_generate

        result = lmforge_generate(
            model=model_path,
            prompt=prompt,
            messages=messages,
            adapter=request.get("adapter"),
            temperature=request.get("temperature", 0.7),
            top_p=request.get("top_p", 0.9),
            max_tokens=request.get("max_tokens", 512),
            trust_remote_code=request.get("trust_remote_code", False),
            seed=request.get("seed"),
        )
        return {
            "text": result.text,
            "num_tokens": result.num_tokens,
            "tokens_per_second": result.tokens_per_second,
            "finish_reason": result.finish_reason,
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@router.get("/status")
def inference_status():
    """Get current inference status (loaded model info)."""
    return {"loaded_model": _loaded_model_info}

"""FastAPI application factory for OpenAI-compatible serving."""

from __future__ import annotations

from fastapi import FastAPI

from mlx_forge.serving.cache_manager import CacheManager
from mlx_forge.serving.model_manager import ModelManager
from mlx_forge.serving.routes import router, set_cache_manager, set_manager


def create_serving_app(
    model: str | None = None,
    adapter: str | None = None,
    draft_model: str | None = None,
    max_conversations: int = 16,
    cache_ttl: float = 600,
) -> FastAPI:
    """Create the OpenAI-compatible serving app.

    Args:
        model: Optional model to pre-load on startup.
        adapter: Optional adapter path.
        draft_model: Optional draft model for speculative decoding.
        max_conversations: Max multi-turn conversation caches.
        cache_ttl: Conversation cache TTL in seconds.
    """
    app = FastAPI(
        title="MLX Forge Serving",
        description="OpenAI-compatible API for locally fine-tuned models",
        version="1.0.0",
    )

    # Create shared cache manager for multi-turn KV persistence
    cache_mgr = CacheManager(
        max_conversations=max_conversations,
        ttl_seconds=cache_ttl,
    )
    set_cache_manager(cache_mgr)

    app.include_router(router)

    if model:

        @app.on_event("startup")
        async def preload_model():
            mgr = ModelManager()
            mgr.load(model, adapter=adapter)
            mgr.snapshot_base_weights()
            if draft_model:
                mgr.load_draft(draft_model)
            set_manager(mgr)

    return app

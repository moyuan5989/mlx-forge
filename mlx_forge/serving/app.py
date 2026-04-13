"""FastAPI application factory for OpenAI-compatible serving."""

from __future__ import annotations

import asyncio
import logging

from fastapi import FastAPI

from mlx_forge.serving.cache_manager import CacheManager
from mlx_forge.serving.model_pool import ModelPool, parse_keep_alive
from mlx_forge.serving.routes import (
    router,
    set_cache_manager,
    set_context_defaults,
    set_manager,
    set_pool,
)

logger = logging.getLogger(__name__)


def create_serving_app(
    model: str | None = None,
    adapter: str | None = None,
    draft_model: str | None = None,
    max_conversations: int = 16,
    cache_ttl: float = 600,
    max_models: int = 1,
    keep_alive: str | int | float = "5m",
    aliases_path: str | None = None,
    context_length: int = 0,
    num_keep: int = 0,
) -> FastAPI:
    """Create the OpenAI-compatible serving app.

    Args:
        model: Optional model to pre-load on startup.
        adapter: Optional adapter path.
        draft_model: Optional draft model for speculative decoding.
        max_conversations: Max multi-turn conversation caches.
        cache_ttl: Conversation cache TTL in seconds.
        max_models: Max models to keep in memory simultaneously.
        keep_alive: Default model keep-alive ("5m", 300, -1, etc.).
        aliases_path: Path to aliases JSON file.
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

    # Create model pool for lifecycle management
    default_ka = parse_keep_alive(keep_alive, 300.0)
    pool = ModelPool(max_models=max_models, default_keep_alive=default_ka)

    # Load aliases if provided
    if aliases_path:
        pool.load_aliases(aliases_path)

    set_pool(pool)
    set_context_defaults(context_length=context_length, num_keep=num_keep)

    app.include_router(router)

    if model:

        @app.on_event("startup")
        async def preload_model():
            from mlx_forge.serving.model_manager import ModelManager

            mgr = ModelManager()
            mgr.load(model, adapter=adapter)
            mgr.snapshot_base_weights()
            if draft_model:
                mgr.load_draft(draft_model)
            set_manager(mgr)

            # Also register in pool so lifecycle management applies
            from mlx_forge.serving.model_pool import ManagedModel

            managed = ManagedModel(
                manager=mgr,
                model_id=mgr.model_id,
                keep_alive=default_ka,
            )
            pool._models[mgr.model_id] = managed

    @app.on_event("startup")
    async def start_eviction_loop():
        """Periodically evict expired models."""

        async def _evict_loop():
            while True:
                await asyncio.sleep(30)
                evicted = pool.tick()
                if evicted:
                    logger.info("Evicted models: %s", evicted)

        asyncio.create_task(_evict_loop())

    return app

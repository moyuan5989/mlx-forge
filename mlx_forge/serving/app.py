"""FastAPI application factory for OpenAI-compatible serving."""

from __future__ import annotations

from fastapi import FastAPI

from mlx_forge.serving.model_manager import ModelManager
from mlx_forge.serving.routes import router, set_manager


def create_serving_app(
    model: str | None = None,
    adapter: str | None = None,
) -> FastAPI:
    """Create the OpenAI-compatible serving app.

    Args:
        model: Optional model to pre-load on startup.
        adapter: Optional adapter path.
    """
    app = FastAPI(
        title="MLX Forge Serving",
        description="OpenAI-compatible API for locally fine-tuned models",
        version="1.0.0",
    )

    app.include_router(router)

    if model:

        @app.on_event("startup")
        async def preload_model():
            mgr = ModelManager()
            mgr.load(model, adapter=adapter)
            set_manager(mgr)

    return app

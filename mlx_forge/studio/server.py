"""FastAPI application for MLX Forge Studio.

Mounts REST API routes and WebSocket hubs.
"""

from __future__ import annotations

import asyncio
from pathlib import Path

from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles

from mlx_forge._version import __version__
from mlx_forge.studio.api import (
    config_schema,
    data_library,
    datasets,
    inference,
    memory,
    models,
    queue,
    recipes,
    runs,
    training,
)
from mlx_forge.studio.services.metrics_watcher import MetricsWatcher


def create_app(runs_dir: str = "~/.mlxforge/runs") -> FastAPI:
    """Create and configure the FastAPI application.

    Args:
        runs_dir: Root directory for training runs.

    Returns:
        Configured FastAPI app.
    """
    app = FastAPI(
        title="MLX Forge Studio",
        description="Fine-tune LLMs on your Mac with MLX",
        version=__version__,
    )

    # CORS for local development
    app.add_middleware(
        CORSMiddleware,
        allow_origins=["http://localhost:*", "http://127.0.0.1:*"],
        allow_origin_regex=r"http://(localhost|127\.0\.0\.1)(:\d+)?",
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Mount REST API routers — V1
    app.include_router(runs.router)
    app.include_router(models.router)
    app.include_router(datasets.router)
    app.include_router(training.router)
    app.include_router(inference.router)

    # Mount REST API routers — V2
    app.include_router(recipes.router)
    app.include_router(queue.router)
    app.include_router(memory.router)
    app.include_router(config_schema.router)
    app.include_router(models.router_v2)
    app.include_router(data_library.router)

    # Configure services with the runs directory
    from mlx_forge.studio.services.run_service import RunService
    runs.set_run_service(RunService(runs_dir))

    # WebSocket: training metrics streaming
    @app.websocket("/ws/training/{run_id}")
    async def ws_training(websocket: WebSocket, run_id: str):
        await websocket.accept()

        runs_path = Path(runs_dir).expanduser()
        metrics_path = runs_path / run_id / "logs" / "metrics.jsonl"

        if not (runs_path / run_id).exists():
            await websocket.send_json({"type": "error", "detail": f"Run '{run_id}' not found"})
            await websocket.close()
            return

        watcher = MetricsWatcher(metrics_path)

        try:
            while True:
                # Poll for new metrics first
                new_metrics = watcher.poll()
                for metric in new_metrics:
                    await websocket.send_json({"type": "metric", "data": metric})

                # Check for incoming messages (non-blocking)
                try:
                    msg = await asyncio.wait_for(websocket.receive_json(), timeout=1.0)
                    if msg.get("type") == "stop":
                        # Final poll before stopping
                        for metric in watcher.poll():
                            await websocket.send_json({"type": "metric", "data": metric})
                        await websocket.send_json({"type": "stopped"})
                        break
                except asyncio.TimeoutError:
                    pass
        except WebSocketDisconnect:
            pass

    # WebSocket: inference streaming
    @app.websocket("/ws/inference")
    async def ws_inference(websocket: WebSocket):
        await websocket.accept()

        try:
            while True:
                msg = await websocket.receive_json()

                if msg.get("type") != "generate":
                    await websocket.send_json({"type": "error", "detail": "Expected type 'generate'"})
                    continue

                model_path = msg.get("model")
                if not model_path:
                    await websocket.send_json({"type": "error", "detail": "'model' field required"})
                    continue

                messages = msg.get("messages")
                prompt = msg.get("prompt")
                config = msg.get("config", {})

                try:
                    from mlx_forge import generate as mlx_forge_generate

                    token_gen = mlx_forge_generate(
                        model=model_path,
                        prompt=prompt,
                        messages=messages,
                        adapter=config.get("adapter"),
                        temperature=config.get("temperature", 0.7),
                        top_p=config.get("top_p", 0.9),
                        max_tokens=config.get("max_tokens", 512),
                        trust_remote_code=config.get("trust_remote_code", False),
                        seed=config.get("seed"),
                        stream=True,
                    )

                    num_tokens = 0
                    for token_text in token_gen:
                        num_tokens += 1
                        await websocket.send_json({"type": "token", "text": token_text})

                    await websocket.send_json({
                        "type": "done",
                        "stats": {"num_tokens": num_tokens},
                    })
                except Exception as e:
                    await websocket.send_json({"type": "error", "detail": str(e)})

        except WebSocketDisconnect:
            pass

    # Serve built frontend SPA
    frontend_dir = Path(__file__).parent / "frontend"
    if frontend_dir.exists() and (frontend_dir / "index.html").exists():
        # Mount static assets (JS, CSS)
        assets_dir = frontend_dir / "assets"
        if assets_dir.exists():
            app.mount("/assets", StaticFiles(directory=str(assets_dir)), name="assets")

        @app.get("/{path:path}")
        async def spa_fallback(path: str):
            """Serve index.html for all non-API routes (SPA client-side routing)."""
            # Security: reject path traversal and null bytes
            if "\x00" in path or ".." in path.split("/"):
                return HTMLResponse("Forbidden", status_code=403)
            # Try to serve the exact file first
            file_path = (frontend_dir / path).resolve()
            if not file_path.is_relative_to(frontend_dir.resolve()):
                return HTMLResponse("Forbidden", status_code=403)
            if path and file_path.exists() and file_path.is_file():
                return FileResponse(str(file_path))
            # Fall back to index.html for client-side routing
            return FileResponse(str(frontend_dir / "index.html"))
    else:
        @app.get("/", response_class=HTMLResponse)
        async def index():
            return (
                "<!DOCTYPE html><html><head><title>MLX Forge Studio</title></head>"
                "<body style='font-family:system-ui;text-align:center;margin:80px auto;max-width:600px'>"
                "<h1>MLX Forge Studio</h1>"
                "<p>Frontend not built. Run <code>npm run build</code> in studio-frontend/</p>"
                "<p>API: <a href='/docs'>/docs</a></p>"
                "</body></html>"
            )

    return app


def run_server(host: str = "127.0.0.1", port: int = 8741, runs_dir: str = "~/.mlxforge/runs"):
    """Start the Studio server.

    Args:
        host: Bind address.
        port: Port number.
        runs_dir: Root directory for training runs.
    """
    import uvicorn

    app = create_app(runs_dir=runs_dir)
    print(f"MLX Forge Studio starting on http://{host}:{port}")
    print(f"Runs directory: {Path(runs_dir).expanduser()}")
    uvicorn.run(app, host=host, port=port)

"""CLI command: serve — start OpenAI-compatible API server."""

from __future__ import annotations


def run_serve(args) -> None:
    """Start the OpenAI-compatible serving server."""
    import uvicorn

    from mlx_forge.serving.app import create_serving_app

    app = create_serving_app(
        model=args.model,
        adapter=getattr(args, "adapter", None),
        draft_model=getattr(args, "draft_model", None),
        max_models=getattr(args, "max_models", 1),
        keep_alive=getattr(args, "keep_alive", "5m"),
        aliases_path=getattr(args, "aliases", None),
    )

    print(f"Starting MLX Forge serving on {args.host}:{args.port}")
    if args.model:
        print(f"Pre-loading model: {args.model}")
    if getattr(args, "draft_model", None):
        print(f"Draft model (speculative): {args.draft_model}")
    max_m = getattr(args, "max_models", 1)
    ka = getattr(args, "keep_alive", "5m")
    print(f"Max models: {max_m}, keep-alive: {ka}")
    print()
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1")
    print("  POST /v1/chat/completions")
    print("  POST /v1/completions")
    print("  GET  /v1/models")
    print("  GET  /v1/models/ps")
    print("  GET  /health")
    print()

    uvicorn.run(app, host=args.host, port=args.port)

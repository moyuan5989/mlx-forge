"""CLI command: serve — start OpenAI-compatible API server."""

from __future__ import annotations


def run_serve(args) -> None:
    """Start the OpenAI-compatible serving server."""
    import uvicorn

    from mlx_forge.serving.app import create_serving_app

    app = create_serving_app(
        model=args.model,
        adapter=getattr(args, "adapter", None),
    )

    print(f"Starting MLX Forge serving on {args.host}:{args.port}")
    if args.model:
        print(f"Pre-loading model: {args.model}")
    print()
    print(f"OpenAI-compatible API: http://{args.host}:{args.port}/v1")
    print("  POST /v1/chat/completions")
    print("  POST /v1/completions")
    print("  GET  /v1/models")
    print()

    uvicorn.run(app, host=args.host, port=args.port)

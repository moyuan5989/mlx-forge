"""Handler for 'mlx_forge generate' CLI command."""

from __future__ import annotations

import time


def run_generate(args) -> None:
    """Execute the generate command from parsed CLI args."""
    from mlx_forge.inference.engine import generate, load_for_inference

    print("Loading model...")
    model, tokenizer = load_for_inference(
        args.model,
        adapter_path=args.adapter,
        trust_remote_code=args.trust_remote_code,
    )
    if args.adapter:
        print(f"Model: {args.model} + adapter: {args.adapter}")
    else:
        print(f"Model: {args.model}")
    print()

    # Speculative decoding mode
    if getattr(args, "draft_model", None) and args.prompt:
        _run_speculative(model, tokenizer, args)
        return

    if args.prompt:
        # Single-shot generation
        result = generate(
            model,
            tokenizer,
            prompt=args.prompt,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=getattr(args, "top_k", 0),
            min_p=getattr(args, "min_p", 0.0),
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=getattr(args, "frequency_penalty", 0.0),
            presence_penalty=getattr(args, "presence_penalty", 0.0),
            seed=args.seed,
        )
        print(result.text)
        stats = f"\n[{result.num_tokens} tokens, {result.tokens_per_second:.1f} tok/s"
        if result.metrics:
            stats += f", TTFT: {result.metrics.ttft_ms:.0f}ms"
        stats += f", finish: {result.finish_reason}]"
        print(stats)
    else:
        # Interactive chat mode
        _run_interactive(model, tokenizer, args)


def _run_speculative(model, tokenizer, args) -> None:
    """Single-shot generation with speculative decoding."""
    from mlx_forge.inference.engine import load_for_inference
    from mlx_forge.inference.speculative import speculative_generate_tokens

    print(f"Loading draft model: {args.draft_model}")
    draft_model, _ = load_for_inference(
        args.draft_model,
        trust_remote_code=getattr(args, "trust_remote_code", False),
    )

    prompt_tokens = tokenizer.encode(args.prompt)
    t0 = time.perf_counter()
    generated_text = ""
    token_buffer = []
    num_tokens = 0
    num_accepted = 0

    for token_id, from_draft in speculative_generate_tokens(
        model,
        draft_model,
        prompt_tokens,
        tokenizer,
        num_draft=getattr(args, "num_draft", 5),
        temperature=args.temperature,
        top_p=args.top_p,
        max_tokens=args.max_tokens,
        repetition_penalty=args.repetition_penalty,
    ):
        num_tokens += 1
        if from_draft:
            num_accepted += 1
        token_buffer.append(token_id)
        text = tokenizer.decode(token_buffer)
        if text and "\ufffd" not in text:
            print(text, end="", flush=True)
            generated_text += text
            token_buffer.clear()

    if token_buffer:
        text = tokenizer.decode(token_buffer)
        print(text, end="", flush=True)

    elapsed = time.perf_counter() - t0
    tok_s = num_tokens / elapsed if elapsed > 0 else 0.0
    accept_rate = num_accepted / num_tokens * 100 if num_tokens > 0 else 0
    print(f"\n[{num_tokens} tokens, {tok_s:.1f} tok/s, "
          f"draft accepted: {accept_rate:.0f}%]")


def _run_interactive(model, tokenizer, args) -> None:
    """Interactive chat REPL."""
    from mlx_forge.inference.engine import generate_tokens

    print("MLX Forge Interactive Generation")
    print("Type 'quit' to exit, 'clear' to reset context.")
    print()

    messages = []
    while True:
        try:
            user_input = input("You: ")
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if user_input.strip().lower() == "quit":
            break
        if user_input.strip().lower() == "clear":
            messages = []
            print("[Context cleared]\n")
            continue
        if not user_input.strip():
            continue

        messages.append({"role": "user", "content": user_input})
        prompt_tokens = tokenizer.apply_chat_template(
            messages, add_generation_prompt=True
        )
        if not isinstance(prompt_tokens, list):
            prompt_tokens = prompt_tokens["input_ids"]

        print("Assistant: ", end="", flush=True)
        generated_text = ""
        t0 = time.perf_counter()
        num_tokens = 0

        token_buffer = []
        for token_id in generate_tokens(
            model,
            prompt_tokens,
            tokenizer,
            temperature=args.temperature,
            top_p=args.top_p,
            top_k=getattr(args, "top_k", 0),
            min_p=getattr(args, "min_p", 0.0),
            max_tokens=args.max_tokens,
            repetition_penalty=args.repetition_penalty,
            frequency_penalty=getattr(args, "frequency_penalty", 0.0),
            presence_penalty=getattr(args, "presence_penalty", 0.0),
        ):
            num_tokens += 1
            token_buffer.append(token_id)
            text = tokenizer.decode(token_buffer)
            if text and "\ufffd" not in text:
                print(text, end="", flush=True)
                generated_text += text
                token_buffer.clear()
        if token_buffer:
            text = tokenizer.decode(token_buffer)
            print(text, end="", flush=True)
            generated_text += text

        elapsed = time.perf_counter() - t0
        tok_s = num_tokens / elapsed if elapsed > 0 else 0.0
        print(f"\n[{num_tokens} tokens, {tok_s:.1f} tok/s]\n")

        messages.append({"role": "assistant", "content": generated_text})

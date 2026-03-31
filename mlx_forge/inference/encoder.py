"""Encoder inference for embedding extraction.

Single forward pass (no autoregressive loop, no KV cache).
Supports CLS pooling and mean pooling with optional L2 normalization.
"""

from __future__ import annotations

import mlx.core as mx


def encode(
    model,
    tokenizer,
    texts: list[str],
    *,
    pooling: str = "cls",
    normalize: bool = True,
    batch_size: int = 32,
) -> list[mx.array]:
    """Extract embeddings from an encoder model.

    Args:
        model: Loaded encoder model (model_category == "encoder").
        tokenizer: Tokenizer instance.
        texts: List of text strings to encode.
        pooling: Pooling strategy — "cls" (position 0) or "mean" (average non-padding).
        normalize: Whether to L2-normalize the output embeddings.
        batch_size: Number of texts to process at once.

    Returns:
        List of embedding vectors (mx.array of shape (D,)).
    """
    all_embeddings = []

    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]

        # Tokenize with padding
        encoded = tokenizer(
            batch_texts,
            padding=True,
            truncation=True,
            return_tensors="np",
        )

        input_ids = mx.array(encoded["input_ids"])
        attention_mask = mx.array(encoded["attention_mask"])

        # Single forward pass
        hidden_states = model(input_ids, attention_mask=attention_mask)

        # Pool
        if pooling == "cls":
            embeddings = hidden_states[:, 0, :]  # (B, D)
        elif pooling == "mean":
            # Mean over non-padding tokens
            mask = attention_mask[:, :, None].astype(hidden_states.dtype)  # (B, T, 1)
            summed = (hidden_states * mask).sum(axis=1)  # (B, D)
            counts = mask.sum(axis=1)  # (B, 1)
            embeddings = summed / mx.maximum(counts, 1)
        else:
            raise ValueError(f"Unknown pooling strategy: {pooling}. Use 'cls' or 'mean'.")

        # Normalize
        if normalize:
            norms = mx.sqrt((embeddings * embeddings).sum(axis=-1, keepdims=True))
            embeddings = embeddings / mx.maximum(norms, 1e-12)

        mx.eval(embeddings)

        for j in range(embeddings.shape[0]):
            all_embeddings.append(embeddings[j])

    return all_embeddings

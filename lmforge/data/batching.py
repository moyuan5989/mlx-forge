"""Sort-by-length, fixed-batch, pad-to-32 iterator for LMForge v0."""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def iterate_batches(dataset, config):
    """Yield (batch_tokens, lengths) tuples per V0_DESIGN_FREEZE.md §2.2.

    batch_tokens: mx.array, dtype=int32, shape=(B, T)
    lengths:      mx.array, dtype=int32, shape=(B, 2)

    - B is config.training.batch_size
    - T is padded to nearest multiple of 32, capped at config.data.max_seq_length
    - lengths[:, 0] is prompt offset, lengths[:, 1] is total unpadded length
    - Padding value is 0

    Steps:
    1. Sort samples by length (descending) for efficient batching
    2. Group into fixed-size batches
    3. Pad to nearest multiple of 32 within each batch

    Note: This iterates once through the dataset (one epoch). Use itertools.cycle
    in the training loop to repeat for multiple epochs.
    """
    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    # Sort samples by length (descending) for efficient padding
    sorted_samples = sorted(
        dataset,
        key=lambda s: len(s["tokens"]) if hasattr(s["tokens"], "__len__") else s["tokens"].size,
        reverse=True,
    )

    # Group into batches
    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]

        # Skip incomplete final batch if it's too small
        if len(batch_samples) < batch_size:
            continue

        # Find max length in this batch
        max_len_in_batch = max(
            len(s["tokens"]) if hasattr(s["tokens"], "__len__") else s["tokens"].size
            for s in batch_samples
        )

        # Pad to nearest multiple of 32, capped at max_seq_length
        padded_length = _round_up_to_multiple(max_len_in_batch, 32)
        padded_length = min(padded_length, max_seq_length)

        # Build batch arrays
        batch_tokens = np.zeros((batch_size, padded_length), dtype=np.int32)
        batch_lengths = np.zeros((batch_size, 2), dtype=np.int32)

        for i, sample in enumerate(batch_samples):
            tokens = sample["tokens"]
            offset = sample["offset"]

            # Convert to numpy if needed (for mx.array)
            if hasattr(tokens, "tolist"):
                tokens = tokens.tolist()
            elif hasattr(tokens, "__iter__"):
                tokens = list(tokens)

            # Truncate if needed
            if len(tokens) > max_seq_length:
                tokens = tokens[:max_seq_length]

            # Copy tokens into batch
            batch_tokens[i, : len(tokens)] = tokens

            # Set lengths: [prompt_offset, total_unpadded_length]
            batch_lengths[i, 0] = min(offset, len(tokens))
            batch_lengths[i, 1] = len(tokens)

        # Convert to MLX arrays
        batch_tokens_mx = mx.array(batch_tokens, dtype=mx.int32)
        batch_lengths_mx = mx.array(batch_lengths, dtype=mx.int32)

        yield batch_tokens_mx, batch_lengths_mx


def iterate_packed_batches(dataset, config):
    """Yield (batch_tokens, segment_ids, offsets) tuples for packed training.

    Packs multiple sequences per row to eliminate padding waste.

    batch_tokens:  mx.array, dtype=int32, shape=(B, T)
    segment_ids:   mx.array, dtype=int32, shape=(B, T)  — segment index per token, -1 for padding
    offsets:       mx.array, dtype=int32, shape=(B, max_segments, 2)  — (prompt_end, seq_end) per segment

    Note: This iterates once through the packed dataset. Use itertools.cycle
    in the training loop to repeat for multiple epochs.
    """
    from lmforge.data.packing import pack_sequences

    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    packed = pack_sequences(dataset, max_seq_length)

    # Group packed sequences into batches
    for batch_start in range(0, len(packed), batch_size):
        batch_packed = packed[batch_start : batch_start + batch_size]

        if len(batch_packed) < batch_size:
            continue

        # Find max token length and max segments in this batch
        max_len_in_batch = max(len(p.tokens) for p in batch_packed)
        max_segments = max(len(p.offsets) for p in batch_packed)

        padded_length = _round_up_to_multiple(max_len_in_batch, 32)
        padded_length = min(padded_length, max_seq_length)

        batch_tokens = np.zeros((batch_size, padded_length), dtype=np.int32)
        batch_seg_ids = np.full((batch_size, padded_length), -1, dtype=np.int32)
        batch_offsets = np.zeros((batch_size, max_segments, 2), dtype=np.int32)

        for i, ps in enumerate(batch_packed):
            tlen = min(len(ps.tokens), padded_length)
            batch_tokens[i, :tlen] = ps.tokens[:tlen]
            batch_seg_ids[i, :tlen] = ps.segment_ids[:tlen]

            for j, (prompt_end, seq_end) in enumerate(ps.offsets):
                batch_offsets[i, j, 0] = prompt_end
                batch_offsets[i, j, 1] = min(seq_end, padded_length)

        yield (
            mx.array(batch_tokens, dtype=mx.int32),
            mx.array(batch_seg_ids, dtype=mx.int32),
            mx.array(batch_offsets, dtype=mx.int32),
        )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple

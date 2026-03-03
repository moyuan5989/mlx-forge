"""Sort-by-length, fixed-batch, pad-to-32 iterator for LMForge V2.

Yields (input_ids, labels) tuples with -100 label masking.
"""

from __future__ import annotations

import mlx.core as mx
import numpy as np


def _get_length(sample, key="input_ids"):
    """Get length of a sample's token sequence."""
    tokens = sample[key]
    if hasattr(tokens, "__len__"):
        return len(tokens)
    if hasattr(tokens, "size"):
        return tokens.size
    return len(list(tokens))


def _to_list(tokens):
    """Convert tokens to a plain list."""
    if hasattr(tokens, "tolist"):
        return tokens.tolist()
    elif isinstance(tokens, list):
        return tokens
    else:
        return list(tokens)


def iterate_batches(dataset, config):
    """Yield (input_ids, labels) tuples.

    input_ids: mx.array, dtype=int32, shape=(B, T)
    labels:    mx.array, dtype=int32, shape=(B, T)

    - B is config.training.batch_size
    - T is padded to nearest multiple of 32, capped at config.data.max_seq_length
    - input_ids padded with 0, labels padded with -100
    """
    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    # Sort samples by length (descending) for efficient padding
    sorted_samples = sorted(
        dataset,
        key=lambda s: _get_length(s, "input_ids"),
        reverse=True,
    )

    # Group into batches
    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]

        if len(batch_samples) < batch_size:
            continue

        # Find max length in this batch
        max_len_in_batch = max(_get_length(s, "input_ids") for s in batch_samples)

        # Pad to nearest multiple of 32, capped at max_seq_length
        padded_length = _round_up_to_multiple(max_len_in_batch, 32)
        padded_length = min(padded_length, max_seq_length)

        # Build batch arrays
        batch_input_ids = np.zeros((batch_size, padded_length), dtype=np.int32)
        batch_labels = np.full((batch_size, padded_length), -100, dtype=np.int32)

        for i, sample in enumerate(batch_samples):
            input_ids = _to_list(sample["input_ids"])
            labels = _to_list(sample["labels"])

            # Truncate if needed
            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]
                labels = labels[:max_seq_length]

            batch_input_ids[i, :len(input_ids)] = input_ids
            batch_labels[i, :len(labels)] = labels

        yield (
            mx.array(batch_input_ids, dtype=mx.int32),
            mx.array(batch_labels, dtype=mx.int32),
        )


def iterate_packed_batches(dataset, config):
    """Yield (input_ids, labels, segment_ids) tuples for packed training.

    input_ids:   mx.array, dtype=int32, shape=(B, T)
    labels:      mx.array, dtype=int32, shape=(B, T)
    segment_ids: mx.array, dtype=int32, shape=(B, T)  — segment index per token, -1 for padding
    """
    from lmforge.data.packing import pack_sequences

    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    packed = pack_sequences(dataset, max_seq_length)

    for batch_start in range(0, len(packed), batch_size):
        batch_packed = packed[batch_start : batch_start + batch_size]

        if len(batch_packed) < batch_size:
            continue

        max_len_in_batch = max(len(p.input_ids) for p in batch_packed)

        padded_length = _round_up_to_multiple(max_len_in_batch, 32)
        padded_length = min(padded_length, max_seq_length)

        batch_input_ids = np.zeros((batch_size, padded_length), dtype=np.int32)
        batch_labels = np.full((batch_size, padded_length), -100, dtype=np.int32)
        batch_seg_ids = np.full((batch_size, padded_length), -1, dtype=np.int32)

        for i, ps in enumerate(batch_packed):
            tlen = min(len(ps.input_ids), padded_length)
            batch_input_ids[i, :tlen] = ps.input_ids[:tlen]
            batch_labels[i, :tlen] = ps.labels[:tlen]
            batch_seg_ids[i, :tlen] = ps.segment_ids[:tlen]

        yield (
            mx.array(batch_input_ids, dtype=mx.int32),
            mx.array(batch_labels, dtype=mx.int32),
            mx.array(batch_seg_ids, dtype=mx.int32),
        )


def iterate_preference_batches(dataset, config):
    """Yield (chosen_ids, chosen_labels, rejected_ids, rejected_labels) tuples.

    For DPO training with preference pairs.
    All arrays are mx.array, dtype=int32, shape=(B, T).
    chosen/rejected labels padded with -100.
    """
    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    # Sort by max of chosen/rejected length (descending)
    sorted_samples = sorted(
        dataset,
        key=lambda s: max(
            _get_length(s, "chosen_input_ids"),
            _get_length(s, "rejected_input_ids"),
        ),
        reverse=True,
    )

    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]

        if len(batch_samples) < batch_size:
            continue

        # Find max length across both chosen and rejected
        max_len = 0
        for s in batch_samples:
            for key in ("chosen_input_ids", "rejected_input_ids"):
                max_len = max(max_len, _get_length(s, key))

        padded_length = _round_up_to_multiple(max_len, 32)
        padded_length = min(padded_length, max_seq_length)

        chosen_ids = np.zeros((batch_size, padded_length), dtype=np.int32)
        chosen_labels = np.full((batch_size, padded_length), -100, dtype=np.int32)
        rejected_ids = np.zeros((batch_size, padded_length), dtype=np.int32)
        rejected_labels = np.full((batch_size, padded_length), -100, dtype=np.int32)

        for i, sample in enumerate(batch_samples):
            for ids_key, labels_key, out_ids, out_labels in [
                ("chosen_input_ids", "chosen_labels", chosen_ids, chosen_labels),
                ("rejected_input_ids", "rejected_labels", rejected_ids, rejected_labels),
            ]:
                ids = _to_list(sample[ids_key])
                lbls = _to_list(sample[labels_key])

                if len(ids) > max_seq_length:
                    ids = ids[:max_seq_length]
                    lbls = lbls[:max_seq_length]

                out_ids[i, :len(ids)] = ids
                out_labels[i, :len(lbls)] = lbls

        yield (
            mx.array(chosen_ids, dtype=mx.int32),
            mx.array(chosen_labels, dtype=mx.int32),
            mx.array(rejected_ids, dtype=mx.int32),
            mx.array(rejected_labels, dtype=mx.int32),
        )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple

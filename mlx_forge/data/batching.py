"""Sort-by-length, fixed-batch, pad-to-32 iterator for MLX Forge V2.

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

    Supports both list datasets (sort-by-length) and iterators (streaming).
    """
    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    # If dataset is an iterator (e.g., MixedDatasetIterator), use streaming path
    if not hasattr(dataset, '__len__'):
        yield from _iterate_batches_streaming(dataset, batch_size, max_seq_length)
        return

    # Sort samples by length (descending) for efficient padding
    sorted_samples = sorted(
        dataset,
        key=lambda s: _get_length(s, "input_ids"),
        reverse=True,
    )

    # Group into batches (partial last batch is included with padding)
    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]

        # First element is longest (already sorted descending)
        max_len_in_batch = _get_length(batch_samples[0], "input_ids")

        # Pad to nearest multiple of 32, capped at max_seq_length
        padded_length = _round_up_to_multiple(max_len_in_batch, 32)
        padded_length = min(padded_length, max_seq_length)

        # Build batch arrays (pad rows use 0 for input_ids, -100 for labels)
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
    from mlx_forge.data.packing import pack_sequences

    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    packed = pack_sequences(dataset, max_seq_length)

    for batch_start in range(0, len(packed), batch_size):
        batch_packed = packed[batch_start : batch_start + batch_size]

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


def _iterate_batches_streaming(dataset_iter, batch_size, max_seq_length):
    """Streaming batch iterator for dataset iterators (no sorting).

    Collects batch_size samples from the iterator, pads, and yields.
    Used for MixedDatasetIterator and other infinite iterators.
    Partial last batch is included (padded rows have all -100 labels).
    """
    batch_samples = []
    for sample in dataset_iter:
        batch_samples.append(sample)
        if len(batch_samples) < batch_size:
            continue

        yield _build_batch(batch_samples, batch_size, max_seq_length)
        batch_samples = []

    # Yield remaining partial batch
    if batch_samples:
        yield _build_batch(batch_samples, batch_size, max_seq_length)


def _build_batch(batch_samples, batch_size, max_seq_length):
    """Build a padded (input_ids, labels) batch from a list of samples."""
    max_len_in_batch = max(_get_length(s, "input_ids") for s in batch_samples)
    padded_length = _round_up_to_multiple(max_len_in_batch, 32)
    padded_length = min(padded_length, max_seq_length)

    batch_input_ids = np.zeros((batch_size, padded_length), dtype=np.int32)
    batch_labels = np.full((batch_size, padded_length), -100, dtype=np.int32)

    for i, sample in enumerate(batch_samples):
        input_ids = _to_list(sample["input_ids"])
        labels = _to_list(sample["labels"])
        if len(input_ids) > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
        batch_input_ids[i, :len(input_ids)] = input_ids
        batch_labels[i, :len(labels)] = labels

    return (
        mx.array(batch_input_ids, dtype=mx.int32),
        mx.array(batch_labels, dtype=mx.int32),
    )


def iterate_mlm_batches(dataset, config):
    """Yield (input_ids, labels, attention_mask) tuples for MLM training.

    Applies dynamic MLM masking: each epoch, different tokens are masked.
    Input dataset should contain {"input_ids": [...], "labels": [...]} where
    labels may be identical to input_ids (from _tokenize_text) — masking is
    applied here at batch time.

    input_ids:       mx.array, dtype=int32, shape=(B, T)
    labels:          mx.array, dtype=int32, shape=(B, T)
    attention_mask:  mx.array, dtype=int32, shape=(B, T) — 1 for real, 0 for padding
    """
    import random

    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length
    mlm_probability = getattr(config.training, 'mlm_probability', 0.15)

    sorted_samples = sorted(
        dataset,
        key=lambda s: _get_length(s, "input_ids"),
        reverse=True,
    )

    pad_token_id = 0  # Default padding
    mask_token_id = 103  # BERT [MASK] default; overridden if available
    vocab_size = 30522

    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]

        max_len_in_batch = _get_length(batch_samples[0], "input_ids")
        padded_length = _round_up_to_multiple(max_len_in_batch, 32)
        padded_length = min(padded_length, max_seq_length)

        batch_input_ids = np.full((batch_size, padded_length), pad_token_id, dtype=np.int32)
        batch_labels = np.full((batch_size, padded_length), -100, dtype=np.int32)
        batch_attention_mask = np.zeros((batch_size, padded_length), dtype=np.int32)

        for i, sample in enumerate(batch_samples):
            input_ids = _to_list(sample["input_ids"])

            if len(input_ids) > max_seq_length:
                input_ids = input_ids[:max_seq_length]

            # Apply dynamic MLM masking
            masked_ids = list(input_ids)
            labels = [-100] * len(input_ids)

            # Special token IDs to skip (0=pad, first/last tokens are typically CLS/SEP)
            special_positions = {0, len(input_ids) - 1} if len(input_ids) > 1 else set()

            for j in range(len(input_ids)):
                if j in special_positions:
                    continue
                if input_ids[j] == pad_token_id:
                    continue
                if random.random() < mlm_probability:
                    labels[j] = input_ids[j]
                    r = random.random()
                    if r < 0.8:
                        masked_ids[j] = mask_token_id
                    elif r < 0.9:
                        masked_ids[j] = random.randint(0, vocab_size - 1)
                    # else: keep original (10%)

            batch_input_ids[i, :len(masked_ids)] = masked_ids
            batch_labels[i, :len(labels)] = labels
            batch_attention_mask[i, :len(input_ids)] = 1

        yield (
            mx.array(batch_input_ids, dtype=mx.int32),
            mx.array(batch_labels, dtype=mx.int32),
            mx.array(batch_attention_mask, dtype=mx.int32),
        )


def iterate_seq2seq_batches(dataset, config):
    """Yield (encoder_ids, decoder_ids, decoder_labels, encoder_attention_mask) tuples.

    For encoder-decoder (T5/BART) training. Pads encoder and decoder independently.
    """
    batch_size = config.training.batch_size
    max_seq_length = config.data.max_seq_length

    sorted_samples = sorted(
        dataset,
        key=lambda s: max(
            _get_length(s, "encoder_input_ids"),
            _get_length(s, "decoder_input_ids"),
        ),
        reverse=True,
    )

    for batch_start in range(0, len(sorted_samples), batch_size):
        batch_samples = sorted_samples[batch_start : batch_start + batch_size]

        max_enc_len = max(_get_length(s, "encoder_input_ids") for s in batch_samples)
        max_dec_len = max(_get_length(s, "decoder_input_ids") for s in batch_samples)

        enc_padded = _round_up_to_multiple(max_enc_len, 32)
        enc_padded = min(enc_padded, max_seq_length)
        dec_padded = _round_up_to_multiple(max_dec_len, 32)
        dec_padded = min(dec_padded, max_seq_length)

        batch_enc_ids = np.zeros((batch_size, enc_padded), dtype=np.int32)
        batch_dec_ids = np.zeros((batch_size, dec_padded), dtype=np.int32)
        batch_dec_labels = np.full((batch_size, dec_padded), -100, dtype=np.int32)
        batch_enc_mask = np.zeros((batch_size, enc_padded), dtype=np.int32)

        for i, sample in enumerate(batch_samples):
            enc_ids = _to_list(sample["encoder_input_ids"])
            dec_ids = _to_list(sample["decoder_input_ids"])
            dec_labels = _to_list(sample["decoder_labels"])

            if len(enc_ids) > max_seq_length:
                enc_ids = enc_ids[:max_seq_length]
            if len(dec_ids) > max_seq_length:
                dec_ids = dec_ids[:max_seq_length]
                dec_labels = dec_labels[:max_seq_length]

            batch_enc_ids[i, :len(enc_ids)] = enc_ids
            batch_dec_ids[i, :len(dec_ids)] = dec_ids
            batch_dec_labels[i, :len(dec_labels)] = dec_labels
            batch_enc_mask[i, :len(enc_ids)] = 1

        yield (
            mx.array(batch_enc_ids, dtype=mx.int32),
            mx.array(batch_dec_ids, dtype=mx.int32),
            mx.array(batch_dec_labels, dtype=mx.int32),
            mx.array(batch_enc_mask, dtype=mx.int32),
        )


def _round_up_to_multiple(value: int, multiple: int) -> int:
    """Round value up to the nearest multiple."""
    return ((value + multiple - 1) // multiple) * multiple

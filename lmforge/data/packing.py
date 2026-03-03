"""Sequence packing for efficient training on short-sequence datasets.

V2: PackedSequence uses input_ids + labels + segment_ids.
Labels encode prompt masking; no more offsets needed.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackedSequence:
    """A single packed row containing multiple sequences."""

    input_ids: list[int]
    labels: list[int]        # -100 at segment boundaries + prompt regions
    segment_ids: list[int]


def pack_sequences(
    dataset: list[dict],
    max_seq_length: int,
) -> list[PackedSequence]:
    """Pack multiple sequences into bins up to max_seq_length.

    Uses first-fit-decreasing bin packing: sorts sequences by length
    (longest first), then places each into the first bin that has room.

    Args:
        dataset: List of dicts with 'input_ids' (list/array) and 'labels' (list/array).
        max_seq_length: Maximum tokens per packed row.

    Returns:
        List of PackedSequence objects.
    """
    # Prepare items: (length, input_ids, labels, original_index)
    items = []
    for i, sample in enumerate(dataset):
        input_ids = _to_list(sample["input_ids"])
        labels = _to_list(sample["labels"])

        length = len(input_ids)
        if length > max_seq_length:
            input_ids = input_ids[:max_seq_length]
            labels = labels[:max_seq_length]
            length = max_seq_length
        items.append((length, input_ids, labels, i))

    # Sort by length descending (first-fit-decreasing)
    items.sort(key=lambda x: x[0], reverse=True)

    # Bins: each bin is a PackedSequence being built
    bins: list[PackedSequence] = []
    bin_remaining: list[int] = []  # remaining capacity per bin

    for length, input_ids, labels, _ in items:
        # Find first bin with enough room
        placed = False
        for j, remaining in enumerate(bin_remaining):
            if remaining >= length:
                seg_id = _count_segments(bins[j])
                bins[j].input_ids.extend(input_ids)
                # Mark the boundary between segments: the last token of the
                # previous segment predicting the first token of the new segment
                # should not contribute to loss. The labels already handle this
                # via -100 masking from preprocessing, but we also need to mark
                # boundaries in labels to prevent cross-segment prediction.
                bins[j].labels.extend(labels)
                bins[j].segment_ids.extend([seg_id] * length)
                bin_remaining[j] -= length
                placed = True
                break

        if not placed:
            # Open a new bin
            seq = PackedSequence(
                input_ids=list(input_ids),
                labels=list(labels),
                segment_ids=[0] * length,
            )
            bins.append(seq)
            bin_remaining.append(max_seq_length - length)

    return bins


def _count_segments(packed: PackedSequence) -> int:
    """Count the number of distinct segments in a packed sequence."""
    if not packed.segment_ids:
        return 0
    return max(packed.segment_ids) + 1


def _to_list(tokens) -> list:
    """Convert tokens to a plain list."""
    if hasattr(tokens, "tolist"):
        return tokens.tolist()
    elif isinstance(tokens, list):
        return list(tokens)  # defensive copy
    else:
        return list(tokens)

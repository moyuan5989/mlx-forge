"""Sequence packing for efficient training on short-sequence datasets.

Packs multiple short sequences into a single row up to max_seq_length,
eliminating padding waste. Uses first-fit-decreasing bin packing.
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass
class PackedSequence:
    """A single packed row containing multiple sequences."""

    tokens: list[int]
    segment_ids: list[int]
    offsets: list[tuple[int, int]]  # (prompt_end, seq_end) per segment


def pack_sequences(
    dataset: list[dict],
    max_seq_length: int,
) -> list[PackedSequence]:
    """Pack multiple sequences into bins up to max_seq_length.

    Uses first-fit-decreasing bin packing: sorts sequences by length
    (longest first), then places each into the first bin that has room.

    Args:
        dataset: List of dicts with 'tokens' (list/array) and 'offset' (int).
        max_seq_length: Maximum tokens per packed row.

    Returns:
        List of PackedSequence objects.
    """
    # Prepare items: (length, tokens, offset, original_index)
    items = []
    for i, sample in enumerate(dataset):
        tokens = sample["tokens"]
        if hasattr(tokens, "tolist"):
            tokens = tokens.tolist()
        elif hasattr(tokens, "__iter__") and not isinstance(tokens, list):
            tokens = list(tokens)
        length = len(tokens)
        if length > max_seq_length:
            tokens = tokens[:max_seq_length]
            length = max_seq_length
        items.append((length, tokens, sample["offset"], i))

    # Sort by length descending (first-fit-decreasing)
    items.sort(key=lambda x: x[0], reverse=True)

    # Bins: each bin is a PackedSequence being built
    bins: list[PackedSequence] = []
    bin_remaining: list[int] = []  # remaining capacity per bin

    for length, tokens, offset, _ in items:
        # Find first bin with enough room
        placed = False
        for j, remaining in enumerate(bin_remaining):
            if remaining >= length:
                seg_id = len(bins[j].offsets)
                pos = len(bins[j].tokens)
                bins[j].tokens.extend(tokens)
                bins[j].segment_ids.extend([seg_id] * length)
                bins[j].offsets.append((pos + offset, pos + length))
                bin_remaining[j] -= length
                placed = True
                break

        if not placed:
            # Open a new bin
            seq = PackedSequence(
                tokens=list(tokens),
                segment_ids=[0] * length,
                offsets=[(offset, length)],
            )
            bins.append(seq)
            bin_remaining.append(max_seq_length - length)

    return bins

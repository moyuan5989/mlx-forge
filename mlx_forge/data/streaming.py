"""Streaming data pipeline for large datasets.

Two capabilities:
1. StreamingHFDataset: HuggingFace streaming (datasets.load_dataset(..., streaming=True))
2. StreamingJSONLDataset: Memory-mapped JSONL reading for large local files

Both yield tokenized samples without loading all data into RAM.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Optional


class StreamingHFDataset:
    """Wraps HuggingFace IterableDataset for MLX Forge training.

    - No __len__ (signals streaming to batching.py)
    - Tokenizes on-the-fly
    - Optional shuffle buffer

    Args:
        dataset_id: HuggingFace dataset ID
        split: Dataset split
        tokenizer: Tokenizer instance
        subset: Optional dataset subset/config
        columns: Optional column mapping
        max_seq_length: Maximum sequence length
        mask_prompt: Whether to mask prompt tokens
        shuffle_buffer: Shuffle buffer size (0 to disable)
    """

    def __init__(
        self,
        dataset_id: str,
        split: str = "train",
        tokenizer=None,
        *,
        subset: Optional[str] = None,
        columns: Optional[dict[str, str]] = None,
        max_seq_length: int = 2048,
        mask_prompt: bool = True,
        shuffle_buffer: int = 1000,
    ):
        from datasets import load_dataset

        self.ds = load_dataset(
            dataset_id,
            name=subset,
            split=split,
            streaming=True,
        )
        if shuffle_buffer > 0:
            self.ds = self.ds.shuffle(buffer_size=shuffle_buffer)

        self.tokenizer = tokenizer
        self.columns = columns
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt

    def __iter__(self):
        for row in self.ds:
            sample = self._convert_row(row)
            if sample is None:
                continue
            tokenized = self._tokenize(sample)
            if tokenized is not None:
                yield tokenized

    def _convert_row(self, row: dict) -> Optional[dict]:
        """Convert HF row to standard format using column mapping."""
        if self.columns:
            result = {}
            for target_key, source_key in self.columns.items():
                if source_key in row:
                    result[target_key] = row[source_key]
            return result if result else None
        return row

    def _tokenize(self, sample: dict) -> Optional[dict]:
        """Tokenize a single sample."""
        from mlx_forge.data.preprocessing import tokenize_single

        try:
            return tokenize_single(
                sample,
                self.tokenizer,
                max_seq_length=self.max_seq_length,
                mask_prompt=self.mask_prompt,
            )
        except Exception:
            return None


class StreamingJSONLDataset:
    """Memory-efficient JSONL reader for large local files.

    Reads line-by-line without loading all into RAM.
    Cycles infinitely for training.

    Args:
        path: Path to JSONL file
        tokenizer: Tokenizer instance
        max_seq_length: Maximum sequence length
        mask_prompt: Whether to mask prompt tokens
    """

    def __init__(
        self,
        path: str,
        tokenizer=None,
        *,
        max_seq_length: int = 2048,
        mask_prompt: bool = True,
    ):
        self.path = Path(path)
        if not self.path.exists():
            raise FileNotFoundError(f"JSONL file not found: {path}")
        self.tokenizer = tokenizer
        self.max_seq_length = max_seq_length
        self.mask_prompt = mask_prompt

    def __iter__(self):
        """Iterate through JSONL file, cycling infinitely."""
        while True:
            with open(self.path) as f:
                for line in f:
                    line = line.strip()
                    if not line:
                        continue
                    try:
                        sample = json.loads(line)
                    except json.JSONDecodeError:
                        continue

                    tokenized = self._tokenize(sample)
                    if tokenized is not None:
                        yield tokenized

    def _tokenize(self, sample: dict) -> Optional[dict]:
        """Tokenize a single sample."""
        from mlx_forge.data.preprocessing import tokenize_single

        try:
            return tokenize_single(
                sample,
                self.tokenizer,
                max_seq_length=self.max_seq_length,
                mask_prompt=self.mask_prompt,
            )
        except Exception:
            return None

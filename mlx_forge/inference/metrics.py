"""Streaming generation metrics: TTFT, prefill/decode throughput."""

from __future__ import annotations

import time
from dataclasses import dataclass


@dataclass
class GenerationMetrics:
    """Metrics from a generation run."""

    ttft_ms: float  # time to first token
    prefill_tokens: int
    prefill_tokens_per_sec: float
    decode_tokens: int
    decode_tokens_per_sec: float
    total_time_ms: float


class MetricsTracker:
    """Tracks timing for prefill and decode phases.

    Usage:
        tracker = MetricsTracker(num_prompt_tokens=100)
        # ... prefill ...
        tracker.mark_prefill_done()
        # ... decode loop ...
        tracker.mark_token()
        tracker.mark_token()
        metrics = tracker.finish()
    """

    def __init__(self, num_prompt_tokens: int):
        self._num_prompt_tokens = num_prompt_tokens
        self._start = time.perf_counter()
        self._prefill_done: float | None = None
        self._decode_tokens = 0

    def mark_prefill_done(self) -> None:
        """Call after the first model forward (prefill) completes."""
        self._prefill_done = time.perf_counter()

    def mark_token(self) -> None:
        """Call after each decode step produces a token."""
        self._decode_tokens += 1

    def finish(self) -> GenerationMetrics:
        """Finalize and return metrics."""
        end = time.perf_counter()
        total_ms = (end - self._start) * 1000

        if self._prefill_done is not None:
            prefill_elapsed = self._prefill_done - self._start
            ttft_ms = prefill_elapsed * 1000
            prefill_tps = (
                self._num_prompt_tokens / prefill_elapsed if prefill_elapsed > 0 else 0.0
            )
            decode_elapsed = end - self._prefill_done
            decode_tps = (
                self._decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
            )
        else:
            ttft_ms = 0.0
            prefill_tps = 0.0
            decode_elapsed = end - self._start
            decode_tps = (
                self._decode_tokens / decode_elapsed if decode_elapsed > 0 else 0.0
            )

        return GenerationMetrics(
            ttft_ms=ttft_ms,
            prefill_tokens=self._num_prompt_tokens,
            prefill_tokens_per_sec=prefill_tps,
            decode_tokens=self._decode_tokens,
            decode_tokens_per_sec=decode_tps,
            total_time_ms=total_ms,
        )

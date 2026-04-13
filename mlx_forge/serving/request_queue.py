"""Async request queue for serializing inference on single-GPU MLX.

MLX uses a single GPU — true parallelism is impossible. This queue keeps
the event loop responsive during generation by cooperative yielding between
tokens, allowing health checks and new submissions to proceed.
"""

from __future__ import annotations

import asyncio
import uuid
from dataclasses import dataclass, field


@dataclass
class InferenceRequest:
    """A queued inference request."""

    id: str = field(default_factory=lambda: uuid.uuid4().hex[:8])
    params: dict = field(default_factory=dict)
    response_queue: asyncio.Queue = field(default_factory=asyncio.Queue)
    cancelled: bool = False


class RequestQueue:
    """Async FIFO queue serializing inference requests.

    Design: NOT continuous batching (too complex for single-GPU). Simple FIFO
    with cooperative yielding so the event loop stays responsive.

    Args:
        max_size: Maximum number of pending requests. 0 = unlimited.
    """

    def __init__(self, max_size: int = 64):
        self._queue: asyncio.Queue[InferenceRequest] = asyncio.Queue(
            maxsize=max_size if max_size > 0 else 0
        )
        self._running = False
        self._current_request: InferenceRequest | None = None

    async def submit(self, params: dict) -> InferenceRequest:
        """Submit a new inference request to the queue.

        Args:
            params: Generation parameters dict.

        Returns:
            InferenceRequest with a response_queue to read results from.

        Raises:
            asyncio.QueueFull: If queue is at max capacity.
        """
        request = InferenceRequest(params=params)
        try:
            self._queue.put_nowait(request)
        except asyncio.QueueFull:
            raise asyncio.QueueFull(
                f"Request queue full (max_size={self._queue.maxsize})"
            )
        return request

    async def run_loop(self, process_fn) -> None:
        """Main processing loop. Pulls requests and calls process_fn.

        Args:
            process_fn: Async callable(request) that processes each request.
                Should put results into request.response_queue and a final
                None sentinel when done.
        """
        self._running = True
        try:
            while self._running:
                try:
                    request = await asyncio.wait_for(
                        self._queue.get(), timeout=1.0
                    )
                except asyncio.TimeoutError:
                    continue

                if request.cancelled:
                    await request.response_queue.put(None)
                    continue

                self._current_request = request
                try:
                    await process_fn(request)
                except Exception as e:
                    await request.response_queue.put({"error": str(e)})
                    await request.response_queue.put(None)
                finally:
                    self._current_request = None
        except asyncio.CancelledError:
            pass
        finally:
            self._running = False

    def stop(self) -> None:
        """Signal the run loop to stop."""
        self._running = False

    @property
    def pending_count(self) -> int:
        """Number of requests waiting in queue."""
        return self._queue.qsize()

    @property
    def is_running(self) -> bool:
        return self._running

"""ASR two-phase async scheduler.

Phase 1: synchronous encoder forward (feature extraction + FastConformer encode)
Phase 2: async decode_stream, gated by asyncio.Semaphore(max_concurrency)

Adapted from the Auralis TwoPhaseScheduler
(xttsv2-vllm-streaming-server/src/auralis/common/scheduling/two_phase_scheduler.py).

Key differences from the Auralis original:
  - One sequence per request: ASR has one audio clip → one decoder stream,
    so there is no ``parallel_inputs`` chunking.  The inner per-sequence queue
    and ``generators_count`` bookkeeping are collapsed to a single queue.
  - Phase 1 is a *synchronous* callable (CUDA dispatch via torch encoder
    forward); it runs directly in the calling coroutine rather than in a
    background worker task.  Multiple coroutines interleave naturally at the
    ``await`` in phase 2 so encodes proceed one at a time per caller while
    GPU kernels overlap asynchronously.
  - Phase 2 is an async generator bounded by ``asyncio.Semaphore(max_concurrency)``.
  - Ordered output is delivered via ``asyncio.Queue`` + the ``_END_OF_STREAM``
    sentinel (identity check), mirroring the Auralis pattern so results arrive
    correctly even if we later add batching or interleaved requests.
"""
from __future__ import annotations

import asyncio
import uuid
from typing import Any, AsyncIterator, Callable

# Sentinel pushed into the output queue by the phase-2 producer to signal that
# the generator has produced its final item.  The consumer loop checks for
# ``is _END_OF_STREAM`` (identity, not equality) before exiting — the same
# convention as the Auralis reference.
_END_OF_STREAM = object()


class TwoPhaseScheduler:
    """Two-phase async scheduler for ASR requests.

    Phase 1 (encode) is a synchronous callable executed directly in the
    coroutine — it dispatches CUDA kernels that execute asynchronously on the
    GPU.  Phase 2 (decode_stream) is an async-generator callable whose
    concurrency across simultaneous requests is capped by a shared
    ``asyncio.Semaphore(max_concurrency)``.

    The output queue + ``_END_OF_STREAM`` sentinel pattern mirrors Auralis:
    a background producer task fills the queue; the ``run()`` async generator
    drains it, yielding items to the caller as they arrive.  This keeps the
    scheduler extensible (future: encoder micro-batching in phase 1, or
    multiple sub-sequences per request in phase 2).

    Args:
        max_concurrency: Maximum number of simultaneous phase-2 (decode)
            operations.  Corresponds to ``second_phase_concurrency`` in
            Auralis.  Default 8.
    """

    def __init__(self, max_concurrency: int = 8) -> None:
        if max_concurrency <= 0:
            raise ValueError(
                f"max_concurrency must be >= 1, got {max_concurrency!r}"
            )
        self._max_concurrency = max_concurrency
        # Created lazily inside the running event loop so that constructing
        # ASREngine at import time (before an event loop exists) is safe.
        self._sem: asyncio.Semaphore | None = None

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _get_sem(self) -> asyncio.Semaphore:
        """Return the phase-2 semaphore, creating it on first call.

        Must be called from within a running event loop so that the
        Semaphore is bound to the correct loop.
        """
        if self._sem is None:
            self._sem = asyncio.Semaphore(self._max_concurrency)
        return self._sem

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def run(
        self,
        inputs: Any,
        phase1_fn: Callable[[Any], Any],
        phase2_fn: Callable[[Any, str], AsyncIterator[Any]],
        request_id: str | None = None,
    ) -> AsyncIterator[Any]:
        """Run a request through both phases and yield phase-2 outputs.

        Phase 1 (``phase1_fn``) is called synchronously — it dispatches
        CUDA kernels quickly and returns before GPU execution completes.
        Phase 2 (``phase2_fn``) is an async generator that runs under the
        concurrency semaphore.

        Results are delivered via an ``asyncio.Queue``; the producer task
        pushes the ``_END_OF_STREAM`` sentinel when done so the consumer loop
        terminates without polling.  This mirrors the
        ``_yield_ordered_outputs`` / ``_process_generator`` split in Auralis.

        Args:
            inputs: Raw input passed to ``phase1_fn`` (e.g. ``np.ndarray``).
            phase1_fn: Synchronous callable ``(inputs) -> phase1_result``.
                       For ASR this is the encoder (feature extraction +
                       FastConformer forward).
            phase2_fn: Async-generator callable
                       ``(phase1_result, request_id) -> AsyncIterator``.
                       For ASR this is ``backend.decode_stream``.
            request_id: Optional identifier forwarded to ``phase2_fn``.

        Yields:
            Items emitted by ``phase2_fn`` (``TokenDelta`` objects for ASR).

        Raises:
            Exception: Any exception raised inside the phase-2 producer is
                re-raised in the consumer after the queue is drained.
        """
        sem = self._get_sem()
        request_id = request_id or str(uuid.uuid4())

        # ── Phase 1: encode ───────────────────────────────────────────────────
        # Synchronous call; dispatches CUDA kernels that run asynchronously on
        # the GPU.  Calling directly (not via run_in_executor) keeps all GPU
        # state on the main thread, avoiding cross-thread CUDA context issues.
        phase1_result = phase1_fn(inputs)

        # ── Phase 2: decode, semaphore-bounded ────────────────────────────────
        # Each request gets its own queue.  The producer task acquires the
        # semaphore, runs the async generator, pushes items, and finally pushes
        # _END_OF_STREAM.  The consumer (this coroutine) drains the queue.
        output_q: asyncio.Queue[Any] = asyncio.Queue()
        error_holder: list[BaseException] = []

        async def _producer() -> None:
            """Acquire semaphore, run phase-2 generator, push to output_q."""
            async with sem:
                try:
                    async for item in phase2_fn(phase1_result, request_id):
                        await output_q.put(item)
                except Exception as exc:  # noqa: BLE001
                    error_holder.append(exc)
                finally:
                    # Always push sentinel — even on cancel/exception — so the
                    # consumer does not hang.  Mirrors Auralis _process_generator.
                    await output_q.put(_END_OF_STREAM)

        producer_task = asyncio.create_task(_producer())

        try:
            while True:
                item = await output_q.get()
                if item is _END_OF_STREAM:
                    break
                yield item
        finally:
            # If consumer exits early (exception or generator.aclose()), cancel
            # and await the producer so we don't leak tasks.
            # asyncio.shield() makes the gather itself immune to an outer
            # CancelledError (e.g. HTTP request timeout cancelling transcribe()),
            # ensuring the producer is definitively awaited before propagating
            # the cancellation — otherwise the gather is itself a cancellation
            # point and the producer can keep running while holding its semaphore
            # slot.
            if not producer_task.done():
                producer_task.cancel()
            await asyncio.shield(asyncio.gather(producer_task, return_exceptions=True))

        # Surface any error the producer stashed, after the queue is drained.
        if error_holder:
            raise error_holder[0]

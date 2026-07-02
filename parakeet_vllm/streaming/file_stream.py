"""SSE formatting and word-timestamp building for Parakeet ASR.

``partials_to_sse`` wraps any async iterator of partial-transcript strings
into Server-Sent Events, suitable for a ``StreamingResponse``.

``build_word_timestamps`` calls the processor's timestamp-aware decode path
and normalizes the result to the OpenAI ``words`` shape.
"""
from __future__ import annotations

import json
from typing import AsyncIterator


async def partials_to_sse(aiter: AsyncIterator[str]) -> AsyncIterator[str]:
    """Wrap an async iterator of partial transcripts as SSE chunks.

    Each partial is emitted as::

        data: {"text": "<partial>"}\n\n

    A terminal sentinel is emitted at the end::

        data: [DONE]\n\n

    Args:
        aiter: Async iterator of partial-transcript strings (non-decreasing).

    Yields:
        SSE-formatted strings ready for a ``StreamingResponse``.
    """
    async for partial in aiter:
        yield f"data: {json.dumps({'text': partial})}\n\n"
    yield "data: [DONE]\n\n"


def build_word_timestamps(
    tokens: list[int],
    durations: list[float],
    processor,
) -> list[dict]:
    """Build OpenAI-compatible word timestamps for a single utterance.

    Delegates to ``processor.decode(..., durations=...)`` which returns
    ``(texts, timestamps)`` where ``timestamps`` is a list (one per batch item)
    of ``[{"token": str, "start": float, "end": float}, ...]``.

    The result is normalised to the OpenAI ``words`` shape:
    ``[{"word": str, "start": float, "end": float}, ...]``.

    Args:
        tokens: Token-ID list for one utterance (no batch dimension).
        durations: Frame-duration list aligned with ``tokens``.
        processor: HuggingFace processor whose ``decode`` method accepts a
            ``durations`` keyword and returns ``(texts, timestamps)``.

    Returns:
        List of ``{"word": str, "start": float, "end": float}`` dicts,
        one entry per decoded token (special tokens excluded).
    """
    import torch

    # Wrap single utterance into a batch-1 tensor pair.
    seq = torch.tensor([tokens])
    dur = torch.tensor([durations])

    _texts, timestamps = processor.decode(seq, durations=dur, skip_special_tokens=True)

    # ``timestamps`` is a list with one element per batch item.
    ts_list: list[dict] = timestamps[0]

    # Normalise from {"token": ..., "start": ..., "end": ...}
    # to OpenAI   {"word":  ..., "start": ..., "end": ...}.
    return [
        {"word": t["token"], "start": t["start"], "end": t["end"]}
        for t in ts_list
    ]

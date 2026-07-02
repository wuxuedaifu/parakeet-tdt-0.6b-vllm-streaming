"""Shared TDT joint-output math, used by both the reference and vLLM backends.

The transducer joint emits a width ``vocab_size + len(durations)`` vector per
step. The emitted token is the argmax over the vocab slice (the duration slots
are suppressed in ``model.generate``, so the full-width argmax equals the
vocab-slice argmax); the frame advance is ``config.durations[argmax(dur_slice)]``,
with a blank prediction of duration 0 forced to advance one frame.

Keeping this in one place guarantees the reference backend and any in-engine
vLLM port apply identical selection semantics.
"""

from __future__ import annotations

import torch


def split_joint(
    logits_row: torch.Tensor,
    *,
    vocab_size: int,
    blank_id: int,
    durations: list[int],
) -> tuple[int, int]:
    """Split one joint-output row into ``(token_id, frame_advance)``.

    ``logits_row`` is a 1-D tensor of width ``vocab_size + len(durations)``.
    """
    row = logits_row.float()
    token_id = int(row[:vocab_size].argmax().item())
    dur_idx = int(row[vocab_size:].argmax().item())
    duration = int(durations[dur_idx])
    if token_id == blank_id and duration == 0:
        duration = 1
    return token_id, duration

"""Pack encoder frames as vLLM multimodal data for the in-engine spike.

vLLM V1 serialises ``multi_modal_data`` via msgpack over an IPC to the
EngineCore process (see the Auralis XTTSv2 reference,
``XTTSv2.py:1128-1148``). Tensors must therefore be plain, CPU-resident, and
contiguous so they survive the round-trip and can be re-materialised on the
worker device.

This helper produces the ``{"audio": ...}``-style dict a custom multimodal
model would receive. It is intentionally minimal: the fundamental blocker (see
findings) is not the packing but that the model cannot recover *which* request a
decode-step row belongs to, so it cannot look these frames back up per step.
"""

from __future__ import annotations

from typing import Any

import torch


def pack_encoder_frames(encoder_frames: torch.Tensor) -> dict[str, Any]:
    """Return a ``multi_modal_data`` dict carrying raw encoder frames.

    ``encoder_frames``: ``[1, T, hidden]`` float tensor (the encoder
    ``last_hidden_state``; the model projects it to ``decoder_hidden_size``
    itself, matching the reference backend).
    """
    if encoder_frames.dim() != 3 or encoder_frames.shape[0] != 1:
        raise ValueError(
            f"expected [1, T, hidden] encoder frames, got {tuple(encoder_frames.shape)}"
        )
    frames = encoder_frames.squeeze(0).detach().to("cpu").contiguous()
    return {"parakeet_encoder_frames": frames}

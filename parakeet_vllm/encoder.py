"""Batched FastConformer encoder runner (Task 7).

Exposes ``encode(input_features, attention_mask)`` as the single entry point
for encoding mel-spectrogram features into per-frame hidden states.

Backend selection is controlled by ``PARAKEET_ENCODER_BACKEND`` (env var,
read via ``parakeet_vllm.config.ENCODER_BACKEND``):

- ``"torch"`` (default) — direct batched PyTorch forward through the HF
  ``model.encoder``.  Always works, parity-verified by the test suite.
- ``"vllm"`` — delegates to ``VLLMEncoder``, which raises ``VLLMEncoderNoGo``
  (the encoder-in-vLLM port is a NO-GO for Phase 1; this path is preserved as
  the explicit opt-in so the rejected route stays reproducible).  See
  ``parakeet_vllm/vllm_engine/encoder_engine.py`` and the findings doc.
"""

from __future__ import annotations

import torch

from .config import ENCODER_BACKEND
from .model_loader import get_reference_model


@torch.inference_mode()
def _torch_encode(
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Torch (HF) encoder forward.  Returns ``(frames, lengths)``."""
    model = get_reference_model()
    out = model.encoder(
        input_features=input_features,
        attention_mask=attention_mask,
        output_attention_mask=True,
    )
    frames = out.last_hidden_state  # [B, T', 1024]

    # Derive per-item valid frame lengths from the encoder's output attention
    # mask when present; fall back to the full T' for all items.
    out_mask = getattr(out, "attention_mask", None)
    if out_mask is not None:
        lengths = out_mask.sum(dim=-1).long()  # [B]
    else:
        B, T = frames.shape[0], frames.shape[1]
        lengths = torch.full((B,), T, dtype=torch.long, device=frames.device)

    return frames, lengths


def encode(
    input_features: torch.Tensor,
    attention_mask: torch.Tensor | None = None,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Encode mel features into encoder frames and valid frame lengths.

    Factory that dispatches on ``ENCODER_BACKEND``:

    - ``"vllm"``  → ``VLLMEncoder`` (raises ``VLLMEncoderNoGo`` — explicit
                     NO-GO opt-in; deferred for Phase 1).
    - ``"torch"`` → batched PyTorch HF encoder forward (Phase 1 default).

    Args:
        input_features: ``[B, n_mels, T]`` mel-spectrogram features.
        attention_mask:  ``[B, T]`` padding mask (1 = valid, 0 = pad).

    Returns:
        frames:  ``torch.FloatTensor[B, T', 1024]`` — raw FastConformer
                 per-frame hidden states (``last_hidden_state``).
        lengths: ``torch.LongTensor[B]`` — number of valid frames per item.
    """
    if ENCODER_BACKEND == "vllm":
        from .vllm_engine.encoder_engine import VLLMEncoder
        # Construction raises VLLMEncoderNoGo immediately on the current NO-GO.
        # A future GO implementation would call enc.encode_async here instead.
        enc = VLLMEncoder()  # noqa: F841 — unreachable past this line
        import asyncio  # pragma: no cover
        return asyncio.run(enc.encode_async(input_features, attention_mask))  # pragma: no cover

    return _torch_encode(input_features, attention_mask)

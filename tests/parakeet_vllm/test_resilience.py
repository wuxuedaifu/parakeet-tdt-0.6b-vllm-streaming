"""Resilience tests for the parakeet_vllm service (Task 13).

All tests are marked ``@pytest.mark.gpu`` — run with ``-m gpu``.

Guards tested:
- ``test_silence_returns_empty``: silent audio short-circuits before encode/decode.
- ``test_runaway_decode_is_capped``: a monkeypatched always-duration-0 _split_joint
  cannot make decode_stream loop forever; total emissions are bounded by the
  per-frame symbol cap (max_symbols_per_step guard).
"""
import asyncio

import numpy as np
import pytest


@pytest.mark.gpu
def test_silence_returns_empty():
    """An all-zero waveform must return an empty Transcription without touching the
    encoder or decoder (silence short-circuit in ASREngine.transcribe)."""
    from parakeet_vllm.engine.asr_engine import ASREngine

    eng = ASREngine()
    out = asyncio.run(eng.transcribe(np.zeros(1600, dtype="float32"), "sil"))
    assert out.text.strip() == ""


@pytest.mark.gpu
def test_runaway_decode_is_capped(monkeypatch):
    """Per-frame symbol cap prevents an infinite decode loop.

    _split_joint is patched to always return (5, 0): non-blank token with
    duration 0.  Without the cap the frame pointer would never advance and
    decode_stream would loop forever.  With the cap, after ``max_symbols``
    non-advancing emissions the frame is force-advanced, so total emissions
    are bounded by ``T * max_symbols``.
    """
    from parakeet_vllm.decode import reference_tdt as r

    eng_mod = r.ReferenceTDTBackend
    monkeypatch.setattr(eng_mod, "_split_joint", lambda self, logits: (5, 0))

    from parakeet_vllm.model_loader import get_reference_model
    import torch

    b = eng_mod(get_reference_model())
    frames = torch.zeros(1, 20, 1024, device="cuda")
    lengths = torch.tensor([20], device="cuda")

    async def run():
        n = 0
        async for d in b.decode_stream("x", frames, lengths):
            n += len(d.token_ids)
            if n > 10000:
                break
        return n

    n = asyncio.run(run())
    assert n <= 20 * b.max_symbols + 10, (
        f"Expected capped at {20 * b.max_symbols + 10}, got {n}"
    )

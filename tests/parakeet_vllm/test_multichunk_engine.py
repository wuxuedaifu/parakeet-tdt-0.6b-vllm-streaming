"""Mock-based end-to-end multi-chunk transcribe path (Task 12, GPU-free).

Covers three guarantees introduced / hardened in Task 12:

(a) ``decode_stream`` receives the real per-chunk encoder length (not T_max),
    proving padded frames beyond the real length are never decoded.
(b) ``Transcription.text`` is the two chunks' texts joined with a space.
(c) ``Transcription.words`` holds each chunk's words with ``start``/``end``
    offset-corrected by ``offset_samples / 16000``.

All GPU operations are mocked out; the test runs on CPU only.
"""
from __future__ import annotations

import asyncio

import numpy as np
import pytest
import torch

# ── Constants ────────────────────────────────────────────────────────────────

T_MAX = 20       # padded frame dimension returned by the mock encoder
LEN_0 = 12       # real frame count for chunk 0  (<  T_MAX)
LEN_1 = 8        # real frame count for chunk 1  (<  T_MAX, ≠ LEN_0)
SR = 16_000
OFFSET_0 = 0     # sample offset of chunk 0 → 0.0 s
OFFSET_1 = SR    # sample offset of chunk 1 → 1.0 s


# ── Fake decode backend ──────────────────────────────────────────────────────

class _FakeBackend:
    """Async-generator decode backend that records each call and emits fixed tokens."""

    def __init__(self) -> None:
        self.calls: list[dict] = []

    async def decode_stream(self, request_id, encoder_frames, encoder_lengths):
        from parakeet_vllm.decode.backend import TokenDelta

        self.calls.append(
            {
                "request_id": request_id,
                "encoder_length": int(encoder_lengths[0].item()),
                "frames_T": encoder_frames.shape[1],
            }
        )

        if "chunk0" in request_id:
            yield TokenDelta(token_ids=[10, 11], durations=[3, 3], finished=False)
        else:
            yield TokenDelta(token_ids=[20, 21], durations=[2, 2], finished=False)
        yield TokenDelta(token_ids=[], durations=[], finished=True)


# ── Fake processor ───────────────────────────────────────────────────────────

class _FakeProcessor:
    """Translates token-ids [10, 11] → "hello" and [20, 21] → "world"."""

    def decode(self, tokens, skip_special_tokens=True, **kwargs):
        try:
            token_list = tokens[0].tolist()
        except Exception:
            token_list = []
        return "hello" if 10 in token_list else "world"


# ── Test ─────────────────────────────────────────────────────────────────────


def test_multichunk_per_chunk_lengths_and_text(monkeypatch):
    """
    End-to-end mock test for the multi-chunk transcribe path.

    (a) ``decode_stream`` is called once per chunk with
        ``int(encoder_lengths[0])`` equal to the real per-chunk frame count
        (LEN_0 or LEN_1), NOT the padded T_max.
    (b) ``Transcription.text == "hello world"`` (chunks joined with a space).
    (c) ``Transcription.words`` has each chunk's words offset-corrected by
        ``offset_samples / 16000``:
          - chunk 0 offset = 0 s  → start/end unchanged
          - chunk 1 offset = 1 s  → start/end shifted by 1.0 s
    """
    import parakeet_vllm.engine.asr_engine as _asr_mod
    import parakeet_vllm.streaming.chunker as _chunker_mod
    import parakeet_vllm.streaming.file_stream as _fs_mod

    fake_backend = _FakeBackend()
    fake_processor = _FakeProcessor()

    # ── Monkeypatches ─────────────────────────────────────────────────────
    # Replace model-loading singletons; no GPU / HF download needed.
    monkeypatch.setattr(_asr_mod, "get_decode_backend", lambda: fake_backend)
    monkeypatch.setattr(_asr_mod, "get_processor", lambda: fake_processor)

    # Two chunks with different lengths (ensures LEN_0 ≠ LEN_1 matters).
    chunk_wave_0 = np.zeros(SR * 5, dtype=np.float32)   # 5 s
    chunk_wave_1 = np.zeros(SR * 3, dtype=np.float32)   # 3 s
    monkeypatch.setattr(
        _chunker_mod,
        "split_on_silence",
        lambda audio, max_chunk_s: [
            (OFFSET_0, chunk_wave_0),
            (OFFSET_1, chunk_wave_1),
        ],
    )

    # Feature extraction: shape is irrelevant because encode is mocked.
    def _fake_extract_features(chunks):
        B = len(chunks)
        return {
            "input_features": torch.zeros(B, 80, 100),
            "attention_mask": torch.ones(B, 100, dtype=torch.long),
        }

    monkeypatch.setattr(_asr_mod, "extract_features", _fake_extract_features)

    # Encoder: return padded [B, T_MAX, 1024] frames with per-chunk real lengths.
    def _fake_encode(input_features, attention_mask):
        B = input_features.shape[0]
        frames = torch.zeros(B, T_MAX, 1024)
        lengths = torch.tensor([LEN_0, LEN_1], dtype=torch.long)
        return frames, lengths

    monkeypatch.setattr(_asr_mod, "encode", _fake_encode)

    # build_word_timestamps: return predictable per-chunk word timestamps.
    def _fake_bwt(tokens, durs, processor):
        if 10 in tokens:
            return [{"word": "hello", "start": 0.1, "end": 0.5}]
        return [{"word": "world", "start": 0.0, "end": 0.3}]

    monkeypatch.setattr(_fs_mod, "build_word_timestamps", _fake_bwt)

    # ── Engine construction + transcribe ──────────────────────────────────
    from parakeet_vllm.engine.asr_engine import ASREngine

    # 35 s audio > max_chunk_s=10 → triggers the multi-chunk path.
    # Use non-zero amplitude so the silence short-circuit in transcribe() is
    # not triggered (the test mocks all backends; the audio content is unused).
    long_audio = np.full(SR * 35, 0.1, dtype=np.float32)

    async def _run():
        engine = ASREngine(max_concurrency=1, max_chunk_s=10.0)
        return await engine.transcribe(long_audio, "test_req")

    result = asyncio.run(_run())

    # ── (a) Per-chunk encoder lengths ─────────────────────────────────────
    assert len(fake_backend.calls) == 2, (
        f"Expected exactly 2 decode_stream calls, got {len(fake_backend.calls)}"
    )

    call_0 = next(c for c in fake_backend.calls if "chunk0" in c["request_id"])
    call_1 = next(c for c in fake_backend.calls if "chunk1" in c["request_id"])

    assert call_0["encoder_length"] == LEN_0, (
        f"chunk0 encoder_length={call_0['encoder_length']}, expected {LEN_0} (real length)"
    )
    assert call_1["encoder_length"] == LEN_1, (
        f"chunk1 encoder_length={call_1['encoder_length']}, expected {LEN_1} (real length)"
    )
    # Confirm neither chunk received the padded T_MAX length.
    assert call_0["encoder_length"] != T_MAX, "chunk0 must NOT receive T_MAX as encoder_length"
    assert call_1["encoder_length"] != T_MAX, "chunk1 must NOT receive T_MAX as encoder_length"

    # ── (b) Text is joined with a space ───────────────────────────────────
    assert result.text == "hello world", (
        f"Expected text='hello world', got {result.text!r}"
    )

    # ── (c) Words are offset-corrected ────────────────────────────────────
    assert isinstance(result.words, list), (
        "multi-chunk Transcription.words must always be a list (never None)"
    )
    assert len(result.words) == 2, (
        f"Expected 2 word entries, got {len(result.words)}"
    )

    offset_0_s = OFFSET_0 / SR   # 0.0 s
    offset_1_s = OFFSET_1 / SR   # 1.0 s

    w_hello = next(w for w in result.words if w["word"] == "hello")
    w_world = next(w for w in result.words if w["word"] == "world")

    assert abs(w_hello["start"] - (0.1 + offset_0_s)) < 1e-6, (
        f"hello start={w_hello['start']}, expected {0.1 + offset_0_s}"
    )
    assert abs(w_hello["end"] - (0.5 + offset_0_s)) < 1e-6, (
        f"hello end={w_hello['end']}, expected {0.5 + offset_0_s}"
    )
    assert abs(w_world["start"] - (0.0 + offset_1_s)) < 1e-6, (
        f"world start={w_world['start']}, expected {0.0 + offset_1_s}"
    )
    assert abs(w_world["end"] - (0.3 + offset_1_s)) < 1e-6, (
        f"world end={w_world['end']}, expected {0.3 + offset_1_s}"
    )


def test_short_audio_words_is_none(monkeypatch):
    """Short-audio path must still yield words=None (single-chunk, unchanged)."""
    import parakeet_vllm.engine.asr_engine as _asr_mod

    fake_backend = _FakeBackend()
    fake_processor = _FakeProcessor()

    monkeypatch.setattr(_asr_mod, "get_decode_backend", lambda: fake_backend)
    monkeypatch.setattr(_asr_mod, "get_processor", lambda: fake_processor)

    # Short audio: 5 s < max_chunk_s=30 s → short path, never touches multi-chunk.
    def _fake_phase1(audio):
        frames = torch.zeros(1, T_MAX, 1024)
        lengths = torch.tensor([LEN_0], dtype=torch.long)
        return frames, lengths

    from parakeet_vllm.engine.asr_engine import ASREngine

    # Non-zero amplitude to avoid triggering the silence short-circuit.
    short_audio = np.full(SR * 5, 0.1, dtype=np.float32)   # 5 s

    async def _run():
        engine = ASREngine(max_concurrency=1, max_chunk_s=30.0)
        # Patch the scheduler so phase1 doesn't call real GPU ops.
        engine._phase1_encode = _fake_phase1
        return await engine.transcribe(short_audio, "short_req")

    result = asyncio.run(_run())
    assert result.words is None, (
        f"Short-audio Transcription.words must be None, got {result.words!r}"
    )

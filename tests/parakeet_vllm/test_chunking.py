"""Tests for VAD-based audio chunking (Task 12)."""
from __future__ import annotations

import numpy as np
import pytest

from parakeet_vllm.streaming.chunker import split_on_silence


def _real_speech_16k():
    """Load the first utterance from hf-internal-testing/librispeech_asr_dummy resampled to 16 kHz.

    The clip is ~5.86 s with one continuous speech span (0.5 s–5.5 s) that
    Silero ONNX reliably detects as speech, unlike synthetic sine tones.
    """
    from datasets import load_dataset, Audio  # type: ignore
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy",
        "clean",
        split="validation",
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds["audio"][0]["array"].astype(np.float32)


def test_splits_long_audio_with_silence():
    """Real speech clips separated by 0.6 s silence gaps must produce >= 2 chunks.

    Silero ONNX detects speech in real librispeech audio (unlike a 200 Hz sine
    tone).  Three ~5.86 s clips interleaved with 0.6 s silence gaps give an
    ~18.8 s waveform; the packing + hard-split logic should produce many chunks
    well above the >= 2 threshold.
    """
    sr = 16000
    clip = _real_speech_16k()
    # 0.6 s silence — longer than min_silence_duration_ms=300 so Silero
    # treats each clip as a separate speech region.
    sil = np.zeros(int(0.6 * sr), dtype=np.float32)
    audio = np.concatenate([clip, sil, clip, sil, clip])
    chunks = split_on_silence(audio, max_chunk_s=1.5)
    assert len(chunks) >= 2
    assert all(c.dtype == np.float32 for _, c in chunks)
    assert chunks[0][0] == 0


def test_short_audio_returns_single_chunk():
    """Audio shorter than max_chunk_s must be returned as a single chunk at offset 0."""
    sr = 16000
    audio = (0.1 * np.random.randn(sr)).astype("float32")
    chunks = split_on_silence(audio, max_chunk_s=5.0)
    assert len(chunks) == 1
    assert chunks[0][0] == 0
    np.testing.assert_array_equal(chunks[0][1], audio)


def test_offset_samples_are_non_decreasing():
    """Offsets must be monotonically non-decreasing."""
    sr = 16000
    speech = (0.2 * np.sin(2 * np.pi * 300 * np.arange(sr) / sr)).astype("float32")
    sil = np.zeros(sr // 2, dtype="float32")
    audio = np.concatenate([speech, sil, speech, sil, speech, sil, speech])
    chunks = split_on_silence(audio, max_chunk_s=1.5)
    offsets = [off for off, _ in chunks]
    assert offsets == sorted(offsets), f"Offsets not sorted: {offsets}"


# ---------------------------------------------------------------------------
# Fix I3: hard-split long continuous chunks
# ---------------------------------------------------------------------------

def test_hard_split_continuous_speech_no_silence():
    """Fix I3: a long continuous speech span longer than max_chunk_s is hard-split into
    contiguous sub-chunks each <= max_chunk_s samples, covering the full signal.

    Two librispeech clips are concatenated without any silence gap so the total
    audio is ~11.7 s.  With max_chunk_s=1.0 s each detected speech region
    (~5 s) far exceeds the cap, ensuring the hard-split path fires.  The test
    verifies chunk sizes, contiguity, full coverage, and first-offset == 0.
    """
    sr = 16000
    max_chunk_s = 1.0
    max_samples = int(max_chunk_s * sr)

    # Two real-speech clips concatenated with NO silence gap between them.
    # This guarantees a long speech span that exceeds max_chunk_s.
    clip = _real_speech_16k()
    audio = np.concatenate([clip, clip])
    n_samples = audio.size
    assert n_samples > max_samples, "Test setup: audio must be longer than max_chunk_s"

    chunks = split_on_silence(audio, max_chunk_s=max_chunk_s)

    # 1. Every chunk must fit within the cap.
    for off, chunk in chunks:
        assert chunk.size <= max_samples, (
            f"Chunk at offset {off} has {chunk.size} samples > cap {max_samples}"
        )

    # 2. Chunks must be contiguous and cover the whole signal.
    expected_start = 0
    for off, chunk in chunks:
        assert off == expected_start, (
            f"Gap/overlap detected: expected offset {expected_start}, got {off}"
        )
        expected_start += chunk.size
    assert expected_start == n_samples, (
        f"Chunks cover {expected_start} samples but audio has {n_samples}"
    )

    # 3. First offset must be 0.
    assert chunks[0][0] == 0

    # 4. Multiple chunks must be produced (it's longer than the cap).
    assert len(chunks) >= 2, f"Expected >= 2 chunks, got {len(chunks)}"

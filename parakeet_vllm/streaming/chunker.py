"""VAD-based audio chunker for long-file transcription (Task 12).

Splits a long waveform on silence boundaries using Silero VAD (or an energy
fallback) and returns ``(offset_samples, chunk_array)`` pairs ready for
batched encoding.

The splitting logic follows the pause-midpoint approach in
``parakeet_service/chunker.py``: speech segments are packed until a chunk
would exceed ``max_chunk_s``; at that point the waveform is cut at the
midpoint of the silence that separates the last packed segment from the next
one.  This ensures cuts happen on silence rather than in the middle of speech.
"""
from __future__ import annotations

import logging
from typing import List, Tuple

import numpy as np

_SR = 16_000
_VAD_THRESHOLD = 0.5
_VAD_MIN_SILENCE_MS = 300
_VAD_SPEECH_PAD_MS = 30

logger = logging.getLogger("parakeet_vllm.streaming.chunker")

_vad_model = None  # lazy singleton


def _get_vad():
    global _vad_model
    if _vad_model is None:
        try:
            from silero_vad import load_silero_vad  # type: ignore
            _vad_model = load_silero_vad(onnx=True)
            logger.info("Loaded Silero VAD (ONNX backend) for parakeet_vllm chunker")
        except Exception as exc:
            logger.warning(
                "Silero VAD unavailable (%s); falling back to energy VAD", exc
            )
            _vad_model = "energy"
    return _vad_model


# ---------------------------------------------------------------------------
# Speech segment detection
# ---------------------------------------------------------------------------

def _silero_speech_segments(wav: np.ndarray) -> List[Tuple[int, int]]:
    """Return (start_sample, end_sample) speech spans via Silero VAD."""
    model = _get_vad()
    if model == "energy":
        return _energy_speech_segments(wav)

    from silero_vad import get_speech_timestamps  # type: ignore
    import torch  # silero-vad pulls torch even for onnx mode

    t = torch.from_numpy(wav)
    ts = get_speech_timestamps(
        t,
        model,
        sampling_rate=_SR,
        threshold=_VAD_THRESHOLD,
        min_silence_duration_ms=_VAD_MIN_SILENCE_MS,
        speech_pad_ms=_VAD_SPEECH_PAD_MS,
        return_seconds=False,
    )
    return [(int(s["start"]), int(s["end"])) for s in ts]


def _energy_speech_segments(wav: np.ndarray) -> List[Tuple[int, int]]:
    """Cheap RMS-based fallback when Silero is unavailable."""
    frame = int(0.02 * _SR)  # 20 ms frames
    if frame <= 0 or wav.size < frame:
        return [(0, wav.size)]
    n = wav.size // frame
    framed = wav[: n * frame].reshape(n, frame)
    rms = np.sqrt((framed * framed).mean(axis=1) + 1e-12)
    thr = max(1e-3, rms.mean() * 0.4)
    voiced = rms > thr
    min_sil = max(1, int(_VAD_MIN_SILENCE_MS / 20))
    segs: List[Tuple[int, int]] = []
    i = 0
    while i < n:
        if voiced[i]:
            start = i
            j = i
            sil = 0
            while j < n:
                if voiced[j]:
                    sil = 0
                else:
                    sil += 1
                    if sil >= min_sil:
                        break
                j += 1
            end = min(j - sil, n)
            segs.append((start * frame, end * frame))
            i = j
        else:
            i += 1
    return segs or [(0, wav.size)]


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def split_on_silence(
    audio: np.ndarray,
    max_chunk_s: float,
) -> List[Tuple[int, np.ndarray]]:
    """Split *audio* on silence boundaries into chunks of at most *max_chunk_s*.

    Chunks are cut at the midpoint of the silence gap between consecutive
    speech segments (the pause-midpoint approach from the legacy chunker),
    so boundaries never fall inside speech.

    Short audio (``len(audio) / 16000 <= max_chunk_s``) is returned as a
    single element list: ``[(0, audio)]``.

    Args:
        audio:       Float32 waveform sampled at 16 kHz.
        max_chunk_s: Maximum chunk duration in seconds.  Any run of speech
                     that would cause a chunk to exceed this length triggers a
                     cut at the preceding silence midpoint.

    Returns:
        List of ``(offset_samples, chunk)`` tuples where ``offset_samples`` is
        the position of the chunk's first sample in the original *audio* array
        and ``chunk`` is a ``float32`` NumPy array.  The list is in temporal
        order; ``offsets[0] == 0`` always.
    """
    total = audio.size
    max_samples = int(max_chunk_s * _SR)

    # Short clip: return as-is (single chunk, offset 0).
    if total <= max_samples:
        return [(0, audio.astype(np.float32, copy=False))]

    segs = _silero_speech_segments(audio.astype(np.float32, copy=False))

    if not segs:
        # Pure silence — single chunk.
        return [(0, audio.astype(np.float32, copy=False))]

    # Pack speech segments into chunks, cutting at silence midpoints.
    cut_points: List[int] = [segs[0][0]]  # start of first segment
    cur_start: int = segs[0][0]
    cur_end: int = segs[0][1]

    for s, e in segs[1:]:
        if e - cur_start <= max_samples:
            # Still fits — extend the current chunk.
            cur_end = e
        else:
            # Would exceed max_chunk_s; cut at midpoint of the silence gap
            # (cur_end .. s).
            mid = (cur_end + s) // 2
            cut_points.append(mid)
            cur_start = mid
            cur_end = e

    # Build (start, end) pairs from cut_points + total length.
    cut_points.append(total)

    result: List[Tuple[int, np.ndarray]] = []
    for i in range(len(cut_points) - 1):
        s = cut_points[i]
        e = cut_points[i + 1]
        chunk = audio[s:e].astype(np.float32, copy=False)
        result.append((s, chunk))

    # The first offset must always be 0; correct if VAD pushed the first
    # speech segment forward (leading silence).
    if result and result[0][0] != 0:
        off, chunk = result[0]
        # Prepend leading silence to the first chunk so offset stays 0.
        result[0] = (0, audio[0:cut_points[1]].astype(np.float32, copy=False))

    # ------------------------------------------------------------------
    # Fix I3: hard-split any chunk that is STILL longer than max_chunk_s
    # after the silence-based packing pass.  This handles continuous speech
    # runs with no usable silence gaps (e.g. recordings with no pauses).
    # Sub-chunks are contiguous fixed-size slices; absolute offsets into the
    # original array are preserved.
    # ------------------------------------------------------------------
    final_result: List[Tuple[int, np.ndarray]] = []
    for base_off, chunk in result:
        if chunk.size <= max_samples:
            final_result.append((base_off, chunk))
        else:
            pos = 0
            while pos < chunk.size:
                sub = chunk[pos : pos + max_samples].astype(np.float32, copy=False)
                final_result.append((base_off + pos, sub))
                pos += max_samples
    return final_result

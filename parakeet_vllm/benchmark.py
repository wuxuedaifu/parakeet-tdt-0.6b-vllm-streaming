"""Throughput benchmark for the parakeet_vllm service (Task 13).

Measures RTF (Real-Time Factor = wall_time / audio_duration, lower = faster)
and throughput (total_audio_seconds / wall_seconds, higher = better) at
concurrency levels 1, 8, and 16.

Two measurement scenarios:
  1. Single long request  – one 60-second synthetic file transcribed alone.
  2. Concurrent requests  – N × 10-second clips submitted simultaneously with
     asyncio.gather.

RTF < 1.0 means faster-than-real-time.

Run directly:
    . .venv-vllm/bin/activate
    python -m parakeet_vllm.benchmark

Or with the helper entrypoint (if configured):
    python parakeet_vllm/benchmark.py
"""
from __future__ import annotations

import asyncio
import time
from typing import Sequence

import numpy as np

_SR = 16_000  # 16 kHz


# ─────────────────────────────────────────────────────────────────────────────
# Synthetic audio helpers
# ─────────────────────────────────────────────────────────────────────────────


def _speech_like(duration_s: float, amplitude: float = 0.15) -> np.ndarray:
    """Generate a speech-like synthetic waveform.

    Produces alternating 2-second tone bursts and 0.3-second silences so that
    Silero VAD can segment the audio properly for long-clip VAD-chunking paths.
    The amplitude (0.15) is well above the silence threshold (1e-4) used in
    ASREngine.transcribe.

    Args:
        duration_s: Desired waveform length in seconds.
        amplitude:  Peak amplitude of the tone bursts.

    Returns:
        float32 waveform at 16 kHz.
    """
    rng = np.random.default_rng(42)
    burst_s = 2.0
    gap_s = 0.3
    period_s = burst_s + gap_s
    n_total = int(duration_s * _SR)
    audio = np.zeros(n_total, dtype=np.float32)
    t = 0
    while t < n_total:
        burst_end = min(t + int(burst_s * _SR), n_total)
        n = burst_end - t
        freq = rng.uniform(120.0, 300.0)  # random fundamental in speech range
        tone = (
            np.sin(2 * np.pi * freq * np.arange(n) / _SR)
            + 0.3 * rng.standard_normal(n)
        ).astype(np.float32)
        audio[t:burst_end] = (amplitude * tone).astype(np.float32)
        t = burst_end + int(gap_s * _SR)
    return audio


# ─────────────────────────────────────────────────────────────────────────────
# Benchmark helpers
# ─────────────────────────────────────────────────────────────────────────────


async def _transcribe_one(eng, audio: np.ndarray, request_id: str) -> float:
    """Return wall-clock seconds for a single transcribe call."""
    t0 = time.perf_counter()
    await eng.transcribe(audio, request_id)
    return time.perf_counter() - t0


async def _concurrent(
    eng,
    clips: Sequence[np.ndarray],
    base_id: str = "bench",
) -> float:
    """Submit all clips concurrently and return total wall-clock seconds."""
    t0 = time.perf_counter()
    await asyncio.gather(
        *[eng.transcribe(clip, f"{base_id}-{i}") for i, clip in enumerate(clips)]
    )
    return time.perf_counter() - t0


def _rtf(wall_s: float, audio_s: float) -> float:
    return wall_s / audio_s


def _throughput(wall_s: float, audio_s: float) -> float:
    return audio_s / wall_s


# ─────────────────────────────────────────────────────────────────────────────
# ONNX service comparison (best-effort)
# ─────────────────────────────────────────────────────────────────────────────


def _try_onnx_benchmark(clip_10s: np.ndarray, concurrencies: list[int]) -> dict | None:
    """Attempt to benchmark the legacy ONNX service.  Returns None on any error."""
    try:
        import parakeet_service  # noqa: F401 — checks importability only
        from parakeet_service.asr import transcribe as onnx_transcribe
    except Exception as exc:
        print(f"  [onnx] Skipped — legacy ONNX service not importable: {exc}")
        return None

    results: dict[int, dict] = {}
    for c in concurrencies:
        clips = [clip_10s] * c
        try:
            t0 = time.perf_counter()
            for clip in clips:
                onnx_transcribe(clip)
            wall = time.perf_counter() - t0
            audio = 10.0 * c
            results[c] = {"wall_s": wall, "rtf": _rtf(wall, audio), "throughput": _throughput(wall, audio)}
            print(f"  [onnx] c={c:2d}: wall={wall:.2f}s  RTF={results[c]['rtf']:.3f}  throughput={results[c]['throughput']:.2f}x")
        except Exception as exc:
            print(f"  [onnx] c={c}: error — {exc}")
    return results if results else None


# ─────────────────────────────────────────────────────────────────────────────
# Main benchmark routine
# ─────────────────────────────────────────────────────────────────────────────


def run_benchmark() -> dict:
    """Run the full benchmark and return a results dict."""
    from parakeet_vllm.engine.asr_engine import ASREngine

    print("=" * 60)
    print("parakeet_vllm throughput benchmark")
    print("=" * 60)

    # Construct engine once (loads model into GPU memory).
    print("\nLoading model...")
    t_load = time.perf_counter()
    eng = ASREngine()
    print(f"  Model ready in {time.perf_counter() - t_load:.1f}s")

    results: dict = {"vllm": {}}

    # ── Scenario 1: single long request ──────────────────────────────────────
    print("\n--- Scenario 1: single 60-second request ---")
    audio_60s = _speech_like(60.0)
    wall_60 = asyncio.run(_transcribe_one(eng, audio_60s, "long-0"))
    rtf_60 = _rtf(wall_60, 60.0)
    tput_60 = _throughput(wall_60, 60.0)
    print(f"  wall={wall_60:.2f}s  RTF={rtf_60:.3f}  throughput={tput_60:.2f}x realtime")
    results["vllm"]["single_60s"] = {
        "audio_s": 60.0,
        "wall_s": wall_60,
        "rtf": rtf_60,
        "throughput": tput_60,
    }

    # ── Scenario 2: concurrent 10-second clips ────────────────────────────────
    print("\n--- Scenario 2: concurrent 10-second clips ---")
    concurrencies = [1, 8, 16]
    audio_10s = _speech_like(10.0)

    for c in concurrencies:
        clips = [audio_10s] * c
        wall = asyncio.run(_concurrent(eng, clips, base_id=f"c{c}"))
        audio_total = 10.0 * c
        rtf = _rtf(wall, audio_total)
        tput = _throughput(wall, audio_total)
        print(f"  c={c:2d}: wall={wall:.2f}s  RTF={rtf:.3f}  throughput={tput:.2f}x realtime  ({c} clips × 10s)")
        results["vllm"][f"c{c}"] = {
            "n_clips": c,
            "audio_s": audio_total,
            "wall_s": wall,
            "rtf": rtf,
            "throughput": tput,
        }

    # ── Optional: ONNX comparison ─────────────────────────────────────────────
    print("\n--- Legacy ONNX comparison (best-effort) ---")
    audio_10s_onnx = audio_10s  # same clip
    onnx_results = _try_onnx_benchmark(audio_10s_onnx, concurrencies)
    if onnx_results:
        results["onnx"] = onnx_results
    else:
        results["onnx"] = None

    print("\n" + "=" * 60)
    print("Summary (parakeet_vllm):")
    print(f"  60s single :  RTF={rtf_60:.3f}  throughput={tput_60:.2f}x")
    for c in concurrencies:
        r = results["vllm"][f"c{c}"]
        print(f"  c={c:2d}        :  RTF={r['rtf']:.3f}  throughput={r['throughput']:.2f}x")
    if onnx_results is None:
        print("  ONNX comparison: skipped (not importable in this env)")
    print("=" * 60)

    return results


if __name__ == "__main__":
    run_benchmark()

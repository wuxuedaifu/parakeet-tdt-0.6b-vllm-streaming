"""bench_live_concurrency.py — measure serialization in the live decode path.

Quantifies whether concurrent decode_window calls overlap or serialize on the
asyncio event loop.

The live path (parakeet_vllm.realtime.decoder_stream.decode_window) calls
extract_features / encode / decode_stream synchronously with no semaphore guard.
This means N concurrent callers each block the event loop in turn, giving
fully serialized execution rather than true parallelism.

This script measures that effect directly:
  - Runs ONE decode_window call and records wall-clock latency.
  - Runs N=8 decode_window calls via asyncio.gather and records wall-clock.
  - Computes serialization_factor = time_8way / time_1way.
    ~1.0  → true parallelism (overlapping GPU work)
    ~8.0  → fully serialized (each call runs after the previous)

Usage:
    . .venv-vllm/bin/activate
    VLLM_USE_V1=1 python scripts/bench_live_concurrency.py
"""
from __future__ import annotations

import asyncio
import time

import numpy as np

_SR = 16_000
_CLIP_DURATION_S = 2.0  # short hop-sized window — matches live decode window


def _make_clip(duration_s: float = _CLIP_DURATION_S) -> np.ndarray:
    """Return a speech-like float32 waveform at 16 kHz."""
    rng = np.random.default_rng(0)
    n = int(duration_s * _SR)
    t = np.arange(n, dtype=np.float32) / _SR
    # Superimposed tones in speech fundamental-frequency range
    audio = (
        0.12 * np.sin(2 * np.pi * 180.0 * t)
        + 0.06 * np.sin(2 * np.pi * 360.0 * t)
        + 0.04 * rng.standard_normal(n).astype(np.float32)
    ).astype(np.float32)
    return audio


async def _time_one(clip: np.ndarray, request_id: str) -> float:
    """Return wall-clock seconds for a single decode_window call."""
    from parakeet_vllm.realtime.decoder_stream import decode_window
    t0 = time.perf_counter()
    await decode_window(clip, request_id)
    return time.perf_counter() - t0


async def _time_n_concurrent(clip: np.ndarray, n: int) -> float:
    """Return wall-clock seconds for n concurrent decode_window calls."""
    from parakeet_vllm.realtime.decoder_stream import decode_window
    t0 = time.perf_counter()
    await asyncio.gather(
        *[decode_window(clip, f"bench-live-{i}") for i in range(n)]
    )
    return time.perf_counter() - t0


async def main() -> None:
    # Warm up the backend (loads model into GPU memory on first call)
    print("Warming up model (first call loads weights)...")
    clip = _make_clip()
    _ = await _time_one(clip, "warmup")
    print("Warm-up complete.\n")

    # Single call
    latency_1 = await _time_one(clip, "single-0")
    print(f"Single-call latency:          {latency_1:.3f}s")

    # 8-way concurrent
    n = 8
    wall_8 = await _time_n_concurrent(clip, n)
    effective_per_call = wall_8 / n
    serialization_factor = wall_8 / latency_1

    print(f"8-way concurrent wall-clock:  {wall_8:.3f}s")
    print(f"Per-call effective latency:   {effective_per_call:.3f}s")
    print(f"Serialization factor (8/1):   {serialization_factor:.2f}x")
    print()

    # Print table
    print("-" * 60)
    print(f"{'Metric':<35} {'Value':>10}")
    print("-" * 60)
    print(f"{'Clip duration (s)':<35} {_CLIP_DURATION_S:>10.1f}")
    print(f"{'Single-call latency (s)':<35} {latency_1:>10.3f}")
    print(f"{'8-way concurrent wall-clock (s)':<35} {wall_8:>10.3f}")
    print(f"{'Per-call effective latency (s)':<35} {effective_per_call:>10.3f}")
    print(f"{'Serialization factor':<35} {serialization_factor:>10.2f}x")
    print("-" * 60)
    print()

    if serialization_factor > 6.0:
        print(
            "CONCLUSION: Live decode path is FULLY SERIALIZED.\n"
            "  decode_window blocks the asyncio event loop (no semaphore,\n"
            "  no asyncio.to_thread offload). N concurrent clients queue\n"
            "  one-at-a-time. See docs/superpowers/findings/\n"
            "  2026-07-02-phase2-live-followups.md for the recommended fix."
        )
    elif serialization_factor > 2.0:
        print(
            f"CONCLUSION: Live decode path shows significant serialization\n"
            f"  (factor {serialization_factor:.2f}x). Some overlap but not true parallelism."
        )
    else:
        print(
            "CONCLUSION: Live decode path shows good parallelism "
            f"(factor {serialization_factor:.2f}x)."
        )


if __name__ == "__main__":
    asyncio.run(main())

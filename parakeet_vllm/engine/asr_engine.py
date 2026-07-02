from __future__ import annotations

import os
import uuid
from dataclasses import dataclass, field
from typing import AsyncIterator

import numpy as np

from ..features import extract_features
from ..encoder import encode
from ..model_loader import get_decode_backend, get_processor
from ..config import DEVICE
from ..scheduling import TwoPhaseScheduler

# Default maximum chunk length (seconds) for long-audio VAD chunking.
# Audio longer than this triggers split_on_silence → batched encode → per-chunk
# decode.  Override with the PARAKEET_MAX_CHUNK_S env var.
_DEFAULT_MAX_CHUNK_S = float(os.getenv("PARAKEET_MAX_CHUNK_S", "30"))


@dataclass
class Transcription:
    text: str
    tokens: list[int]
    durations: list[int]
    # Pre-computed, offset-corrected word timestamps populated by the
    # multi-chunk path in transcribe().  None for short (single-chunk) audio,
    # in which case callers should use build_word_timestamps() as before.
    words: list[dict] | None = field(default=None)


class ASREngine:
    """End-to-end ASR engine with two-phase concurrent scheduling.

    Phase 1 (encode): synchronous feature extraction + FastConformer encoder
    forward.  Phase 2 (decode): async TDT decode stream, bounded by
    ``asyncio.Semaphore(max_concurrency)`` so many requests can decode
    concurrently without overwhelming GPU memory.

    For audio longer than *max_chunk_s* seconds, ``transcribe`` automatically
    splits the audio on silence boundaries (VAD chunking), encodes **all**
    chunks in a single batched forward pass (the throughput win), and then
    decodes each chunk sequentially.  Chunk word timestamps are offset-corrected
    before being stored in ``Transcription.words``.

    Each ``decode_stream`` call keeps ALL per-request state local (frame_ptr,
    decoder_cache, prev_token), so concurrent phase-2 calls are independent;
    the shared ``ParakeetForTDT`` module is only read.

    Args:
        max_concurrency: Maximum simultaneous phase-2 (decode) operations.
            Defaults to the ``PARAKEET_MAX_CONCURRENCY`` env-var or 8.
        max_chunk_s: Audio longer than this (seconds) is VAD-chunked before
            encoding.  Defaults to ``PARAKEET_MAX_CHUNK_S`` env-var or 30.
    """

    def __init__(
        self,
        max_concurrency: int | None = None,
        max_chunk_s: float | None = None,
    ) -> None:
        self.backend = get_decode_backend()
        self.processor = get_processor()
        if max_concurrency is None:
            max_concurrency = int(os.getenv("PARAKEET_MAX_CONCURRENCY", "8"))
        self.max_concurrency = max_concurrency
        self.max_chunk_s: float = (
            max_chunk_s if max_chunk_s is not None else _DEFAULT_MAX_CHUNK_S
        )
        self._scheduler = TwoPhaseScheduler(max_concurrency=max_concurrency)

    # ------------------------------------------------------------------
    # Phase functions (passed to the scheduler — single-chunk path)
    # ------------------------------------------------------------------

    def _phase1_encode(self, audio: np.ndarray):
        """Phase 1: feature extraction + encoder forward.

        Synchronous; dispatches CUDA kernels that run asynchronously on the
        GPU.  Returns ``(frames, lengths)`` ready for ``decode_stream``.
        """
        feats = extract_features([audio])
        frames, lengths = encode(
            feats["input_features"].to(DEVICE),
            feats["attention_mask"].to(DEVICE),
        )
        return frames, lengths

    async def _phase2_decode(self, phase1_result, request_id: str):
        """Phase 2: TDT decode stream.

        Async generator that yields ``TokenDelta`` objects.  All per-request
        state (frame_ptr, decoder_cache, prev_token) lives inside
        ``decode_stream``; concurrent calls are independent.
        """
        frames, lengths = phase1_result
        async for delta in self.backend.decode_stream(request_id, frames, lengths):
            yield delta

    # ------------------------------------------------------------------
    # Internal: multi-chunk batched encode + per-chunk decode
    # ------------------------------------------------------------------

    def _batch_encode_chunks(
        self, chunks: list[np.ndarray]
    ):
        """Batch-encode multiple chunks in one encoder forward pass.

        Args:
            chunks: List of float32 waveforms (one per VAD chunk).

        Returns:
            ``(frames, lengths)`` where ``frames`` is ``[B, T', 1024]`` and
            ``lengths`` is ``[B]`` — the same shapes the single-chunk path
            produces, but for a batch of B chunks.
        """
        feats = extract_features(chunks)
        frames, lengths = encode(
            feats["input_features"].to(DEVICE),
            feats["attention_mask"].to(DEVICE),
        )
        return frames, lengths

    async def _decode_single(
        self, frames, lengths, request_id: str
    ) -> tuple[list[int], list[int]]:
        """Decode one set of encoder frames and collect all tokens + durations."""
        toks: list[int] = []
        durs: list[int] = []
        async for delta in self.backend.decode_stream(request_id, frames, lengths):
            toks.extend(delta.token_ids)
            durs.extend(delta.durations)
        return toks, durs

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def transcribe(self, audio: np.ndarray, request_id: str) -> Transcription:
        """Run end-to-end ASR and return the full Transcription.

        **Short audio** (``len(audio) / 16000 <= max_chunk_s``): routes through
        the two-phase scheduler exactly as before (single-chunk path, unchanged
        behaviour).

        **Long audio** (``len(audio) / 16000 > max_chunk_s``): splits on
        silence via Silero VAD, encodes ALL chunks in one batched forward pass
        (throughput win), decodes each chunk sequentially, joins texts with
        spaces, and stores offset-corrected word timestamps in
        ``Transcription.words``.

        Routes through the two-phase scheduler: phase 1 encodes, phase 2
        streams tokens under the concurrency semaphore.  Collecting all
        TokenDelta items and decoding with ``skip_special_tokens=True``
        reproduces the oracle text from ``ParakeetForTDT.generate()`` exactly.
        """
        import torch

        # Silence / too-short short-circuit: avoid encode + decode entirely.
        # 160 samples = one feature-extractor hop at 16 kHz (10 ms).
        _HOP = 160
        if len(audio) < _HOP or float(np.abs(audio).max()) < 1e-4:
            return Transcription(text="", tokens=[], durations=[], words=None)

        duration_s = len(audio) / 16_000

        # ── Short-audio path (unchanged) ─────────────────────────────────────
        if duration_s <= self.max_chunk_s:
            toks: list[int] = []
            durs: list[int] = []
            async for delta in self._scheduler.run(
                inputs=audio,
                phase1_fn=self._phase1_encode,
                phase2_fn=self._phase2_decode,
                request_id=request_id,
            ):
                toks.extend(delta.token_ids)
                durs.extend(delta.durations)

            text = self.processor.decode(
                torch.tensor([toks]), skip_special_tokens=True
            )
            text = text[0] if isinstance(text, list) else text
            # words=None → caller (routes) uses build_word_timestamps as before.
            return Transcription(text=text, tokens=toks, durations=durs, words=None)

        # ── Long-audio path: VAD chunk → batched encode → per-chunk decode ───
        from ..streaming.chunker import split_on_silence
        from ..streaming.file_stream import build_word_timestamps

        vad_chunks = split_on_silence(audio, max_chunk_s=self.max_chunk_s)
        offsets = [off for off, _ in vad_chunks]
        chunk_waves = [c for _, c in vad_chunks]

        # Phase 1: ONE batched encoder forward for ALL chunks (throughput win).
        all_frames, all_lengths = self._batch_encode_chunks(chunk_waves)

        texts: list[str] = []
        all_words: list[dict] = []
        all_toks: list[int] = []
        all_durs: list[int] = []

        for i, offset_samples in enumerate(offsets):
            # Slice this chunk's encoder output (batch dim i).
            frames_i = all_frames[i : i + 1]   # [1, T', 1024]
            lengths_i = all_lengths[i : i + 1]  # [1]

            chunk_request_id = f"{request_id}__chunk{i}"
            toks_i, durs_i = await self._decode_single(
                frames_i, lengths_i, chunk_request_id
            )

            # Decode text for this chunk.
            text_i = self.processor.decode(
                torch.tensor([toks_i]), skip_special_tokens=True
            )
            text_i = text_i[0] if isinstance(text_i, list) else text_i
            texts.append(text_i)

            # Build per-chunk word timestamps, then shift by the chunk's
            # start time (offset_samples / 16000 seconds).
            offset_s: float = offset_samples / 16_000.0
            try:
                words_i = build_word_timestamps(toks_i, durs_i, self.processor)
                for w in words_i:
                    all_words.append(
                        {
                            "word": w["word"],
                            "start": w["start"] + offset_s,
                            "end": w["end"] + offset_s,
                        }
                    )
            except Exception:
                # Timestamp computation is best-effort; don't fail the
                # transcription if it errors (e.g. blank-only chunk).
                pass

            all_toks.extend(toks_i)
            all_durs.extend(durs_i)

        joined_text = " ".join(t for t in texts if t).strip()
        # Always return a list (possibly empty) for multi-chunk audio so that
        # routes.py never falls back to build_word_timestamps on the
        # concatenated cross-chunk token/duration stream, which would produce
        # wrong absolute timestamps.  The short-audio path still yields
        # words=None (see above) so callers can distinguish the two cases.
        return Transcription(
            text=joined_text,
            tokens=all_toks,
            durations=all_durs,
            words=all_words,
        )

    async def transcribe_stream(
        self, audio: np.ndarray, request_id: str
    ) -> AsyncIterator[str]:
        """Stream partial transcription text as tokens arrive.

        Routes through the two-phase scheduler so streaming requests also
        participate in bounded concurrency.

        Note: long-audio chunking is NOT applied on the streaming path; callers
        should prefer the non-streaming ``transcribe`` for very long files.
        """
        import torch

        toks: list[int] = []
        durs: list[int] = []
        async for delta in self._scheduler.run(
            inputs=audio,
            phase1_fn=self._phase1_encode,
            phase2_fn=self._phase2_decode,
            request_id=request_id,
        ):
            toks.extend(delta.token_ids)
            durs.extend(delta.durations)
            if delta.token_ids:
                partial = self.processor.decode(
                    torch.tensor([toks]), skip_special_tokens=True
                )
                yield partial[0] if isinstance(partial, list) else partial

from __future__ import annotations
import asyncio, logging, uuid
import numpy as np
from .protocol import server_event
from .localagreement import LocalAgreement
from .vad import StreamingVAD

logger = logging.getLogger(__name__)

# Sentinel placed on self._out by close() after any in-flight decode completes.
# events() exits only when it dequeues this object — no poll/timeout needed.
_CLOSE = object()


class LiveSession:
    """Per-connection ASR engine: hop loop, server_vad endpointing, LocalAgreement deltas.

    ``decode_fn(audio_16k, request_id) -> (words, text, ts)`` is injected so the
    session is fully testable without a GPU (pass a mock; use Task-4's
    ``decode_window`` in production).

    Outbound server events are queued and drained via ``async for ev in session.events()``.

    Assumes a single ``feed_pcm`` producer; ``_lock`` serializes decode/LocalAgreement
    state so concurrent re-entrancy from the producer and ``close()`` is safe.
    """

    def __init__(
        self,
        *,
        decode_fn,
        vad=None,
        hop_ms: int = 500,
        agreement_n: int = 2,
        max_segment_s: int = 25,
        sample_rate: int = 16000,
    ):
        self._decode = decode_fn
        self._vad = vad if vad is not None else StreamingVAD(sample_rate=sample_rate)
        self._sr = sample_rate
        self._hop = int(sample_rate * hop_ms / 1000)
        self._max_seg = int(sample_rate * max_segment_s)
        self._agreement_n = agreement_n
        self._out: asyncio.Queue = asyncio.Queue()
        self._seg = np.zeros(0, dtype=np.float32)   # current segment audio
        self._since_hop = 0
        self._speaking = False
        self._la = LocalAgreement(n=agreement_n)
        self._rid = uuid.uuid4().hex
        self._closed = False
        self._manual = False   # turn_detection is null → manual commit mode
        self._lock = asyncio.Lock()

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    async def _emit(self, type: str, **fields) -> None:
        await self._out.put(server_event(type, **fields))

    def _reset_segment(self) -> None:
        self._seg = np.zeros(0, dtype=np.float32)
        self._since_hop = 0
        self._speaking = False
        self._la = LocalAgreement(n=self._agreement_n)
        self._rid = uuid.uuid4().hex
        try:
            self._vad.reset()
        except Exception:
            logger.warning("VAD reset failed", exc_info=True)

    # ------------------------------------------------------------------
    # Decode helpers
    # ------------------------------------------------------------------

    async def _hop_decode(self) -> None:
        async with self._lock:
            words, _text, _ts = await self._decode(self._seg, self._rid)
            newly = self._la.commit(words)
            if newly:
                # prepend a space separator when there are already committed words before
                # these newly committed ones (append-only stream; space-join on delta side)
                prefix_sep = " " if self._la.committed[: -len(newly)] else ""
                await self._emit(
                    "conversation.item.input_audio_transcription.delta",
                    delta=prefix_sep + " ".join(newly),
                )

    async def _finalize(self) -> None:
        # Approach-A note: the committed LocalAgreement deltas are append-only across
        # hops, but the authoritative `completed.transcript` comes from a full-segment
        # re-decode here.  The delta word-list being a strict prefix of the final
        # transcript is an empirical Approach-A property (holds when growing-window
        # decode is prefix-stable), not a hard code invariant.
        async with self._lock:
            if self._seg.size == 0:
                await self._emit("input_audio_buffer.committed", item_id=self._rid)
                await self._emit(
                    "conversation.item.input_audio_transcription.completed",
                    item_id=self._rid,
                    transcript="",
                )
                self._reset_segment()
                return
            words, text, ts = await self._decode(self._seg, self._rid)
            await self._emit("input_audio_buffer.committed", item_id=self._rid)
            await self._emit(
                "conversation.item.input_audio_transcription.completed",
                item_id=self._rid,
                transcript=text,
                words=ts,
            )
            self._reset_segment()

    # ------------------------------------------------------------------
    # Public API
    # ------------------------------------------------------------------

    async def handle_client_event(self, ev: dict) -> None:
        """Process a parsed client event (from ``protocol.parse_client_event``)."""
        t = ev["type"]
        if t == "session.update":
            sess = ev.get("session", {}) or {}
            self._manual = ("turn_detection" in sess) and (sess["turn_detection"] is None)
            if any("logprobs" in str(x) for x in (ev.get("include") or [])):
                await self._emit("error", error={"message": "logprobs not supported"})
            await self._emit("session.updated", session=sess)
        elif t == "input_audio_buffer.commit":
            await self._finalize()
        elif t == "input_audio_buffer.clear":
            self._reset_segment()
        # input_audio_buffer.append is handled externally via feed_pcm
        # (the protocol layer decodes the base64 payload and resamples)

    async def feed_pcm(self, pcm_16k: np.ndarray) -> None:
        """Push a PCM chunk (float32, 16 kHz) into the session."""
        if self._closed:
            return
        for ev in self._vad.push(pcm_16k):
            if ev.kind == "speech_started" and not self._speaking:
                self._speaking = True
                if not self._manual:   # manual mode: accumulate but don't surface VAD events
                    await self._emit("input_audio_buffer.speech_started", audio_start_ms=0)
            elif ev.kind == "speech_stopped" and self._speaking and not self._manual:
                await self._emit("input_audio_buffer.speech_stopped", audio_end_ms=0)
                await self._finalize()
                continue  # _reset_segment already ran; skip accumulation below
        if self._speaking or self._manual:
            self._seg = np.concatenate([self._seg, pcm_16k])
            self._since_hop += pcm_16k.size
            if self._since_hop >= self._hop and self._seg.size:
                self._since_hop = 0
                await self._hop_decode()
            if self._seg.size >= self._max_seg:   # runaway segment cap
                # Mirror the VAD-stop path: clients in auto/server_vad mode
                # expect speech_stopped BEFORE committed+completed.  Manual mode
                # never emits VAD turn events, so skip it there.
                if self._speaking and not self._manual:
                    await self._emit("input_audio_buffer.speech_stopped", audio_end_ms=0)
                await self._finalize()

    async def events(self):
        """Async generator that yields outbound server-event dicts until the sentinel.

        Exits only when it dequeues the ``_CLOSE`` sentinel, which ``close()`` places
        after any in-flight decode has finished — guaranteeing no ``committed`` or
        ``completed`` event is ever orphaned.
        """
        while True:
            ev = await self._out.get()
            if ev is _CLOSE:
                break
            yield ev

    async def close(self) -> None:
        """Signal end-of-session; waits for any in-flight decode, then enqueues sentinel."""
        self._closed = True
        async with self._lock:
            pass   # await any in-flight _hop_decode / _finalize to finish
        await self._out.put(_CLOSE)

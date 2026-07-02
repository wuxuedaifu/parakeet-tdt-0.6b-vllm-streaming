import asyncio, numpy as np, pytest
from parakeet_vllm.realtime.session import LiveSession


class FakeVAD:
    """Deterministic VAD: emits start on first push, stop when fed the sentinel."""
    def __init__(self): self.started = False
    def reset(self): self.started = False
    def push(self, pcm):
        from parakeet_vllm.realtime.vad import VadEvent
        evs = []
        if not self.started and pcm.size:
            self.started = True; evs.append(VadEvent("speech_started", 0))
        if pcm.size and float(np.max(np.abs(pcm))) == 0.0 and self.started:
            self.started = False; evs.append(VadEvent("speech_stopped", 0))
        return evs


def make_decode_fn(script):
    calls = {"i": 0}
    async def decode_fn(audio, rid):
        i = min(calls["i"], len(script)-1); calls["i"] += 1
        words = script[i]
        return words, " ".join(words), []
    return decode_fn


def drain(session):
    out = []
    async def _c():
        async for ev in session.events():
            out.append(ev)
    return _c, out


def test_emits_openai_event_sequence():
    # hypotheses grow; LocalAgreement-2 commits "the cat" then "sat"
    script = [["the","cat"], ["the","cat"], ["the","cat","sat"], ["the","cat","sat"]]
    session = LiveSession(decode_fn=make_decode_fn(script), vad=FakeVAD(),
                          hop_ms=100, agreement_n=2)
    async def run():
        consumer, out = drain(session)
        task = asyncio.create_task(consumer())
        # 100ms hops at 16k = 1600 samples/hop; push 4 hops of speech then silence
        for _ in range(4):
            await session.feed_pcm(np.full(1600, 0.2, dtype=np.float32))
        await session.feed_pcm(np.zeros(1600, dtype=np.float32))   # silence → stop
        await session.close()
        await task
        return out

    out = asyncio.run(run())
    types = [e["type"] for e in out]

    # Basic event-type presence
    assert types[0] == "input_audio_buffer.speech_started"
    assert "conversation.item.input_audio_transcription.delta" in types
    assert "input_audio_buffer.committed" in types
    assert types[-1] == "conversation.item.input_audio_transcription.completed"

    # (a) Event ordering invariant:
    #     speech_started → delta* → speech_stopped → committed → completed
    delta_idxs = [i for i, t in enumerate(types) if "transcription.delta" in t]
    stopped_idx = types.index("input_audio_buffer.speech_stopped")
    committed_idx = types.index("input_audio_buffer.committed")
    completed_idx = next(i for i, t in enumerate(types) if "transcription.completed" in t)
    assert all(i < stopped_idx for i in delta_idxs), (
        f"all delta events must precede speech_stopped; delta idxs={delta_idxs}, stopped={stopped_idx}"
    )
    assert stopped_idx < committed_idx < completed_idx, (
        f"expected speech_stopped({stopped_idx}) < committed({committed_idx}) < completed({completed_idx})"
    )

    # (b) Deltas are non-empty
    deltas = [e["delta"] for e in out if e["type"].endswith("transcription.delta")]
    assert len(deltas) > 0, "expected at least one transcription.delta event"

    # (c) Word-list prefix invariant: concatenated delta words must be a prefix
    #     of the completed transcript's words (append-only, never retract).
    completed = next(e for e in out if e["type"].endswith("transcription.completed"))
    delta_words = " ".join(deltas).split()
    completed_words = completed["transcript"].split()
    assert len(delta_words) > 0, "concatenated delta text must contain at least one word"
    assert completed_words[:len(delta_words)] == delta_words, (
        f"delta words {delta_words!r} are not a prefix of completed words {completed_words!r}"
    )
    assert "the cat" in completed["transcript"]


def test_manual_commit_mode():
    script = [["hello"], ["hello","world"], ["hello","world"]]
    from parakeet_vllm.realtime.session import LiveSession
    session = LiveSession(decode_fn=make_decode_fn(script), vad=FakeVAD(),
                          hop_ms=100, agreement_n=2)
    async def run():
        consumer, out = drain(session); task = asyncio.create_task(consumer())
        await session.handle_client_event({"type":"session.update",
                                           "session":{"turn_detection": None}})
        for _ in range(3):
            await session.feed_pcm(np.full(1600, 0.2, dtype=np.float32))
        await session.handle_client_event({"type":"input_audio_buffer.commit"})
        await session.close(); await task; return out
    out = asyncio.run(run())
    assert any(e["type"].endswith("transcription.completed") for e in out)
    # non-VAD feeds must not produce extra committed events
    assert [e["type"] for e in out].count("input_audio_buffer.committed") == 1


def test_clear_resets_segment():
    from parakeet_vllm.realtime.session import LiveSession
    session = LiveSession(decode_fn=make_decode_fn([["x"]]), vad=FakeVAD(), hop_ms=100)
    async def run():
        consumer, out = drain(session); task = asyncio.create_task(consumer())
        await session.feed_pcm(np.full(1600, 0.2, dtype=np.float32))
        await session.handle_client_event({"type":"input_audio_buffer.clear"})
        await session.close(); await task; return out
    out = asyncio.run(run())
    # after clear, no completed emitted (segment discarded)
    assert not any(e["type"].endswith("transcription.completed") for e in out)


def test_max_segment_cap_force_finalizes():
    from parakeet_vllm.realtime.session import LiveSession
    long_script = [["w"]*10]*100
    # FakeVAD fires speech_started on first non-empty push, never fires speech_stopped
    # (all chunks have amplitude 0.2, not zero), so the cap is what triggers finalization.
    session = LiveSession(decode_fn=make_decode_fn(long_script), vad=FakeVAD(),
                          hop_ms=100, max_segment_s=1)  # 1s cap @16k = 16000 samples
    async def run():
        consumer, out = drain(session); task = asyncio.create_task(consumer())
        for _ in range(12):   # 12*1600 = 19200 > 16000 → cap fires
            await session.feed_pcm(np.full(1600, 0.2, dtype=np.float32))
        await session.close(); await task; return out
    out = asyncio.run(run())
    types = [e["type"] for e in out]
    # Cap path must preserve the Task-5 invariant:
    #   speech_stopped → committed → completed
    assert "input_audio_buffer.speech_stopped" in types, (
        f"cap path must emit speech_stopped before finalize; got {types}"
    )
    assert "input_audio_buffer.committed" in types, (
        f"cap path must emit committed; got {types}"
    )
    assert any(e["type"].endswith("transcription.completed") for e in out), (
        f"cap path must emit completed; got {types}"
    )
    stopped_idx = types.index("input_audio_buffer.speech_stopped")
    committed_idx = types.index("input_audio_buffer.committed")
    completed_idx = next(i for i, t in enumerate(types) if "transcription.completed" in t)
    assert stopped_idx < committed_idx < completed_idx, (
        f"expected speech_stopped({stopped_idx}) < committed({committed_idx}) < "
        f"completed({completed_idx}); got {types}"
    )


def test_manual_mode_no_vad_events():
    """In manual mode, VAD-driven speech_started/speech_stopped must NOT be emitted.
    Finalization is driven only by input_audio_buffer.commit.
    """
    from parakeet_vllm.realtime.session import LiveSession
    script = [["hello"], ["hello","world"], ["hello","world"]]
    session = LiveSession(decode_fn=make_decode_fn(script), vad=FakeVAD(),
                          hop_ms=100, agreement_n=2)
    async def run():
        consumer, out = drain(session); task = asyncio.create_task(consumer())
        await session.handle_client_event({"type":"session.update",
                                           "session":{"turn_detection": None}})
        for _ in range(3):
            await session.feed_pcm(np.full(1600, 0.2, dtype=np.float32))
        await session.handle_client_event({"type":"input_audio_buffer.commit"})
        await session.close(); await task; return out
    out = asyncio.run(run())
    types = [e["type"] for e in out]
    # No VAD turn events in manual mode
    assert "input_audio_buffer.speech_started" not in types, (
        f"speech_started must not be emitted in manual mode; got {types}"
    )
    assert "input_audio_buffer.speech_stopped" not in types, (
        f"speech_stopped must not be emitted in manual mode; got {types}"
    )
    # commit produces committed + completed
    assert "input_audio_buffer.committed" in types
    assert any(e["type"].endswith("transcription.completed") for e in out)


def test_close_during_finalize_does_not_lose_events():
    """Regression: close() racing an IN-FLIGHT _finalize() decode must not
    orphan committed/completed events.

    The old timeout-based events() (``while not (self._closed and self._out.empty())``,
    0.25 s wait_for timeout) exits as soon as it sees closed=True AND queue empty —
    even if _finalize() is still mid-decode.  With a finalize decode that sleeps 0.4 s
    (longer than the 0.25 s wait_for timeout), the old events() times out, sees the
    closed+empty condition, and exits before committed/completed are enqueued.

    The sentinel fix: close() acquires _lock (held by _finalize during its decode),
    so it cannot enqueue _CLOSE until _finalize has emitted committed+completed.
    events() exits only on the sentinel — guaranteeing the full event set is drained.

    This test is structured so close() genuinely races an in-flight finalize:
    the silence feed (which triggers _finalize) is created as a background task;
    close() is called while _finalize's decode is still awaiting.
    """
    call_count = {"n": 0}

    async def slow_decode_fn(audio, rid):
        call_count["n"] += 1
        if call_count["n"] >= 2:
            # 2nd call is from _finalize(); sleep > the old events() 0.25 s wait_for
            # timeout so the old code's timeout fires BEFORE finalize completes.
            await asyncio.sleep(0.4)
        return ["hello"], "hello", []

    session = LiveSession(
        decode_fn=slow_decode_fn,
        vad=FakeVAD(),
        hop_ms=100,
        agreement_n=2,
    )

    async def _consume(session, out):
        async for ev in session.events():
            out.append(ev)

    async def run():
        collected = []
        # Start consumer before feeding so it drains events concurrently.
        consumer_task = asyncio.create_task(_consume(session, collected))

        # One hop of speech → speech_started + hop_decode (call #1, instant).
        await session.feed_pcm(np.full(1600, 0.2, dtype=np.float32))

        # Trigger finalize as a background task: _finalize acquires the lock and
        # calls slow_decode_fn (call #2) which sleeps 0.4 s.  The task will NOT
        # return until well after close() is called below.
        fin = asyncio.create_task(
            session.feed_pcm(np.zeros(1600, dtype=np.float32))
        )
        # Yield control so the fin task enters _finalize's decode await (~0.1 s
        # is well within the 0.4 s decode sleep).
        await asyncio.sleep(0.1)

        # close() now races the still-sleeping finalize decode.
        # Fixed code: acquires _lock → waits for _finalize → then enqueues _CLOSE.
        # Old code: just sets _closed=True; events() times out at 0.25 s and exits
        #           before _finalize completes at 0.4 s → committed/completed orphaned.
        await session.close()
        await fin
        await consumer_task
        return collected

    out = asyncio.run(run())
    types = [e["type"] for e in out]

    assert "input_audio_buffer.committed" in types, (
        f"committed event orphaned by close-race; got types={types}"
    )
    assert any("transcription.completed" in t for t in types), (
        f"completed event orphaned by close-race; got types={types}"
    )

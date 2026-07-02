import numpy as np
import pytest
from parakeet_vllm.realtime.vad import StreamingVAD, VadEvent


def _sil(sec, sr=16000):
    return np.zeros(int(sec * sr), dtype=np.float32)


def _real_speech_16k():
    """Load the first utterance from hf-internal-testing/librispeech_asr_dummy resampled to 16 kHz."""
    from datasets import load_dataset, Audio
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation", trust_remote_code=True)
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    return ds["audio"][0]["array"].astype(np.float32)


def test_detects_start_then_stop():
    """Real speech clip surrounded by silence should emit speech_started then speech_stopped."""
    speech = _real_speech_16k()
    vad = StreamingVAD()
    events = []
    for seg in [_sil(0.5), speech, _sil(1.5)]:
        for i in range(0, len(seg), 1600):  # 100ms pushes
            events += vad.push(seg[i:i + 1600])
    kinds = [e.kind for e in events]
    assert "speech_started" in kinds, f"Expected speech_started but got events: {events}"
    assert kinds.index("speech_started") < kinds.index("speech_stopped"), (
        f"speech_started must precede speech_stopped, got: {kinds}"
    )


def test_pure_silence_no_events():
    vad = StreamingVAD()
    events = []
    for i in range(0, len(_sil(2.0)), 1600):
        events += vad.push(_sil(2.0)[i:i + 1600])
    assert events == []

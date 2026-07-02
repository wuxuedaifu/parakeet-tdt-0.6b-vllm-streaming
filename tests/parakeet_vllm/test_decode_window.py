import asyncio, pytest
pytest.importorskip("torch")

@pytest.mark.gpu
def test_decode_window_matches_offline():
    from datasets import load_dataset, Audio
    from parakeet_vllm.realtime.decoder_stream import decode_window
    from parakeet_vllm.engine.asr_engine import ASREngine
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    audio = ds["audio"][0]["array"].astype("float32")

    offline = asyncio.run(ASREngine().transcribe(audio, "off")).text
    words, text, ts = asyncio.run(decode_window(audio, "win"))
    assert text.strip() == offline.strip()
    assert words == text.split()
    assert len(ts) >= 1

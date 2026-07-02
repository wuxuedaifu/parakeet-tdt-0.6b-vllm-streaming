import asyncio, json, pytest


def test_stream_grows_monotonically(monkeypatch):
    from parakeet_vllm.streaming.file_stream import partials_to_sse

    async def fake_partials():
        for t in ["he", "hello", "hello world"]:
            yield t

    async def collect():
        return [c async for c in partials_to_sse(fake_partials())]

    chunks = asyncio.run(collect())

    # Collect data: lines, split off the terminal [DONE]
    data_lines = [c.strip() for c in chunks if c.startswith("data:")]
    assert len(data_lines) >= 3

    non_done = [ln for ln in data_lines if ln != "data: [DONE]"]
    assert data_lines[-1] == "data: [DONE]", "last event must be the [DONE] sentinel"

    # Parse partial text out of each JSON event
    texts = [json.loads(ln[len("data: "):])["text"] for ln in non_done]

    # Text length must be non-decreasing (monotonically growing)
    assert all(len(a) <= len(b) for a, b in zip(texts, texts[1:])), (
        f"stream text lengths are not non-decreasing: {[len(t) for t in texts]}"
    )

    # Last partial text must match the final expected value
    assert texts[-1] == "hello world"


@pytest.mark.gpu
def test_timestamps_match_processor():
    from datasets import load_dataset, Audio
    from parakeet_vllm.model_loader import get_processor, get_reference_model
    from parakeet_vllm.streaming.file_stream import build_word_timestamps

    proc, model = get_processor(), get_reference_model()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    inputs = proc(ds["audio"][0]["array"], sampling_rate=16000, return_tensors="pt",
                  return_attention_mask=True).to(model.device, dtype=model.dtype)
    out = model.generate(**inputs, return_dict_in_generate=True)
    _txt, ref_ts = proc.decode(out.sequences, durations=out.durations, skip_special_tokens=True)
    ours = build_word_timestamps(out.sequences[0].tolist(), out.durations[0].tolist(), proc)
    assert len(ours) == len(ref_ts[0])

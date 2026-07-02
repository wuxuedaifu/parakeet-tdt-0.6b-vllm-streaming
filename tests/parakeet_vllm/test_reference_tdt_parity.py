import asyncio

import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("transformers")

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


@pytest.mark.gpu
def test_reference_backend_matches_generate():
    from transformers import AutoModelForTDT, AutoProcessor
    from datasets import load_dataset, Audio
    from parakeet_vllm.decode.reference_tdt import ReferenceTDTBackend

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(MODEL_ID, device_map="auto").eval()
    ds = load_dataset(
        "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
    )
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    sample = ds["audio"][0]["array"]
    inputs = processor(sample, sampling_rate=16000).to(model.device, dtype=model.dtype)

    # Oracle
    oracle = model.generate(**inputs, return_dict_in_generate=True)
    oracle_ids = oracle.sequences[0].tolist()

    # Reference backend: encoder outside, then step-driven decode
    enc = model.encoder(
        input_features=inputs["input_features"],
        attention_mask=inputs.get("attention_mask"),
    )
    frames = enc.last_hidden_state
    lengths = torch.tensor([frames.shape[1]], device=frames.device)

    backend = ReferenceTDTBackend(model)

    async def run():
        ids = []
        async for d in backend.decode_stream("r0", frames, lengths):
            ids.extend(d.token_ids)
        return ids

    got = asyncio.run(run())
    assert got == oracle_ids

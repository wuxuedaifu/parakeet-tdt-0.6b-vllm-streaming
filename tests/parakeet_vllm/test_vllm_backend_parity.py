import asyncio
import pytest

torch = pytest.importorskip("torch")
pytest.importorskip("vllm")

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


@pytest.mark.gpu
@pytest.mark.slow
def test_vllm_backend_matches_oracle():
    from transformers import AutoModelForTDT, AutoProcessor
    from datasets import load_dataset, Audio
    from parakeet_vllm.decode.vllm_backend import VLLMDecodeBackend

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    ref = AutoModelForTDT.from_pretrained(MODEL_ID, device_map="auto").eval()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    sample = ds["audio"][0]["array"]
    inputs = processor(sample, sampling_rate=16000).to(ref.device, dtype=ref.dtype)
    oracle_ids = ref.generate(**inputs, return_dict_in_generate=True).sequences[0].tolist()
    enc = ref.encoder(input_features=inputs["input_features"],
                      attention_mask=inputs.get("attention_mask"))
    frames, lengths = enc.last_hidden_state, torch.tensor([enc.last_hidden_state.shape[1]])

    backend = VLLMDecodeBackend(model_id=MODEL_ID)  # boots AsyncLLMEngine (V1)

    async def run():
        ids = []
        async for d in backend.decode_stream("r0", frames, lengths):
            ids.extend(d.token_ids)
        return ids

    assert asyncio.run(run()) == oracle_ids


@pytest.mark.gpu
def test_vllm_backend_records_nogo():
    """Task 3 NO-GO gate: in-engine vLLM V1 TDT decode is not expressible.

    The custom model + backend are implemented (see parakeet_vllm/vllm_engine/
    and decode/vllm_backend.py) and reach the documented blocker. This test
    pins that decision so a future vLLM version that lifts the blocker will make
    it fail and prompt a re-evaluation.
    """
    from parakeet_vllm.decode.vllm_backend import VLLMDecodeBackend, VLLMInEngineNoGo

    with pytest.raises(VLLMInEngineNoGo):
        VLLMDecodeBackend(model_id=MODEL_ID)


def test_factory_defaults_to_reference():
    """PARAKEET_VLLM_BACKEND factory: NO-GO selects 'reference' as default."""
    from parakeet_vllm.decode import DEFAULT_BACKEND

    assert DEFAULT_BACKEND == "reference"

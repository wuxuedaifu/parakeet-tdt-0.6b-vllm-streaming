import asyncio, pytest
torch = pytest.importorskip("torch")
pytest.importorskip("vllm")
MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


@pytest.mark.gpu
@pytest.mark.slow
def test_vllm_encoder_frames_match_hf_and_decode_matches_oracle():
    from transformers import AutoModelForTDT, AutoProcessor
    from datasets import load_dataset, Audio
    from parakeet_vllm.vllm_engine.encoder_engine import VLLMEncoder
    from parakeet_vllm.decode.reference_tdt import ReferenceTDTBackend

    proc = AutoProcessor.from_pretrained(MODEL_ID)
    ref = AutoModelForTDT.from_pretrained(MODEL_ID, device_map="auto").eval()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    sample = ds["audio"][0]["array"]
    inputs = proc(sample, sampling_rate=16000, return_tensors="pt",
                  return_attention_mask=True).to(ref.device, dtype=ref.dtype)

    hf_frames = ref.encoder(input_features=inputs["input_features"],
                            attention_mask=inputs["attention_mask"]).last_hidden_state
    oracle_ids = ref.generate(**inputs, return_dict_in_generate=True).sequences[0].tolist()

    enc = VLLMEncoder(model_id=MODEL_ID)
    frames, lengths = asyncio.run(enc.encode_async(inputs["input_features"],
                                                   inputs["attention_mask"]))
    # 1) frame-level numerical parity with the HF encoder
    assert frames.shape[1] == hf_frames.shape[1]
    assert torch.allclose(frames.to(hf_frames.dtype), hf_frames, atol=1e-2)
    # 2) end-to-end: vLLM-encoder frames through the reference decode == oracle
    backend = ReferenceTDTBackend(ref)
    async def run():
        ids = []
        async for d in backend.decode_stream("r0", frames.to(ref.device), lengths.to(ref.device)):
            ids.extend(d.token_ids)
        return ids
    assert asyncio.run(run()) == oracle_ids


def test_vllm_encoder_records_nogo():
    """Pin the Task-3b NO-GO decision (see the findings doc).

    Unlike Task 3 (in-engine decode), per-frame pooling *is* supported by vLLM
    0.24.0 V1 (tokwise ``AllPool`` + multimodal). The NO-GO is that there is no
    standalone vLLM Parakeet-encoder pooling architecture, so ``VLLMEncoder``
    cannot boot one and instead raises ``VLLMEncoderNoGo``. This test also asserts
    that vLLM still auto-resolves this checkpoint to the generic
    ``TransformersForCausalLM`` backend (i.e. no native Parakeet-encoder arch);
    if a future vLLM ships one, this test breaks and prompts a GO re-evaluation.
    """
    from parakeet_vllm.vllm_engine.encoder_engine import (
        VLLMEncoder,
        VLLMEncoderNoGo,
        PARAKEET_ENCODER_BACKEND_DEFAULT,
    )

    # Project default encoder backend is the batched-PyTorch fallback.
    assert PARAKEET_ENCODER_BACKEND_DEFAULT == "torch"

    with pytest.raises(VLLMEncoderNoGo):
        VLLMEncoder(model_id=MODEL_ID)


def test_vllm_still_lacks_native_parakeet_encoder_arch():
    """Empirical re-open sentinel: vLLM has no registered ``Parakeet*``
    architecture, so ``ParakeetForTDT`` cannot resolve to a native standalone
    encoder pooling model. If a future vLLM ships one, this breaks and prompts a
    GO re-evaluation.
    """
    pytest.importorskip("vllm")
    from vllm import ModelRegistry

    supported = set(ModelRegistry.get_supported_archs())
    assert "ParakeetForTDT" not in supported
    assert [a for a in supported if "Parakeet" in a] == []

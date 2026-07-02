import pytest
torch = pytest.importorskip("torch")
pytest.importorskip("transformers")


@pytest.mark.gpu
def test_encoder_matches_hf_and_returns_lengths():
    from datasets import load_dataset, Audio
    from parakeet_vllm.model_loader import get_processor, get_reference_model
    from parakeet_vllm.encoder import encode
    proc, model = get_processor(), get_reference_model()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    inputs = proc(ds["audio"][0]["array"], sampling_rate=16000, return_tensors="pt",
                  return_attention_mask=True).to(model.device, dtype=model.dtype)
    frames, lengths = encode(inputs["input_features"], inputs["attention_mask"])
    ref = model.encoder(input_features=inputs["input_features"],
                        attention_mask=inputs["attention_mask"]).last_hidden_state
    assert torch.allclose(frames, ref, atol=1e-3)
    assert lengths.shape[0] == 1 and int(lengths[0]) <= frames.shape[1]

import asyncio
import pytest

pytest.importorskip("transformers")


@pytest.mark.gpu
def test_engine_text_matches_oracle():
    from datasets import load_dataset, Audio
    from parakeet_vllm.model_loader import get_processor, get_reference_model
    from parakeet_vllm.engine.asr_engine import ASREngine

    proc, model = get_processor(), get_reference_model()
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    sample = ds["audio"][0]["array"]
    inputs = proc(sample, sampling_rate=16000, return_tensors="pt",
                  return_attention_mask=True).to(model.device, dtype=model.dtype)
    oracle = proc.decode(model.generate(**inputs, return_dict_in_generate=True).sequences,
                         skip_special_tokens=True)
    oracle = oracle[0] if isinstance(oracle, list) else oracle

    eng = ASREngine()
    got = asyncio.run(eng.transcribe(sample.astype("float32"), "r0")).text
    assert got.strip() == oracle.strip()

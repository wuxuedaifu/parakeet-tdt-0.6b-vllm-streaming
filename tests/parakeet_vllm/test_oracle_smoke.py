import re
import pytest

torch = pytest.importorskip("torch")
transformers = pytest.importorskip("transformers")

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


@pytest.mark.gpu
def test_oracle_transcribes_known_clip():
    """ParakeetForTDT.generate must transcribe the librispeech dummy clip.

    Expected text is documented in the transformers Parakeet model card.
    """
    from transformers import AutoModelForTDT, AutoProcessor
    from datasets import load_dataset, Audio

    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(MODEL_ID, device_map="auto")

    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=processor.feature_extractor.sampling_rate))
    sample = ds["audio"][0]["array"]

    inputs = processor(sample, sampling_rate=processor.feature_extractor.sampling_rate)
    inputs = inputs.to(model.device, dtype=model.dtype)
    out = model.generate(**inputs, return_dict_in_generate=True)
    text = processor.decode(out.sequences, skip_special_tokens=True)
    text = text[0] if isinstance(text, list) else text

    norm = re.sub(r"[^a-z ]", "", text.lower())
    assert "quilter is the apostle of the middle classes" in norm

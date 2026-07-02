import numpy as np, pytest
pytest.importorskip("transformers")

@pytest.mark.gpu
def test_batched_features_have_mask():
    from parakeet_vllm.features import extract_features
    a = np.zeros(16000, dtype="float32")
    b = np.zeros(8000, dtype="float32")
    feats = extract_features([a, b])
    assert feats["input_features"].shape[0] == 2
    assert "attention_mask" in feats
    # ParakeetFeatureExtractor uses feature_size=128 (not 80 like Whisper).
    # Brief specified 80 but the actual model config has feature_size=128.
    assert feats["input_features"].shape[-1] == 128

import pytest
pytest.importorskip("transformers")


def test_config_defaults(monkeypatch):
    monkeypatch.delenv("PARAKEET_MODEL_ID", raising=False)
    monkeypatch.delenv("PARAKEET_VLLM_BACKEND", raising=False)
    monkeypatch.delenv("PARAKEET_ENCODER_BACKEND", raising=False)
    import importlib, parakeet_vllm.config as c
    importlib.reload(c)
    assert c.MODEL_ID == "nvidia/parakeet-tdt-0.6b-v3"
    assert c.BACKEND == "reference"
    assert c.ENCODER_BACKEND == "torch"


@pytest.mark.gpu
def test_loader_singletons():
    from parakeet_vllm.model_loader import get_processor, get_reference_model
    assert get_processor() is get_processor()
    assert get_reference_model() is get_reference_model()


@pytest.mark.gpu
def test_loader_returns_parakeet_tdt():
    from parakeet_vllm.model_loader import get_reference_model
    m = get_reference_model()
    assert type(m).__name__ == "ParakeetForTDT"
    # sanity: the submodules ReferenceTDTBackend relies on exist
    assert hasattr(m, "encoder") and hasattr(m, "encoder_projector")
    assert list(m.config.durations) == [0, 1, 2, 3, 4]

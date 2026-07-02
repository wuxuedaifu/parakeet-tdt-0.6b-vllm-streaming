from __future__ import annotations
import functools

from .config import MODEL_ID, BACKEND, DEVICE


@functools.lru_cache(maxsize=1)
def get_processor():
    """Return a singleton AutoProcessor for the configured model."""
    from transformers import AutoProcessor
    return AutoProcessor.from_pretrained(MODEL_ID)


@functools.lru_cache(maxsize=1)
def get_reference_model():
    """Return a singleton ParakeetForTDT model on the configured device."""
    # ParakeetForTDT is registered under AutoModelForTDT (transformers >= 5.x).
    # Fall back to the explicit class path only if the Auto mapping is unavailable.
    try:
        from transformers import AutoModelForTDT
        return AutoModelForTDT.from_pretrained(
            MODEL_ID, device_map=DEVICE
        ).eval()
    except ImportError:
        from transformers.models.parakeet.modeling_parakeet import ParakeetForTDT
        return ParakeetForTDT.from_pretrained(
            MODEL_ID, device_map=DEVICE
        ).eval()


@functools.lru_cache(maxsize=1)
def get_decode_backend():
    """Return a singleton DecodeBackend honoring the BACKEND config.

    - BACKEND == "vllm"      → VLLMDecodeBackend (raises VLLMInEngineNoGo — explicit
                                opt-in to the NO-GO path for reproducibility)
    - anything else (default "reference") → ReferenceTDTBackend
    """
    if BACKEND == "vllm":
        from .decode.vllm_backend import VLLMDecodeBackend
        return VLLMDecodeBackend(model_id=MODEL_ID)
    from .decode.reference_tdt import ReferenceTDTBackend
    return ReferenceTDTBackend(get_reference_model())

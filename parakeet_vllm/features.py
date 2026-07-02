from __future__ import annotations
import numpy as np
from .model_loader import get_processor


def extract_features(waves: list[np.ndarray]):
    proc = get_processor()
    return proc(
        waves,
        sampling_rate=16000,
        return_tensors="pt",
        return_attention_mask=True,
        padding="longest",
    )

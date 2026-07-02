from __future__ import annotations
import numpy as np
import torch
from ..config import DEVICE
from ..features import extract_features
from ..encoder import encode
from ..model_loader import get_decode_backend, get_processor
from ..streaming.file_stream import build_word_timestamps

async def decode_window(audio_16k: np.ndarray, request_id: str):
    feats = extract_features([audio_16k])
    frames, lengths = encode(
        feats["input_features"].to(DEVICE), feats["attention_mask"].to(DEVICE)
    )
    backend = get_decode_backend()
    processor = get_processor()
    toks: list[int] = []
    durs: list[float] = []
    async for d in backend.decode_stream(request_id, frames, lengths):
        toks.extend(d.token_ids)
        durs.extend(d.durations)
    text = processor.decode(torch.tensor([toks]), skip_special_tokens=True)
    if isinstance(text, list):
        text = text[0]
    ts = build_word_timestamps(toks, durs, processor) if toks else []
    return text.split(), text, ts

from __future__ import annotations
import io
import numpy as np
import soundfile as sf
import librosa

TARGET_SR = 16000

def decode_to_16k_mono(data: bytes) -> np.ndarray:
    audio, sr = sf.read(io.BytesIO(data), dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != TARGET_SR:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=TARGET_SR)
    return np.ascontiguousarray(audio, dtype=np.float32)

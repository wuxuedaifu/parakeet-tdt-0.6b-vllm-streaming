"""Manual oracle smoke test. Run: python spike/oracle_smoke.py path/to.wav"""
from __future__ import annotations
import sys
import soundfile as sf
import librosa
from transformers import AutoModelForTDT, AutoProcessor

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def main(path: str) -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(MODEL_ID, device_map="auto")
    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    inputs = processor(audio, sampling_rate=16000).to(model.device, dtype=model.dtype)
    out = model.generate(**inputs, return_dict_in_generate=True)
    print(processor.decode(out.sequences, skip_special_tokens=True))


if __name__ == "__main__":
    main(sys.argv[1])

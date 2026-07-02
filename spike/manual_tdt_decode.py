"""Manual TDT decode parity spike / debugger.

Compares model.generate() (oracle) against ReferenceTDTBackend driven from
precomputed encoder frames, printing per-step (token_id, duration, frame_ptr)
and asserting the token sequences match.

Run:
    python spike/manual_tdt_decode.py path/to.wav
    python spike/manual_tdt_decode.py            # uses librispeech dummy clip[0]
"""
from __future__ import annotations

import asyncio
import os
import sys

# Make the project root importable when run directly (python spike/...).
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import torch
from transformers import AutoModelForTDT, AutoProcessor

from parakeet_vllm.decode.reference_tdt import ReferenceTDTBackend

MODEL_ID = "nvidia/parakeet-tdt-0.6b-v3"


def _load_audio(path: str | None):
    if path is None:
        from datasets import Audio, load_dataset

        ds = load_dataset(
            "hf-internal-testing/librispeech_asr_dummy", "clean", split="validation"
        )
        ds = ds.cast_column("audio", Audio(sampling_rate=16000))
        return ds["audio"][0]["array"]
    import librosa
    import soundfile as sf

    audio, sr = sf.read(path, dtype="float32")
    if audio.ndim > 1:
        audio = audio.mean(axis=1)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000)
    return audio


def main(path: str | None) -> None:
    processor = AutoProcessor.from_pretrained(MODEL_ID)
    model = AutoModelForTDT.from_pretrained(MODEL_ID, device_map="auto").eval()

    audio = _load_audio(path)
    inputs = processor(audio, sampling_rate=16000).to(model.device, dtype=model.dtype)

    oracle = model.generate(**inputs, return_dict_in_generate=True)
    oracle_ids = oracle.sequences[0].tolist()

    enc = model.encoder(
        input_features=inputs["input_features"],
        attention_mask=inputs.get("attention_mask"),
    )
    frames = enc.last_hidden_state
    lengths = torch.tensor([frames.shape[1]], device=frames.device)

    backend = ReferenceTDTBackend(model)

    async def run():
        ids, durs = [], []
        frame_ptr = 0
        async for d in backend.decode_stream("spike", frames, lengths):
            if d.finished:
                break
            for tok, dur in zip(d.token_ids, d.durations):
                print(f"step token_id={tok:>5} duration={dur} frame_ptr={frame_ptr}")
                frame_ptr += dur
                ids.append(tok)
                durs.append(dur)
        return ids, durs

    got, durations = asyncio.run(run())

    print(f"\nT (encoder frames) = {int(lengths[0].item())}")
    print(f"oracle ids ({len(oracle_ids)}): {oracle_ids}")
    print(f"got    ids ({len(got)}): {got}")
    print(f"text: {processor.decode(oracle.sequences, skip_special_tokens=True)}")

    if got != oracle_ids:
        for i, (a, b) in enumerate(zip(got, oracle_ids)):
            if a != b:
                print(f"FIRST DIVERGENCE at index {i}: got={a} oracle={b}")
                break
        raise SystemExit("MISMATCH: reference decode != generate()")
    print("\nPARITY OK: got == oracle_ids")


if __name__ == "__main__":
    main(sys.argv[1] if len(sys.argv) > 1 else None)

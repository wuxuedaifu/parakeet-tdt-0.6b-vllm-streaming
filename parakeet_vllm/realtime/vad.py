from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import torch

_WINDOW = 512  # samples @16k that Silero expects per step


@dataclass
class VadEvent:
    kind: str   # "speech_started" | "speech_stopped"
    sample: int


class StreamingVAD:
    def __init__(self, sample_rate: int = 16000, threshold: float = 0.5):
        from silero_vad import load_silero_vad, VADIterator
        self.sr = sample_rate
        self._model = load_silero_vad(onnx=True)
        self._it = VADIterator(self._model, threshold=threshold, sampling_rate=sample_rate)
        self._buf = np.zeros(0, dtype=np.float32)

    def reset(self) -> None:
        self._it.reset_states()
        self._buf = np.zeros(0, dtype=np.float32)

    def push(self, pcm_16k: np.ndarray) -> list[VadEvent]:
        self._buf = np.concatenate([self._buf, pcm_16k.astype(np.float32, copy=False)])
        events: list[VadEvent] = []
        while len(self._buf) >= _WINDOW:
            window = self._buf[:_WINDOW]
            self._buf = self._buf[_WINDOW:]
            out = self._it(torch.from_numpy(window), return_seconds=False)
            if out:
                if "start" in out:
                    events.append(VadEvent("speech_started", int(out["start"])))
                if "end" in out:
                    events.append(VadEvent("speech_stopped", int(out["end"])))
        return events

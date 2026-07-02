from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Optional


@dataclass
class TDTRequestState:
    """Per-request recurrent state for in-engine TDT decode.

    This is the non-KV state a transducer step needs and that vLLM V1 does not
    manage for us: an externally-driven frame pointer, the count of symbols
    already emitted on the current frame (0-duration re-emission), the stateful
    LSTM prediction-network cache (``ParakeetRNNTDecoderCache``), and the full
    encoder frames captured at prefill so a single frame can be selected per
    step.
    """

    frame_ptr: int = 0
    symbols_this_frame: int = 0
    decoder_cache: Optional[Any] = None      # ParakeetRNNTDecoderCache
    encoder_frames: Any = None               # [1, T, hidden] (set at prefill)
    encoder_length: int = 0
    last_token: Optional[int] = None         # previous emitted token id
    finished: bool = False


class TDTStateStore:
    """``request_id -> TDTRequestState``. Lives in the EngineCore process.

    NOTE (Task 3 finding): keying by ``request_id`` presumes the model's
    ``forward`` / ``compute_logits`` can observe the request id of each row it
    processes. vLLM V1 does not thread request ids into those calls, which is
    the central reason the in-engine port is a NO-GO (see the findings doc).
    """

    def __init__(self) -> None:
        self._states: dict[str, TDTRequestState] = {}

    def create(self, rid: str, frames, length: int) -> TDTRequestState:
        st = TDTRequestState(encoder_frames=frames, encoder_length=length)
        self._states[rid] = st
        return st

    def get(self, rid: str) -> TDTRequestState:
        return self._states[rid]

    def drop(self, rid: str) -> None:
        self._states.pop(rid, None)

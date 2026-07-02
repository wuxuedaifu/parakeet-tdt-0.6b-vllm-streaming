from __future__ import annotations

import abc
from dataclasses import dataclass
from typing import AsyncIterator


@dataclass
class TokenDelta:
    token_ids: list[int]      # new token ids since last yield (greedy: usually 1)
    durations: list[int]      # frame-spans for each token (TDT)
    finished: bool


@dataclass
class DecodeResult:
    token_ids: list[int]
    durations: list[int]


class DecodeBackend(abc.ABC):
    @abc.abstractmethod
    async def decode_stream(
        self, request_id: str, encoder_frames, encoder_lengths
    ) -> AsyncIterator[TokenDelta]:
        """Yield TokenDelta objects as the TDT decode emits tokens.

        encoder_frames: torch.FloatTensor [1, T, hidden]
        encoder_lengths: torch.LongTensor [1]
        """
        raise NotImplementedError

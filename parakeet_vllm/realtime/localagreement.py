from __future__ import annotations
from collections import deque

def _common_prefix(seqs: list[list[str]]) -> list[str]:
    if not seqs:
        return []
    out: list[str] = []
    for tup in zip(*seqs):
        first = tup[0]
        if all(x == first for x in tup):
            out.append(first)
        else:
            break
    return out

class LocalAgreement:
    """Commit only the word-prefix agreed across the last ``n`` hypotheses.

    Emitted (committed) text is append-only and never retracts.
    """
    def __init__(self, n: int = 2):
        if n < 2:
            raise ValueError("LocalAgreement n must be >= 2")
        self.n = n
        self._recent: deque[list[str]] = deque(maxlen=n)
        self._committed: list[str] = []

    @property
    def committed(self) -> list[str]:
        return list(self._committed)

    def commit(self, hypothesis_words: list[str]) -> list[str]:
        self._recent.append(list(hypothesis_words))
        if len(self._recent) < self.n:
            return []
        agreed = _common_prefix(list(self._recent))
        # never shrink: only extend committed if the agreed prefix is longer
        if len(agreed) <= len(self._committed):
            return []
        # guard: agreed must extend the existing committed prefix
        if agreed[: len(self._committed)] != self._committed:
            return []
        newly = agreed[len(self._committed):]
        self._committed = agreed
        return newly

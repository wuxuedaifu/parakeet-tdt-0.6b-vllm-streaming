"""Decode backends (Task 3/4 contract).

``PARAKEET_VLLM_BACKEND`` selects the decode backend:

- ``reference`` (default): :class:`ReferenceTDTBackend`, which reproduces
  ``ParakeetForTDT.generate()`` token-for-token. Selected by the Task 3 NO-GO.
- ``vllm``: :class:`VLLMDecodeBackend`, the in-engine spike. Constructing it
  raises :class:`VLLMInEngineNoGo` on vLLM 0.24.0 (see the findings doc); it is
  kept for reproducibility of the decision.

Note: ``make_backend`` was removed (Fix M1) — it was dead code called nowhere.
"""

from __future__ import annotations

from .backend import DecodeBackend, DecodeResult, TokenDelta

DEFAULT_BACKEND = "reference"

__all__ = [
    "DecodeBackend",
    "DecodeResult",
    "TokenDelta",
    "DEFAULT_BACKEND",
]

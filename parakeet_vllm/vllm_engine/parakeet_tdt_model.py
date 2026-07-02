"""Custom vLLM V1 model attempting to host the TDT greedy decode loop.

This module implements the artifact the Task 3 brief calls for and registers it
with ``ModelRegistry.register_model``. It is written to be *faithful*, not
merely a stub: it wires the real V1 model surface (``embed_input_ids``,
``forward(input_ids, positions, intermediate_tensors, inputs_embeds, **kw)``,
``compute_logits(hidden_states)``), declares the closest-matching interfaces
(``IsAttentionFree`` + ``HasInnerState`` + ``SupportsMultiModal``), and holds a
``TDTStateStore`` for the LSTM cache / frame pointer.

Doing so surfaces the concrete, version-pinned blocker empirically: the V1
model-execution contract never hands the model the *identity* of the request a
given decode-step row belongs to, so the per-request LSTM state + encoder frames
+ frame pointer in ``TDTStateStore`` cannot be indexed inside ``forward`` /
``compute_logits``. See
``docs/superpowers/findings/2026-06-30-vllm-inengine-decision.md``.
"""

from __future__ import annotations

from collections.abc import Iterable

import torch
import torch.nn as nn

from .state import TDTStateStore


class RequestIdentityUnavailable(RuntimeError):
    """Raised at the exact point the port cannot proceed.

    vLLM V1 calls ``forward(input_ids, positions, intermediate_tensors,
    inputs_embeds, **model_kwargs)`` and ``compute_logits(hidden_states)`` with
    no request id, no seq id, and no per-request key. ``_init_model_kwargs()``
    returns ``{}`` for non-pooling models and ``_extract_mm_kwargs()`` returns
    ``{}`` unless the model is ``is_multimodal_raw_input_only`` (and even then
    only batched, not request-keyed, tensors on scheduled-mm steps). The
    ``ForwardContext`` exposes only ``no_compile_layers`` / ``attn_metadata``
    (keyed by layer name, positional) / ``dp_metadata``. So there is no hook to
    map a hidden-state row back to its ``TDTRequestState``.
    """


class ParakeetTDTForVLLM(nn.Module):
    """Attempted in-engine TDT model.

    The transducer prediction network (stateful LSTM), the joint network, and
    the encoder projector are the real HF submodules; only the plumbing to
    vLLM's execution loop is novel. The state store carries the non-KV recurrent
    state vLLM does not manage.
    """

    # Closest-matching V1 interface flags. TDT has no attention over frames and
    # carries constant-per-token recurrent state, so IsAttentionFree/HasInnerState
    # are the natural declarations -- but see the blocker below.
    is_attention_free = True
    has_inner_state = True
    supports_multimodal = True

    def __init__(self, *, vllm_config, prefix: str = "") -> None:
        super().__init__()
        self.vllm_config = vllm_config
        self.config = vllm_config.model_config.hf_config
        self.state_store = TDTStateStore()

        # These are populated from the HF checkpoint in load_weights(); the
        # transducer decoder (LSTM), joint, and encoder projector are reused
        # verbatim -- the decode math is identical to ReferenceTDTBackend and
        # lives in parakeet_vllm/decode/tdt_step.py.
        self.decoder: nn.Module | None = None
        self.joint: nn.Module | None = None
        self.encoder_projector: nn.Module | None = None

    # -- V1 model surface -------------------------------------------------

    def embed_input_ids(self, input_ids: torch.Tensor, **kwargs) -> torch.Tensor:
        # For a transducer the "embedding" is the prediction-network step, which
        # requires the per-request LSTM cache -> same identity problem as forward.
        raise RequestIdentityUnavailable(
            "TDT prediction-network step needs the per-request LSTM cache; "
            "embed_input_ids receives only token ids, no request id."
        )

    def forward(
        self,
        input_ids: torch.Tensor | None = None,
        positions: torch.Tensor | None = None,
        intermediate_tensors=None,
        inputs_embeds: torch.Tensor | None = None,
        **model_kwargs,
    ) -> torch.Tensor:
        # To run the TDT step for each row we would need, per row:
        #   1. the request's TDTRequestState (LSTM cache, frame_ptr, frames)
        #   2. to select encoder_frames[frame_ptr] and run decoder+joint
        #   3. to write the updated cache/pointer back
        # (1) is impossible: none of forward's arguments identify the request.
        raise RequestIdentityUnavailable(
            "vLLM V1 forward() cannot map a hidden-state row to its request_id, "
            "so the per-request TDTRequestState (LSTM cache + frame pointer + "
            "encoder frames) cannot be indexed. This is the NO-GO blocker."
        )

    def compute_logits(self, hidden_states: torch.Tensor) -> torch.Tensor:
        # Even if forward produced joint outputs, the sampler consumes a single
        # width-vocab_size logit row and feeds back one token; the TDT duration
        # head (needed to advance the frame pointer) has nowhere to go, and again
        # there is no request id to store it against.
        raise RequestIdentityUnavailable(
            "compute_logits(hidden_states) has no request id and no channel for "
            "the per-step TDT duration output."
        )

    def load_weights(self, weights: Iterable[tuple[str, torch.Tensor]]) -> set[str]:
        # A real port would load decoder/joint/encoder_projector here. Left
        # unimplemented on purpose: the port is blocked upstream of weight
        # loading (see forward), so wiring a full loader would be dead code.
        raise NotImplementedError(
            "in-engine TDT port is a NO-GO; weights are not loaded (see findings)."
        )


def register() -> None:
    """Register the custom model architecture with vLLM."""
    from vllm import ModelRegistry

    ModelRegistry.register_model(
        "ParakeetTDTForVLLM",
        "parakeet_vllm.vllm_engine.parakeet_tdt_model:ParakeetTDTForVLLM",
    )


# Register at import so ``AsyncLLMEngine`` can find the architecture.
try:  # pragma: no cover - registration side effect
    register()
except Exception:  # noqa: BLE001 - vLLM may be absent in some envs
    pass

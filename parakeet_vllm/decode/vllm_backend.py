"""In-engine vLLM V1 TDT decode backend (NO-GO).

Task 3 evaluated hosting the Parakeet TDT greedy decode loop inside vLLM V1's
``AsyncLLMEngine`` as a custom registered model. The conclusion is **NO-GO**:
the V1 model-execution contract cannot express an external-frame-driven,
stateful-LSTM transducer step. See
``docs/superpowers/findings/2026-06-30-vllm-inengine-decision.md`` for the full
analysis and the exact blocking APIs.

This module keeps the intended interface (``VLLMDecodeBackend(DecodeBackend)``)
and the intended engine configuration so the decision is reproducible, but
``__init__`` raises :class:`VLLMInEngineNoGo` rather than silently degrading.
Production decode uses :class:`~parakeet_vllm.decode.reference_tdt.ReferenceTDTBackend`
selected via the ``PARAKEET_VLLM_BACKEND`` factory (see ``decode/__init__.py``).
"""

from __future__ import annotations

from typing import AsyncIterator

from .backend import DecodeBackend, TokenDelta


class VLLMInEngineNoGo(RuntimeError):
    """Raised because in-engine vLLM V1 TDT decode is not expressible.

    Blocking vLLM 0.24.0 (V1) behaviours, each verified against source:

    1. No per-request identity in the model. ``forward(input_ids, positions,
       intermediate_tensors, inputs_embeds, **model_kwargs)`` and
       ``compute_logits(hidden_states)`` receive no request/seq id.
       ``GPUModelRunner._init_model_kwargs()`` returns ``{}`` for non-pooling
       models; ``_extract_mm_kwargs()`` returns ``{}`` unless
       ``is_multimodal_raw_input_only`` (and then only batched tensors on
       scheduled-mm steps). ``ForwardContext`` exposes only ``no_compile_layers``
       / ``attn_metadata`` (layer-keyed, positional) / ``dp_metadata``. So a
       side dict keyed by ``request_id`` (``TDTStateStore``) cannot be indexed.

    2. No external, duration-driven frame pointer. Multimodal embeddings are
       scattered into ``inputs_embeds`` at prefill placeholder positions
       (``_gather_mm_embeddings`` / ``embed_input_ids``); the decode-step input
       is strictly the embedding of the previously sampled token. There is no
       hook to feed a single pointer-selected encoder frame per step, nor to
       advance that pointer by a model-emitted duration (incl. 0-duration
       re-emission on the same frame).

    3. No LSTM recurrent-state facility. The only non-KV per-request state in V1
       is the Mamba/SSM cache (``MambaSpec`` + ``MambaMixer``), addressed
       positionally via ``state_indices_tensor`` and updated by mamba kernels
       (causal_conv1d / chunk_scan). It cannot host an LSTM (hidden+cell+last
       output with the blank fast-path in-place update) nor store variable-length
       ``[T, hidden]`` encoder frames.

    4. Token/duration joint + stopping mismatch. The sampler selects one token
       from a width-``vocab_size`` logit row and feeds it back; the TDT duration
       head has no channel and stopping is frame-exhaustion driven (not EOS),
       which would inject a spurious trailing EOS token.
    """


class VLLMDecodeBackend(DecodeBackend):
    # Intended engine configuration, recorded for reproducibility (Auralis
    # XTTSv2.py:304-329 rationale): V1 engine, prefix caching off (recurrent
    # LSTM state is not prefix-shareable), tokenizer init skipped (we operate on
    # raw token ids), eager configurable (the TDT step needs python control flow
    # + a side state dict, incompatible with CUDA-graph capture).
    ENGINE_ARGS = dict(
        enable_prefix_caching=False,
        skip_tokenizer_init=True,
    )

    def __init__(
        self,
        model_id: str = "nvidia/parakeet-tdt-0.6b-v3",
        *,
        enforce_eager: bool = True,
    ) -> None:
        self.model_id = model_id
        self.enforce_eager = enforce_eager
        raise VLLMInEngineNoGo(
            "In-engine vLLM V1 TDT decode is a NO-GO in vLLM 0.24.0. "
            "Use ReferenceTDTBackend (PARAKEET_VLLM_BACKEND=reference). "
            "See docs/superpowers/findings/2026-06-30-vllm-inengine-decision.md."
        )

    async def decode_stream(
        self, request_id: str, encoder_frames, encoder_lengths
    ) -> AsyncIterator[TokenDelta]:  # pragma: no cover - unreachable (init raises)
        raise VLLMInEngineNoGo(
            "In-engine vLLM V1 TDT decode is a NO-GO; see findings doc."
        )
        yield  # make this an async generator

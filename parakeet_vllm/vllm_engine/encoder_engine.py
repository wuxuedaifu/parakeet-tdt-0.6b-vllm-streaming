"""Encoder-in-vLLM spike + backend (Task 3b, the NEW GATE).

Goal: host the Parakeet FastConformer **encoder** inside vLLM 0.24.0 V1 and
return **full, variable-length, per-frame** hidden states ``[T, 1024]`` (+ per-
request frame lengths) per request — the raw ``last_hidden_state``, NOT a pooled
vector and NOT the VLM projection — so the frames feed straight into the
already-parity-proven :class:`~parakeet_vllm.decode.reference_tdt.ReferenceTDTBackend`.

Outcome: **NO-GO for Phase 1** (vLLM encoder path deferred; fallback =
batched-PyTorch encoder, plan Task 7). See
``docs/superpowers/findings/2026-07-01-encoder-in-vllm-decision.md``.

Why NO-GO — the honest, source-verified summary (this is *not* an API wall like
Task 3's decode NO-GO; the capability exists, the standalone integration does
not):

  * The enabling half WORKS. vLLM 0.24.0 V1 can return per-position hidden states
    via tokwise ``AllPool`` (``config/pooler.py`` ``TokenPoolingType='ALL'``,
    task ``token_embed``): the active ``v1/worker/gpu_model_runner._pool`` calls
    ``model.pooler(hidden_states[:num_scheduled_tokens], ...)`` and ``AllPool``
    returns ``torch.split(hidden_states, num_scheduled_tokens_per_seq)`` — the
    full variable-length per-frame sequence. Direct AUDIO precedent:
    ``Qwen3ASRForcedAlignerForTokenClassification`` (multimodal audio tower +
    ``@default_pooling_type(tok_pooling_type='ALL')`` + per-frame outputs, whose
    processor even accepts pre-extracted ``input_audio_features``).

  * The blocker is the STANDALONE model. There is no registered vLLM
    architecture for ``ParakeetForTDT`` / ``parakeet_tdt``. vLLM's only Parakeet
    code (``models/parakeet.py`` ``ProjectedParakeet``) is a VLM sub-module that
    *applies a projection to ``llm_hidden_size``* (not the raw 1024-d frames) and
    needs VLM-only config fields (``llm_hidden_size``/``projection_hidden_size``)
    absent from the TDT checkpoint. Auto-resolution of this checkpoint in pooling
    mode falls back to the generic ``TransformersForCausalLM`` backend (verified:
    ``ModelConfig(..., runner='pooling')`` logs ``Resolved architecture:
    TransformersForCausalLM``, ``--convert embed``, ``Encoder-decoder model
    detected``), which treats it as a token causal-LM and never exposes the bare
    FastConformer ``last_hidden_state`` per frame.

  * A GO would require a full custom vLLM model integration (custom arch + a
    ModelConfig-satisfying config shim for a *bare encoder-only conformer* +
    a multimodal processor with exact subsampled placeholder accounting + an
    attention-free/identity backbone with no KV cache + a headless
    ``TokenPooler(AllPool, head=None)``), i.e. a new-model PR, not a spike — for
    marginal benefit over a batched PyTorch encoder, since the encoder is a
    single non-autoregressive forward with no KV cache and gains nothing from
    vLLM's paged-attention continuous-batching decode machinery.

This module therefore *faithfully attempts* the boot (mirroring Task 3's
approach of surfacing the wall in code, not merely arguing it): ``VLLMEncoder``
probes vLLM's architecture resolution for the checkpoint and raises
:class:`VLLMEncoderNoGo` at the exact point a native standalone Parakeet-encoder
pooling architecture is required and absent. The intended output contract
(``encode_async -> ([1,T,1024], [1])``) is defined so a future GO can drop in
behind the Task-7 ``encode()`` factory with zero downstream change.
"""

from __future__ import annotations

import torch

# Project default encoder backend selected by this gate. The Task-7
# ``parakeet_vllm/encoder.py::encode()`` factory reads ``PARAKEET_ENCODER_BACKEND``
# (env), defaulting to this value.
PARAKEET_ENCODER_BACKEND_DEFAULT = "torch"

MODEL_ID_DEFAULT = "nvidia/parakeet-tdt-0.6b-v3"

# Encoder shape facts (transformers 5.12.1, nvidia/parakeet-tdt-0.6b-v3).
ENCODER_HIDDEN_SIZE = 1024
SUBSAMPLING_FACTOR = 8


class VLLMEncoderNoGo(RuntimeError):
    """Raised at the exact point the encoder-in-vLLM port cannot proceed.

    vLLM 0.24.0 V1 has no standalone Parakeet-encoder pooling architecture; the
    checkpoint auto-resolves to the generic ``TransformersForCausalLM`` backend,
    which cannot emit the raw FastConformer per-frame ``last_hidden_state``. See
    the module docstring and the findings doc for the full evidence.
    """


class VLLMEncoder:
    """Attempted vLLM-hosted Parakeet encoder returning per-frame hidden states.

    Interface (value-compatible with ``model.encoder(...).last_hidden_state``):

        ``async def encode_async(self, input_features, attention_mask)``
            ``-> (frames: torch.FloatTensor[1, T, 1024], lengths: torch.LongTensor[1])``

    On this NO-GO, construction surfaces the concrete blocker by probing vLLM's
    architecture resolution and raising :class:`VLLMEncoderNoGo`.
    """

    def __init__(self, model_id: str = MODEL_ID_DEFAULT) -> None:
        self.model_id = model_id
        self._assert_native_encoder_arch_or_nogo()
        # Unreachable on NO-GO. A GO implementation would boot the AsyncLLMEngine
        # here (runner="pooling", pooling_task="token_embed", a headless
        # TokenPooler(AllPool), a custom registered Parakeet-encoder arch, and a
        # config shim), storing the engine handle for encode_async.
        raise VLLMEncoderNoGo(  # pragma: no cover - defensive; probe raises first
            "unreachable: native Parakeet-encoder pooling arch probe should have "
            "already raised."
        )

    def _assert_native_encoder_arch_or_nogo(self) -> None:
        """Probe vLLM's arch registry for this checkpoint; raise on NO-GO.

        A GO requires a *native standalone Parakeet-encoder pooling* architecture
        registered with vLLM. In vLLM 0.24.0 V1 none exists: the checkpoint's
        architecture (``ParakeetForTDT``) is not in
        ``ModelRegistry.get_supported_archs()`` and there is no ``Parakeet*`` arch
        at all, so the checkpoint auto-resolves to the generic
        ``TransformersForCausalLM`` backend (a token causal-LM), which cannot emit
        the raw FastConformer per-frame ``last_hidden_state``.
        """
        try:
            from transformers import AutoConfig
            from vllm import ModelRegistry
        except Exception as exc:  # noqa: BLE001 - vLLM/transformers absent
            raise VLLMEncoderNoGo(f"vLLM is not importable: {exc!r}") from exc

        try:
            hf_cfg = AutoConfig.from_pretrained(self.model_id, trust_remote_code=True)
            checkpoint_archs = list(getattr(hf_cfg, "architectures", None) or [])
        except Exception as exc:  # noqa: BLE001 - config load failed
            raise VLLMEncoderNoGo(
                f"cannot load HF config for {self.model_id!r}: {exc!r}"
            ) from exc

        supported = set(ModelRegistry.get_supported_archs())
        # A native GO arch would be a registered Parakeet-*encoder* architecture
        # (not the VLM-only ProjectedParakeet sub-module, which is unregistered).
        native_parakeet_archs = sorted(
            a for a in supported if "Parakeet" in a
        )
        registered = [a for a in checkpoint_archs if a in supported]

        if not registered and not native_parakeet_archs:
            raise VLLMEncoderNoGo(
                "NO-GO: vLLM 0.24.0 V1 has no standalone Parakeet-encoder pooling "
                f"architecture. Checkpoint {self.model_id!r} declares "
                f"architectures={checkpoint_archs!r}, none registered with vLLM "
                "(ModelRegistry has no 'Parakeet*' arch), so it auto-resolves to the "
                "generic TransformersForCausalLM backend, which treats the model as "
                "a token causal-LM and cannot emit the raw FastConformer per-frame "
                "last_hidden_state. Per-frame pooling itself IS supported (tokwise "
                "AllPool + multimodal, cf. Qwen3ASRForcedAligner); the blocker is "
                "the missing standalone encoder arch/config. Hosting it would "
                "require a full custom vLLM model integration (custom arch + config "
                "shim + multimodal processor + attention-free identity backbone + "
                "headless TokenPooler). Fallback: PARAKEET_ENCODER_BACKEND=torch "
                "(batched-PyTorch encoder, plan Task 7). See the findings doc."
            )

    async def encode_async(
        self,
        input_features: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> tuple[torch.Tensor, torch.Tensor]:
        """Return ``(frames[1, T, 1024], lengths[1])`` — raw encoder frames.

        Unreachable on this NO-GO (construction raises first). A GO
        implementation would: submit ``{"prompt_token_ids": [<audio placeholder>
        * T], "multi_modal_data": {"audio": (input_features, attention_mask)}}``
        to the pooling engine with ``pooling_task="token_embed"`` (CPU-pinned
        multimodal tensors for V1 IPC), receive the per-frame pooled output
        ``[T, 1024]`` from a headless ``TokenPooler(AllPool)``, and reshape to
        ``[1, T, 1024]`` with ``lengths = tensor([T])``. ``T`` equals
        ``subsample(mel_len)`` and must match
        ``model.encoder(...).last_hidden_state.shape[1]`` exactly.
        """
        raise VLLMEncoderNoGo(  # pragma: no cover - construction raises first
            "encode_async is unreachable: encoder-in-vLLM is a NO-GO for Phase 1 "
            "(PARAKEET_ENCODER_BACKEND=torch). See the Task-3b findings doc."
        )

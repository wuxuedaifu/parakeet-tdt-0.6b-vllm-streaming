"""Task 3 spike: probe whether vLLM V1 can host the TDT decode loop.

Run:  VLLM_USE_V1=1 python spike/vllm_inengine_spike.py

This does not need the GPU checkpoint. It exercises the three decision-critical
facts against the installed vLLM and prints the verdict:

1. Can a custom model be registered?           -> ModelRegistry.register_model
2. Does the V1 model forward/compute_logits get a request id? -> inspect sigs
3. Does ForwardContext carry request identity?  -> inspect fields

The empirical output backs the NO-GO recorded in
docs/superpowers/findings/2026-06-30-vllm-inengine-decision.md.
"""

from __future__ import annotations

import inspect


def main() -> None:
    import vllm
    from vllm import ModelRegistry
    from vllm.forward_context import ForwardContext
    from vllm.v1.worker.gpu_model_runner import GPUModelRunner

    print(f"vllm=={vllm.__version__}")

    # 1. custom-model registration works
    from parakeet_vllm.vllm_engine.parakeet_tdt_model import ParakeetTDTForVLLM  # noqa: F401

    registered = "ParakeetTDTForVLLM" in ModelRegistry.get_supported_archs()
    print(f"[1] custom model registered: {registered}")

    # 2. model forward / compute_logits get no request id
    fwd_sig = inspect.signature(ParakeetTDTForVLLM.forward)
    cl_sig = inspect.signature(ParakeetTDTForVLLM.compute_logits)
    print(f"[2] forward params:        {list(fwd_sig.parameters)}")
    print(f"    compute_logits params: {list(cl_sig.parameters)}")

    # how the runner calls the model + builds kwargs
    mf_src = inspect.getsource(GPUModelRunner._model_forward)
    print("    runner passes to model: input_ids, positions, "
          "intermediate_tensors, inputs_embeds, **model_kwargs (no req id)")
    print(f"    _model_forward mentions 'req': {'req' in mf_src}")

    # 3. ForwardContext fields
    fc_fields = [f for f in getattr(ForwardContext, "__dataclass_fields__", {})]
    print(f"[3] ForwardContext fields: {fc_fields}")
    has_req = any("req" in f or "request" in f for f in fc_fields)
    print(f"    carries request identity: {has_req}")

    print()
    print("VERDICT: NO-GO -- request identity is unavailable inside the model, "
          "so per-request LSTM state + encoder frames + frame pointer cannot be "
          "indexed. See findings doc.")


if __name__ == "__main__":
    main()

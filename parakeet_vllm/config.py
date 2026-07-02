from __future__ import annotations
import logging
import os

logger = logging.getLogger("parakeet_vllm")

MODEL_ID = os.getenv("PARAKEET_MODEL_ID", "nvidia/parakeet-tdt-0.6b-v3")

# Default is "reference" because both vLLM integration paths (in-engine TDT
# decode and encoder-in-vLLM) were proven NO-GO in Task 3.  Set
# PARAKEET_VLLM_BACKEND=vllm explicitly to opt-in and receive the VLLMInEngineNoGo
# error (intentional — keeps the rejected path reproducible).
BACKEND = os.getenv("PARAKEET_VLLM_BACKEND", "reference")

# Encoder backend: "torch" (direct HF forward) or "vllm" (encoder-in-vLLM, NO-GO).
ENCODER_BACKEND = os.getenv("PARAKEET_ENCODER_BACKEND", "torch")

DEVICE = os.getenv("PARAKEET_DEVICE", "cuda")

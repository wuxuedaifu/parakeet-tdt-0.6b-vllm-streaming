# Knowledge Transfer — Parakeet TDT Streaming ASR Server

A self-contained engineering handoff for anyone picking up this project. Read
this before making changes. It covers what the system is, how it's built, the
non-obvious decisions and *why* they were made, the gotchas that will bite you,
and where to extend.

---

## 1. What this is

A speech-to-text server for NVIDIA `nvidia/parakeet-tdt-0.6b-v3` (a FastConformer
encoder + **Token-and-Duration Transducer (TDT)** decoder), running on
**PyTorch / HuggingFace `transformers`**. Two OpenAI-compatible surfaces:

- **`POST /v1/audio/transcriptions`** — batch/offline (OpenAI Audio API shape).
- **`GET /v1/realtime`** — live incremental transcription over WebSocket
  (OpenAI Realtime transcription event schema).

There are **two independent services** in the repo:
- `parakeet_vllm/` — the PyTorch streaming service (this doc's subject).
- `parakeet_service/` — the **legacy ONNX** FastAPI service. Untouched, still
  works, not part of the new work. Don't confuse the two.

---

## 2. The single most important thing to understand: the vLLM decision

The project's original goal was to run Parakeet **inside vLLM's engine**. That
goal was **abandoned after proving it infeasible** — twice, with source-verified
evidence. If you're tempted to "just put it on vLLM," read this first.

**Why TDT does not fit vLLM (0.24.0):**
- vLLM is built for **autoregressive transformer decoders** with a KV cache and
  a single-token sampler.
- Parakeet's heavy compute is a **non-autoregressive FastConformer encoder**
  (processes the whole utterance in one pass — not a decoder).
- Parakeet's autoregressive part is a **tiny LSTM prediction network** driven
  frame-synchronously with a **duration head** — not a KV-cached transformer.
- Concretely, vLLM V1's model contract has **no per-request identity** in
  `forward`/`compute_logits`, **no external duration-driven frame pointer**, and
  **no LSTM recurrent-state facility** (only the Mamba/SSM fixed-shape cache),
  and the **token+duration joint** does not fit the single-token sampler.

**The encoder-in-vLLM fallback** (run only the Conformer as a vLLM pooling model)
is technically possible — vLLM's tokwise `AllPool` can emit per-frame states —
but vLLM ships **no standalone Parakeet-encoder architecture**, so a GO would
need a full custom model integration (arch registration + config shim + MM
processor + attention-free backbone + headless pooler) for **marginal** benefit:
the encoder is a single non-AR forward that gains nothing from paged-attention
continuous-decode machinery.

**Conclusion (shipped):** run a **PyTorch decode backend** (`ReferenceTDTBackend`)
with our **own batched encoder** and a **two-phase async scheduler**. Throughput
comes from **cross-request encoder batching** + parallel decode, not vLLM. The
inert vLLM probe code remains in-tree (`parakeet_vllm/vllm_engine/`,
`decode/vllm_backend.py`) as reproducible evidence; both raise a `NoGo` unless
explicitly opted into via env, and default config never touches them.

**Takeaway:** the repo name says "vllm-streaming" for continuity with the
original goal. The runtime does **not** use vLLM. Don't reintroduce it without
re-checking these blockers against a newer vLLM release.

---

## 3. Architecture

### Offline (batch) path
```
upload → decode/resample 16k mono → AutoProcessor (log-mel, 128 bins)
   ├─ (long file) Silero-VAD chunk  → cross-chunk BATCHED encode  (throughput)
   ▼
FastConformer encoder (PyTorch)  → frames [B, T, 1024], lengths [B]
   ▼
ReferenceTDTBackend.decode_stream  → greedy TDT (LSTM pred-net + joint + duration)
   ▼
processor.decode → text (+ word timestamps from durations)
```
Orchestrated by `ASREngine.transcribe` → `TwoPhaseScheduler` (phase 1 = encode,
phase 2 = decode) with an `asyncio.Semaphore(PARAKEET_MAX_CONCURRENCY)`.

### Live (streaming) path — "Approach A"
```
WS /v1/realtime → LiveSession (per connection)
  input_audio_buffer.append → base64 PCM16 → 16k mono → ring buffer → Silero VAD
  every ~0.5s while speaking:
     re-decode the GROWING segment from its start  (decode_window)
     LocalAgreement-2 → emit only newly-agreed words as transcription.delta
  VAD silence / manual commit / max-segment cap:
     re-decode full segment → transcription.completed (+ words); reset segment
```
The offline checkpoint is full-context (attends the whole utterance), so live
streaming **re-decodes the growing segment each hop** and stabilizes partials
with **LocalAgreement-n** (commit only the word-prefix agreed across the last n
hypotheses → deltas are append-only, never retract). This is the standard way to
stream an offline ASR model (cf. Whisper-Streaming).

### The decode is parity-locked to `generate()`
`ReferenceTDTBackend` is a hand-port of the transformers greedy TDT loop
(`generation_parakeet.py`). It drives `ParakeetForTDT.forward` step-by-step
(`decoder_input_ids` + `ParakeetRNNTDecoderCache` + precomputed `encoder_outputs`)
and produces **token-for-token identical** output to
`AutoModelForTDT.generate()`. This equality is the project's **test oracle** —
if you change decode, the parity tests must still pass.

---

## 4. Repository map (`parakeet_vllm/`)

| File | Responsibility |
|---|---|
| `config.py` | Env config: `MODEL_ID`, `BACKEND` (default `reference`), `ENCODER_BACKEND` (default `torch`), `DEVICE`. |
| `model_loader.py` | Singletons: `get_processor()`, `get_reference_model()` (**`AutoModelForTDT`** — see gotcha #4), `get_decode_backend()` factory. |
| `audio.py` | `decode_to_16k_mono(bytes) -> np.float32` (soundfile + librosa). |
| `features.py` | `extract_features(list[np.ndarray])` → `ParakeetProcessor` batch (128-dim log-mel + attention_mask). |
| `encoder.py` | `encode(input_features, attention_mask) -> (frames[B,T,1024], lengths[B])`; factory over `{torch, vllm}` (vllm = NoGo). |
| `decode/backend.py` | `DecodeBackend` ABC (`async decode_stream(request_id, frames, lengths)`), `TokenDelta`, `DecodeResult`. |
| `decode/reference_tdt.py` | **The decode.** Greedy TDT, parity with `generate()`. No-grad safe. Per-frame `max_symbols` cap. |
| `decode/tdt_step.py` | Shared joint-split (token vs duration) math. |
| `decode/vllm_backend.py` | **Inert.** `VLLMDecodeBackend` raises `NoGo` (records the decode-in-vLLM finding). |
| `vllm_engine/` | **Inert.** Encoder-in-vLLM probe + `NoGo`; reproducible evidence of the second finding. |
| `scheduling/two_phase_scheduler.py` | Phase-1 encode → phase-2 decode; semaphore-bounded; race-free cleanup. |
| `engine/asr_engine.py` | `ASREngine.transcribe` / `transcribe_stream`; silence short-circuit; VAD chunking + batched multi-chunk encode. |
| `streaming/chunker.py` | `split_on_silence` (Silero) + hard-split of over-long chunks. |
| `streaming/file_stream.py` | SSE (`partials_to_sse`) + `build_word_timestamps`. |
| `api/routes.py` | `create_app()`, `POST /v1/audio/transcriptions`, upload cap, timestamps, SSE. Wires the realtime WS. |
| `server.py` | uvicorn entrypoint (`PARAKEET_HOST`/`PARAKEET_PORT`, default `0.0.0.0:5092`). |
| `benchmark.py` | Offline throughput benchmark. |
| **`realtime/protocol.py`** | OpenAI Realtime event parse/build; unknown → `ProtocolError`. |
| **`realtime/localagreement.py`** | `LocalAgreement(n)` — append-only commit policy. Pure, no torch. |
| **`realtime/vad.py`** | `StreamingVAD` — Silero `VADIterator` wrapper (512-sample windows). |
| **`realtime/decoder_stream.py`** | `async decode_window(audio, rid)` — one growing-window decode over the shipped pipeline. |
| **`realtime/session.py`** | `LiveSession` — per-connection state machine (ring buffer, VAD, hop loop, endpointing). Headless/testable. |
| **`realtime/ws_app.py`** | `add_realtime_ws(app)` — the `/v1/realtime` WebSocket bridge. |

`scripts/bench_live_concurrency.py` measures live-session serialization.

---

## 5. Environment & workflow

- **Venv:** `.venv-vllm` (gitignored). `uv venv .venv-vllm --python 3.12` +
  `uv pip install -r requirements-vllm.txt`. Always activate it in commands.
- **Key deps:** `transformers>=5.12` (provides `AutoModelForTDT`), `torch`,
  `silero-vad`, `onnxruntime` (silero onnx mode), `fastapi`, `uvicorn`,
  `librosa`, `soundfile`, `datasets` (tests). Model downloads (~2.4 GB) on first
  run and caches under `~/.cache/huggingface`.
- **Run:** `python parakeet_vllm/server.py`.
- **Tests:**
  - CPU-only fast suite: `python -m pytest tests/parakeet_vllm/ -m "not gpu" -q`
  - Full (GPU parity + e2e): `VLLM_USE_V1=1 python -m pytest tests/parakeet_vllm/ -q`
  - GPU tests carry `@pytest.mark.gpu` and auto-skip on machines without CUDA
    (see `conftest.py`).

**Testing philosophy:** the oracle is `AutoModelForTDT.generate()`. Correctness
tests assert **exact** token/text equality against it
(`test_reference_tdt_parity`, `test_asr_engine_parity`, `test_decode_window`),
and the live e2e (`test_realtime_ws`) asserts the WebSocket path **converges to
the offline transcript**. Non-GPU tests mock the decode to check orchestration
and OpenAI event-sequence conformance.

---

## 6. Performance & concurrency (measured, honest)

On **1× NVIDIA H200 NVL** (FP32):

| Workload | Throughput |
|---|---|
| Single 60 s file | 38.6× realtime |
| c=1 (10 s) | 107.7× realtime |
| c=8 | 161.7× realtime |
| c=16 | **164.8× realtime** |

- Offline throughput scales with concurrency because the **encoder is batched
  across requests**. N concurrent transcriptions == sequential results (verified).
- **Live path is NOT concurrency-optimized.** Measured 8-way concurrent
  live-decode serialization factor ≈ **7.2×**: `decode_window` runs a synchronous
  decode on the asyncio event loop and bypasses the scheduler, so concurrent
  `/v1/realtime` sessions queue. One live session is comfortably real-time; many
  are not. **This is the #1 roadmap item** (see §8).

---

## 7. Gotchas (things that already bit us — don't relearn the hard way)

1. **Feature dim is 128, not 80.** The plan/docs originally said 80 (copied from
   Whisper); the real `ParakeetFeatureExtractor` uses **128** mel bins. Never
   hardcode it — it flows through `extract_features`. Encoder hidden size is 1024.
2. **`inference_mode` vs autograd.** `encode()` is `@torch.inference_mode()`;
   its output tensors carry the inference-mode flag. `ReferenceTDTBackend` runs
   its projector/decode under `torch.inference_mode()` so it's safe — but if you
   add code that touches encoder frames **outside** a no-grad context, autograd
   will try to save inference-mode tensors and raise. Keep decode no-grad.
3. **onnx-Silero rejects synthetic tones.** Silero VAD in onnx mode does **not**
   flag a pure sine tone as speech. Tests that need "speech" must use a **real**
   clip (librispeech dummy). A prior sine-tone fixture silently passed only when
   `onnxruntime` was absent (torch fallback); installing onnxruntime exposed it.
   Use real audio for any VAD/chunking test.
4. **Load the model via `AutoModelForTDT`.** Not `AutoModelForCausalLM` (returns
   a wrong head that the decode backend can't drive). `get_reference_model()` has
   a regression test asserting `type(m).__name__ == "ParakeetForTDT"`.
5. **vLLM V1 prefix caching would corrupt output** for this kind of prompt — but
   this is moot now since vLLM isn't in the runtime path. Documented for anyone
   re-attempting the vLLM route.
6. **LocalAgreement operates on WORDS, not raw TDT tokens.** Token alignment
   shifts hop-to-hop; word-level agreement is stable. Deltas must stay
   append-only — the tests enforce this.
7. **Live session lifecycle is race-sensitive.** `LiveSession` uses an
   `asyncio.Lock` around decode and a `_CLOSE` sentinel so `close()` (on client
   disconnect) never loses in-flight `committed`/`completed` events. Don't add an
   `await` inside `decode_stream` or reintroduce timeout-based `events()`
   termination — either would break the race-free guarantee.
8. **Realtime sample rate.** Input defaults to **24 kHz** (OpenAI default);
   clients declare `audio.input.format.rate` in `session.update` to override.
   Internally everything runs at the model's 16 kHz (resampled).
9. **GPU tests need the HF librispeech dummy dataset** (network on first run,
   then cached). CI must have network or a warm cache.

---

## 8. Roadmap / known limitations

Priority order:

1. **Live-session concurrency** (biggest): route `decode_window` through the
   shared scheduler semaphore and offload the synchronous decode to a worker
   thread (`asyncio.to_thread`/executor) so concurrent `/v1/realtime` sessions
   stop serializing. Re-run `scripts/bench_live_concurrency.py` to confirm the
   factor drops toward ~1.
2. **Cache-aware streaming checkpoint** (latency): Approach A re-decodes the
   growing segment each hop (O(n²)/segment, bounded by VAD + max-segment cap). A
   true streaming FastConformer would cut latency/compute — but it's a different
   (English-only) NeMo model, so it leaves the transformers stack.
3. Streaming SSE path has no error-event wrapping; a mid-decode exception aborts
   the connection with no terminal `[DONE]`.
4. `include: logprobs` unsupported (greedy TDT has no cheap logprobs).
5. Multi-value `timestamp_granularities` beyond `word`; full g711 handling.
6. Upload cap bounds heap RAM but Starlette still spools the full multipart body
   to disk before rejection (a `Content-Length` pre-check would reject earlier).

---

## 9. How to extend

- **New decode backend:** implement `DecodeBackend` (`decode/backend.py`) —
  `async decode_stream(request_id, encoder_frames[1,T,1024], encoder_lengths[1])`
  yielding `TokenDelta(token_ids, durations, finished)` — and wire it into
  `get_decode_backend()` behind a `PARAKEET_VLLM_BACKEND` value. Everything
  downstream (engine, scheduler, streaming, WS) is backend-agnostic.
- **New encoder backend:** same pattern behind `PARAKEET_ENCODER_BACKEND` in
  `encoder.py::encode`.
- **Tune streaming:** hop cadence, LocalAgreement `n`, and max-segment cap are
  `LiveSession.__init__` args (defaults 500 ms / 2 / 25 s).
- **Golden rule:** any change to the decode/encode path must keep the
  `generate()`-parity GPU tests green. Run them before you commit.

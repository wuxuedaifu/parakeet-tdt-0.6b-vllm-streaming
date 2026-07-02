# Parakeet TDT 0.6B v3 — Streaming ASR Server

[![Python 3.12](https://img.shields.io/badge/python-3.12-blue.svg)](https://www.python.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A high-throughput, **streaming** speech-to-text server for NVIDIA's
[`parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3)
(FastConformer encoder + Token-and-Duration Transducer decoder), running on
**PyTorch / HuggingFace `transformers`**. It exposes two OpenAI-compatible
surfaces:

- **`POST /v1/audio/transcriptions`** — batch/offline transcription (OpenAI
  Audio API compatible), with VAD chunking, SSE output-streaming, and word
  timestamps.
- **`GET /v1/realtime`** — **live** incremental transcription over WebSocket,
  compatible with the **OpenAI Realtime transcription** event schema.

The greedy TDT decode is **token-for-token identical** to
`transformers`' `AutoModelForTDT.generate()` (parity-tested), so accuracy
matches the reference model exactly. Multilingual: 25 languages with automatic
language detection.

---

## Why this exists (and the vLLM story)

The project set out to run Parakeet TDT **inside vLLM's engine** for streaming +
throughput. After building it, two feasibility spikes proved that
**vLLM 0.24.0 cannot host this model's decode**: a Token-and-Duration Transducer
is not a KV-cached transformer decoder (no per-request identity in the model
forward, no external duration-driven frame pointer, no LSTM recurrent-state
facility, and the token+duration joint doesn't fit the single-token sampler).
Running only the encoder inside vLLM was possible but required a full custom
pooling-model integration for marginal benefit.

**So this server runs a PyTorch decode backend** (`ReferenceTDTBackend`,
parity-proven against `generate()`) with our **own batched encoder** and a
**two-phase async scheduler**. vLLM's paged-attention/continuous-decode machinery
simply doesn't map onto a transducer ASR model — the throughput win here comes
from cross-request **encoder batching** + parallel decode, not from vLLM. The
name is kept for continuity with the original goal; the vLLM integration code
remains in-tree (inert by default) as reproducible evidence of the finding.

---

## Features

- ✅ **Offline REST** — OpenAI `POST /v1/audio/transcriptions` (multipart upload),
  with `response_format`, optional `stream=true` (SSE), and word-level
  `timestamp_granularities[]`.
- ✅ **Live WebSocket** — OpenAI Realtime transcription schema at `/v1/realtime`:
  `input_audio_buffer.append/commit/clear` in, `speech_started/stopped`,
  `…transcription.delta` (stable partials) and `…transcription.completed` out.
- ✅ **Stable partials** — LocalAgreement-2 makes streamed partials append-only
  (they never retract).
- ✅ **Endpointing** — Silero VAD (`server_vad`) or manual commit.
- ✅ **Long files** — Silero-VAD chunking with cross-chunk batched encoding.
- ✅ **Accuracy parity** — decode matches `AutoModelForTDT.generate()` exactly.
- ✅ **Multilingual** — 25 languages, automatic language detection.

---

## Performance & Concurrency

Measured on **1× NVIDIA H200 NVL** (`nvidia/parakeet-tdt-0.6b-v3`, FP32,
`transformers` 5.12 / PyTorch). Offline batch path
(`ASREngine` → two-phase scheduler → batched encoder):

| Workload | RTF | Throughput |
|---|---:|---:|
| Single 60 s file | 0.026 | **38.6× realtime** |
| 1× 10 s clip (c=1) | 0.009 | **107.7× realtime** |
| 8× 10 s concurrent (c=8) | 0.006 | **161.7× realtime** |
| 16× 10 s concurrent (c=16) | 0.006 | **164.8× realtime** |

- **Correctness under concurrency:** N concurrent transcriptions produce
  byte-identical results to running them sequentially (verified in tests).
  Throughput scales with concurrency because the encoder forward is
  cross-request batched.

**Live streaming honesty note:** the live `/v1/realtime` decode path is **not yet
concurrency-optimized**. A measured 8-way concurrent live-decode test shows a
**serialization factor of ~7.2×** — concurrent live sessions currently serialize
because the decode loop is synchronous and runs on the asyncio event loop
(bypassing the batched offline scheduler). A single live session is comfortably
real-time; many simultaneous live sessions will queue. Routing the live decode
through the shared scheduler + a worker thread is the top item on the roadmap
below.

Reproduce: `python parakeet_vllm/benchmark.py` (offline) and
`python scripts/bench_live_concurrency.py` (live serialization factor).

---

## Requirements

- Linux, **NVIDIA GPU + CUDA** (reference: H200 / RTX 3090-class or better)
- Python 3.12
- `transformers >= 5.12` (provides `AutoModelForTDT`), PyTorch, `silero-vad`,
  FastAPI/uvicorn, librosa, soundfile, onnxruntime — pinned in
  [`requirements-vllm.txt`](requirements-vllm.txt).

## Installation

```bash
git clone https://github.com/wuxuedaifu/parakeet-tdt-0.6b-vllm-streaming.git
cd parakeet-tdt-0.6b-vllm-streaming

# with uv (recommended)
uv venv .venv-vllm --python 3.12
. .venv-vllm/bin/activate
uv pip install -r requirements-vllm.txt
```

The model (~2.4 GB) is downloaded from HuggingFace on first run and cached.

## Running the server

```bash
. .venv-vllm/bin/activate
python parakeet_vllm/server.py      # serves on 0.0.0.0:5092
```

### Configuration (environment variables)

| Var | Default | Purpose |
|---|---|---|
| `PARAKEET_HOST` / `PARAKEET_PORT` | `0.0.0.0` / `5092` | bind address |
| `PARAKEET_MODEL_ID` | `nvidia/parakeet-tdt-0.6b-v3` | HF model id |
| `PARAKEET_DEVICE` | `cuda` | torch device |
| `PARAKEET_MAX_CONCURRENCY` | `8` | max parallel decodes (offline scheduler) |
| `PARAKEET_MAX_UPLOAD_MB` | `100` | REST upload size cap (413 over-cap) |
| `PARAKEET_MAX_CHUNK_S` | `30` | long-file VAD chunk length |
| `PARAKEET_VLLM_BACKEND` | `reference` | decode backend (`reference`; `vllm` = NO-GO probe) |
| `PARAKEET_ENCODER_BACKEND` | `torch` | encoder backend (`torch`; `vllm` = NO-GO probe) |

## Usage

### Offline transcription (OpenAI-compatible)

```bash
curl -s http://localhost:5092/v1/audio/transcriptions \
  -F "file=@audio.wav" \
  -F "model=parakeet-tdt-0.6b-v3"
# {"text": "..."}
```

With the OpenAI Python SDK:

```python
from openai import OpenAI
client = OpenAI(base_url="http://localhost:5092/v1", api_key="not-needed")
with open("audio.wav", "rb") as f:
    print(client.audio.transcriptions.create(model="parakeet-tdt-0.6b-v3", file=f).text)
```

Word timestamps: add `-F "timestamp_granularities[]=word"`. Streamed output:
add `-F "stream=true"` (Server-Sent Events).

### Live streaming (OpenAI Realtime transcription)

Connect a WebSocket to `ws://localhost:5092/v1/realtime` and speak the Realtime
transcription protocol:

```
→ session.update            {"session": {"turn_detection": {...} | null,
                                          "audio": {"input": {"format": {"rate": 16000}}}}}
→ input_audio_buffer.append {"audio": "<base64 PCM16>"}
   ... (server_vad auto-commits on silence, or send:)
→ input_audio_buffer.commit

← session.created / session.updated
← input_audio_buffer.speech_started / speech_stopped     (server_vad mode)
← conversation.item.input_audio_transcription.delta      {"delta": "<new words>"}
← input_audio_buffer.committed
← conversation.item.input_audio_transcription.completed  {"transcript": "...", "words": [...]}
```

Notes: input default is **24 kHz** PCM16 (OpenAI default); declare
`audio.input.format.rate` in `session.update` for other rates (audio is
resampled to the model's 16 kHz internally). `logprobs` is not supported for
greedy TDT.

---

## Architecture

```
audio ─▶ decode/resample 16k ─▶ AutoProcessor (log-mel)
                                     │
        (offline) VAD chunk ────────┤
                                     ▼
        FastConformer encoder (batched, PyTorch)  ── frames [T, 1024]
                                     ▼
        ReferenceTDTBackend  ── greedy TDT decode (LSTM pred-net + joint +
                                duration head); == generate() token-for-token
                                     ▼
        text (+ word timestamps)
```

- **Offline:** `ASREngine.transcribe` → two-phase scheduler (`asyncio.Semaphore`)
  → batched encoder → decode. Throughput comes from batching the encoder across
  concurrent requests.
- **Live:** `LiveSession` accumulates audio, runs Silero VAD, and every ~0.5 s
  re-decodes the growing segment (Approach A); LocalAgreement-2 emits stable
  append-only partials; VAD or manual `commit` finalizes each segment.

Package layout: `parakeet_vllm/{model_loader, features, encoder, decode/,
engine/, scheduling/, streaming/, api/, realtime/}`.

---

## Testing

```bash
. .venv-vllm/bin/activate
python -m pytest tests/parakeet_vllm/ -m "not gpu" -q     # fast, CPU-only
VLLM_USE_V1=1 python -m pytest tests/parakeet_vllm/ -q    # includes GPU parity + e2e
```

GPU tests assert token/text parity with `AutoModelForTDT.generate()` and that
the live WebSocket path converges to the offline transcript.

---

## Roadmap / Known limitations

- **Live-session concurrency** (top priority): route the live decode through the
  shared scheduler and offload the synchronous decode to a worker thread so
  concurrent `/v1/realtime` sessions stop serializing on the event loop
  (currently ~7.2× serialization at 8 sessions).
- Streaming uses the **offline full-context checkpoint** re-decoded per hop
  (O(n²)/segment, bounded by VAD + a max-segment cap). A true cache-aware
  streaming checkpoint would lower latency/compute — future work.
- `include: logprobs` and full g711 handling are not implemented.
- Multi-value `timestamp_granularities` beyond `word` is not implemented.

---

## Acknowledgments

- NVIDIA NeMo team for [`parakeet-tdt-0.6b-v3`](https://huggingface.co/nvidia/parakeet-tdt-0.6b-v3).
- HuggingFace `transformers` Parakeet integration (`AutoModelForTDT`).
- [Silero VAD](https://github.com/snakers4/silero-vad) for endpointing/chunking.

## License

MIT

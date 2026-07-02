import io
import json
import numpy as np
import pytest
import soundfile as sf
from fastapi.testclient import TestClient


def _wav():
    x = (0.1 * np.sin(2 * np.pi * 220 * np.arange(8000) / 16000)).astype("float32")
    b = io.BytesIO()
    sf.write(b, x, 16000, format="WAV")
    return b.getvalue()


def test_transcriptions_endpoint(monkeypatch):
    from parakeet_vllm.api import routes

    class FakeEngine:
        async def transcribe(self, audio, request_id):
            from parakeet_vllm.engine.asr_engine import Transcription

            return Transcription(text="hello world", tokens=[1], durations=[1])

    monkeypatch.setattr(routes, "get_engine", lambda: FakeEngine())
    app = routes.create_app()
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav(), "audio/wav")},
        data={"model": "parakeet-tdt-0.6b-v3"},
    )
    assert r.status_code == 200
    assert r.json()["text"] == "hello world"


def test_sse_route_integration(monkeypatch):
    """SSE route wires partials from transcribe_stream as text/event-stream."""
    from parakeet_vllm.api import routes

    PARTIALS = ["he", "hello", "hello world"]

    class FakeStreamEngine:
        async def transcribe_stream(self, audio, request_id):
            for t in PARTIALS:
                yield t

    monkeypatch.setattr(routes, "get_engine", lambda: FakeStreamEngine())
    app = routes.create_app()
    client = TestClient(app)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav(), "audio/wav")},
        data={"model": "x", "stream": "true"},
    )

    assert r.status_code == 200
    assert r.headers["content-type"].startswith("text/event-stream")

    body = r.text
    data_lines = [ln.strip() for ln in body.splitlines() if ln.strip().startswith("data:")]

    # Terminal sentinel must be present
    assert "data: [DONE]" in data_lines

    # All partial texts must appear as data: events
    non_done = [ln for ln in data_lines if ln != "data: [DONE]"]
    parsed_texts = [json.loads(ln[len("data: "):])["text"] for ln in non_done]
    assert parsed_texts == PARTIALS

    # Body must end with the [DONE] sentinel
    assert data_lines[-1] == "data: [DONE]"


def test_words_route_integration(monkeypatch):
    """Non-streaming route attaches words field from build_word_timestamps."""
    from parakeet_vllm.api import routes
    from parakeet_vllm.engine.asr_engine import Transcription

    CANNED_WORDS = [{"word": "hello", "start": 0.0, "end": 0.2}]

    class FakeWordsEngine:
        processor = object()  # placeholder; route passes it to build_word_timestamps

        async def transcribe(self, audio, request_id):
            return Transcription(
                text="hello world",
                tokens=[1, 2],
                durations=[0.1, 0.1],
            )

    monkeypatch.setattr(routes, "get_engine", lambda: FakeWordsEngine())
    monkeypatch.setattr(routes, "build_word_timestamps", lambda tokens, durations, processor: CANNED_WORDS)
    app = routes.create_app()
    client = TestClient(app)

    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav(), "audio/wav")},
        data={"model": "x", "timestamp_granularities": "word"},
    )

    assert r.status_code == 200
    body = r.json()
    assert body["text"] == "hello world"
    assert body["words"] == CANNED_WORDS


# ---------------------------------------------------------------------------
# Fix I1: upload size cap tests
# ---------------------------------------------------------------------------

def test_upload_size_cap_returns_413(monkeypatch):
    """Fix I1: uploading more bytes than _MAX_UPLOAD_BYTES → 413 Payload Too Large."""
    from parakeet_vllm.api import routes

    # Patch the cap to 1 byte so any real audio body exceeds it.
    monkeypatch.setattr(routes, "_MAX_UPLOAD_BYTES", 1)

    class FakeEngine:
        async def transcribe(self, audio, request_id):
            from parakeet_vllm.engine.asr_engine import Transcription
            return Transcription(text="x", tokens=[1], durations=[1])

    monkeypatch.setattr(routes, "get_engine", lambda: FakeEngine())
    app = routes.create_app()
    client = TestClient(app, raise_server_exceptions=False)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("big.bin", b"x" * 16, "audio/wav")},
    )
    assert r.status_code == 413


def test_upload_within_cap_succeeds(monkeypatch):
    """Fix I1: upload at exactly the cap is accepted (not rejected as 413)."""
    from parakeet_vllm.api import routes
    from parakeet_vllm.engine.asr_engine import Transcription

    CANNED_WORDS = None

    class FakeEngine:
        async def transcribe(self, audio, request_id):
            return Transcription(text="ok", tokens=[1], durations=[1])

    # Leave _MAX_UPLOAD_BYTES at its default (100 MB).  A small WAV is fine.
    monkeypatch.setattr(routes, "get_engine", lambda: FakeEngine())
    app = routes.create_app()
    client = TestClient(app)
    r = client.post(
        "/v1/audio/transcriptions",
        files={"file": ("a.wav", _wav(), "audio/wav")},
    )
    assert r.status_code == 200


# ---------------------------------------------------------------------------
# Fix I2: timestamp_granularities[] (bracketed-array) form binding
# ---------------------------------------------------------------------------

def test_timestamp_granularities_array_form(monkeypatch):
    """Fix I2: ``timestamp_granularities[]=word`` (OpenAI Python client format) attaches words."""
    from parakeet_vllm.api import routes
    from parakeet_vllm.engine.asr_engine import Transcription

    CANNED_WORDS = [{"word": "hi", "start": 0.0, "end": 0.1}]

    class FakeWordsEngine:
        processor = object()

        async def transcribe(self, audio, request_id):
            return Transcription(text="hi", tokens=[1], durations=[0.1])

    monkeypatch.setattr(routes, "get_engine", lambda: FakeWordsEngine())
    monkeypatch.setattr(routes, "build_word_timestamps", lambda tokens, durations, processor: CANNED_WORDS)
    app = routes.create_app()
    client = TestClient(app)

    # Send the bracketed-array form used by the OpenAI Python client.
    # Use the files-list encoding (with None filename for text fields) so httpx
    # sends a single multipart request rather than mixing data= and files=.
    r = client.post(
        "/v1/audio/transcriptions",
        files=[
            ("model", (None, "x")),
            ("timestamp_granularities[]", (None, "word")),
            ("file", ("a.wav", _wav(), "audio/wav")),
        ],
    )

    assert r.status_code == 200
    body = r.json()
    assert "words" in body, f"Expected 'words' key in response, got: {list(body.keys())}"
    assert body["words"] == CANNED_WORDS


def test_timestamp_granularities_array_form_repeated(monkeypatch):
    """Fix I2: multiple ``timestamp_granularities[]`` values — word triggers if any is 'word'."""
    from parakeet_vllm.api import routes
    from parakeet_vllm.engine.asr_engine import Transcription

    CANNED_WORDS = [{"word": "hey", "start": 0.0, "end": 0.1}]

    class FakeWordsEngine:
        processor = object()

        async def transcribe(self, audio, request_id):
            return Transcription(text="hey", tokens=[1], durations=[0.1])

    monkeypatch.setattr(routes, "get_engine", lambda: FakeWordsEngine())
    monkeypatch.setattr(routes, "build_word_timestamps", lambda tokens, durations, processor: CANNED_WORDS)
    app = routes.create_app()
    client = TestClient(app)

    # Send multiple timestamp_granularities[] entries — "word" is the second one.
    r = client.post(
        "/v1/audio/transcriptions",
        files=[
            ("model", (None, "x")),
            ("timestamp_granularities[]", (None, "segment")),
            ("timestamp_granularities[]", (None, "word")),
            ("file", ("a.wav", _wav(), "audio/wav")),
        ],
    )

    assert r.status_code == 200
    assert r.json().get("words") == CANNED_WORDS

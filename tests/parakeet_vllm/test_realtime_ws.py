import base64, json, numpy as np, pytest


def _pcm16_b64(samples: np.ndarray) -> str:
    pcm = (np.clip(samples, -1, 1) * 32767).astype("<i2").tobytes()
    return base64.b64encode(pcm).decode()


def test_ws_malformed_event_returns_error(monkeypatch):
    from parakeet_vllm.realtime import ws_app
    async def fake_decode(audio, rid): return (["x"], "x", [])
    monkeypatch.setattr(ws_app, "decode_window", fake_decode)
    from parakeet_vllm.api import routes
    from fastapi.testclient import TestClient
    client = TestClient(routes.create_app())
    with client.websocket_connect("/v1/realtime") as ws:
        created = ws.receive_json()
        assert created["type"] == "session.created"
        ws.send_json({"type": "nonsense.event"})
        err = ws.receive_json()
        assert err["type"] == "error"


def test_ws_append_missing_audio_key_sends_error_session_survives(monkeypatch):
    """F1a: append with no 'audio' key → error event, socket still usable."""
    from parakeet_vllm.realtime import ws_app
    async def fake_decode(audio, rid): return (["x"], "x", [])
    monkeypatch.setattr(ws_app, "decode_window", fake_decode)
    from parakeet_vllm.api import routes
    from fastapi.testclient import TestClient
    client = TestClient(routes.create_app())
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()  # session.created
        # Send append with no 'audio' key
        ws.send_json({"type": "input_audio_buffer.append"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "audio" in err["error"]["message"].lower() or "field" in err["error"]["message"].lower()
        # Session is still alive: a valid event still works
        ws.send_json({"type": "input_audio_buffer.clear"})
        # No crash — send a second malformed event to confirm the loop continues
        ws.send_json({"type": "input_audio_buffer.append"})
        err2 = ws.receive_json()
        assert err2["type"] == "error"


def test_ws_append_bad_base64_sends_error_session_survives(monkeypatch):
    """F1b: append with non-base64 'audio' → error event, session survives."""
    from parakeet_vllm.realtime import ws_app
    async def fake_decode(audio, rid): return (["x"], "x", [])
    monkeypatch.setattr(ws_app, "decode_window", fake_decode)
    from parakeet_vllm.api import routes
    from fastapi.testclient import TestClient
    client = TestClient(routes.create_app())
    with client.websocket_connect("/v1/realtime") as ws:
        ws.receive_json()  # session.created
        # Send append with garbage base64
        ws.send_json({"type": "input_audio_buffer.append", "audio": "!!!not-base64!!!"})
        err = ws.receive_json()
        assert err["type"] == "error"
        assert "audio" in err["error"]["message"].lower() or "payload" in err["error"]["message"].lower()
        # Session is still alive: send another bad payload
        ws.send_json({"type": "input_audio_buffer.append", "audio": "!!!bad!!!"})
        err2 = ws.receive_json()
        assert err2["type"] == "error"


@pytest.mark.gpu
def test_ws_end_to_end_converges_to_offline():
    from datasets import load_dataset, Audio
    from parakeet_vllm.api import routes
    from parakeet_vllm.engine.asr_engine import ASREngine
    import asyncio
    from fastapi.testclient import TestClient
    ds = load_dataset("hf-internal-testing/librispeech_asr_dummy", "clean", split="validation")
    ds = ds.cast_column("audio", Audio(sampling_rate=16000))
    audio = ds["audio"][0]["array"].astype("float32")
    offline = asyncio.run(ASREngine().transcribe(audio, "off")).text.strip()

    client = TestClient(routes.create_app())
    deltas, completed = [], None
    with client.websocket_connect("/v1/realtime") as ws:
        assert ws.receive_json()["type"] == "session.created"
        ws.send_json({"type":"session.update","session":{
            "turn_detection": None,
            "audio": {"input": {"format": {"rate": 16000}}},
        }})
        for i in range(0, len(audio), 8000):   # 0.5s chunks
            ws.send_json({"type":"input_audio_buffer.append",
                          "audio": _pcm16_b64(audio[i:i+8000])})
        ws.send_json({"type":"input_audio_buffer.commit"})
        # drain until completed
        for _ in range(200):
            ev = ws.receive_json()
            if ev["type"].endswith("transcription.delta"): deltas.append(ev["delta"])
            if ev["type"].endswith("transcription.completed"):
                completed = ev["transcript"]; break
    assert completed is not None
    assert completed.strip() == offline
    # intermediate hops must have fired for a multi-second clip
    assert len(deltas) > 0
    # deltas append-only: concatenated committed text is a prefix of final
    joined = " ".join(d.strip() for d in deltas).split()
    assert completed.split()[:len(joined)] == joined

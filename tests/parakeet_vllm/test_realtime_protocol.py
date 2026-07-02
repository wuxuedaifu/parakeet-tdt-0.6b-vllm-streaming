import pytest
from parakeet_vllm.realtime.protocol import (
    parse_client_event, server_event, ProtocolError, CLIENT_EVENTS, SERVER_EVENTS,
)

def test_parse_known_client_event():
    ev = parse_client_event({"type": "input_audio_buffer.append", "audio": "AAAA"})
    assert ev["type"] == "input_audio_buffer.append"
    assert ev["audio"] == "AAAA"

def test_parse_session_update_passthrough():
    ev = parse_client_event({"type": "session.update", "session": {"turn_detection": None}})
    assert ev["session"]["turn_detection"] is None

def test_unknown_event_raises():
    with pytest.raises(ProtocolError):
        parse_client_event({"type": "does.not.exist"})

def test_missing_type_raises():
    with pytest.raises(ProtocolError):
        parse_client_event({"no_type": 1})

def test_server_event_has_type_and_id():
    ev = server_event("input_audio_buffer.speech_started", audio_start_ms=120)
    assert ev["type"] == "input_audio_buffer.speech_started"
    assert "event_id" in ev and ev["audio_start_ms"] == 120

def test_event_type_sets_are_disjoint_and_populated():
    assert CLIENT_EVENTS and SERVER_EVENTS
    assert "error" in SERVER_EVENTS


def test_client_and_server_event_sets_are_disjoint():
    assert not (CLIENT_EVENTS & SERVER_EVENTS), (
        f"Event types appear in both sets: {CLIENT_EVENTS & SERVER_EVENTS}"
    )


def test_server_event_unknown_type_raises_protocol_error():
    with pytest.raises(ProtocolError):
        server_event("bogus.type")

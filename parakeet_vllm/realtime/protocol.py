from __future__ import annotations
import itertools


class ProtocolError(ValueError):
    """Raised for malformed or unknown client events."""


CLIENT_EVENTS = {
    "session.update",
    "input_audio_buffer.append",
    "input_audio_buffer.commit",
    "input_audio_buffer.clear",
}
SERVER_EVENTS = {
    "session.created", "session.updated",
    "input_audio_buffer.speech_started", "input_audio_buffer.speech_stopped",
    "input_audio_buffer.committed",
    "conversation.item.input_audio_transcription.delta",
    "conversation.item.input_audio_transcription.completed",
    "error",
}

_counter = itertools.count(1)


def parse_client_event(msg: dict) -> dict:
    if not isinstance(msg, dict) or "type" not in msg:
        raise ProtocolError("event missing 'type'")
    t = msg["type"]
    if t not in CLIENT_EVENTS:
        raise ProtocolError(f"unknown client event type: {t!r}")
    return dict(msg)


def server_event(type: str, **fields) -> dict:
    if type not in SERVER_EVENTS:
        raise ProtocolError(f"not a server event type: {type!r}")
    return {"type": type, "event_id": f"evt_{next(_counter)}", **fields}

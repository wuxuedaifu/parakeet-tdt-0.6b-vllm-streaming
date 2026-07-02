from __future__ import annotations
import asyncio, base64
import numpy as np
from fastapi import WebSocket, WebSocketDisconnect
from .protocol import parse_client_event, server_event, ProtocolError
from .session import LiveSession
from .decoder_stream import decode_window


def _pcm16_to_f32_16k(b64: str, src_rate: int = 24000) -> np.ndarray:
    raw = base64.b64decode(b64)
    x = np.frombuffer(raw, dtype="<i2").astype(np.float32) / 32768.0
    if src_rate != 16000:
        import librosa
        x = librosa.resample(x, orig_sr=src_rate, target_sr=16000)
    return np.ascontiguousarray(x, dtype=np.float32)


def add_realtime_ws(app) -> None:
    @app.websocket("/v1/realtime")
    async def realtime(ws: WebSocket):
        await ws.accept()
        session = LiveSession(decode_fn=decode_window)
        await ws.send_json(server_event("session.created", session={}))

        async def pump():
            async for ev in session.events():
                await ws.send_json(ev)
        pump_task = asyncio.create_task(pump())
        # 24000 = OpenAI Realtime default input rate; overridden when the client declares a format.
        src_rate = 24000
        try:
            while True:
                msg = await ws.receive_json()
                try:
                    ev = parse_client_event(msg)
                except ProtocolError as e:
                    await ws.send_json(server_event("error", error={"message": str(e)}))
                    continue
                if ev["type"] == "session.update":
                    fmt = (((ev.get("session") or {}).get("audio") or {})
                           .get("input") or {}).get("format")
                    if isinstance(fmt, dict) and "rate" in fmt:
                        src_rate = int(fmt["rate"])
                    await session.handle_client_event(ev)
                elif ev["type"] == "input_audio_buffer.append":
                    try:
                        pcm = _pcm16_to_f32_16k(ev["audio"], src_rate)
                        await session.feed_pcm(pcm)
                    except KeyError:
                        await ws.send_json(server_event(
                            "error",
                            error={"message": "input_audio_buffer.append: missing 'audio' field"},
                        ))
                    except Exception as exc:
                        await ws.send_json(server_event(
                            "error",
                            error={"message": f"input_audio_buffer.append: bad audio payload: {exc}"},
                        ))
                else:
                    await session.handle_client_event(ev)
        except WebSocketDisconnect:
            pass
        finally:
            await session.close()
            # Give the pump a chance to drain the _CLOSE sentinel that session.close()
            # just enqueued, so no events are silently dropped on clean disconnect.
            try:
                await asyncio.wait_for(pump_task, timeout=2.0)
            except (asyncio.TimeoutError, asyncio.CancelledError, Exception):
                pump_task.cancel()
            try:
                await pump_task
            except (asyncio.CancelledError, Exception):
                pass

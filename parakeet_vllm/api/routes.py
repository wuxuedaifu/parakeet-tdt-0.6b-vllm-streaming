"""OpenAI-compatible transcription routes."""
from __future__ import annotations

import os
import uuid
from typing import Optional

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import JSONResponse, StreamingResponse

from ..audio import decode_to_16k_mono
from ..streaming.file_stream import build_word_timestamps

# ---------------------------------------------------------------------------
# Upload size cap (Fix I1)
# ---------------------------------------------------------------------------

_MAX_UPLOAD_MB = int(os.getenv("PARAKEET_MAX_UPLOAD_MB", "100"))
_MAX_UPLOAD_BYTES = _MAX_UPLOAD_MB * 1024 * 1024

# ---------------------------------------------------------------------------
# Engine singleton
# ---------------------------------------------------------------------------

_engine = None


def get_engine():
    """Return the global ASREngine, creating it on first call."""
    global _engine
    if _engine is None:
        from ..engine.asr_engine import ASREngine

        _engine = ASREngine()
    return _engine


# ---------------------------------------------------------------------------
# App factory
# ---------------------------------------------------------------------------


def create_app() -> FastAPI:
    """Create and return the FastAPI application."""
    app = FastAPI(title="Parakeet ASR", version="0.1.0")

    @app.post("/v1/audio/transcriptions")
    async def transcriptions(
        request: Request,
        file: UploadFile = File(...),
        model: Optional[str] = Form(None),
        response_format: Optional[str] = Form(None),
        stream: Optional[bool] = Form(None),
        timestamp_granularities: Optional[str] = Form(None),
    ):
        """OpenAI-compatible POST /v1/audio/transcriptions.

        Accepts multipart form data with:
          - ``file``: audio file bytes
          - ``model``: model name (informational, ignored for dispatch)
          - ``response_format``: e.g. ``json`` (default) or ``text``
          - ``stream``: when ``true``, returns a ``text/event-stream`` SSE
            response emitting ``data: {"text": <partial>}`` events followed
            by a terminal ``data: [DONE]``.
          - ``timestamp_granularities`` or ``timestamp_granularities[]``:
            when any value equals ``"word"``, the non-streaming response
            includes a ``words`` list with OpenAI-compatible word timestamps.
            Both the scalar form and the bracketed-array form sent by the
            official OpenAI Python client are accepted.

        Returns ``{"text": "..."}`` (non-streaming) or an SSE stream.
        """
        # ------------------------------------------------------------------
        # Fix I1: cap upload size to avoid unbounded RAM consumption (DoS).
        # Read one byte more than the cap; if we get it the upload is too big.
        # ------------------------------------------------------------------
        audio_bytes = await file.read(_MAX_UPLOAD_BYTES + 1)
        if len(audio_bytes) > _MAX_UPLOAD_BYTES:
            raise HTTPException(
                status_code=413,
                detail=f"Upload exceeds maximum allowed size ({_MAX_UPLOAD_MB} MB)",
            )
        if not audio_bytes:
            raise HTTPException(status_code=422, detail="Uploaded file is empty")

        # ------------------------------------------------------------------
        # Fix I2: accept both ``timestamp_granularities`` (scalar) and
        # ``timestamp_granularities[]`` (bracketed-array sent by the OpenAI
        # Python client).  Starlette caches parsed form data on the request,
        # so calling request.form() here is safe alongside UploadFile/Form.
        # ------------------------------------------------------------------
        form_data = await request.form()
        tg_arr = form_data.getlist("timestamp_granularities[]")
        want_words = (
            (timestamp_granularities is not None and "word" in timestamp_granularities)
            or any("word" in str(v) for v in tg_arr)
        )

        try:
            audio = decode_to_16k_mono(audio_bytes)
        except Exception as exc:
            raise HTTPException(status_code=422, detail=f"Audio decode failed: {exc}") from exc

        engine = get_engine()
        request_id = str(uuid.uuid4())

        # ------------------------------------------------------------------
        # Streaming path
        # ------------------------------------------------------------------
        if stream:
            from ..streaming.file_stream import partials_to_sse

            return StreamingResponse(
                partials_to_sse(engine.transcribe_stream(audio, request_id)),
                media_type="text/event-stream",
            )

        # ------------------------------------------------------------------
        # Non-streaming path
        # ------------------------------------------------------------------
        try:
            result = await engine.transcribe(audio, request_id)
        except Exception as exc:
            raise HTTPException(status_code=500, detail=f"Transcription failed: {exc}") from exc

        response: dict = {"text": result.text}

        # Attach word-level timestamps when the caller requests them.
        # For long audio, transcribe() pre-computes offset-corrected word
        # timestamps and stores them in result.words; use them directly so the
        # multi-chunk timestamps are correct.  Fall back to build_word_timestamps
        # for short (single-chunk) audio where result.words is None.
        if want_words:
            if result.words is not None:
                response["words"] = result.words
            else:
                response["words"] = build_word_timestamps(
                    result.tokens, result.durations, engine.processor
                )

        return JSONResponse(response)

    from ..realtime.ws_app import add_realtime_ws
    add_realtime_ws(app)

    return app

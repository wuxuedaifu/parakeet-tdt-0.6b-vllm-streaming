"""Uvicorn server entrypoint for the Parakeet ASR service."""
from __future__ import annotations

import os

import uvicorn

from .api.routes import create_app

DEFAULT_HOST = "0.0.0.0"
DEFAULT_PORT = 5092


def main() -> None:
    """Start the Parakeet ASR HTTP server.

    Environment variables:
      - ``PARAKEET_HOST``: bind host (default ``0.0.0.0``)
      - ``PARAKEET_PORT``: bind port (default ``5092``)
    """
    host = os.getenv("PARAKEET_HOST", DEFAULT_HOST)
    port = int(os.getenv("PARAKEET_PORT", str(DEFAULT_PORT)))
    app = create_app()
    uvicorn.run(app, host=host, port=port)


if __name__ == "__main__":
    main()

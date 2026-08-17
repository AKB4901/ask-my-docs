"""One-command entrypoint: build the index if missing, then start the server.

    python run.py
"""

from __future__ import annotations

import uvicorn

from app.config import get_settings
from app.rag.ingest import ingest, is_indexed


def main() -> None:
    settings = get_settings()
    if not is_indexed(settings):
        print("Building index (first run only)...")
        ingest(settings)
    uvicorn.run("app.main:app", host=settings.host, port=settings.port, reload=False)


if __name__ == "__main__":
    main()

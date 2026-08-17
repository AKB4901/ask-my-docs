"""
FastAPI application.

Serves both the JSON API and the single-page UI from one process, so the whole
project is `clone -> install -> run` with no separate frontend build step.

Visitors can query the default demo corpus, or upload their own document (PDF,
txt, md) and ask questions about that instead — scoped privately to their
session.
"""

from __future__ import annotations

import json
import logging
import uuid
from contextlib import asynccontextmanager
from pathlib import Path

from fastapi import FastAPI, File, HTTPException, Request, Response, UploadFile
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import FileResponse
from fastapi.staticfiles import StaticFiles

from app import __version__
from app.config import get_settings
from app.rag import llm
from app.rag.ingest import ingest, is_indexed
from app.rag.loaders import (
    EmptyDocument,
    UnsupportedFileType,
    load_upload,
)
from app.rag.pipeline import RagEngine
from app.schemas import AskRequest, AskResponse, CorpusStats, UploadResponse
from app.sessions import SessionStore

logging.basicConfig(level=logging.INFO, format="%(message)s")
logger = logging.getLogger("ask_my_docs")

STATIC_DIR = Path(__file__).parent / "static"
SESSION_COOKIE = "amd_sid"

state: dict = {}


@asynccontextmanager
async def lifespan(app: FastAPI):
    settings = get_settings()
    if not is_indexed(settings):
        logger.info("No index found — building it now (first run only)...")
        ingest(settings)
    state["engine"] = RagEngine(settings)
    state["sessions"] = SessionStore(settings)
    logger.info("Ask My Docs ready.")
    yield
    state.clear()


app = FastAPI(title="Ask My Docs", version=__version__, lifespan=lifespan)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


def _session_id(request: Request, response: Response) -> str:
    sid = request.cookies.get(SESSION_COOKIE)
    if not sid:
        sid = uuid.uuid4().hex
        response.set_cookie(
            SESSION_COOKIE, sid, httponly=True, samesite="lax", max_age=60 * 60 * 24
        )
    return sid


@app.get("/health")
def health():
    return {"status": "ok", "version": __version__}


@app.get("/api/stats", response_model=CorpusStats)
def stats():
    settings = get_settings()
    manifest_path = settings.index_dir / "manifest.json"
    manifest = json.loads(manifest_path.read_text()) if manifest_path.exists() else {}
    return CorpusStats(
        documents=manifest.get("documents", 0),
        chunks=manifest.get("chunks", 0),
        embedding_model=settings.embedding_model.split("/")[-1],
        reranker_model=settings.reranker_model.split("/")[-1],
        indexed=is_indexed(settings),
    )


@app.post("/api/upload", response_model=UploadResponse)
async def upload(request: Request, response: Response, file: UploadFile = File(...)):
    sessions: SessionStore = state["sessions"]
    data = await file.read()
    try:
        text = load_upload(file.filename or "document", data)
    except (UnsupportedFileType, EmptyDocument, ValueError) as exc:
        raise HTTPException(status_code=400, detail=str(exc)) from exc

    sid = _session_id(request, response)
    try:
        corpus = sessions.add_document(sid, file.filename or "document", text)
    except Exception as exc:
        logger.exception("upload indexing failed")
        raise HTTPException(status_code=500, detail=f"Could not index file: {exc}") from exc

    return UploadResponse(
        doc_id=file.filename or "document",
        documents=corpus.num_docs,
        chunks=corpus.num_chunks,
        using_upload=True,
    )


@app.post("/api/reset")
def reset(request: Request, response: Response):
    sessions: SessionStore = state["sessions"]
    sid = request.cookies.get(SESSION_COOKIE)
    if sid:
        sessions.clear(sid)
    return {"using_upload": False}


@app.post("/api/ask", response_model=AskResponse)
def ask(req: AskRequest, request: Request, response: Response):
    engine: RagEngine = state.get("engine")
    sessions: SessionStore = state.get("sessions")
    if engine is None:
        raise HTTPException(status_code=503, detail="Engine not ready yet.")

    # If this session uploaded a document, query that; else the demo corpus.
    corpus = None
    sid = request.cookies.get(SESSION_COOKIE)
    if sid and sessions:
        corpus = sessions.get_corpus(sid)

    try:
        return engine.answer(req.question, top_k=req.top_k, corpus=corpus)
    except llm.LLMError as exc:
        raise HTTPException(status_code=502, detail=str(exc)) from exc
    except Exception as exc:
        logger.exception("ask failed")
        raise HTTPException(status_code=500, detail=f"Internal error: {exc}") from exc


@app.get("/")
def index():
    return FileResponse(STATIC_DIR / "index.html")


app.mount("/static", StaticFiles(directory=STATIC_DIR), name="static")

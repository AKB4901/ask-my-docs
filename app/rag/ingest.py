"""
Ingestion pipeline.

Reads every document under ``data/docs``, chunks it, embeds the chunks, and
builds both the FAISS vector index and the BM25 lexical index. Everything is
persisted to ``data/index`` so the server starts instantly and retrieval is
reproducible between runs (important for a stable evaluation baseline).

Run directly:  python -m app.rag.ingest
"""

from __future__ import annotations

import json
from pathlib import Path

from app.config import Settings, get_settings
from app.rag.bm25_store import BM25Store
from app.rag.chunking import Chunk, chunk_document
from app.rag.embeddings import embed_texts
from app.rag.vector_store import VectorStore

SUPPORTED_SUFFIXES = {".md", ".txt", ".markdown"}


def load_documents(data_dir: Path) -> dict[str, str]:
    docs: dict[str, str] = {}
    for path in sorted(data_dir.rglob("*")):
        if path.suffix.lower() in SUPPORTED_SUFFIXES and path.is_file():
            doc_id = path.relative_to(data_dir).as_posix()
            docs[doc_id] = path.read_text(encoding="utf-8", errors="ignore")
    return docs


def build_chunks(docs: dict[str, str], settings: Settings) -> list[Chunk]:
    chunks: list[Chunk] = []
    for doc_id, text in docs.items():
        chunks.extend(
            chunk_document(
                doc_id,
                text,
                chunk_size=settings.chunk_size,
                chunk_overlap=settings.chunk_overlap,
            )
        )
    return chunks


def ingest(settings: Settings | None = None) -> int:
    settings = settings or get_settings()
    docs = load_documents(settings.data_dir)
    if not docs:
        raise SystemExit(f"No documents found in {settings.data_dir}")

    chunks = build_chunks(docs, settings)
    chunk_ids = [c.chunk_id for c in chunks]
    texts = [c.text for c in chunks]

    print(f"Loaded {len(docs)} docs -> {len(chunks)} chunks. Embedding...")
    vectors = embed_texts(texts)

    vs = VectorStore(dim=vectors.shape[1])
    vs.add(chunk_ids, vectors)
    vs.save(settings.index_dir)

    bm25 = BM25Store.build(chunk_ids, texts)
    bm25.save(settings.index_dir)

    # Persist chunk text + metadata so retrieval can hydrate passages.
    meta = {c.chunk_id: {"doc_id": c.doc_id, "text": c.text} for c in chunks}
    (settings.index_dir / "chunks.json").write_text(json.dumps(meta))
    (settings.index_dir / "manifest.json").write_text(
        json.dumps({"documents": len(docs), "chunks": len(chunks)})
    )

    print(f"Index built at {settings.index_dir}")
    return len(chunks)


def is_indexed(settings: Settings) -> bool:
    required = ["vectors.faiss", "bm25.json", "chunks.json", "manifest.json"]
    return all((settings.index_dir / f).exists() for f in required)


if __name__ == "__main__":
    ingest()

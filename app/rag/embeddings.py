"""
Dense embeddings.

We use a small local sentence-transformers model (all-MiniLM-L6-v2, ~90 MB).
It runs comfortably on CPU and needs well under 6 GB of VRAM if a GPU is
present, so this project runs on modest, zero-budget hardware.

Embeddings are L2-normalized so that a FAISS inner-product index is equivalent
to cosine similarity.
"""

from __future__ import annotations

import threading

import numpy as np

from app.config import get_settings

_model = None
_lock = threading.Lock()


def _get_model():
    """Load the embedding model once, on first use (keeps startup fast)."""
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from sentence_transformers import SentenceTransformer

                _model = SentenceTransformer(get_settings().embedding_model)
    return _model


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of documents. Returns a (n, dim) float32 array."""
    model = _get_model()
    vectors = model.encode(
        texts,
        batch_size=32,
        convert_to_numpy=True,
        normalize_embeddings=True,
        show_progress_bar=False,
    )
    return vectors.astype("float32")


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns a (dim,) float32 array."""
    return embed_texts([text])[0]

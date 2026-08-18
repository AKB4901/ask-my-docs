"""
Dense embeddings (ONNX via fastembed).

We use all-MiniLM-L6-v2 through fastembed, which runs the model on ONNX Runtime
instead of PyTorch. Same model, same quality — but no torch dependency, which
cuts the container's memory footprint by hundreds of MB and lets the whole app
fit a free 512 MB host.

Embeddings are L2-normalized so a FAISS inner-product index equals cosine
similarity.
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
                from fastembed import TextEmbedding

                _model = TextEmbedding(model_name=get_settings().embedding_model)
    return _model


def _normalize(vectors: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(vectors, axis=1, keepdims=True)
    return (vectors / np.clip(norms, 1e-9, None)).astype("float32")


def embed_texts(texts: list[str]) -> np.ndarray:
    """Embed a batch of documents. Returns a (n, dim) float32 array."""
    model = _get_model()
    vectors = np.array(list(model.embed(texts)), dtype="float32")
    return _normalize(vectors)


def embed_query(text: str) -> np.ndarray:
    """Embed a single query. Returns a (dim,) float32 array."""
    return embed_texts([text])[0]

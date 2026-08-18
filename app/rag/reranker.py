"""
Cross-encoder reranking (ONNX via fastembed).

First-stage retrieval (BM25 + vectors) optimizes for recall: cast a wide net.
The top of that net is noisy, so a cross-encoder reads the query and each
candidate *together* and scores true relevance far more accurately than
first-stage similarity. It's too slow to run over the whole corpus, which is
why we only rerank the ~20 fused candidates.

This uses the same ms-marco-MiniLM-L-6-v2 model as a PyTorch cross-encoder,
served through ONNX Runtime — so scores are on the same scale (unbounded
logits, higher = more relevant) and the abstention threshold behaves the same.
"""

from __future__ import annotations

import threading

from app.config import get_settings

_model = None
_lock = threading.Lock()


def _get_model():
    global _model
    if _model is None:
        with _lock:
            if _model is None:
                from fastembed.rerank.cross_encoder import TextCrossEncoder

                _model = TextCrossEncoder(model_name=get_settings().reranker_model)
    return _model


def rerank(query: str, candidates: list[tuple[str, str]]) -> list[tuple[str, float]]:
    """Score (chunk_id, text) candidates against the query.

    Returns (chunk_id, score) sorted best-first.
    """
    if not candidates:
        return []
    model = _get_model()
    scores = list(model.rerank(query, [text for _, text in candidates]))
    ranked = sorted(
        ((cid, float(s)) for (cid, _), s in zip(candidates, scores)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked

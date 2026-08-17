"""
Cross-encoder reranking.

First-stage retrieval (BM25 + vectors) optimizes for recall: cast a wide net.
But the top of that net is noisy. A cross-encoder reads the query and each
candidate *together* and scores true relevance far more accurately than the
bi-encoder similarity used for first-stage search. It is too slow to run over
the whole corpus, which is exactly why we only rerank the ~20 fused candidates.

This precision step is what lets us send just 4 passages to the LLM instead of
15 — cheaper, faster, and less prone to distraction.
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
                from sentence_transformers import CrossEncoder

                _model = CrossEncoder(get_settings().reranker_model)
    return _model


def rerank(
    query: str, candidates: list[tuple[str, str]]
) -> list[tuple[str, float]]:
    """Score (chunk_id, text) candidates against the query.

    Returns (chunk_id, score) sorted best-first. Scores are cross-encoder
    logits — higher is more relevant; they are unbounded and can be negative.
    """
    if not candidates:
        return []
    model = _get_model()
    pairs = [[query, text] for _, text in candidates]
    scores = model.predict(pairs)
    ranked = sorted(
        ((cid, float(s)) for (cid, _), s in zip(candidates, scores)),
        key=lambda kv: kv[1],
        reverse=True,
    )
    return ranked

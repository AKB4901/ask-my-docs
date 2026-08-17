"""
Hybrid fusion via Reciprocal Rank Fusion (RRF).

BM25 (lexical) and dense vectors (semantic) fail in different ways: BM25 misses
paraphrases, vectors miss exact keywords, IDs, and rare terms. RRF combines two
ranked lists without needing their scores to be on the same scale — it only
looks at *rank position*, which makes it robust and dependency-free.

    score(d) = sum over lists of  1 / (k + rank_in_list(d))

Lower ranks (nearer the top) contribute more. ``k`` (default 60, from the
original RRF paper) damps the influence of any single list's ordering.
"""

from __future__ import annotations

from collections import defaultdict


def reciprocal_rank_fusion(
    ranked_lists: list[list[str]], *, k: int = 60
) -> list[tuple[str, float]]:
    """Fuse several ranked lists of ids into one, best first.

    Args:
        ranked_lists: each inner list is ids ordered best-to-worst.
        k: RRF damping constant.

    Returns:
        (id, fused_score) pairs sorted by descending score.
    """
    scores: dict[str, float] = defaultdict(float)
    for ranked in ranked_lists:
        for rank, doc_id in enumerate(ranked):
            scores[doc_id] += 1.0 / (k + rank)
    return sorted(scores.items(), key=lambda kv: kv[1], reverse=True)


def label_retrieval(doc_id: str, lexical: set[str], semantic: set[str]) -> str:
    """Explain *why* a passage surfaced — useful signal in the UI and traces."""
    in_lex, in_sem = doc_id in lexical, doc_id in semantic
    if in_lex and in_sem:
        return "hybrid"
    if in_sem:
        return "vector"
    return "lexical"

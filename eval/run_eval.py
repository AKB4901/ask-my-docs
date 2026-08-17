"""
Evaluation harness + CI gate.

Two layers of metrics:

1. Retrieval (always runs, no LLM, deterministic)
   - hit@k : did the gold document appear among the final reranked passages?
   - MRR   : how high did it rank? (1/rank of the first correct passage)

2. End-to-end (runs only when an LLM is configured)
   - answer_keyword_recall : did the answer contain the expected fact?
   - citation_validity     : were grounded answers backed by valid citations?

Every metric has a floor in ``thresholds.yaml``. If any measured metric is below
its floor, this script exits 1 — which is what turns "quality" into a build
gate instead of a vibe. This is the single most under-shown skill in RAG
portfolios, so it is deliberately front-and-center here.

Usage:
    python -m eval.run_eval              # retrieval-only unless LLM configured
    python -m eval.run_eval --e2e        # force end-to-end (requires an LLM)
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import yaml

from app.config import get_settings
from app.rag.bm25_store import BM25Store
from app.rag.hybrid import reciprocal_rank_fusion
from app.rag.ingest import is_indexed
from app.rag.reranker import rerank
from app.rag.vector_store import VectorStore

EVAL_DIR = Path(__file__).parent


def load_gold() -> list[dict]:
    lines = (EVAL_DIR / "gold.jsonl").read_text().strip().splitlines()
    return [json.loads(line) for line in lines]


def load_thresholds() -> dict:
    return yaml.safe_load((EVAL_DIR / "thresholds.yaml").read_text())


def retrieve_docs(question: str, settings, vs, bm25, chunks) -> list[str]:
    """Return the ordered list of *doc ids* for the final reranked passages."""
    from app.rag.embeddings import embed_query

    lexical = [cid for cid, _ in bm25.search(question, settings.bm25_top_k)]
    qvec = embed_query(question)
    vector = [cid for cid, _ in vs.search(qvec, settings.vector_top_k)]
    fused = reciprocal_rank_fusion([lexical, vector], k=settings.rrf_k)
    candidates = [cid for cid, _ in fused[: settings.rerank_candidates]]
    reranked = rerank(question, [(cid, chunks[cid]["text"]) for cid in candidates])
    top = [cid for cid, _ in reranked[: settings.final_top_k]]
    return [chunks[cid]["doc_id"] for cid in top]


def evaluate(force_e2e: bool = False) -> int:
    settings = get_settings()
    if not is_indexed(settings):
        print("Index missing — run `python -m app.rag.ingest` first.")
        return 1

    gold = load_gold()
    thresholds = load_thresholds()
    vs = VectorStore.load(settings.index_dir)
    bm25 = BM25Store.load(settings.index_dir)
    chunks = json.loads((settings.index_dir / "chunks.json").read_text())

    hits, reciprocal_ranks = 0, []
    per_question_docs: list[list[str]] = []

    for item in gold:
        docs = retrieve_docs(item["question"], settings, vs, bm25, chunks)
        per_question_docs.append(docs)
        expected = item["expected_doc"]
        if expected in docs:
            hits += 1
            reciprocal_ranks.append(1.0 / (docs.index(expected) + 1))
        else:
            reciprocal_ranks.append(0.0)

    n = len(gold)
    metrics = {
        "retrieval_hit_at_k": round(hits / n, 3),
        "retrieval_mrr": round(sum(reciprocal_ranks) / n, 3),
    }

    run_e2e = force_e2e or bool(settings.groq_api_key) or not settings.uses_groq
    if run_e2e:
        e2e = _run_e2e(gold)
        metrics.update(e2e)

    # -- report ----------------------------------------------------------
    print("\n=== Evaluation ===")
    failed = []
    for name, value in metrics.items():
        floor = thresholds.get(name)
        if floor is None:
            print(f"  {name:24s} {value}")
            continue
        ok = value >= floor
        flag = "PASS" if ok else "FAIL"
        print(f"  {name:24s} {value:>6}  (floor {floor})  [{flag}]")
        if not ok:
            failed.append(name)

    if failed:
        print(f"\nFAILED gate on: {', '.join(failed)}")
        return 1
    print("\nAll gates passed.")
    return 0


def _run_e2e(gold: list[dict]) -> dict:
    """Full-pipeline metrics — needs a working LLM."""
    from app.rag.pipeline import RagEngine

    engine = RagEngine()
    keyword_hits, grounded_valid, grounded_total = 0, 0, 0
    for item in gold:
        resp = engine.answer(item["question"])
        text = resp.answer.lower()
        if any(kw.lower() in text for kw in item["answer_keywords"]):
            keyword_hits += 1
        if resp.grounded:
            grounded_total += 1
            if resp.sources:
                grounded_valid += 1

    n = len(gold)
    return {
        "answer_keyword_recall": round(keyword_hits / n, 3),
        "citation_validity": round(
            grounded_valid / grounded_total if grounded_total else 1.0, 3
        ),
    }


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--e2e", action="store_true", help="force end-to-end eval")
    args = parser.parse_args()
    sys.exit(evaluate(force_e2e=args.e2e))

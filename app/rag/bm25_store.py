"""
Lexical store (BM25).

BM25 is the half of retrieval that dense vectors are worst at: exact terms,
product names, error codes, acronyms, numbers. Cheap to run, no model to load,
and it catches the queries where semantic search quietly returns something
plausible-but-wrong.
"""

from __future__ import annotations

import json
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    return _TOKEN_RE.findall(text.lower())


class BM25Store:
    def __init__(self, chunk_ids: list[str], corpus_tokens: list[list[str]]):
        self.chunk_ids = chunk_ids
        self.corpus_tokens = corpus_tokens
        self.bm25 = BM25Okapi(corpus_tokens) if corpus_tokens else None

    @classmethod
    def build(cls, chunk_ids: list[str], texts: list[str]) -> BM25Store:
        return cls(chunk_ids, [tokenize(t) for t in texts])

    def search(self, query: str, top_k: int) -> list[tuple[str, float]]:
        if not self.bm25:
            return []
        scores = self.bm25.get_scores(tokenize(query))
        ranked = sorted(
            zip(self.chunk_ids, scores), key=lambda kv: kv[1], reverse=True
        )
        return [(cid, float(s)) for cid, s in ranked[:top_k] if s > 0]

    def save(self, index_dir: Path) -> None:
        index_dir.mkdir(parents=True, exist_ok=True)
        payload = {"chunk_ids": self.chunk_ids, "corpus_tokens": self.corpus_tokens}
        (index_dir / "bm25.json").write_text(json.dumps(payload))

    @classmethod
    def load(cls, index_dir: Path) -> BM25Store:
        payload = json.loads((index_dir / "bm25.json").read_text())
        return cls(payload["chunk_ids"], payload["corpus_tokens"])

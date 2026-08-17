"""
Vector store (FAISS).

A flat inner-product index over normalized embeddings == exact cosine search.
Flat is the right default here: corpora that fit a portfolio/enterprise-docs
use case are small enough that exact search is instant, and we avoid the
recall cliffs of approximate indexes. Swapping in IVF/HNSW later is a one-line
change behind this interface.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np


class VectorStore:
    def __init__(self, dim: int):
        import faiss

        self.dim = dim
        self.index = faiss.IndexFlatIP(dim)
        self.chunk_ids: list[str] = []

    def add(self, chunk_ids: list[str], vectors: np.ndarray) -> None:
        assert vectors.shape[0] == len(chunk_ids)
        self.index.add(vectors)
        self.chunk_ids.extend(chunk_ids)

    def search(self, query_vec: np.ndarray, top_k: int) -> list[tuple[str, float]]:
        if self.index.ntotal == 0:
            return []
        scores, idxs = self.index.search(query_vec.reshape(1, -1), top_k)
        out = []
        for score, idx in zip(scores[0], idxs[0]):
            if idx == -1:
                continue
            out.append((self.chunk_ids[idx], float(score)))
        return out

    def save(self, index_dir: Path) -> None:
        import faiss

        index_dir.mkdir(parents=True, exist_ok=True)
        faiss.write_index(self.index, str(index_dir / "vectors.faiss"))
        (index_dir / "vector_ids.txt").write_text("\n".join(self.chunk_ids))

    @classmethod
    def load(cls, index_dir: Path) -> VectorStore:
        import faiss

        index = faiss.read_index(str(index_dir / "vectors.faiss"))
        store = cls.__new__(cls)
        store.index = index
        store.dim = index.d
        ids_file = index_dir / "vector_ids.txt"
        store.chunk_ids = ids_file.read_text().splitlines() if ids_file.exists() else []
        return store

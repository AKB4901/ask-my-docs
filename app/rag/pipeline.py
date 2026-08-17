"""
The RAG engine.

Orchestrates the full path of a question, timing every stage:

    query
      ├─ lexical (BM25)      ─┐
      ├─ semantic (vectors)  ─┤─ Reciprocal Rank Fusion ─ rerank (cross-encoder)
      │                                                        │
      └────────────────── top-k passages ─────────────────────┘
                                    │
                            LLM (cite-only prompt)
                                    │
                        citation validation / abstain
                                    │
                                 answer

Retrieval runs against a `Corpus` — either the default document set loaded from
disk, or an in-memory corpus built from a user's uploaded file. Same pipeline,
same guarantees, either way.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

from app.config import Settings, get_settings
from app.observability.tracing import Trace
from app.rag import citations, llm
from app.rag.bm25_store import BM25Store
from app.rag.chunking import chunk_document
from app.rag.hybrid import label_retrieval, reciprocal_rank_fusion
from app.rag.ingest import is_indexed
from app.rag.reranker import rerank
from app.rag.vector_store import VectorStore
from app.schemas import AskResponse, Source, StageTiming
from app.schemas import Trace as TraceSchema


@dataclass
class Corpus:
    """A searchable set of documents: the two indexes plus chunk text."""

    vector_store: VectorStore
    bm25: BM25Store
    chunks: dict[str, dict]  # chunk_id -> {"doc_id": str, "text": str}

    def text(self, chunk_id: str) -> str:
        return self.chunks[chunk_id]["text"]

    def doc(self, chunk_id: str) -> str:
        return self.chunks[chunk_id]["doc_id"]

    @property
    def num_chunks(self) -> int:
        return len(self.chunks)

    @property
    def num_docs(self) -> int:
        return len({c["doc_id"] for c in self.chunks.values()})

    @classmethod
    def load(cls, index_dir: Path) -> Corpus:
        """Load a persisted corpus from disk (the default document set)."""
        return cls(
            vector_store=VectorStore.load(index_dir),
            bm25=BM25Store.load(index_dir),
            chunks=json.loads((index_dir / "chunks.json").read_text()),
        )

    @classmethod
    def from_documents(cls, docs: dict[str, str], settings: Settings) -> Corpus:
        """Build an in-memory corpus from raw text (used for uploads)."""
        from app.rag.embeddings import embed_texts

        chunks = []
        for doc_id, text in docs.items():
            chunks.extend(
                chunk_document(
                    doc_id,
                    text,
                    chunk_size=settings.chunk_size,
                    chunk_overlap=settings.chunk_overlap,
                )
            )
        if not chunks:
            raise ValueError("Document produced no indexable chunks.")

        ids = [c.chunk_id for c in chunks]
        texts = [c.text for c in chunks]
        vectors = embed_texts(texts)

        vs = VectorStore(dim=vectors.shape[1])
        vs.add(ids, vectors)
        bm25 = BM25Store.build(ids, texts)
        chunkmap = {c.chunk_id: {"doc_id": c.doc_id, "text": c.text} for c in chunks}
        return cls(vector_store=vs, bm25=bm25, chunks=chunkmap)


class RagEngine:
    def __init__(self, settings: Settings | None = None):
        self.settings = settings or get_settings()
        if not is_indexed(self.settings):
            raise RuntimeError("Index not found. Run `python -m app.rag.ingest` first.")
        self.default_corpus = Corpus.load(self.settings.index_dir)

    def answer(
        self,
        question: str,
        top_k: int | None = None,
        corpus: Corpus | None = None,
    ) -> AskResponse:
        s = self.settings
        corpus = corpus or self.default_corpus
        final_k = top_k or s.final_top_k
        trace = Trace()

        # 1) First-stage retrieval — lexical + semantic.
        with trace.stage("lexical"):
            lexical_hits = corpus.bm25.search(question, s.bm25_top_k)
        with trace.stage("semantic"):
            from app.rag.embeddings import embed_query

            qvec = embed_query(question)
            vector_hits = corpus.vector_store.search(qvec, s.vector_top_k)

        lexical_ids = [cid for cid, _ in lexical_hits]
        vector_ids = [cid for cid, _ in vector_hits]

        # 2) Fuse the two ranked lists.
        with trace.stage("fuse"):
            fused = reciprocal_rank_fusion([lexical_ids, vector_ids], k=s.rrf_k)
            candidate_ids = [cid for cid, _ in fused[: s.rerank_candidates]]

        # 3) Rerank the fused candidates for precision.
        with trace.stage("rerank"):
            reranked = rerank(question, [(cid, corpus.text(cid)) for cid in candidate_ids])
            top = [(cid, score) for cid, score in reranked[:final_k]]

        # Guard: abstain before spending an LLM call on weak passages.
        strong = [(cid, sc) for cid, sc in top if sc >= s.min_rerank_score]
        if not strong:
            trace.emit(question=question, grounded=False, abstained=True)
            return self._build_response(
                question, citations.ABSTAIN_SENTENCE, [], [], trace,
                grounded=False, abstained=True,
            )

        lex_set, sem_set = set(lexical_ids), set(vector_ids)
        passages = [corpus.text(cid) for cid, _ in strong]

        # 4) Generate under a strict cite-only prompt.
        with trace.stage("generate"):
            user_prompt = citations.build_user_prompt(question, passages)
            result = llm.generate(s, citations.SYSTEM_PROMPT, user_prompt)
            trace.set_usage(result.prompt_tokens, result.completion_tokens, result.cost_usd)

        # 5) Validate citations / detect abstention.
        with trace.stage("verify"):
            validated = citations.validate_answer(result.text, num_sources=len(passages))

        sources = self._sources_for(strong, validated, lex_set, sem_set, corpus)
        trace.emit(question=question, grounded=validated.grounded, abstained=validated.abstained)
        return self._build_response(
            question, validated.answer, strong, sources, trace,
            grounded=validated.grounded, abstained=validated.abstained,
        )

    def _sources_for(self, strong, validated, lex_set, sem_set, corpus) -> list[Source]:
        sources: list[Source] = []
        for i, (cid, score) in enumerate(strong, start=1):
            if i not in validated.cited_indices:
                continue
            sources.append(
                Source(
                    index=i,
                    doc_id=corpus.doc(cid),
                    chunk_id=cid,
                    text=corpus.text(cid),
                    rerank_score=round(score, 4),
                    retrieval=label_retrieval(cid, lex_set, sem_set),
                )
            )
        return sources

    def _build_response(self, question, answer, strong, sources, trace, *, grounded, abstained) -> AskResponse:
        return AskResponse(
            question=question,
            answer=answer,
            grounded=grounded,
            abstained=abstained,
            sources=sources,
            trace=TraceSchema(
                stages=[StageTiming(name=n, ms=round(ms, 1)) for n, ms in trace.stages],
                total_ms=round(trace.total_ms, 1),
                prompt_tokens=trace.prompt_tokens,
                completion_tokens=trace.completion_tokens,
                cost_usd=trace.cost_usd,
            ),
            provider=self.settings.llm_provider,
            model=llm.active_model(self.settings),
        )

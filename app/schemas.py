"""API request/response contracts.

These models are the boundary between the RAG engine and the outside world.
Typing them explicitly means the frontend, the tests, and the eval harness all
agree on one shape.
"""

from __future__ import annotations

from pydantic import BaseModel, Field


class AskRequest(BaseModel):
    question: str = Field(..., min_length=3, max_length=1000)
    top_k: int | None = Field(default=None, ge=1, le=10)


class Source(BaseModel):
    index: int                # the [n] the answer cites
    doc_id: str
    chunk_id: str
    text: str
    rerank_score: float
    retrieval: str            # "hybrid" | "vector" | "lexical"


class StageTiming(BaseModel):
    name: str
    ms: float


class Trace(BaseModel):
    stages: list[StageTiming]
    total_ms: float
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class AskResponse(BaseModel):
    question: str
    answer: str
    grounded: bool            # did the answer cite valid supporting passages?
    abstained: bool           # did the system decline for lack of evidence?
    sources: list[Source]
    trace: Trace
    provider: str
    model: str


class CorpusStats(BaseModel):
    documents: int
    chunks: int
    embedding_model: str
    reranker_model: str
    indexed: bool


class UploadResponse(BaseModel):
    doc_id: str          # the uploaded file's name
    documents: int       # how many docs are now in this session's corpus
    chunks: int          # how many chunks were indexed
    using_upload: bool   # true once a session has its own docs

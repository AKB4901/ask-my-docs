"""
Central configuration.

Every tunable lives here and can be overridden with an environment variable
(or a `.env` file). Keeping retrieval knobs in one typed place is what lets the
evaluation pipeline pin a configuration and detect regressions when it changes.
"""

from __future__ import annotations

from functools import lru_cache
from pathlib import Path

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict

ROOT_DIR = Path(__file__).resolve().parent.parent


class Settings(BaseSettings):
    model_config = SettingsConfigDict(
        env_file=".env", env_file_encoding="utf-8", extra="ignore"
    )

    # ---- LLM provider ---------------------------------------------------
    # "groq"   -> free, fast hosted inference (needs GROQ_API_KEY)
    # "ollama" -> fully local, zero-network inference (needs Ollama running)
    llm_provider: str = Field(default="groq")

    groq_api_key: str = Field(default="")
    groq_model: str = Field(default="openai/gpt-oss-20b")
    groq_base_url: str = Field(default="https://api.groq.com/openai/v1")

    ollama_model: str = Field(default="llama3.1:8b")
    ollama_base_url: str = Field(default="http://localhost:11434")

    llm_temperature: float = Field(default=0.1)
    llm_max_tokens: int = Field(default=700)
    llm_timeout_s: float = Field(default=60.0)

    # ---- Retrieval models (run locally, CPU-friendly, <6GB VRAM) --------
    embedding_model: str = Field(default="sentence-transformers/all-MiniLM-L6-v2")
    reranker_model: str = Field(default="cross-encoder/ms-marco-MiniLM-L-6-v2")

    # ---- Chunking -------------------------------------------------------
    chunk_size: int = Field(default=900)      # characters
    chunk_overlap: int = Field(default=150)   # characters

    # ---- Hybrid retrieval knobs ----------------------------------------
    bm25_top_k: int = Field(default=20)       # lexical candidates
    vector_top_k: int = Field(default=20)     # semantic candidates
    rrf_k: int = Field(default=60)            # Reciprocal Rank Fusion constant
    rerank_candidates: int = Field(default=20)  # fed into the cross-encoder
    final_top_k: int = Field(default=4)       # passages sent to the LLM

    # A retrieved passage is only trustworthy enough to answer from if the
    # reranker score clears this bar. Below it, we prefer to abstain.
    min_rerank_score: float = Field(default=-6.0)

    # ---- Paths ----------------------------------------------------------
    data_dir: Path = Field(default=ROOT_DIR / "data" / "docs")
    index_dir: Path = Field(default=ROOT_DIR / "data" / "index")

    # ---- Server ---------------------------------------------------------
    host: str = Field(default="0.0.0.0")
    port: int = Field(default=8000)

    @property
    def uses_groq(self) -> bool:
        return self.llm_provider.lower() == "groq"


@lru_cache
def get_settings() -> Settings:
    """Cached singleton so the whole app shares one config instance."""
    return Settings()

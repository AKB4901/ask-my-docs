"""
Per-session uploaded documents.

When a visitor uploads a file, we build a private in-memory corpus for their
session and query that instead of the shared demo corpus. This keeps one
visitor's resume from leaking into another visitor's search results — a small
but real multi-tenancy concern for a public demo.

State is in-memory and ephemeral: it clears on restart, and there is a soft cap
on how many sessions we retain so a public link can't grow memory unbounded.
"""

from __future__ import annotations

import threading
from collections import OrderedDict

from app.config import Settings
from app.rag.pipeline import Corpus

MAX_SESSIONS = 200  # oldest sessions are evicted beyond this


class SessionStore:
    def __init__(self, settings: Settings):
        self.settings = settings
        self._docs: OrderedDict[str, dict[str, str]] = OrderedDict()
        self._corpus: dict[str, Corpus] = {}
        self._lock = threading.Lock()

    def add_document(self, sid: str, doc_id: str, text: str) -> Corpus:
        """Add a document to a session and (re)build its private corpus."""
        with self._lock:
            docs = self._docs.setdefault(sid, {})
            docs[doc_id] = text
            self._docs.move_to_end(sid)
            corpus = Corpus.from_documents(docs, self.settings)
            self._corpus[sid] = corpus
            self._evict_if_needed()
            return corpus

    def get_corpus(self, sid: str) -> Corpus | None:
        return self._corpus.get(sid)

    def clear(self, sid: str) -> None:
        with self._lock:
            self._docs.pop(sid, None)
            self._corpus.pop(sid, None)

    def _evict_if_needed(self) -> None:
        while len(self._docs) > MAX_SESSIONS:
            old_sid, _ = self._docs.popitem(last=False)
            self._corpus.pop(old_sid, None)

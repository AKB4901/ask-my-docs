"""
Chunking.

We split on paragraph boundaries first and only fall back to hard character
cuts when a single paragraph is longer than the target size. Overlap keeps
context from being severed exactly at a chunk boundary, which is a common cause
of "the answer was retrieved but half of it was missing".
"""

from __future__ import annotations

import re
from dataclasses import dataclass, field


@dataclass
class Chunk:
    chunk_id: str
    doc_id: str
    text: str
    ordinal: int
    metadata: dict = field(default_factory=dict)


_PARAGRAPH_SPLIT = re.compile(r"\n\s*\n")


def _hard_split(text: str, size: int, overlap: int) -> list[str]:
    """Slice an over-long block into overlapping windows."""
    if len(text) <= size:
        return [text]
    step = max(1, size - overlap)
    windows = []
    for start in range(0, len(text), step):
        window = text[start : start + size].strip()
        if window:
            windows.append(window)
        if start + size >= len(text):
            break
    return windows


def chunk_document(
    doc_id: str, text: str, *, chunk_size: int, chunk_overlap: int
) -> list[Chunk]:
    """Turn one document's text into a list of Chunks.

    Paragraphs are greedily packed until adding the next one would exceed
    ``chunk_size``; the tail of each packed chunk is carried into the next as
    overlap so no idea is cut cleanly in half.
    """
    if chunk_overlap >= chunk_size:
        raise ValueError("chunk_overlap must be smaller than chunk_size")

    paragraphs = [p.strip() for p in _PARAGRAPH_SPLIT.split(text) if p.strip()]
    raw_chunks: list[str] = []
    buffer = ""

    for para in paragraphs:
        if len(para) > chunk_size:
            # Flush what we have, then hard-split the giant paragraph.
            if buffer:
                raw_chunks.append(buffer.strip())
                buffer = ""
            raw_chunks.extend(_hard_split(para, chunk_size, chunk_overlap))
            continue

        candidate = f"{buffer}\n\n{para}".strip() if buffer else para
        if len(candidate) <= chunk_size:
            buffer = candidate
        else:
            raw_chunks.append(buffer.strip())
            # Start the next buffer with the overlap tail of the previous one.
            tail = buffer[-chunk_overlap:] if chunk_overlap else ""
            buffer = f"{tail}\n\n{para}".strip()

    if buffer.strip():
        raw_chunks.append(buffer.strip())

    return [
        Chunk(
            chunk_id=f"{doc_id}::{i}",
            doc_id=doc_id,
            text=chunk,
            ordinal=i,
            metadata={"chars": len(chunk)},
        )
        for i, chunk in enumerate(raw_chunks)
        if chunk
    ]

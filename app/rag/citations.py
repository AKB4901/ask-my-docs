"""
Citation enforcement.

The single most important property of a production RAG system is that it does
not invent answers. We enforce this on two sides:

1. *Prompt* — the model is instructed to answer only from the numbered sources
   and to cite them inline as [1], [2]; if the sources don't cover the question
   it must say so with the exact abstention sentence.
2. *Validation* — after generation we parse the [n] markers, drop any that
   point at a source that wasn't provided, and decide whether the answer is
   "grounded" (cites at least one real source) or an explicit abstention.

An answer that is neither grounded nor an abstention is flagged ungrounded so
the UI can warn the user rather than present a confident hallucination.
"""

from __future__ import annotations

import re
from dataclasses import dataclass

ABSTAIN_SENTENCE = (
    "I couldn't find enough information in the provided documents to answer that."
)

_CITATION_RE = re.compile(r"\[(\d+)\]")

SYSTEM_PROMPT = f"""You are a precise assistant that answers questions strictly from a set of numbered source passages.

Rules:
- Use ONLY information contained in the sources below. Never use outside knowledge.
- Cite every claim inline with the source number in square brackets, e.g. [1] or [2].
- You may cite multiple sources for one claim, e.g. [1][3].
- If the sources do not contain the answer, reply with exactly this sentence and nothing else: "{ABSTAIN_SENTENCE}"
- Be concise and factual. Do not add caveats, apologies, or meta-commentary.
"""


def build_user_prompt(question: str, passages: list[str]) -> str:
    """Render the numbered-source context block plus the question."""
    blocks = []
    for i, text in enumerate(passages, start=1):
        blocks.append(f"[{i}] {text.strip()}")
    context = "\n\n".join(blocks)
    return f"Sources:\n{context}\n\nQuestion: {question}\n\nAnswer (cite sources inline):"


@dataclass
class CitationResult:
    answer: str
    cited_indices: list[int]   # 1-based, validated against available sources
    grounded: bool
    abstained: bool


def validate_answer(answer: str, num_sources: int) -> CitationResult:
    """Parse and validate citations in a generated answer.

    Any citation pointing outside 1..num_sources is stripped from the text so a
    user never sees a dangling reference.
    """
    answer = answer.strip()

    abstained = ABSTAIN_SENTENCE.lower() in answer.lower() and len(answer) < len(
        ABSTAIN_SENTENCE
    ) + 40

    raw = [int(m) for m in _CITATION_RE.findall(answer)]
    valid = sorted({n for n in raw if 1 <= n <= num_sources})
    invalid = {n for n in raw if n < 1 or n > num_sources}

    # Remove references to sources that don't exist.
    cleaned = answer
    for bad in invalid:
        cleaned = cleaned.replace(f"[{bad}]", "")
    cleaned = re.sub(r"\s{2,}", " ", cleaned).strip()

    grounded = (not abstained) and len(valid) > 0

    return CitationResult(
        answer=cleaned,
        cited_indices=valid,
        grounded=grounded,
        abstained=abstained,
    )

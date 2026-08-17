"""
Lightweight per-request tracing.

Every question is timed stage-by-stage (embed, lexical, vector, fuse, rerank,
generate, verify). We surface that breakdown to the user *and* log it as one
structured JSON line per request. That per-stage latency plus token/cost is the
raw material an observability/eval system (like an agent-reliability harness)
consumes to spot regressions — so it's a first-class output here, not an
afterthought bolted on later.
"""

from __future__ import annotations

import json
import logging
import time
from contextlib import contextmanager
from dataclasses import dataclass, field

logger = logging.getLogger("ask_my_docs.trace")


@dataclass
class Trace:
    stages: list[tuple[str, float]] = field(default_factory=list)
    prompt_tokens: int = 0
    completion_tokens: int = 0
    cost_usd: float = 0.0
    _t0: float = field(default_factory=time.perf_counter)

    @contextmanager
    def stage(self, name: str):
        start = time.perf_counter()
        try:
            yield
        finally:
            self.stages.append((name, (time.perf_counter() - start) * 1000))

    @property
    def total_ms(self) -> float:
        return (time.perf_counter() - self._t0) * 1000

    def set_usage(self, prompt_tokens: int, completion_tokens: int, cost_usd: float):
        self.prompt_tokens = prompt_tokens
        self.completion_tokens = completion_tokens
        self.cost_usd = cost_usd

    def emit(self, *, question: str, grounded: bool, abstained: bool):
        """Write one structured log line. Cheap to ship to any log aggregator."""
        record = {
            "event": "ask",
            "question": question[:200],
            "grounded": grounded,
            "abstained": abstained,
            "total_ms": round(self.total_ms, 1),
            "stages_ms": {n: round(ms, 1) for n, ms in self.stages},
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "cost_usd": self.cost_usd,
        }
        logger.info(json.dumps(record))
        return record

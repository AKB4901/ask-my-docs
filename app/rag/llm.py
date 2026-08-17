"""
LLM provider abstraction.

Two backends behind one interface so the project runs on any budget:
  - Groq   : free hosted API, very fast (default). Needs GROQ_API_KEY.
  - Ollama : fully local/offline. Needs `ollama serve` + a pulled model.

Both return a normalized ``LLMResult`` with token usage so the observability
layer can estimate cost per request.
"""

from __future__ import annotations

from dataclasses import dataclass

import requests

from app.config import Settings

# Approximate Groq pricing (USD per 1M tokens) for cost estimation only.
# Kept as a small table so the number in the trace is honest, not invented.
_GROQ_PRICING = {
    "openai/gpt-oss-20b": (0.10, 0.50),
    "openai/gpt-oss-120b": (0.15, 0.75),
    "llama-3.3-70b-versatile": (0.59, 0.79),
    "llama-3.1-8b-instant": (0.05, 0.08),
}


@dataclass
class LLMResult:
    text: str
    prompt_tokens: int
    completion_tokens: int
    cost_usd: float


class LLMError(RuntimeError):
    pass


def _estimate_cost(model: str, prompt_tokens: int, completion_tokens: int) -> float:
    in_rate, out_rate = _GROQ_PRICING.get(model, (0.0, 0.0))
    return round(
        (prompt_tokens * in_rate + completion_tokens * out_rate) / 1_000_000, 6
    )


def _generate_groq(settings: Settings, system: str, user: str) -> LLMResult:
    if not settings.groq_api_key:
        raise LLMError(
            "GROQ_API_KEY is not set. Add it to .env, or set LLM_PROVIDER=ollama "
            "to run fully locally."
        )
    resp = requests.post(
        f"{settings.groq_base_url}/chat/completions",
        headers={"Authorization": f"Bearer {settings.groq_api_key}"},
        json={
            "model": settings.groq_model,
            "temperature": settings.llm_temperature,
            "max_tokens": settings.llm_max_tokens,
            "messages": [
                {"role": "system", "content": system},
                {"role": "user", "content": user},
            ],
        },
        timeout=settings.llm_timeout_s,
    )
    if resp.status_code != 200:
        raise LLMError(f"Groq API error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = data["choices"][0]["message"]["content"]
    usage = data.get("usage", {})
    pt = int(usage.get("prompt_tokens", 0))
    ct = int(usage.get("completion_tokens", 0))
    return LLMResult(text, pt, ct, _estimate_cost(settings.groq_model, pt, ct))


def _generate_ollama(settings: Settings, system: str, user: str) -> LLMResult:
    try:
        resp = requests.post(
            f"{settings.ollama_base_url}/api/chat",
            json={
                "model": settings.ollama_model,
                "stream": False,
                "options": {
                    "temperature": settings.llm_temperature,
                    "num_predict": settings.llm_max_tokens,
                },
                "messages": [
                    {"role": "system", "content": system},
                    {"role": "user", "content": user},
                ],
            },
            timeout=settings.llm_timeout_s,
        )
    except requests.ConnectionError as exc:
        raise LLMError(
            "Could not reach Ollama at "
            f"{settings.ollama_base_url}. Is `ollama serve` running?"
        ) from exc
    if resp.status_code != 200:
        raise LLMError(f"Ollama error {resp.status_code}: {resp.text[:300]}")
    data = resp.json()
    text = data["message"]["content"]
    pt = int(data.get("prompt_eval_count", 0))
    ct = int(data.get("eval_count", 0))
    # Local inference has no marginal dollar cost.
    return LLMResult(text, pt, ct, 0.0)


def generate(settings: Settings, system: str, user: str) -> LLMResult:
    if settings.uses_groq:
        return _generate_groq(settings, system, user)
    return _generate_ollama(settings, system, user)


def active_model(settings: Settings) -> str:
    return settings.groq_model if settings.uses_groq else settings.ollama_model

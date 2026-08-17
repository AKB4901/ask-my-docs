<div align="center">

# Ask My Docs

**A production-grade hybrid RAG system that gives answers you can check.**

Hybrid retrieval (BM25 + vectors) · cross-encoder reranking · enforced citations · abstention · CI-gated evaluation · per-request observability

</div>

---

## The problem

Most retrieval-augmented generation (RAG) demos are one embedding model, one vector search, and one prompt. They look impressive for thirty seconds and then fail the moment they meet real users:

- **They miss exact terms.** Pure vector search returns something *semantically close* to "error code TS-4019" instead of the passage that actually contains it.
- **They hallucinate confidently.** When the documents don't hold the answer, the model invents one — and nothing catches it.
- **They can't be trusted.** There's no way to see *which* passage an answer came from, so a user can't verify it.
- **They silently degrade.** Someone changes the chunk size or swaps a model, retrieval quality drops, and nobody notices until a customer complains.

For any real use — internal knowledge bases, support, compliance, legal — those four gaps are disqualifying. This project closes all four.

## What it does

Ask a question about a set of documents. Every answer is:

1. **Retrieved with a hybrid** of keyword search (BM25) and semantic search (dense vectors), so exact terms *and* paraphrases both land.
2. **Reranked** by a cross-encoder for precision, so only the most relevant passages reach the model.
3. **Forced to cite** the exact source passages it used — rendered as clickable references in the UI.
4. **Allowed to say "I don't know"** when the documents genuinely don't contain the answer, instead of hallucinating.

And every request is **traced** — you can see the latency of each stage, the tokens used, and the estimated cost.

You can query the built-in sample corpus, or **upload your own document** (PDF, txt, or md — a resume, a paper, a contract) and ask questions about that instead. Uploads are indexed at runtime and scoped privately to your browser session.

## How it works

```
                          ┌─────────────── question ───────────────┐
                          │                                         │
                  ┌───────▼────────┐                     ┌──────────▼─────────┐
                  │  BM25 (lexical) │                     │  Vectors (semantic) │
                  │  exact terms    │                     │  meaning / paraphrase│
                  └───────┬────────┘                     └──────────┬─────────┘
                          │                                         │
                          └──────────────┐         ┌────────────────┘
                                         ▼         ▼
                              ┌────────────────────────────┐
                              │  Reciprocal Rank Fusion     │   combine rankings
                              └──────────────┬─────────────┘
                                             ▼
                              ┌────────────────────────────┐
                              │  Cross-encoder reranker     │   precision: keep top 4
                              └──────────────┬─────────────┘
                                             ▼
                        below relevance floor? ──► abstain (no LLM call)
                                             │
                                             ▼
                              ┌────────────────────────────┐
                              │  LLM — cite-only prompt     │   Groq or local Ollama
                              └──────────────┬─────────────┘
                                             ▼
                              ┌────────────────────────────┐
                              │  Citation validation        │   grounded? valid refs?
                              └──────────────┬─────────────┘
                                             ▼
                                    answer + sources + trace
```

**Why hybrid + rerank?** Lexical and semantic search fail in opposite ways — BM25 misses paraphrases, vectors miss exact tokens, IDs, and rare terms. Fusing them recovers both. The cross-encoder then reads the query and each candidate *together*, scoring true relevance far more accurately than first-stage similarity — so we can send the model 4 passages instead of 15.

**Why citations and abstention?** A production answer has to be *checkable*. The model is prompted to answer only from numbered sources and cite them inline; a validation pass then confirms the citations point at real passages and flags any answer that isn't grounded. If nothing clears the relevance bar, the system abstains before spending an LLM call.

## Architecture

| Layer | Choice | Why |
|---|---|---|
| API + UI | FastAPI, one process | Clone → run, no separate frontend build |
| Embeddings | `all-MiniLM-L6-v2` (local) | Small, CPU-friendly, runs under 6 GB VRAM |
| Vector index | FAISS `IndexFlatIP` | Exact cosine search; instant at this corpus size |
| Lexical | BM25 (`rank-bm25`) | Catches exact terms vectors miss |
| Fusion | Reciprocal Rank Fusion | Scale-free, robust, dependency-light |
| Reranker | `ms-marco-MiniLM-L-6-v2` cross-encoder | Precision over the fused candidates |
| LLM | Groq (free) or Ollama (local) | Zero-budget by default, or fully offline |
| Eval | Gold set + threshold gate | Turns "quality" into a CI check |
| Observability | Per-stage timing + structured logs | The raw signal for regression detection |

## Quick start

```bash
git clone <your-repo-url> ask-my-docs && cd ask-my-docs
pip install -r requirements.txt

cp .env.example .env
# then either:
#   - set GROQ_API_KEY in .env  (free key: https://console.groq.com/keys), or
#   - set LLM_PROVIDER=ollama   (fully local; needs `ollama serve`)

python run.py            # builds the index on first run, then serves
```

Open **http://localhost:8000**. The first run downloads the embedding and reranker models (~170 MB) once; after that it starts instantly.

```bash
make test    # unit tests
make eval    # the evaluation quality gate
make ingest  # rebuild the index after editing data/docs
make docker  # build + run in a container
```

## Bring your own documents

Drop `.md` or `.txt` files into `data/docs/`, run `make ingest`, and ask away. The included sample corpus is a small company handbook (onboarding, security, leave, expenses) chosen to show hybrid retrieval doing real work — exact lookups like "minimum password length" alongside paraphrased questions like "can I paste customer data into an AI tool?".

## Evaluation as a build gate

`eval/gold.jsonl` holds question → expected-source pairs. `python -m eval.run_eval` scores:

- **Retrieval** — `hit@k` and `MRR` (no LLM needed; always gates CI)
- **End-to-end** — answer keyword recall and citation validity (when an LLM is configured)

Every metric has a floor in `eval/thresholds.yaml`. Drop below it — by changing chunk size, swapping a model, tweaking `k` — and the run exits non-zero and **CI fails the build**. This is the single most under-shown skill in RAG portfolios, so it's deliberately front-and-center here.

## Outcomes

- **Grounded, checkable answers** — every claim traces to a passage the user can open.
- **No confident hallucinations** — the system abstains instead of inventing.
- **Regressions can't merge** — the eval gate protects retrieval quality automatically.
- **Full cost/latency visibility** — every request is traced end to end.
- **Runs on a zero budget** — free Groq tier or fully local, under 6 GB VRAM.

## Project layout

```
ask-my-docs/
├── app/
│   ├── main.py              FastAPI app + routes + static UI
│   ├── config.py            typed settings (every knob lives here)
│   ├── schemas.py           API request/response contracts
│   ├── rag/
│   │   ├── chunking.py      paragraph-aware chunking with overlap
│   │   ├── ingest.py        build + persist both indexes
│   │   ├── embeddings.py    local dense embeddings
│   │   ├── vector_store.py  FAISS wrapper
│   │   ├── bm25_store.py    lexical index
│   │   ├── hybrid.py        Reciprocal Rank Fusion
│   │   ├── reranker.py      cross-encoder reranking
│   │   ├── llm.py           Groq / Ollama provider abstraction
│   │   ├── citations.py     cite-only prompt + validation
│   │   ├── loaders.py       PDF / txt / md upload parsing
│   │   └── pipeline.py      the orchestrator + Corpus
│   ├── sessions.py          per-visitor uploaded corpora
│   ├── observability/
│   │   └── tracing.py       per-stage timing + structured logs
│   └── static/              index.html · styles.css · app.js
├── data/docs/               sample corpus (bring your own)
├── eval/                    gold set, thresholds, gate runner
├── tests/                   unit tests
└── .github/workflows/ci.yml lint → test → build → eval gate
```

## Tech stack

Python · FastAPI · FAISS · sentence-transformers · rank-bm25 · Groq / Ollama · pytest · GitHub Actions · Docker

## License

MIT — see [LICENSE](LICENSE).

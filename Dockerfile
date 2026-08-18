FROM python:3.11-slim

WORKDIR /app

# fastembed/faiss ship prebuilt wheels, so no build toolchain is needed.
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Build the index AND warm both ONNX models at image-build time, so the models
# are baked into the image and the first user request isn't slowed by a
# download. Keeps cold starts as short as a free host allows.
RUN python -m app.rag.ingest && \
    python -c "from app.rag.embeddings import embed_texts; from app.rag.reranker import rerank; embed_texts(['warmup']); rerank('q', [('a', 'warmup passage')])"

EXPOSE 8000

# Render (and most PaaS) inject $PORT; fall back to 8000 for local `docker run`.
CMD ["sh", "-c", "uvicorn app.main:app --host 0.0.0.0 --port ${PORT:-8000}"]

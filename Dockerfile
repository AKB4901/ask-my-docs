FROM python:3.11-slim

WORKDIR /app

# System deps kept minimal; faiss-cpu and torch wheels are self-contained.
RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

COPY . .

# Pre-build the index at image build time so the container starts instantly.
RUN python -m app.rag.ingest

EXPOSE 8000
CMD ["uvicorn", "app.main:app", "--host", "0.0.0.0", "--port", "8000"]

FROM python:3.10-slim

WORKDIR /app

RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .

RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu

RUN pip install --no-cache-dir -r requirements.txt

# E5 base (embeddings tristan + encoder del reranker)
RUN python -c "from transformers import AutoTokenizer, AutoModel; AutoTokenizer.from_pretrained('intfloat/multilingual-e5-base').save_pretrained('./hf_model'); AutoModel.from_pretrained('intfloat/multilingual-e5-base').save_pretrained('./hf_model')"

COPY src/ ./src/

# Descargados por cloudbuild.yaml desde GCS antes del build
COPY data/chroma_db ./data/chroma_db
COPY rr_model ./rr_model

ENV CHROMA_PATH=/app/data/chroma_db
ENV CHROMA_COLLECTION=cosora_actas_e5
ENV BM25_PATH=/app/data/chroma_db/bm25.json
ENV HF_MODEL_PATH=/app/hf_model
ENV RR_MODEL_PATH=/app/rr_model
ENV EMBEDDING_STYLE=e5
ENV RERANK_ENABLED=1
ENV RERANK_POOL_N=20
ENV REBUILD_BM25=0
ENV RAG_MODE=cypher_transversal
ENV CYPHER_ROUTE=hybrid

EXPOSE 8080
HEALTHCHECK CMD curl --fail http://localhost:8080/_stcore/health

ENTRYPOINT ["streamlit", "run", "src/app.py", "--server.port=8080", "--server.address=0.0.0.0"]

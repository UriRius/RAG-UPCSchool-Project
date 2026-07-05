import os
from pathlib import Path

# src/rag/config.py → raíz del repo (/app en Docker)
APP_ROOT = Path(__file__).resolve().parent.parent.parent

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
CHROMA_PATH = os.getenv("CHROMA_PATH", str(APP_ROOT / "data" / "chroma_db"))
HF_MODEL_DIR = os.getenv("HF_MODEL_PATH", str(APP_ROOT / "hf_model"))
COLLECTION_NAME = "cosora_actas_e5"
COLLECTION_NAME_V2 = "cosora_actas_e5_v2"
COLLECTION_NAME_TRISTAN = "construction_site_visit_reports"
GCS_GRAPH_PREFIX = os.getenv("GCS_GRAPH_PREFIX", "graph")
DRIVE_GRAPH_FOLDER_ID = os.getenv("DRIVE_GRAPH_FOLDER_ID", "")
GRAPH_DIR = os.getenv("GRAPH_DIR", "./data/graph")

# RAG_MODE: v1 | v2 | v2_table | graph_baseline | cypher_transversal
GRAPH_MODES = ("graph_baseline", "cypher_transversal")
CLASSIC_MODES = ("v1", "v2", "v2_table")
RAG_MODES = (*CLASSIC_MODES, *GRAPH_MODES)
RAG_MODE = os.getenv("RAG_MODE", "v2").lower()

# Neo4j (Graph RAG)
NEO4J_URI = os.getenv("NEO4J_URI", "")
NEO4J_USER = os.getenv("NEO4J_USER", "neo4j")
NEO4J_PASSWORD = os.getenv("NEO4J_PASSWORD", "")
NEO4J_DATABASE = os.getenv("NEO4J_DATABASE", "neo4j")

# Graph catalog / sync (offline jobs — scripts/sync_neo4j_catalog.py)
LEGACY_BATCH1_JSON = "graph_actas_e5_original_sub10_bal_raw.json"
LEGACY_BATCH1_TAG = "sub10"
WIPE_ALL_ON_SYNC = os.getenv("WIPE_ALL_ON_SYNC", "false").lower() in ("1", "true", "yes")
FORCE_RELOAD_TAGS = [
    tag.strip()
    for tag in os.getenv("FORCE_RELOAD_TAGS", "").split(",")
    if tag.strip()
]

# GCP Storage Bucket (Para automatizar descargas de DB)
GCP_BUCKET_NAME = os.getenv("GCP_BUCKET_NAME", "")
MATCH_MIN_OVERLAP = float(os.getenv("MATCH_MIN_OVERLAP", "0.5"))
V1_RETRIEVAL_K = int(os.getenv("V1_RETRIEVAL_K", "5"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "50"))
TOP_N = int(os.getenv("TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_MIN_SCORE = float(os.getenv("RRF_MIN_SCORE", "0.01"))

# Embeddings + reranker (pipeline Tristan)
EMBEDDING_STYLE = os.getenv("EMBEDDING_STYLE", "tristan").lower()
RR_MODEL_PATH = os.getenv("RR_MODEL_PATH", str(APP_ROOT / "rr_model"))
RERANK_POOL_N = int(os.getenv("RERANK_POOL_N", "20"))
RERANK_ENABLED = os.getenv("RERANK_ENABLED", "1").lower() in ("1", "true", "yes")
QUERY_REWRITE_N = int(os.getenv("QUERY_REWRITE_N", "3"))
QUERY_REWRITE_ENABLED = os.getenv("QUERY_REWRITE_ENABLED", "0").lower() in ("1", "true", "yes")

# Graph RAG v2 — Cypher routes: template | llm | hybrid
CYPHER_ROUTE = os.getenv("CYPHER_ROUTE", "hybrid").lower()
CYPHER_LIMIT = int(os.getenv("CYPHER_LIMIT", "100"))
CYPHER_LLM_MODEL = os.getenv("CYPHER_LLM_MODEL", "gpt-4o-mini")
GRAPH_MERGE_MIN_HITS = int(os.getenv("GRAPH_MERGE_MIN_HITS", "1"))


def resolve_collection_name(mode: str | None = None) -> str:
    """Colección Chroma según modo RAG (CHROMA_COLLECTION tiene prioridad)."""
    override = os.getenv("CHROMA_COLLECTION")
    if override:
        return override
    rag_mode = (mode or RAG_MODE).lower()
    if rag_mode == "v2_table":
        return COLLECTION_NAME_V2
    return COLLECTION_NAME

BM25_FILENAME = "bm25.json"
BM25_FILENAME_V2 = "bm25_v2.json"

COLLECTION_BM25 = {
    COLLECTION_NAME: BM25_FILENAME,
    COLLECTION_NAME_V2: BM25_FILENAME_V2,
}


def bm25_path(chroma_path: str | None = None, *, collection_name: str | None = None) -> str:
    base = chroma_path or CHROMA_PATH
    override = os.getenv("BM25_PATH")
    if override:
        return override
    filename = COLLECTION_BM25.get(collection_name or "", BM25_FILENAME)
    return os.path.join(base, filename)

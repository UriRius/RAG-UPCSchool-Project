import os

EMBED_MODEL_NAME = os.getenv("EMBED_MODEL_NAME", "intfloat/multilingual-e5-base")
CHROMA_PATH = os.getenv("CHROMA_PATH", "./data/chroma_db")
COLLECTION_NAME = "cosora_actas_e5"
COLLECTION_NAME_V2 = "cosora_actas_e5_v2"

# RAG_MODE: v1 = baseline (top-5+5, BM25Okapi) | v2 = BM25Log1 + pool 50+50
RAG_MODE = os.getenv("RAG_MODE", "v2").lower()
V1_RETRIEVAL_K = int(os.getenv("V1_RETRIEVAL_K", "5"))

RETRIEVAL_K = int(os.getenv("RETRIEVAL_K", "50"))
TOP_N = int(os.getenv("TOP_N", "5"))
RRF_K = int(os.getenv("RRF_K", "60"))
RRF_MIN_SCORE = float(os.getenv("RRF_MIN_SCORE", "0.01"))


def resolve_collection_name() -> str:
    """Colección Chroma según RAG_MODE (CHROMA_COLLECTION tiene prioridad)."""
    override = os.getenv("CHROMA_COLLECTION")
    if override:
        return override
    if RAG_MODE == "v2_table":
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
    if override and collection_name is None:
        return override
    filename = COLLECTION_BM25.get(collection_name or "", BM25_FILENAME)
    return os.path.join(base, filename)

import json
import math
import re
from pathlib import Path

from rank_bm25 import BM25Okapi

TERM_PATTERN = re.compile(r"\b[a-zA-ZáéíóúñÁÉÍÓÚÑ]+\b")


class BM25Log1(BM25Okapi):
    """BM25 with non-negative IDF: log(1 + (N - df + 0.5) / (df + 0.5))."""

    def __init__(self, chunk_terms, chunk_ids, **kwargs):
        super().__init__(chunk_terms, **kwargs)
        self.chunk_ids = list(chunk_ids)

        df: dict[str, int] = {}
        for doc in self.doc_freqs:
            for term in set(doc):
                df[term] = df.get(term, 0) + 1

        n = self.corpus_size
        for term, dfi in df.items():
            self.idf[term] = math.log(1 + (n - dfi + 0.5) / (dfi + 0.5))

    @staticmethod
    def extract_terms(text: str) -> list[str]:
        return TERM_PATTERN.findall(text.lower())

    def save(self, path: str | Path) -> None:
        data = {
            "chunk_ids": self.chunk_ids,
            "idf": {k: float(v) for k, v in self.idf.items()},
            "doc_freqs": self.doc_freqs,
            "doc_len": self.doc_len,
            "avgdl": self.avgdl,
            "corpus_size": self.corpus_size,
            "k1": self.k1,
            "b": self.b,
        }
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as f:
            json.dump(data, f)

    @classmethod
    def load(cls, path: str | Path) -> "BM25Log1":
        with Path(path).open("r", encoding="utf-8") as f:
            data = json.load(f)

        bm25 = cls.__new__(cls)
        bm25.chunk_ids = data["chunk_ids"]
        bm25.idf = {k: float(v) for k, v in data["idf"].items()}
        bm25.doc_freqs = data["doc_freqs"]
        bm25.doc_len = data["doc_len"]
        bm25.avgdl = data["avgdl"]
        bm25.corpus_size = data["corpus_size"]
        bm25.k1 = data["k1"]
        bm25.b = data["b"]
        return bm25


def build_bm25_index(documents: list[str], chunk_ids: list[str]) -> BM25Log1:
    chunk_terms = [BM25Log1.extract_terms(doc) for doc in documents]
    return BM25Log1(chunk_terms, chunk_ids)


def load_or_build_bm25(
    chroma_path: str,
    documents: list[str],
    chunk_ids: list[str],
    *,
    bm25_file: str | None = None,
    logger=None,
) -> BM25Log1:
    from rag.config import bm25_path

    path = bm25_file or bm25_path(chroma_path)
    if Path(path).is_file():
        bm25 = BM25Log1.load(path)
        if len(bm25.chunk_ids) == len(documents):
            if logger:
                logger.info("BM25 cargado desde %s (%d docs)", path, len(bm25.chunk_ids))
            return bm25
        if logger:
            logger.warning(
                "bm25.json desincronizado (%d vs %d chunks); reconstruyendo.",
                len(bm25.chunk_ids),
                len(documents),
            )
    elif logger:
        logger.warning("bm25.json no encontrado en %s; reconstruyendo desde Chroma.", path)

    return build_bm25_index(documents, chunk_ids)

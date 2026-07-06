"""Query rewriting para la rama híbrida (BM25 + E5)."""

from __future__ import annotations

from typing import TYPE_CHECKING

from rag.bm25_index import BM25Log1
from rag.config import CYPHER_LLM_MODEL, QUERY_REWRITE_N, RERANK_POOL_N, RETRIEVAL_K, RRF_K, TOP_N
from rag.embeddings import E5Embedder
from rag.retrieval import apply_reranker, retrieve_hybrid

if TYPE_CHECKING:
    from e5_reranker import E5Reranker
    from openai import OpenAI

_REWRITE_PROMPT = """Eres un experto en sistemas RAG aplicados a actas de visita semanal de obras de construcción ferroviaria española.

Tu tarea es reescribir la siguiente pregunta generando {n} variantes que mejoren la recuperación en un sistema de búsqueda híbrida (BM25 + embeddings semánticos). Cada variante cumple un rol distinto.

CONTEXTO DEL CORPUS: Actas semanales de obra para mejora de accesibilidad en estación ferroviaria (rampas, ascensores, edículos, andenes, edificio de viajeros). Roles habituales: DC, DF, DO, DEO, UTE, CSS. Elementos frecuentes: micropilotes, encofrado, hormigonado, impermeabilización, solera, arquetas, canalizaciones ADIF/RENFE, pilares metálicos HEB, pletina, losa, batache, pavimento podoTáctil.

TIPOS DE VARIANTES (genera una por tipo, en este orden):
1. KEYWORDS — Lista de términos técnicos exactos separados por comas: sustantivos, códigos de plano, materiales, roles y estados del dominio. Sin forma de pregunta.
2. EXPANSIÓN SEMÁNTICA — Reformula la pregunta con sinónimos y conceptos relacionados que capturen el mismo significado desde ángulos distintos. Útil para embeddings.
3. ESPECIFICACIÓN TÉCNICA — Versión más concreta que introduce terminología de construcción, estructura o instalaciones precisa al elemento preguntado.
4. PERSPECTIVA DE AGENTE O ESTADO — Reformula desde el punto de vista del responsable (DF, UTE, DO...) o del estado del elemento (ejecutado, pendiente, aprobado, incidencia abierta...).

PREGUNTA ORIGINAL: {query}

Devuelve exactamente {n} variantes en el orden de los tipos anteriores, una por línea, sin numeración ni prefijos. Solo en español."""


def rewrite_query(query: str, openai_client: "OpenAI", *, n: int = QUERY_REWRITE_N) -> list[str]:
    """Genera n variantes de la pregunta vía LLM (solo rama híbrida)."""
    prompt = _REWRITE_PROMPT.format(n=n, query=query)
    resp = openai_client.chat.completions.create(
        model=CYPHER_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.7,
        max_tokens=400,
    )
    content = (resp.choices[0].message.content or "").strip()
    lines = [line.strip() for line in content.split("\n") if line.strip()]
    return lines[:n]


def retrieve_multi_hybrid(
    queries: list[str],
    collection,
    embedder: E5Embedder,
    bm25: BM25Log1,
    all_docs: list[str],
    all_metas: list[dict],
    *,
    original_query: str,
    retrieval_k: int = RETRIEVAL_K,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
    reranker: "E5Reranker | None" = None,
    rerank_pool_n: int = RERANK_POOL_N,
) -> list[dict]:
    """
    Retrieval híbrido sobre varias queries + RRF de 2º nivel.
    Reranker (si activo) se aplica una vez sobre el pool fusionado.
    """
    scores: dict[str, dict] = {}
    for q in queries:
        hits = retrieve_hybrid(
            q,
            collection,
            embedder,
            bm25,
            all_docs,
            all_metas,
            retrieval_k=retrieval_k,
            top_n=retrieval_k,
            rrf_k=rrf_k,
            reranker=None,
        )
        for rank, hit in enumerate(hits):
            chunk_id = hit["meta"]["chunk_id"]
            if chunk_id not in scores:
                scores[chunk_id] = {**hit, "score": 0.0}
            scores[chunk_id]["score"] += 1.0 / (rrf_k + rank + 1)

    ranked = sorted(scores.values(), key=lambda item: item["score"], reverse=True)
    pool_n = rerank_pool_n if reranker is not None else top_n
    pool = ranked[:pool_n]
    if reranker is not None:
        return apply_reranker(pool, original_query, reranker, top_k=top_n)
    return pool[:top_n]


def hybrid_retrieve(
    query: str,
    collection,
    embedder: E5Embedder,
    bm25: BM25Log1,
    all_docs: list[str],
    all_metas: list[dict],
    *,
    openai_client: "OpenAI | None" = None,
    query_rewrite: bool = False,
    rewrite_n: int = QUERY_REWRITE_N,
    retrieval_k: int = RETRIEVAL_K,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
    reranker: "E5Reranker | None" = None,
    rerank_pool_n: int = RERANK_POOL_N,
) -> tuple[list[dict], dict]:
    """Rama híbrida con rewriting opcional. Devuelve hits + debug."""
    debug: dict = {"query_rewrite_used": False, "rewritten_queries": []}
    if query_rewrite and openai_client is not None:
        rewritten = rewrite_query(query, openai_client, n=rewrite_n)
        debug["rewritten_queries"] = rewritten
        debug["query_rewrite_used"] = True
        hits = retrieve_multi_hybrid(
            [query] + rewritten,
            collection,
            embedder,
            bm25,
            all_docs,
            all_metas,
            original_query=query,
            retrieval_k=retrieval_k,
            top_n=top_n,
            rrf_k=rrf_k,
            reranker=reranker,
            rerank_pool_n=rerank_pool_n,
        )
        return hits, debug

    hits = retrieve_hybrid(
        query,
        collection,
        embedder,
        bm25,
        all_docs,
        all_metas,
        retrieval_k=retrieval_k,
        top_n=top_n,
        rrf_k=rrf_k,
        reranker=reranker,
        rerank_pool_n=rerank_pool_n,
    )
    return hits, debug

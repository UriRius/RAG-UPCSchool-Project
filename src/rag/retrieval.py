from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np
from rank_bm25 import BM25Okapi

from rag.bm25_index import BM25Log1
from rag.config import RERANK_POOL_N, RETRIEVAL_K, RRF_K, TOP_N, V1_RETRIEVAL_K
from rag.embeddings import E5Embedder

if TYPE_CHECKING:
    from e5_reranker import E5Reranker


def normalize_meta(meta: dict) -> dict:
    out = dict(meta)
    if "doc_id" not in out and "docx_id" in out:
        out["doc_id"] = out["docx_id"]
    return out


def apply_reranker(
    hits: list[dict],
    query: str,
    reranker: "E5Reranker",
    *,
    top_k: int,
) -> list[dict]:
    if not hits:
        return []
    passages = [h["text"] for h in hits]
    ranked = reranker.rerank(query, passages, top_k=min(top_k, len(passages)))
    used: set[int] = set()
    reranked: list[dict] = []
    for passage, score in ranked:
        for i, hit in enumerate(hits):
            if i in used or hit["text"] != passage:
                continue
            used.add(i)
            reranked.append(
                {
                    **hit,
                    "score": float(score),
                    "rerank_score": float(score),
                    "rrf_score": hit.get("score"),
                }
            )
            break
    return reranked


def rrf_score(ranks: list[int], k: int = RRF_K) -> float:
    if not ranks:
        raise ValueError("At least one rank is required.")
    score = 0.0
    for rank in ranks:
        if rank <= 0:
            raise ValueError("All ranks must be greater than 0.")
        score += 1.0 / (rank + k)
    return score


def semantic_search(
    query_text: str,
    collection,
    embedder: E5Embedder,
    top_k: int = RETRIEVAL_K,
) -> list[tuple[str, float]]:
    query_embedding = embedder.embed_query(query_text)
    chroma_results = collection.query(
        query_embeddings=[query_embedding],
        n_results=top_k,
        include=["distances"],
    )
    return [
        (chunk_id, 1.0 - distance)
        for chunk_id, distance in zip(
            chroma_results["ids"][0],
            chroma_results["distances"][0],
        )
    ]


def lexical_search(
    query_text: str,
    bm25: BM25Log1,
    top_k: int = RETRIEVAL_K,
) -> list[tuple[str, float]]:
    query_terms = BM25Log1.extract_terms(query_text)
    bm25_scores = bm25.get_scores(query_terms)
    sorted_idx = sorted(
        range(len(bm25_scores)),
        key=lambda i: bm25_scores[i],
        reverse=True,
    )[:top_k]
    return [(bm25.chunk_ids[i], float(bm25_scores[i])) for i in sorted_idx]


def hybrid_ranking(
    bm25_results: list[tuple[str, float]],
    semantic_results: list[tuple[str, float]],
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
) -> list[tuple[str, float, int | None, int | None]]:
    bm25_rank = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(bm25_results)}
    semantic_rank = {chunk_id: rank + 1 for rank, (chunk_id, _) in enumerate(semantic_results)}
    all_chunk_ids = set(bm25_rank) | set(semantic_rank)

    ranked: list[tuple[str, float, int | None, int | None]] = []
    for chunk_id in all_chunk_ids:
        ranks: list[int] = []
        b_rank = bm25_rank.get(chunk_id)
        s_rank = semantic_rank.get(chunk_id)
        if b_rank is not None:
            ranks.append(b_rank)
        if s_rank is not None:
            ranks.append(s_rank)
        ranked.append((chunk_id, rrf_score(ranks, k=rrf_k), b_rank, s_rank))

    ranked.sort(key=lambda x: x[1], reverse=True)
    return ranked[:top_n]


def retrieve_hybrid(
    query: str,
    collection,
    embedder: E5Embedder,
    bm25: BM25Log1,
    all_docs: list[str],
    all_metas: list[dict],
    *,
    retrieval_k: int = RETRIEVAL_K,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
    reranker: "E5Reranker | None" = None,
    rerank_pool_n: int = RERANK_POOL_N,
) -> list[dict]:
    pool_n = rerank_pool_n if reranker is not None else top_n
    semantic_results = semantic_search(query, collection, embedder, top_k=retrieval_k)
    bm25_results = lexical_search(query, bm25, top_k=retrieval_k)
    hybrid_results = hybrid_ranking(
        bm25_results, semantic_results, top_n=pool_n, rrf_k=rrf_k
    )

    id_to_idx = {meta["chunk_id"]: i for i, meta in enumerate(all_metas)}
    final_hits: list[dict] = []
    for chunk_id, score, bm25_r, sem_r in hybrid_results:
        idx = id_to_idx.get(chunk_id)
        if idx is None:
            continue
        final_hits.append(
            {
                "score": score,
                "text": all_docs[idx],
                "meta": normalize_meta(all_metas[idx]),
                "bm25_rank": bm25_r,
                "semantic_rank": sem_r,
            }
        )
    if reranker is not None:
        return apply_reranker(final_hits, query, reranker, top_k=top_n)
    return final_hits


def retrieve_hybrid_v1(
    query: str,
    collection,
    embedder: E5Embedder,
    bm25: BM25Okapi,
    all_docs: list[str],
    all_metas: list[dict],
    *,
    retrieval_k: int = V1_RETRIEVAL_K,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
) -> list[dict]:
    """Baseline v1: top-k dense + top-k BM25 (split), fusión RRF."""
    q_emb = embedder.embed_query(query)
    dense_results = collection.query(query_embeddings=[q_emb], n_results=retrieval_k)
    dense_ids = dense_results["ids"][0]

    tokenized_query = query.lower().split()
    bm25_scores = bm25.get_scores(tokenized_query)
    top_bm25_indices = np.argsort(bm25_scores)[::-1][:retrieval_k]

    rrf_scores: dict[str, float] = {}
    for rank, doc_id in enumerate(dense_ids):
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (rrf_k + rank + 1))
    for rank, idx in enumerate(top_bm25_indices):
        doc_id = all_metas[idx]["chunk_id"]
        rrf_scores[doc_id] = rrf_scores.get(doc_id, 0) + (1.0 / (rrf_k + rank + 1))

    sorted_rrf = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)[:top_n]
    id_to_idx = {meta["chunk_id"]: i for i, meta in enumerate(all_metas)}
    final_hits: list[dict] = []
    for doc_id, score in sorted_rrf:
        idx = id_to_idx.get(doc_id)
        if idx is None:
            continue
        final_hits.append(
            {
                "score": score,
                "text": all_docs[idx],
                "meta": normalize_meta(all_metas[idx]),
            }
        )
    return final_hits


def retrieve(
    query: str,
    mode: str,
    collection,
    embedder: E5Embedder,
    bm25_v2: BM25Log1 | None,
    bm25_v1: BM25Okapi | None,
    all_docs: list[str],
    all_metas: list[dict],
    *,
    retrieval_k: int = RETRIEVAL_K,
    top_n: int = TOP_N,
    rrf_k: int = RRF_K,
    reranker: "E5Reranker | None" = None,
    rerank_pool_n: int = RERANK_POOL_N,
) -> list[dict]:
    """Despacha al pipeline v1 o v2 según RAG_MODE (v2 y v2_table usan retrieval v2)."""
    if mode == "v1":
        if bm25_v1 is None:
            raise ValueError("bm25_v1 requerido para RAG_MODE=v1")
        return retrieve_hybrid_v1(
            query,
            collection,
            embedder,
            bm25_v1,
            all_docs,
            all_metas,
            retrieval_k=retrieval_k,
            top_n=top_n,
            rrf_k=rrf_k,
        )
    if bm25_v2 is None:
        raise ValueError("bm25_v2 requerido para RAG_MODE v2")
    return retrieve_hybrid(
        query,
        collection,
        embedder,
        bm25_v2,
        all_docs,
        all_metas,
        retrieval_k=retrieval_k,
        top_n=top_n,
        rrf_k=rrf_k,
        reranker=reranker,
        rerank_pool_n=rerank_pool_n,
    )

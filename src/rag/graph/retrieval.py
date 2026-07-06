from __future__ import annotations

import logging

from neo4j import Driver
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag.bm25_index import BM25Log1
from rag.config import GRAPH_MERGE_MIN_HITS, RRF_K, TOP_N
from rag.embeddings import E5Embedder
from rag.graph.cypher import cypher_template_route, rows_to_triples
from rag.graph.cypher_llm import resolve_cypher_query
from rag.graph.neo4j_client import run_query
from rag.graph.text_utils import normalize_doc_key, parse_legacy_chunk_id
from rag.retrieval import retrieve

logger = logging.getLogger(__name__)


def build_chunks_by_doc(
    chunk_by_id: dict[str, tuple[str, dict]],
) -> dict[str, list[tuple[str, str, dict]]]:
    """Índice docx_id normalizado → [(chunk_id, text, meta), ...] por chunk_index."""
    by_doc: dict[str, list[tuple[str, str, dict]]] = {}
    for chunk_id, (text, meta) in chunk_by_id.items():
        doc_key = normalize_doc_key(meta.get("docx_id") or meta.get("doc_id") or "")
        if not doc_key:
            continue
        by_doc.setdefault(doc_key, []).append((chunk_id, text, meta))
    for chunks in by_doc.values():
        chunks.sort(key=lambda item: item[2].get("chunk_index", 0))
    return by_doc


def resolve_doc_chunks(
    source_doc: str,
    chunks_by_doc: dict[str, list[tuple[str, str, dict]]],
) -> list[tuple[str, str, dict]]:
    """Resuelve source_doc de Neo4j a chunks Tristan (match exacto o por prefijo)."""
    key = normalize_doc_key(source_doc)
    if not key:
        return []
    if key in chunks_by_doc:
        return chunks_by_doc[key]

    best: list[tuple[str, str, dict]] = []
    best_len = 0
    for doc_key, chunks in chunks_by_doc.items():
        if key in doc_key or doc_key in key:
            if len(doc_key) > best_len:
                best_len = len(doc_key)
                best = chunks
    return best


def chunks_from_provenance(
    records: list[dict],
    chunk_by_id: dict[str, tuple[str, dict]],
    *,
    max_chunks: int = TOP_N,
    max_chunks_per_doc: int = 2,
    bridge_stats: dict | None = None,
) -> list[dict]:
    """
    Triples Neo4j → chunks Chroma.

    1. ID exacto en chunk_by_id (legacy o UUID).
    2. ID legacy ``doc__cNNNN`` → chunk_index en el mismo docx_id.
    3. Fallback por ``source_doc`` → docx_id (índice Tristan distinto al del grafo).
    """
    chunks_by_doc = build_chunks_by_doc(chunk_by_id)
    seen: set[str] = set()
    hits: list[dict] = []
    stats = {"exact": 0, "legacy_index": 0, "by_doc": 0, "miss": 0}

    def add_hit(text: str, meta: dict, bridge: str) -> bool:
        chunk_id = meta.get("chunk_id") or ""
        if not chunk_id or chunk_id in seen:
            return False
        seen.add(chunk_id)
        hits.append(
            {
                "text": text,
                "meta": {**meta, "graph_bridge": bridge},
                "score": 0.0,
                "from_graph": True,
            }
        )
        stats[bridge] += 1
        return True

    for row in records:
        if "cnt" in row and "subject" not in row:
            continue
        if len(hits) >= max_chunks:
            break

        legacy_id = (row.get("source_chunk_id") or "").strip()
        if legacy_id in chunk_by_id:
            text, meta = chunk_by_id[legacy_id]
            add_hit(text, meta, "exact")
            continue

        parsed = parse_legacy_chunk_id(legacy_id)
        if parsed:
            doc_key, legacy_idx = parsed
            doc_chunks = chunks_by_doc.get(doc_key) or resolve_doc_chunks(doc_key, chunks_by_doc)
            matched = False
            for chunk_id, text, meta in doc_chunks:
                idx = meta.get("chunk_index")
                if idx in (legacy_idx, legacy_idx - 1):
                    if add_hit(text, meta, "legacy_index"):
                        matched = True
                        break
            if matched:
                continue

        source_doc = row.get("source_doc") or (parsed[0] if parsed else "")
        doc_chunks = resolve_doc_chunks(source_doc, chunks_by_doc)
        added = 0
        for _chunk_id, text, meta in doc_chunks:
            if add_hit(text, meta, "by_doc"):
                added += 1
            if added >= max_chunks_per_doc or len(hits) >= max_chunks:
                break
        if not added and legacy_id:
            stats["miss"] += 1

    if bridge_stats is not None:
        bridge_stats.update(stats)
        bridge_stats["n_hits"] = len(hits)

    return hits


def rrf_merge_lists(
    graph_hits: list[dict],
    baseline_hits: list[dict],
    *,
    rrf_k: int = RRF_K,
    top_n: int = TOP_N,
) -> list[dict]:
    scores: dict[str, dict] = {}
    for rank, hit in enumerate(graph_hits):
        chunk_id = hit["meta"]["chunk_id"]
        scores.setdefault(chunk_id, {**hit, "score": 0.0})
        scores[chunk_id]["score"] += 1.0 / (rrf_k + rank + 1)
    for rank, hit in enumerate(baseline_hits):
        chunk_id = hit["meta"]["chunk_id"]
        scores.setdefault(chunk_id, {**hit, "score": 0.0})
        scores[chunk_id]["score"] += 1.0 / (rrf_k + rank + 1)
    return sorted(scores.values(), key=lambda item: item["score"], reverse=True)[:top_n]


def execute_cypher(
    driver: Driver,
    query: str,
    *,
    route: str | None = None,
    openai_client: OpenAI | None = None,
) -> tuple[list[dict], str, str]:
    route_eff = (route or "hybrid").lower()
    cypher, params, route_used = resolve_cypher_query(
        query, route=route, openai_client=openai_client
    )
    rows = run_query(driver, cypher, **params)
    if route_eff == "hybrid" and not rows and route_used == "llm":
        logger.info("hybrid: LLM 0 filas → fallback template")
        cypher, params = cypher_template_route(query)
        rows = run_query(driver, cypher, **params)
        route_used = "template"
    return rows, cypher, route_used


def retrieve_cypher_rag(
    query: str,
    *,
    driver: Driver,
    chunk_by_id: dict[str, tuple[str, dict]],
    collection,
    embedder: E5Embedder,
    bm25_v2: BM25Log1,
    bm25_v1: BM25Okapi,
    all_docs: list[str],
    all_metas: list[dict],
    top_n: int = TOP_N,
    retrieval_k: int | None = None,
    rrf_k: int = RRF_K,
    graph_merge_min_hits: int = GRAPH_MERGE_MIN_HITS,
    cypher_route: str | None = None,
    openai_client: OpenAI | None = None,
    reranker=None,
    rerank_pool_n: int | None = None,
    query_rewrite: bool = False,
) -> tuple[list[dict], list[tuple], dict]:
    debug: dict = {
        "cypher_route": cypher_route or "hybrid",
        "bridge_mode": "provenance+doc_fallback",
    }

    records, cypher_used, route_used = execute_cypher(
        driver,
        query,
        route=cypher_route,
        openai_client=openai_client,
    )
    debug.update(cypher=cypher_used, cypher_route_used=route_used, n_triples=len(records))

    bridge_stats: dict = {}
    graph_chunks = chunks_from_provenance(
        records,
        chunk_by_id,
        max_chunks=top_n,
        bridge_stats=bridge_stats,
    )
    debug["n_graph_chunks"] = len(graph_chunks)
    debug["graph_bridge"] = bridge_stats

    from rag.config import RERANK_POOL_N, RETRIEVAL_K

    rk = retrieval_k if retrieval_k is not None else RETRIEVAL_K
    pool_n = rerank_pool_n if rerank_pool_n is not None else RERANK_POOL_N

    from rag.query_rewrite import hybrid_retrieve

    baseline, rewrite_dbg = hybrid_retrieve(
        query,
        collection,
        embedder,
        bm25_v2,
        all_docs,
        all_metas,
        openai_client=openai_client,
        query_rewrite=query_rewrite,
        retrieval_k=rk,
        top_n=top_n,
        rrf_k=rrf_k,
        reranker=reranker,
        rerank_pool_n=pool_n,
    )
    debug.update(rewrite_dbg)
    debug["baseline_hits"] = baseline

    if len(graph_chunks) >= graph_merge_min_hits:
        merged = rrf_merge_lists(graph_chunks, baseline, top_n=top_n, rrf_k=rrf_k)
        debug["fallback_baseline"] = False
        debug["graph_hits"] = graph_chunks
    else:
        merged = baseline
        debug["fallback_baseline"] = True
        debug["graph_hits"] = graph_chunks

    triples = rows_to_triples(records)
    return merged, triples, debug

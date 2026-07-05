from __future__ import annotations

from dataclasses import dataclass, field

from openai import OpenAI

from rag.config import CYPHER_ROUTE, GRAPH_MERGE_MIN_HITS, GRAPH_MODES, RETRIEVAL_K, RRF_K, TOP_N


@dataclass
class GraphRetrievalResult:
    hits: list[dict]
    triples: list[tuple] = field(default_factory=list)
    debug: dict = field(default_factory=dict)


def retrieve_graph(
    query: str,
    mode: str,
    *,
    driver=None,
    collection,
    embedder,
    bm25_v2,
    bm25_v1,
    all_docs: list[str],
    all_metas: list[dict],
    chunk_by_id: dict[str, tuple[str, dict]] | None = None,
    top_n: int = TOP_N,
    retrieval_k: int = RETRIEVAL_K,
    rrf_k: int = RRF_K,
    graph_merge_min_hits: int = GRAPH_MERGE_MIN_HITS,
    cypher_route: str | None = None,
    openai_client: OpenAI | None = None,
    reranker=None,
    rerank_pool_n: int | None = None,
    query_rewrite: bool = False,
) -> GraphRetrievalResult:
    from rag.config import RERANK_POOL_N
    from rag.graph.retrieval import retrieve_cypher_rag
    from rag.query_rewrite import hybrid_retrieve

    pool_n = rerank_pool_n if rerank_pool_n is not None else RERANK_POOL_N

    if mode not in GRAPH_MODES:
        raise ValueError(f"Modo graph desconocido: {mode}")

    if mode == "graph_baseline":
        hits, debug = hybrid_retrieve(
            query,
            collection,
            embedder,
            bm25_v2,
            all_docs,
            all_metas,
            openai_client=openai_client,
            query_rewrite=query_rewrite,
            retrieval_k=retrieval_k,
            top_n=top_n,
            rrf_k=rrf_k,
            reranker=reranker,
            rerank_pool_n=pool_n,
        )
        debug["baseline_hits"] = hits
        return GraphRetrievalResult(hits=hits, debug=debug)

    if driver is None:
        raise ValueError("Neo4j driver requerido para modos Cypher")

    if chunk_by_id is None:
        chunk_by_id = {
            m["chunk_id"]: (doc, m) for doc, m in zip(all_docs, all_metas)
        }

    hits, triples, debug = retrieve_cypher_rag(
        query,
        driver=driver,
        chunk_by_id=chunk_by_id,
        collection=collection,
        embedder=embedder,
        bm25_v2=bm25_v2,
        bm25_v1=bm25_v1,
        all_docs=all_docs,
        all_metas=all_metas,
        top_n=top_n,
        retrieval_k=retrieval_k,
        rrf_k=rrf_k,
        graph_merge_min_hits=graph_merge_min_hits,
        cypher_route=cypher_route or CYPHER_ROUTE,
        openai_client=openai_client,
        reranker=reranker,
        rerank_pool_n=pool_n,
        query_rewrite=query_rewrite,
    )
    return GraphRetrievalResult(hits=hits, triples=triples, debug=debug)

"""Textos UI para modos RAG y ayuda al usuario."""

DEMO_RAG_MODES = ("cypher_transversal", "v2", "graph_baseline")

RAG_MODE_LABELS: dict[str, str] = {
    "cypher_transversal": "Graph RAG completo (Neo4j + actas)",
    "v2": "Híbrido sin grafo (BM25 + E5)",
    "graph_baseline": "Solo Chroma (comparación grafo)",
}

RAG_MODE_HELP: dict[str, str] = {
    "cypher_transversal": (
        "Consulta entidades en Neo4j, recupera chunks por provenance y fusiona con retrieval híbrido."
    ),
    "v2": "BM25 + vectores E5 + RRF (+ reranker opcional). Sin grafo.",
    "graph_baseline": "Mismo índice Chroma que Graph RAG, pero sin consultar Neo4j.",
}


def _hybrid_node(query_rewrite: bool) -> str:
    if query_rewrite:
        return "Rewrite → BM25 + E5 → RRF"
    return "BM25 + E5 → RRF"


def pipeline_mermaid(rag_mode: str, *, rerank_enabled: bool, query_rewrite: bool = False) -> str:
    """Diagrama Mermaid del pipeline activo (dos ramas → fusión RRF)."""
    h = _hybrid_node(query_rewrite)
    if rag_mode == "cypher_transversal":
        if rerank_enabled:
            body = f"""
flowchart TD
    Q[Pregunta original]
    Q --> G[Cypher hybrid → Neo4j]
    Q --> H[{h}]
    G --> GC[Chunks del grafo]
    H --> RR[Reranker E5]
    RR --> HC[Chunks híbridos]
    GC --> M[Fusión RRF → top-N]
    HC --> M
    M --> GPT[GPT-4o-mini → respuesta]
"""
        else:
            body = f"""
flowchart TD
    Q[Pregunta original]
    Q --> G[Cypher hybrid → Neo4j]
    Q --> H[{h}]
    G --> GC[Chunks del grafo]
    H --> HC[Chunks híbridos]
    GC --> M[Fusión RRF → top-N]
    HC --> M
    M --> GPT[GPT-4o-mini → respuesta]
"""
    elif rag_mode == "v2":
        if rerank_enabled:
            body = f"""
flowchart TD
    Q[Pregunta] --> H[{h}]
    H --> RR[Reranker E5]
    RR --> TOP[top-N]
    TOP --> GPT[GPT-4o-mini → respuesta]
"""
        else:
            body = f"""
flowchart TD
    Q[Pregunta] --> H[{h}]
    H --> TOP[top-N]
    TOP --> GPT[GPT-4o-mini → respuesta]
"""
    else:
        body = f"""
flowchart TD
    Q[Pregunta] --> H[{h}]
    H --> TOP[top-N]
    TOP --> GPT[GPT-4o-mini → respuesta]
"""
    return body.strip()

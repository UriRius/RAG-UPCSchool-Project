"""Presets de pipeline para demo."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

from rag.config import (
    CYPHER_ROUTE,
    GRAPH_MERGE_MIN_HITS,
    RAG_MODE,
    RERANK_ENABLED,
    RERANK_POOL_N,
    RETRIEVAL_K,
    RRF_K,
    RRF_MIN_SCORE,
    TOP_N,
)

# IDs antiguos → nuevos (sesiones Streamlit previas al redeploy).
LEGACY_PRESET_IDS: dict[str, str] = {
    "hybrid_only": "no_graph",
    "ablation_no_graph": "no_graph",
    "ablation_no_rerank": "graph_no_rerank",
    "graph_chroma_only": "chroma_only",
    "production_rewrite": "production",
}


@dataclass(frozen=True)
class PipelinePreset:
    id: str
    label: str
    description: str
    rag_mode: str
    cypher_route: str
    rerank_enabled: bool
    query_rewrite_enabled: bool
    rerank_pool_n: int
    top_n: int
    retrieval_k: int
    rrf_k: int
    rrf_min_score: float
    graph_merge_min_hits: int


def _preset(
    id: str,
    label: str,
    description: str,
    *,
    rag_mode: str,
    cypher_route: str | None = None,
    rerank_enabled: bool = True,
    query_rewrite_enabled: bool = False,
    **overrides: Any,
) -> PipelinePreset:
    return PipelinePreset(
        id=id,
        label=label,
        description=description,
        rag_mode=rag_mode,
        cypher_route=cypher_route or CYPHER_ROUTE,
        rerank_enabled=rerank_enabled,
        query_rewrite_enabled=query_rewrite_enabled,
        rerank_pool_n=int(overrides.get("rerank_pool_n", RERANK_POOL_N)),
        top_n=int(overrides.get("top_n", TOP_N)),
        retrieval_k=int(overrides.get("retrieval_k", RETRIEVAL_K)),
        rrf_k=int(overrides.get("rrf_k", RRF_K)),
        rrf_min_score=float(overrides.get("rrf_min_score", RRF_MIN_SCORE)),
        graph_merge_min_hits=int(overrides.get("graph_merge_min_hits", GRAPH_MERGE_MIN_HITS)),
    )


PRESETS: tuple[PipelinePreset, ...] = (
    _preset(
        "production",
        "Producción",
        "Graph RAG completo: Neo4j, híbrido, reranker y query rewriting.",
        rag_mode="cypher_transversal",
        cypher_route="hybrid",
        rerank_enabled=True,
        query_rewrite_enabled=True,
    ),
    _preset(
        "production_no_rewrite",
        "Producción sin rewriting",
        "Graph RAG completo sin variantes de pregunta en la rama híbrida.",
        rag_mode="cypher_transversal",
        cypher_route="hybrid",
        rerank_enabled=True,
        query_rewrite_enabled=False,
    ),
    _preset(
        "no_graph",
        "Sin grafo",
        "Búsqueda híbrida en actas (BM25 + E5 + reranker).",
        rag_mode="v2",
        rerank_enabled=True,
        query_rewrite_enabled=False,
    ),
    _preset(
        "graph_no_rerank",
        "Graph RAG sin reranker",
        "Graph RAG completo sin cross-encoder rerank.",
        rag_mode="cypher_transversal",
        cypher_route="hybrid",
        rerank_enabled=False,
    ),
    _preset(
        "chroma_only",
        "Solo Chroma",
        "Búsqueda vectorial básica, sin grafo ni reranker.",
        rag_mode="graph_baseline",
        rerank_enabled=False,
    ),
    _preset(
        "custom",
        "Personalizado",
        "Ajusta modo y parámetros manualmente.",
        rag_mode=RAG_MODE if RAG_MODE in ("cypher_transversal", "v2", "graph_baseline") else "cypher_transversal",
        cypher_route=CYPHER_ROUTE,
        rerank_enabled=RERANK_ENABLED,
    ),
)

PRESET_BY_ID = {p.id: p for p in PRESETS}


def resolve_preset_id(preset_id: str) -> str:
    if preset_id in PRESET_BY_ID:
        return preset_id
    return LEGACY_PRESET_IDS.get(preset_id, "production")


def default_preset_id() -> str:
    if RAG_MODE == "cypher_transversal" and RERANK_ENABLED:
        return "production"
    if RAG_MODE == "v2":
        return "no_graph"
    return "custom"


def apply_preset_to_session(preset: PipelinePreset, session: Any) -> None:
    """Escribe valores de preset en st.session_state (claves de widgets)."""
    session["cfg_preset"] = preset.id
    session["rag_mode"] = preset.rag_mode
    session["cfg_rerank"] = preset.rerank_enabled
    session["cfg_query_rewrite"] = preset.query_rewrite_enabled
    session["cfg_rerank_pool"] = preset.rerank_pool_n
    session["cfg_top_n"] = preset.top_n
    session["cfg_retrieval_k"] = preset.retrieval_k
    session["cfg_rrf_k"] = preset.rrf_k
    session["cfg_rrf_min"] = preset.rrf_min_score
    session["cfg_graph_merge"] = preset.graph_merge_min_hits

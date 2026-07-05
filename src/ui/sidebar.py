"""Sidebar Streamlit: presets, pipeline modular y estado."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable

import streamlit as st

from rag.config import (
    CHROMA_PATH,
    EMBEDDING_STYLE,
    GRAPH_MERGE_MIN_HITS,
    GRAPH_MODES,
    QUERY_REWRITE_ENABLED,
    RAG_MODE,
    RERANK_ENABLED,
    RERANK_POOL_N,
    RETRIEVAL_K,
    RRF_K,
    RRF_MIN_SCORE,
    TOP_N,
    resolve_collection_name,
)
from ui.graph_explorer import render_graph_explorer
from ui.labels import (
    DEMO_RAG_MODES,
    RAG_MODE_HELP,
    RAG_MODE_LABELS,
    pipeline_mermaid,
)
from ui.presets import PRESETS, PRESET_BY_ID, apply_preset_to_session, default_preset_id, resolve_preset_id

# Fijo en prod/demo: LLM con fallback a plantilla (no expuesto en UI).
CYPHER_ROUTE_FIXED = "hybrid"


@dataclass
class RagSettings:
    rag_mode: str
    cypher_route: str
    rerank_enabled: bool
    rerank_pool_n: int
    top_n: int
    retrieval_k: int
    rrf_k: int
    rrf_min_score: float
    graph_merge_min_hits: int
    query_rewrite_enabled: bool
    preset_id: str


@dataclass
class SidebarStatus:
    collection_name: str
    n_chunks: int
    embedding_style: str
    neo4j_status: str
    reranker_loaded: bool
    graph_stats: dict | None = None
    last_debug: dict | None = None
    fetch_ego: Callable[[str], list[tuple[str, str, str]]] | None = None
    fetch_batch_hubs: Callable[[str], list[dict]] | None = None


def _init_session_defaults() -> None:
    if "cfg_preset" in st.session_state:
        st.session_state.cfg_preset = resolve_preset_id(st.session_state.cfg_preset)
    else:
        pid = default_preset_id()
        apply_preset_to_session(PRESET_BY_ID[pid], st.session_state)


def _on_preset_change() -> None:
    pid = resolve_preset_id(st.session_state.get("cfg_preset", "custom"))
    st.session_state.cfg_preset = pid
    if pid != "custom" and pid in PRESET_BY_ID:
        apply_preset_to_session(PRESET_BY_ID[pid], st.session_state)


def _mark_custom() -> None:
    st.session_state.cfg_preset = "custom"


def render_sidebar_config() -> RagSettings:
    """Controles de pipeline (parte superior del sidebar)."""
    _init_session_defaults()

    st.subheader("Perfil")
    preset_ids = [p.id for p in PRESETS]
    preset_labels = {p.id: p.label for p in PRESETS}

    st.selectbox(
        "Preset",
        preset_ids,
        format_func=lambda x: preset_labels[x] + (" ★" if x == "production" else ""),
        key="cfg_preset",
        on_change=_on_preset_change,
    )
    preset = PRESET_BY_ID[resolve_preset_id(st.session_state.cfg_preset)]
    locked = resolve_preset_id(st.session_state.cfg_preset) != "custom"

    if preset.id != "custom":
        st.caption(preset.description)
    if locked:
        st.info("Pipeline fijado por el preset. Elige **Personalizado** para editar.")

    st.divider()
    st.subheader("Pipeline")

    with st.expander("Flujo del sistema", expanded=False):
        st.caption("Dos ramas en paralelo que convergen en fusión RRF (neo4j_graph_rag_v2).")
        st.markdown(
            f"```mermaid\n{pipeline_mermaid(st.session_state.get('rag_mode', 'cypher_transversal'), rerank_enabled=bool(st.session_state.get('cfg_rerank', RERANK_ENABLED)), query_rewrite=bool(st.session_state.get('cfg_query_rewrite', QUERY_REWRITE_ENABLED)))}\n```"
        )

    mode_options = list(DEMO_RAG_MODES)

    default_mode = st.session_state.get("rag_mode", RAG_MODE)
    if default_mode not in mode_options:
        default_mode = mode_options[0]
        st.session_state.rag_mode = default_mode

    rag_mode = st.selectbox(
        "Modo RAG",
        mode_options,
        index=mode_options.index(st.session_state.get("rag_mode", default_mode)),
        format_func=lambda m: RAG_MODE_LABELS.get(m, m),
        key="rag_mode",
        on_change=_mark_custom,
        disabled=locked,
    )
    if rag_mode in RAG_MODE_HELP:
        st.caption(RAG_MODE_HELP[rag_mode])

    use_graph = rag_mode in GRAPH_MODES
    if use_graph:
        st.caption("Neo4j: activo en este modo")
    else:
        st.caption("Neo4j: no se consulta")

    rerank_default = bool(st.session_state.get("cfg_rerank", RERANK_ENABLED))
    rerank_enabled = st.toggle(
        "Reranker E5 (fine-tuned)",
        value=rerank_default,
        key="cfg_rerank",
        on_change=_mark_custom,
        disabled=locked,
    )

    rerank_pool_n = RERANK_POOL_N
    if rerank_enabled:
        rerank_pool_n = st.slider(
            "Pool reranker (candidatos)",
            5,
            50,
            int(st.session_state.get("cfg_rerank_pool", RERANK_POOL_N)),
            key="cfg_rerank_pool",
            on_change=_mark_custom,
            disabled=locked,
        )

    rewrite_default = bool(st.session_state.get("cfg_query_rewrite", QUERY_REWRITE_ENABLED))
    query_rewrite_enabled = st.toggle(
        "Query rewriting (rama híbrida)",
        value=rewrite_default,
        key="cfg_query_rewrite",
        on_change=_mark_custom,
        disabled=locked,
        help="Genera variantes de la pregunta para BM25+E5. Neo4j usa la pregunta original.",
    )

    with st.expander("Avanzado", expanded=False):
        st.slider(
            "TOP_N (chunks al LLM)",
            3,
            15,
            int(st.session_state.get("cfg_top_n", TOP_N)),
            key="cfg_top_n",
            on_change=_mark_custom,
            disabled=locked,
        )
        st.slider(
            "RETRIEVAL_K (pool BM25+dense)",
            20,
            100,
            int(st.session_state.get("cfg_retrieval_k", RETRIEVAL_K)),
            key="cfg_retrieval_k",
            on_change=_mark_custom,
            disabled=locked,
        )
        st.slider(
            "RRF_K (fusión)",
            40,
            80,
            int(st.session_state.get("cfg_rrf_k", RRF_K)),
            key="cfg_rrf_k",
            on_change=_mark_custom,
            disabled=locked,
        )
        st.slider(
            "Umbral RRF mínimo",
            0.0,
            0.05,
            float(st.session_state.get("cfg_rrf_min", RRF_MIN_SCORE)),
            step=0.005,
            format="%.3f",
            key="cfg_rrf_min",
            on_change=_mark_custom,
            disabled=locked,
        )
        if rag_mode in GRAPH_MODES:
            st.slider(
                "Mín. chunks grafo para merge",
                0,
                5,
                int(st.session_state.get("cfg_graph_merge", GRAPH_MERGE_MIN_HITS)),
                key="cfg_graph_merge",
                on_change=_mark_custom,
                disabled=locked,
            )

    return RagSettings(
        rag_mode=rag_mode,
        cypher_route=CYPHER_ROUTE_FIXED,
        rerank_enabled=rerank_enabled,
        rerank_pool_n=rerank_pool_n,
        top_n=st.session_state.cfg_top_n,
        retrieval_k=st.session_state.cfg_retrieval_k,
        rrf_k=st.session_state.cfg_rrf_k,
        rrf_min_score=st.session_state.cfg_rrf_min,
        graph_merge_min_hits=st.session_state.get("cfg_graph_merge", GRAPH_MERGE_MIN_HITS),
        query_rewrite_enabled=query_rewrite_enabled,
        preset_id=resolve_preset_id(st.session_state.cfg_preset),
    )


def render_sidebar_status(status: SidebarStatus) -> None:
    st.divider()
    st.subheader("Estado")
    st.text(f"Índice: {status.collection_name}")
    st.text(f"Chunks: {status.n_chunks}")
    st.text(f"Embed: {status.embedding_style}")
    st.text(f"Neo4j: {status.neo4j_status}")
    if status.graph_stats:
        gs = status.graph_stats
        st.caption(
            f"KG: {gs['entities']:,} entidades · {gs['relationships']:,} relaciones · "
            f"{gs['with_provenance']:,} con provenance"
        )
        if gs.get("batches"):
            batch_txt = ", ".join(f"{tag} ({n})" for tag, n in gs["batches"])
            st.caption(f"Batches: {batch_txt}")
    st.text(f"Reranker: {'cargado' if status.reranker_loaded else 'no disponible'}")
    st.caption(f"Chroma: `{CHROMA_PATH}`")

    render_graph_explorer(
        status.graph_stats,
        fetch_ego=status.fetch_ego,
        fetch_batch_hubs=status.fetch_batch_hubs,
    )

    dbg = status.last_debug
    if dbg:
        st.divider()
        st.subheader("Última consulta")
        c1, c2, c3 = st.columns(3)
        c1.metric("Chunks", dbg.get("n_hits", "—"))
        c2.metric("Grafo", dbg.get("n_graph_chunks", "—"))
        c3.metric("Rerank", "ON" if dbg.get("rerank_used") else "OFF")
        if dbg.get("query_rewrite_used"):
            st.caption("Query rewriting: ON (rama híbrida)")
        if dbg.get("bridge"):
            st.caption(f"Bridge: {dbg['bridge']}")
        if dbg.get("fallback_baseline"):
            st.caption("Fallback baseline (sin merge grafo)")


def collection_for_mode(mode: str) -> str:
    return resolve_collection_name(mode)

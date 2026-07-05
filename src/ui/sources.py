"""Trazabilidad de fuentes: grafo, retrieval y top final separados."""

from __future__ import annotations

import streamlit as st

from rag.graph import GraphRetrievalResult
from ui.graph_viz import render_triple_subgraph


def _bridge_metric(bridge: dict | None) -> str:
    if not bridge:
        return "—"
    exact = int(bridge.get("exact", 0))
    legacy = int(bridge.get("legacy_index", 0))
    by_doc = int(bridge.get("by_doc", 0))
    if exact:
        return str(exact)
    if legacy:
        return f"{legacy} legacy"
    if by_doc:
        return f"{by_doc} doc"
    return "0"


def _bridge_caption(bridge: dict | None) -> str | None:
    if not bridge:
        return None
    parts = []
    if bridge.get("exact"):
        parts.append(f"{bridge['exact']} chunk(s) enlazados por ID exacto")
    if bridge.get("legacy_index"):
        parts.append(f"{bridge['legacy_index']} por índice legacy")
    if bridge.get("by_doc"):
        parts.append(f"{bridge['by_doc']} por documento (fallback)")
    if bridge.get("miss"):
        parts.append(f"{bridge['miss']} sin match en Chroma")
    return " · ".join(parts) if parts else None


def _chunk_text(text: str) -> str:
    if text.startswith("passage: "):
        text = text[len("passage: ") :]
    return text[:600] + ("…" if len(text) > 600 else "")


def _doc_label(meta: dict) -> str:
    return meta.get("doc_id") or meta.get("docx_id") or "?"


def _score_label(source: dict) -> str:
    if source.get("rerank_score") is not None:
        return f"rerank {source['score']:.3f}"
    return f"RRF {source['score']:.4f}"


def _render_chunk(source: dict, *, rank: int | None = None) -> None:
    prefix = f"#{rank} · " if rank is not None else ""
    bridge = source.get("meta", {}).get("graph_bridge")
    bridge_tag = f" · bridge={bridge}" if bridge else ""
    st.caption(
        f"{prefix}**{_doc_label(source['meta'])}** · `{source['meta']['chunk_id']}`"
        f"{bridge_tag} ({_score_label(source)})"
    )
    st.text(_chunk_text(source["text"]))


def _render_triples_section(graph_result: GraphRetrievalResult) -> None:
    triples = graph_result.triples
    if not triples:
        return

    st.markdown("#### 1. Relaciones Neo4j (triples)")
    st.caption("Conocimiento estructurado encontrado en el grafo. No van al LLM como texto.")
    for triple in triples[:8]:
        subject, predicate, obj = triple
        if subject == "__agg__":
            st.text(f"  [{predicate}] → {obj}")
        else:
            st.text(f"  ({subject}) -[{predicate}]-> ({obj})")
    if len(triples) > 8:
        st.caption(f"... y {len(triples) - 8} más")

    viz_triples = [t for t in triples[:12] if t[0] != "__agg__"]
    if viz_triples:
        with st.expander("Visualización del subgrafo", expanded=False):
            render_triple_subgraph(viz_triples)


def _render_rewrite_queries(debug: dict) -> None:
    if not debug.get("query_rewrite_used"):
        return
    rewrites = debug.get("rewritten_queries") or []
    if not rewrites:
        return
    st.caption("**Queries reescritas (solo rama híbrida):**")
    for i, q in enumerate(rewrites, 1):
        st.text(f"  {i}. {q}")


def render_sources(sources: list[dict], graph_result: GraphRetrievalResult | None = None) -> None:
    if not sources:
        return

    with st.expander("Ver fuentes y trazabilidad"):
        dbg = graph_result.debug if graph_result else {}
        is_graph_rag = bool(graph_result and graph_result.triples is not None and dbg.get("n_triples") is not None)

        if is_graph_rag:
            bridge = dbg.get("graph_bridge")
            cols = st.columns(3)
            cols[0].metric("Triples Neo4j", dbg.get("n_triples", "—"))
            cols[1].metric("Chunks vía grafo", dbg.get("n_graph_chunks", "—"))
            cols[2].metric("Match exacto actas", _bridge_metric(bridge))
            if dbg.get("fallback_baseline"):
                st.caption("Sin fusión: pocos chunks del grafo → solo retrieval híbrido al LLM.")
            else:
                st.caption("Fusión RRF: se combinan chunks del grafo + retrieval híbrido → top final.")
            bridge_txt = _bridge_caption(bridge)
            if bridge_txt:
                st.caption(bridge_txt)
            if dbg.get("cypher"):
                with st.expander("Cypher ejecutado", expanded=False):
                    st.code(dbg["cypher"], language="cypher")

            _render_triples_section(graph_result)

            graph_hits: list[dict] = dbg.get("graph_hits") or []
            if not graph_hits:
                graph_hits = [s for s in sources if s.get("from_graph")]

            baseline_hits: list[dict] = dbg.get("baseline_hits") or []
            if graph_hits:
                st.divider()
                st.markdown(f"#### 2. Chunks del grafo ({len(graph_hits)})")
                st.caption("Texto de actas enlazado por provenance desde Neo4j.")
                for source in graph_hits:
                    _render_chunk(source)

            if baseline_hits:
                st.divider()
                st.markdown(f"#### 3. Chunks del retrieval híbrido ({len(baseline_hits)})")
                caption = "BM25 + E5 + RRF (+ reranker si está activo). Independiente del grafo."
                if dbg.get("query_rewrite_used"):
                    caption += " Con query rewriting (original + variantes)."
                st.caption(caption)
                _render_rewrite_queries(dbg)
                for source in baseline_hits:
                    _render_chunk(source)

            st.divider()
            st.markdown(f"#### 4. Top final enviado al LLM ({len(sources)})")
            st.caption("Lista fusionada (RRF) que lee GPT para redactar la respuesta.")
            graph_ids = {h["meta"]["chunk_id"] for h in graph_hits}
            for rank, source in enumerate(sources, 1):
                cid = source["meta"]["chunk_id"]
                origin = []
                if source.get("from_graph") or cid in graph_ids:
                    origin.append("grafo")
                if cid in {h["meta"]["chunk_id"] for h in baseline_hits}:
                    origin.append("híbrido")
                origin_tag = f" · {' + '.join(origin)}" if origin else ""
                prefix = f"#{rank}{origin_tag} · "
                bridge = source.get("meta", {}).get("graph_bridge")
                bridge_tag = f" · bridge={bridge}" if bridge else ""
                st.caption(
                    f"{prefix}**{_doc_label(source['meta'])}** · `{cid}`"
                    f"{bridge_tag} ({_score_label(source)})"
                )
                st.text(_chunk_text(source["text"]))
        else:
            dbg = graph_result.debug if graph_result else {}
            st.markdown(f"#### Retrieval híbrido ({len(sources)} chunks)")
            caption = "BM25 + E5 + RRF (+ reranker si está activo). Sin consulta a Neo4j."
            if dbg.get("query_rewrite_used"):
                caption += " Con query rewriting."
            st.caption(caption)
            _render_rewrite_queries(dbg)
            for rank, source in enumerate(sources, 1):
                _render_chunk(source, rank=rank)

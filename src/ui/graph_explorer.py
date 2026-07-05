"""Panel Neo4j: resumen del KG y búsqueda por entidad."""

from __future__ import annotations

from typing import Callable

import streamlit as st

from ui.graph_viz import render_triple_subgraph


def render_graph_summary(overview: dict | None, *, fetch_batch_hubs: Callable[[str], list[dict]] | None = None) -> None:
    if not overview:
        return

    with st.expander("Resumen del grafo (Neo4j)", expanded=False):
        st.caption(
            f"{overview['entities']:,} entidades · {overview['relationships']:,} relaciones · "
            f"{overview['with_provenance']:,} enlazadas a chunks de actas."
        )

        preds = overview.get("top_predicates") or []
        if preds:
            st.markdown("**Tipos de relación más frecuentes**")
            st.bar_chart({row["p"]: row["c"] for row in preds})

        hubs = overview.get("top_entities") or []
        if hubs:
            st.markdown("**Conceptos más conectados en todo el grafo**")
            for row in hubs[:8]:
                st.text(f"· {row['n']} — {row['c']} relaciones")

        batches = overview.get("batches") or []
        if batches and fetch_batch_hubs:
            st.divider()
            st.markdown("**Conceptos destacados por tanda de actas**")
            batch_ids = [tag for tag, _ in batches]
            selected_batch = st.selectbox(
                "Tanda (batch)",
                batch_ids,
                key="kg_batch_select",
                format_func=lambda b: next(
                    (f"{tag} ({n} relaciones)" for tag, n in batches if tag == b),
                    b,
                ),
            )
            batch_hubs = fetch_batch_hubs(selected_batch)
            for row in batch_hubs:
                st.text(f"· {row['n']} — {row['c']} relaciones")


def render_entity_search(fetch_ego: Callable[[str], list[tuple[str, str, str]]] | None) -> None:
    if not fetch_ego:
        return

    with st.expander("Buscar entidad en Neo4j", expanded=False):
        st.caption(
            "Escribe un concepto del dominio (obra, estación, empresa…) y verás "
            "sus vecinos directos en el grafo (máx. 24 conexiones)."
        )
        col_q, col_btn = st.columns([3, 1])
        with col_q:
            entity_q = st.text_input(
                "Entidad",
                key="kg_entity_search",
                placeholder="ej. marquesina, Renfe, andén",
                label_visibility="collapsed",
            )
        with col_btn:
            explore = st.button("Buscar", key="kg_ego_btn", use_container_width=True)

        if explore and entity_q.strip():
            triples = fetch_ego(entity_q.strip())
            if not triples:
                st.warning(f"No hay entidad similar a «{entity_q}» en el grafo.")
            else:
                center = triples[0][0]
                st.caption(f"Entidad: **{center}** · {len(triples)} conexiones mostradas")
                render_triple_subgraph(triples, max_edges=len(triples))


def render_graph_explorer(
    overview: dict | None,
    *,
    fetch_ego: Callable[[str], list[tuple[str, str, str]]] | None = None,
    fetch_batch_hubs: Callable[[str], list[dict]] | None = None,
) -> None:
    render_graph_summary(overview, fetch_batch_hubs=fetch_batch_hubs)
    render_entity_search(fetch_ego)

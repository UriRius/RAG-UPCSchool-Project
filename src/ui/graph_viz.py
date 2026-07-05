"""Visualización ligera de subgrafos (triples de una consulta)."""

from __future__ import annotations

import streamlit.components.v1 as components


def render_triple_subgraph(triples: list[tuple], *, max_edges: int = 12, height: int = 360) -> None:
    """Mini-grafo interactivo a partir de triples (subject, predicate, object)."""
    if not triples:
        return

    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#333333")
    net.barnes_hut(gravity=-8000, central_gravity=0.3, spring_length=120)

    seen_nodes: set[str] = set()
    for subject, predicate, obj in triples[:max_edges]:
        if subject == "__agg__":
            continue
        for node in (subject, obj):
            if node not in seen_nodes:
                label = node if len(node) <= 28 else node[:25] + "…"
                net.add_node(node, label=label, title=node, size=18)
                seen_nodes.add(node)
        net.add_edge(subject, obj, title=predicate, label=predicate[:18] if predicate else "")

    if not seen_nodes:
        return

    net.set_options(
        """
        {
          "physics": { "enabled": true, "stabilization": { "iterations": 80 } },
          "edges": { "font": { "size": 10, "align": "middle" }, "arrows": "to" },
          "interaction": { "hover": true, "navigationButtons": false }
        }
        """
    )
    html = net.generate_html(notebook=False)
    components.html(html, height=height + 20, scrolling=False)

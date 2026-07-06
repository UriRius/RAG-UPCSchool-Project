"""Visualización ligera de subgrafos (triples de una consulta)."""

from __future__ import annotations

import streamlit.components.v1 as components


def render_triple_subgraph(triples: list[tuple], *, max_edges: int = 40, height: int = 520) -> None:
    """Grafo interactivo a partir de triples (subject, predicate, object)."""
    if not triples:
        return

    from pyvis.network import Network

    net = Network(height=f"{height}px", width="100%", directed=True, bgcolor="#ffffff", font_color="#222222")
    net.barnes_hut(gravity=-12000, central_gravity=0.45, spring_length=90, spring_strength=0.04)

    seen_nodes: set[str] = set()
    for subject, predicate, obj in triples[:max_edges]:
        if subject == "__agg__":
            continue
        for node in (subject, obj):
            if node not in seen_nodes:
                label = node if len(node) <= 32 else node[:29] + "…"
                net.add_node(
                    node,
                    label=label,
                    title=node,
                    size=28,
                    font={"size": 14, "face": "arial"},
                )
                seen_nodes.add(node)
        net.add_edge(
            subject,
            obj,
            title=predicate,
            label=predicate[:22] if predicate else "",
            width=1.5,
        )

    if not seen_nodes:
        return

    net.set_options(
        """
        {
          "physics": {
            "enabled": true,
            "stabilization": { "iterations": 150, "fit": true }
          },
          "nodes": { "borderWidth": 1, "borderWidthSelected": 2 },
          "edges": {
            "font": { "size": 11, "align": "middle", "strokeWidth": 0 },
            "arrows": { "to": { "enabled": true, "scaleFactor": 0.7 } },
            "smooth": { "type": "dynamic" }
          },
          "interaction": { "hover": true, "navigationButtons": true, "zoomView": true }
        }
        """
    )
    html = net.generate_html(notebook=False)
    components.html(html, height=height + 24, scrolling=False)

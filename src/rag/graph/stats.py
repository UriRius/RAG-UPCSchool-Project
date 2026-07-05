"""Estadísticas y exploración del grafo Neo4j (para UI demo)."""

from __future__ import annotations

from neo4j import Driver

from rag.graph.neo4j_client import run_query


def fetch_graph_stats(driver: Driver) -> dict:
    """Conteos ligeros del KG cargado en Aura."""
    entities = run_query(driver, "MATCH (e:Entity) RETURN count(e) AS c")[0]["c"]
    rels = run_query(driver, "MATCH ()-[r:RELATED]->() RETURN count(r) AS c")[0]["c"]
    with_prov = run_query(
        driver,
        "MATCH ()-[r:RELATED]->() WHERE r.source_chunk_id IS NOT NULL RETURN count(r) AS c",
    )[0]["c"]
    batches = run_query(
        driver,
        "MATCH ()-[r:RELATED]->() "
        "RETURN r.batch AS tag, count(r) AS c ORDER BY tag",
    )
    return {
        "entities": int(entities),
        "relationships": int(rels),
        "with_provenance": int(with_prov),
        "batches": [(row["tag"], int(row["c"])) for row in batches if row.get("tag")],
    }


def fetch_graph_overview(driver: Driver) -> dict:
    """Stats + analytics agregados (no carga el grafo entero)."""
    overview = fetch_graph_stats(driver)
    top_predicates = run_query(
        driver,
        """
        MATCH ()-[r:RELATED]->()
        RETURN r.predicate AS p, count(*) AS c
        ORDER BY c DESC LIMIT 10
        """,
    )
    top_entities = run_query(
        driver,
        """
        MATCH (e:Entity)-[r:RELATED]-()
        RETURN coalesce(e.name, e.norm) AS n, count(r) AS c
        ORDER BY c DESC LIMIT 10
        """,
    )
    overview["top_predicates"] = [
        {"p": row["p"] or "?", "c": int(row["c"])} for row in top_predicates
    ]
    overview["top_entities"] = [
        {"n": row["n"] or "?", "c": int(row["c"])} for row in top_entities
    ]
    return overview


def fetch_ego_network(driver: Driver, query: str, *, limit: int = 24) -> list[tuple[str, str, str]]:
    """
    Subgrafo 1-hop alrededor de la primera entidad que coincida con `query`.
    Devuelve triples (subject, predicate, object) para visualización.
    """
    q = query.strip().lower()
    if not q:
        return []

    rows = run_query(
        driver,
        """
        MATCH (e:Entity)
        WHERE toLower(coalesce(e.name, '')) CONTAINS $q
           OR e.norm CONTAINS $q
        WITH e, size([(e)-[:RELATED]-() | 1]) AS deg
        ORDER BY deg DESC
        LIMIT 1
        MATCH (e)-[r:RELATED]-(n:Entity)
        RETURN coalesce(e.name, e.norm) AS subject,
               r.predicate AS predicate,
               coalesce(n.name, n.norm) AS object
        LIMIT $limit
        """,
        q=q,
        limit=limit,
    )
    return [(row["subject"], row["predicate"] or "?", row["object"]) for row in rows]


def fetch_batch_hubs(driver: Driver, batch: str, *, limit: int = 8) -> list[dict]:
    """Entidades más conectadas dentro de un batch de ingestión."""
    rows = run_query(
        driver,
        """
        MATCH (e:Entity)-[r:RELATED {batch: $batch}]-()
        RETURN coalesce(e.name, e.norm) AS n, count(r) AS c
        ORDER BY c DESC LIMIT $limit
        """,
        batch=batch,
        limit=limit,
    )
    return [{"n": row["n"] or "?", "c": int(row["c"])} for row in rows]

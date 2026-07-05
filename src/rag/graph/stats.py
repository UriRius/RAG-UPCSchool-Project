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


def _triples_from_rows(rows: list[dict]) -> list[tuple[str, str, str]]:
    return [(row["subject"], row["predicate"] or "?", row["object"]) for row in rows]


def _seed_entities_from_triples(
    triples: list[tuple[str, str, str]], *, max_seeds: int = 4
) -> list[str]:
    from collections import Counter

    counts: Counter[str] = Counter()
    for subject, _, obj in triples:
        if subject != "__agg__":
            counts[subject] += 1
            counts[obj] += 1
    return [name for name, _ in counts.most_common(max_seeds)]


def fetch_subgraph_for_viz(
    driver: Driver,
    triples: list[tuple[str, str, str]],
    *,
    hops: int = 2,
    limit: int = 50,
    max_seeds: int = 4,
) -> list[tuple[str, str, str]]:
    """
    Triples de la consulta + vecindario Neo4j (1–3 saltos) para visualización.
    """
    base = [(s, p, o) for s, p, o in triples if s != "__agg__"]
    hops = max(1, min(3, int(hops)))
    if hops <= 1 or not base:
        return base[:limit]

    seeds = _seed_entities_from_triples(base, max_seeds=max_seeds)
    if not seeds:
        return base[:limit]

    rows = run_query(
        driver,
        f"""
        UNWIND $seeds AS seed
        MATCH (e:Entity)
        WHERE toLower(coalesce(e.name, '')) = toLower(seed)
           OR e.norm = toLower(seed)
           OR toLower(coalesce(e.name, '')) CONTAINS toLower(seed)
        WITH collect(DISTINCT e) AS centers
        UNWIND centers AS e
        MATCH p = (e)-[:RELATED*1..{hops}]-(n:Entity)
        UNWIND relationships(p) AS rel
        WITH DISTINCT startNode(rel) AS s, rel, endNode(rel) AS t
        RETURN coalesce(s.name, s.norm) AS subject,
               rel.predicate AS predicate,
               coalesce(t.name, t.norm) AS object
        LIMIT $limit
        """,
        seeds=seeds,
        limit=limit,
    )

    seen: set[tuple[str, str, str]] = set()
    merged: list[tuple[str, str, str]] = []
    for triple in base + _triples_from_rows(rows):
        if triple not in seen:
            seen.add(triple)
            merged.append(triple)
        if len(merged) >= limit:
            break
    return merged


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
    return _triples_from_rows(rows)


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

from __future__ import annotations

import re

from rag.config import CYPHER_LIMIT
from rag.graph.text_utils import tokenize_match

CYPHER_FACTUAL = """
MATCH (e:Entity)-[r:RELATED]->(o:Entity)
WHERE ANY(seed IN $seeds WHERE e.norm CONTAINS seed OR o.norm CONTAINS seed)
RETURN e.name AS subject, r.predicate AS predicate, o.name AS object,
       r.source_doc AS source_doc, r.source_chunk_id AS source_chunk_id
LIMIT $limit
"""

CYPHER_TRANSVERSAL = """
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
WHERE ANY(kw IN $keywords WHERE r.predicate CONTAINS kw
       OR a.norm CONTAINS kw OR b.norm CONTAINS kw)
RETURN a.name AS subject, r.predicate AS predicate, b.name AS object,
       r.source_doc AS source_doc, r.source_chunk_id AS source_chunk_id
LIMIT $limit
"""

CYPHER_AGG = """
MATCH ()-[r:RELATED]->()
RETURN r.predicate AS predicate, count(*) AS cnt
ORDER BY cnt DESC
LIMIT $limit
"""

ENTITY_KEYWORDS = [
    "talud", "megafonía", "megafonia", "ute", "constructora", "luminaria", "andén",
    "anden", "ar-29", "hormigonado", "zapata", "incidencia", "df", "deo", "pilar",
    "micropilote", "cimentación", "cimentacion", "pararrayos",
]


def extract_seeds(query: str) -> list[str]:
    lowered = query.lower()
    seeds = [kw for kw in ENTITY_KEYWORDS if kw in lowered]
    if seeds:
        return seeds
    tokens = [t for t in tokenize_match(query) if len(t) > 3]
    return tokens[:3]


def cypher_template_route(query: str) -> tuple[str, dict]:
    lowered = query.lower()
    if any(p in lowered for p in ("frecuent", "más común", "más frecuente", "cuáles son las")):
        return CYPHER_AGG, {"limit": 20}
    keywords = extract_seeds(query) or ["incidencia", "problema", "solicitud"]
    return CYPHER_TRANSVERSAL, {"keywords": keywords, "limit": CYPHER_LIMIT}


def rows_to_triples(rows: list[dict]) -> list[tuple]:
    triples: list[tuple] = []
    for row in rows:
        if "subject" in row and "object" in row:
            triples.append((row["subject"], row.get("predicate", ""), row["object"]))
        elif "predicate" in row and "cnt" in row:
            triples.append(("__agg__", row["predicate"], f"count={row['cnt']}"))
    return triples

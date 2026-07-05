from __future__ import annotations

import re
import unicodedata

from neo4j import Driver

from rag.config import NEO4J_DATABASE

SCHEMA_CONSTRAINTS = [
    "CREATE CONSTRAINT entity_norm IF NOT EXISTS FOR (e:Entity) REQUIRE e.norm IS UNIQUE",
]

LOAD_CYPHER = """
UNWIND $rows AS row
MERGE (s:Entity {norm: row.s_norm})
  ON CREATE SET s.name = row.subject, s.batch = row.batch
  ON MATCH SET s.name = coalesce(s.name, row.subject)
MERGE (o:Entity {norm: row.o_norm})
  ON CREATE SET o.name = row.object, o.batch = row.batch
  ON MATCH SET o.name = coalesce(o.name, row.object)
MERGE (s)-[r:RELATED {triple_idx: row.triple_idx, batch: row.batch}]->(o)
  SET r.predicate = row.predicate
"""


def norm_entity(name: str) -> str:
    name = unicodedata.normalize("NFKD", name)
    name = "".join(c for c in name if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", name.lower().strip())


def ensure_schema(driver: Driver) -> None:
    with driver.session(database=NEO4J_DATABASE) as session:
        for query in SCHEMA_CONSTRAINTS:
            session.run(query)


def build_load_rows(rels: list[tuple], batch: str) -> list[dict]:
    rows = []
    for idx, (subject, predicate, obj) in enumerate(rels):
        rows.append(
            {
                "triple_idx": idx,
                "subject": subject,
                "s_norm": norm_entity(subject),
                "predicate": predicate,
                "object": obj,
                "o_norm": norm_entity(obj),
                "batch": batch,
            }
        )
    return rows


def load_graph_to_neo4j(
    driver: Driver,
    graph_dict: dict,
    batch_tag: str,
    *,
    batch_size: int = 100,
) -> int:
    rels = [tuple(r) for r in graph_dict["relations"]]
    rows = build_load_rows(rels, batch=batch_tag)
    with driver.session(database=NEO4J_DATABASE) as session:
        for i in range(0, len(rows), batch_size):
            chunk = rows[i : i + batch_size]
            session.run(LOAD_CYPHER, rows=chunk)
    return len(rows)

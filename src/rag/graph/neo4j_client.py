from __future__ import annotations

from neo4j import GraphDatabase, Driver

from rag.config import NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER


def neo4j_configured() -> bool:
    return bool(NEO4J_URI and NEO4J_PASSWORD)


def get_driver(*, verify: bool = True) -> Driver:
    if not neo4j_configured():
        raise ValueError("Configura NEO4J_URI y NEO4J_PASSWORD en .env")
    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    if verify:
        driver.verify_connectivity()
    return driver


def run_query(driver: Driver, cypher: str, **params) -> list[dict]:
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(cypher, **params)
        return [dict(record) for record in result]

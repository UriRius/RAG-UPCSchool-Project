from __future__ import annotations

import json
import re
from pathlib import Path

from neo4j import Driver

from rag.config import (
    FORCE_RELOAD_TAGS,
    GRAPH_DIR,
    LEGACY_BATCH1_JSON,
    LEGACY_BATCH1_TAG,
    NEO4J_DATABASE,
    WIPE_ALL_ON_SYNC,
)
from rag.graph.schema import ensure_schema, load_graph_to_neo4j


def batch_tag(batch_id: int) -> str:
    return LEGACY_BATCH1_TAG if batch_id == 1 else f"batch{batch_id:02d}"


def refresh_catalog(graph_dir: Path | str | None = None) -> dict:
    graph_dir = Path(graph_dir or GRAPH_DIR)
    batches = []

    path_batch1 = graph_dir / LEGACY_BATCH1_JSON
    if path_batch1.exists():
        with path_batch1.open(encoding="utf-8") as f:
            graph_data = json.load(f)
        batches.append(
            {
                "batch_id": 1,
                "tag": LEGACY_BATCH1_TAG,
                "json_file": LEGACY_BATCH1_JSON,
                "sources": graph_data.get("meta", {}).get("sources") or [],
                "n_triples": len(graph_data.get("relations", [])),
            }
        )

    for path in sorted(graph_dir.glob("graph_actas_e5_original_batch*_bal_raw.json")):
        match = re.search(r"batch(\d+)_", path.name)
        if not match:
            continue
        batch_id = int(match.group(1))
        with path.open(encoding="utf-8") as f:
            graph_data = json.load(f)
        batches.append(
            {
                "batch_id": batch_id,
                "tag": batch_tag(batch_id),
                "json_file": path.name,
                "sources": graph_data.get("meta", {}).get("sources") or [],
                "n_triples": len(graph_data.get("relations", [])),
            }
        )

    by_id = {item["batch_id"]: item for item in batches}
    catalog = {
        "version": 1,
        "batches": sorted(by_id.values(), key=lambda x: x["batch_id"]),
    }
    graph_dir.mkdir(parents=True, exist_ok=True)
    with (graph_dir / "catalog.json").open("w", encoding="utf-8") as f:
        json.dump(catalog, f, ensure_ascii=False, indent=2)
    return catalog


def get_loaded_batch_tags(driver: Driver) -> set[str]:
    with driver.session(database=NEO4J_DATABASE) as session:
        rows = session.run(
            "MATCH ()-[r:RELATED]->() RETURN DISTINCT r.batch AS tag"
        ).data()
    return {row["tag"] for row in rows if row.get("tag")}


def wipe_all_graph(driver: Driver) -> None:
    with driver.session(database=NEO4J_DATABASE) as session:
        session.run("MATCH (n) DETACH DELETE n")


def wipe_batch_relationships(driver: Driver, tag: str) -> int:
    with driver.session(database=NEO4J_DATABASE) as session:
        result = session.run(
            "MATCH ()-[r:RELATED]->() WHERE r.batch = $tag DELETE r RETURN count(r) AS c",
            tag=tag,
        )
        return int(result.single()["c"])


def sync_catalog_to_neo4j(
    driver: Driver,
    catalog: dict,
    *,
    graph_dir: Path | str | None = None,
) -> list[dict]:
    force = set(FORCE_RELOAD_TAGS)
    loaded = set() if WIPE_ALL_ON_SYNC else get_loaded_batch_tags(driver)
    if WIPE_ALL_ON_SYNC:
        wipe_all_graph(driver)

    graph_path = Path(graph_dir or GRAPH_DIR)
    sync_rows: list[dict] = []

    for batch in catalog["batches"]:
        tag = batch["tag"]
        json_file = batch["json_file"]
        path = graph_path / json_file

        if not path.exists():
            sync_rows.append({**batch, "status": "missing"})
            continue
        if tag in loaded and tag not in force:
            sync_rows.append({**batch, "status": "skipped"})
            continue

        wipe_batch_relationships(driver, tag)
        with path.open(encoding="utf-8") as f:
            graph_data = json.load(f)
        n_loaded = load_graph_to_neo4j(driver, graph_data, batch_tag=tag)
        sync_rows.append({**batch, "status": "loaded", "n_loaded": n_loaded})

    return sync_rows


def sync_graph_catalog(
    graph_dir: Path | str | None = None,
    *,
    ensure_schema_first: bool = True,
) -> dict:
    from rag.graph.neo4j_client import get_driver

    graph_path = Path(graph_dir or GRAPH_DIR)
    driver = get_driver()
    if ensure_schema_first:
        ensure_schema(driver)
    catalog = refresh_catalog(graph_path)
    sync_rows = sync_catalog_to_neo4j(driver, catalog, graph_dir=graph_path)
    driver.close()
    return {"catalog": catalog, "sync_rows": sync_rows}

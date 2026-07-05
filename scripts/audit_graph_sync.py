#!/usr/bin/env python3
"""Audita Neo4j vs Chroma y lista JSON de grafo en Drive."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()


def audit_neo4j() -> None:
    import chromadb
    from neo4j import GraphDatabase

    from rag.config import CHROMA_PATH, NEO4J_DATABASE, NEO4J_PASSWORD, NEO4J_URI, NEO4J_USER

    collection = os.getenv("CHROMA_COLLECTION", "construction_site_visit_reports")
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    chroma_n = client.get_collection(collection).count()

    driver = GraphDatabase.driver(NEO4J_URI, auth=(NEO4J_USER, NEO4J_PASSWORD))
    driver.verify_connectivity()
    with driver.session(database=NEO4J_DATABASE) as s:
        nodes = s.run("MATCH (n) RETURN count(n) AS c").single()["c"]
        rels = s.run("MATCH ()-[r]->() RETURN count(r) AS c").single()["c"]
        labels = [r["label"] for r in s.run("CALL db.labels() YIELD label RETURN label").data()]
        chunk_nodes = 0
        if "Chunk" in labels:
            chunk_nodes = s.run("MATCH (c:Chunk) RETURN count(c) AS c").single()["c"]
        batches = s.run(
            "MATCH ()-[r:RELATED]->() RETURN r.batch AS tag, count(r) AS c ORDER BY tag"
        ).data()
        prov = s.run(
            "MATCH ()-[r:RELATED]->() "
            "WHERE r.source_chunk_id IS NOT NULL "
            "RETURN count(r) AS c"
        ).single()["c"]
    driver.close()

    print("=== Chroma (local) ===")
    print(f"  Coleccion {collection!r}: {chroma_n} chunks")
    print()
    print("=== Neo4j ===")
    print(f"  Nodos: {nodes}")
    print(f"  Relaciones: {rels}")
    print(f"  Chunk nodes: {chunk_nodes}")
    print(f"  RELATED con source_chunk_id: {prov}")
    print(f"  Labels: {labels}")
    print("  Batches cargados:")
    for row in batches:
        print(f"    {row['tag']}: {row['c']} rels")
    print()


def audit_drive_graph() -> None:
    from google.oauth2 import service_account
    from googleapiclient.discovery import build

    folder_id = os.getenv("DRIVE_FOLDER_ID") or os.getenv("DRIVE_GRAPH_FOLDER_ID")
    if not folder_id:
        print("WARN: sin DRIVE_FOLDER_ID / DRIVE_GRAPH_FOLDER_ID")
        return

    creds = service_account.Credentials.from_service_account_file(
        "credentials.json",
        scopes=["https://www.googleapis.com/auth/drive.readonly"],
    )
    svc = build("drive", "v3", credentials=creds)

    def children(pid: str) -> list[dict]:
        items: list[dict] = []
        page = None
        while True:
            res = (
                svc.files()
                .list(
                    q=f"'{pid}' in parents and trashed = false",
                    fields="nextPageToken, files(id, name, mimeType, size, modifiedTime)",
                    pageSize=200,
                    pageToken=page,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            items.extend(res.get("files", []))
            page = res.get("nextPageToken")
            if not page:
                break
        return items

    roots = children(folder_id)
    graph_dirs = [
        it
        for it in roots
        if it.get("mimeType", "").endswith("folder")
        and "graph" in it["name"].lower()
    ]
    if not graph_dirs:
        graph_dirs = [it for it in roots if it["name"].lower() == "graph"]

    print("=== Drive: JSON de grafo ===")
    if not graph_dirs:
        print("  No hay carpeta graph/ en la raiz de Drive")
        json_in_root = [it for it in roots if it["name"].endswith(".json") and "graph" in it["name"]]
        for it in sorted(json_in_root, key=lambda x: x["name"]):
            sz = int(it.get("size") or 0)
            print(f"  {it['name']}  {sz // 1024} KB  id={it['id']}")
        return

    for gd in graph_dirs:
        print(f"  Carpeta: {gd['name']}  id={gd['id']}")
        sub = children(gd["id"])
        jsons = [f for f in sub if f["name"].lower().endswith(".json")]
        for it in sorted(jsons, key=lambda x: x["name"]):
            sz = int(it.get("size") or 0)
            mod = (it.get("modifiedTime") or "")[:10]
            print(f"    {it['name']:<55} {sz // 1024:>6} KB  {mod}")
        print(f"  Total JSON: {len(jsons)}")


if __name__ == "__main__":
    audit_neo4j()
    audit_drive_graph()

#!/usr/bin/env python3
"""
Pipeline Graph RAG sin notebooks:
  Drive o GCS (JSON) -> catalog -> Neo4j Aura

Uso local:
  python src/sync_graph.py --from-drive --sync-neo4j
  python src/sync_graph.py --from-gcs --sync-neo4j

Cloud Run Job (por defecto):
  python src/sync_graph.py --from-drive --sync-neo4j --upload-gcs
"""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from dotenv import load_dotenv

load_dotenv()

from rag.config import (
    DRIVE_GRAPH_FOLDER_ID,
    GCS_GRAPH_PREFIX,
    GCP_BUCKET_NAME,
    GRAPH_DIR,
)
from rag.graph.catalog import refresh_catalog, sync_graph_catalog

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)


def _print_sync_summary(result: dict) -> None:
    catalog = result["catalog"]
    print(f"Catalog: {len(catalog['batches'])} batches")
    for row in result["sync_rows"]:
        status = row["status"]
        tag = row["tag"]
        if status == "loaded":
            print(f"  loaded  {tag} ({row.get('n_loaded', '?')} triples)")
        else:
            print(f"  {status:7} {tag}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Sync grafo COSORA: Drive/GCS -> Neo4j")
    parser.add_argument(
        "--graph-dir",
        default=GRAPH_DIR,
        help=f"Directorio local de JSON (default: {GRAPH_DIR})",
    )
    source = parser.add_mutually_exclusive_group()
    source.add_argument(
        "--from-drive",
        action="store_true",
        help="Descargar JSON del grafo desde Google Drive",
    )
    source.add_argument(
        "--from-gcs",
        action="store_true",
        help="Descargar JSON del grafo desde GCS",
    )
    parser.add_argument(
        "--upload-gcs",
        action="store_true",
        help="Subir graph/ local a GCS tras descargar",
    )
    parser.add_argument(
        "--sync-neo4j",
        action="store_true",
        help="Sincronizar catalog -> Neo4j Aura (incremental)",
    )
    parser.add_argument(
        "--catalog-only",
        action="store_true",
        help="Solo regenerar catalog.json (sin Neo4j)",
    )
    parser.add_argument(
        "--drive-folder-id",
        default=DRIVE_GRAPH_FOLDER_ID or os.getenv("DRIVE_FOLDER_ID"),
        help="ID carpeta Drive con los JSON del grafo",
    )
    parser.add_argument(
        "--bucket",
        default=GCP_BUCKET_NAME,
        help="Bucket GCS (default: GCP_BUCKET_NAME)",
    )
    parser.add_argument(
        "--gcs-prefix",
        default=GCS_GRAPH_PREFIX,
        help="Prefijo en bucket para graph/ (default: graph)",
    )
    args = parser.parse_args()

    graph_dir = Path(args.graph_dir)
    graph_dir.mkdir(parents=True, exist_ok=True)

    if args.from_drive:
        if not args.drive_folder_id:
            logger.error(
                "Falta DRIVE_GRAPH_FOLDER_ID (o --drive-folder-id). "
                "Es el ID de la carpeta graph/ en Drive."
            )
            return 1
        from rag.io.drive import download_graph_json_from_drive

        download_graph_json_from_drive(args.drive_folder_id, graph_dir)

    elif args.from_gcs:
        if not args.bucket:
            logger.error("Falta GCP_BUCKET_NAME (o --bucket)")
            return 1
        from rag.io.gcs import download_folder_from_gcs

        download_folder_from_gcs(args.bucket, args.gcs_prefix, graph_dir)

    if args.upload_gcs:
        if not args.bucket:
            logger.error("Falta GCP_BUCKET_NAME para --upload-gcs")
            return 1
        from rag.io.gcs import upload_folder_to_gcs

        upload_folder_to_gcs(args.bucket, graph_dir, args.gcs_prefix)

    if args.catalog_only:
        catalog = refresh_catalog(graph_dir)
        print(f"Catalog: {len(catalog['batches'])} batches -> {graph_dir / 'catalog.json'}")
        for batch in catalog["batches"]:
            print(f"  - {batch['tag']}: {batch['json_file']} ({batch['n_triples']} triples)")
        return 0

    if args.sync_neo4j:
        result = sync_graph_catalog(graph_dir)
        _print_sync_summary(result)
        return 0

    if args.from_drive or args.from_gcs or args.upload_gcs:
        catalog = refresh_catalog(graph_dir)
        print(f"Descarga OK. {len(catalog['batches'])} batches en {graph_dir}")
        print("Ejecuta con --sync-neo4j para cargar en Neo4j.")
        return 0

    parser.error("Indica --from-drive, --from-gcs, --sync-neo4j o --catalog-only")


if __name__ == "__main__":
    raise SystemExit(main())

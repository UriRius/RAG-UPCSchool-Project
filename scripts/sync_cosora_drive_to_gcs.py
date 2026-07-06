#!/usr/bin/env python3
"""
Opción B: Drive (chroma_db COSORA) -> GCS -> Cloud Build.

No requiere copia permanente en data/chroma_db del repo.
"""
from __future__ import annotations

import argparse
import logging
import os
import sqlite3
import sys
import tempfile
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from rag.config import COLLECTION_NAME, GCP_BUCKET_NAME
from rag.io.drive import find_child_file, get_drive_service, upload_folder_from_drive_to_gcs
from rag.io.gcs import delete_gcs_prefix

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GCS_CHROMA_PREFIX = "chroma_db"
DRIVE_CHROMA_FOLDER = os.getenv("DRIVE_CHROMA_FOLDER", "chroma_db")


def verify_gcs_chroma(bucket: str, collection: str) -> int:
    from google.cloud import storage

    client = storage.Client()
    blob = client.bucket(bucket).blob(f"{GCS_CHROMA_PREFIX}/chroma.sqlite3")
    if not blob.exists():
        raise FileNotFoundError(f"Falta gs://{bucket}/{GCS_CHROMA_PREFIX}/chroma.sqlite3")

    with tempfile.NamedTemporaryFile(suffix=".sqlite3", delete=False) as tmp:
        path = tmp.name
    try:
        blob.download_to_filename(path)
        conn = sqlite3.connect(path)
        rows = conn.execute("SELECT id, name FROM collections").fetchall()
        counts = {}
        for cid, name in rows:
            n = conn.execute(
                "SELECT COUNT(*) FROM embeddings "
                "WHERE segment_id IN (SELECT id FROM segments WHERE collection=?)",
                (cid,),
            ).fetchone()[0]
            counts[name] = n
        conn.close()
        logger.info("Colecciones en GCS: %s", counts)
        n = counts.get(collection, 0)
        if n == 0:
            raise RuntimeError(f"Coleccion {collection!r} vacia o ausente en GCS")
        return n
    finally:
        Path(path).unlink(missing_ok=True)


def main() -> int:
    parser = argparse.ArgumentParser(description="Drive chroma_db COSORA -> GCS")
    parser.add_argument("--bucket", default=GCP_BUCKET_NAME or "rag-actas-db-bucket")
    parser.add_argument("--drive-folder", default=DRIVE_CHROMA_FOLDER)
    parser.add_argument("--collection", default=COLLECTION_NAME)
    parser.add_argument("--wipe-gcs", action="store_true", help="Borra chroma_db/ en GCS antes")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    root_id = os.getenv("DRIVE_FOLDER_ID")
    if not root_id:
        logger.error("Falta DRIVE_FOLDER_ID en .env")
        return 1
    if not args.bucket:
        logger.error("Falta GCP_BUCKET_NAME")
        return 1

    service = get_drive_service()
    folder = find_child_file(service, root_id, args.drive_folder)
    if folder is None:
        logger.error("No se encontro carpeta %r en Drive", args.drive_folder)
        return 1
    if folder.get("mimeType") != "application/vnd.google-apps.folder":
        logger.error("%r no es una carpeta", args.drive_folder)
        return 1

    logger.info(
        "Origen: Drive/%s (id=%s) -> gs://%s/%s/",
        args.drive_folder,
        folder["id"],
        args.bucket,
        GCS_CHROMA_PREFIX,
    )

    if args.dry_run:
        print("Dry run — no se sube nada")
        return 0

    if args.wipe_gcs:
        delete_gcs_prefix(args.bucket, GCS_CHROMA_PREFIX)

    n_files = upload_folder_from_drive_to_gcs(
        folder["id"],
        args.bucket,
        GCS_CHROMA_PREFIX,
    )
    n_chunks = verify_gcs_chroma(args.bucket, args.collection)

    print()
    print("=== GCS listo (COSORA opcion B) ===")
    print(f"  gs://{args.bucket}/{GCS_CHROMA_PREFIX}/  ({n_files} archivos)")
    print(f"  Coleccion {args.collection!r}: {n_chunks} chunks")
    print()
    print("Siguiente paso:")
    print("  powershell deploy/scripts/deploy.ps1")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

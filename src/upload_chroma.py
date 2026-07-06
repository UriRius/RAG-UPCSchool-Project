#!/usr/bin/env python3
"""
Sube Chroma local a GCS (para deploy Cloud Run).

Uso:
  python src/upload_chroma.py --verify-gcs
  python src/upload_chroma.py --path ./data/chroma_db --upload-gcs
  python src/upload_chroma.py --from-drive --upload-gcs
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

from rag.config import CHROMA_PATH, COLLECTION_NAME, GCP_BUCKET_NAME

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GCS_CHROMA_PREFIX = "chroma_db"
TARGET_V2_CHUNKS = 1544  # kg_ingest_v2 Colab
MIN_V2_CHUNKS = 1500


def verify_chroma(path: str | Path, *, label: str = "") -> dict[str, int]:
    import chromadb

    path = Path(path)
    client = chromadb.PersistentClient(path=str(path))
    counts = {c.name: c.count() for c in client.list_collections()}
    tag = f"{label} " if label else ""
    for name, n in sorted(counts.items()):
        logger.info("%s%s: %d chunks", tag, name, n)
    main = counts.get(COLLECTION_NAME, 0)
    if main < MIN_V2_CHUNKS:
        logger.warning(
            "%s tiene %d chunks en %r (esperado >= %d para demo v2)",
            tag.strip() or "Chroma",
            main,
            COLLECTION_NAME,
            MIN_V2_CHUNKS,
        )
    return counts


def verify_gcs(bucket: str, prefix: str = GCS_CHROMA_PREFIX) -> dict[str, int]:
    import shutil
    import tempfile

    from rag.io.gcs import download_folder_from_gcs

    tmp = Path(tempfile.mkdtemp(prefix="chroma_gcs_verify_"))
    try:
        logger.info("Descargando gs://%s/%s/ para verificar...", bucket, prefix)
        n = download_folder_from_gcs(bucket, prefix, tmp)
        logger.info("Archivos en GCS: %d", n)
        return verify_chroma(tmp, label="GCS")
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def upload_to_gcs(local_path: str | Path, bucket: str, prefix: str = GCS_CHROMA_PREFIX) -> int:
    from rag.io.gcs import upload_folder_to_gcs

    local_path = Path(local_path)
    if not local_path.is_dir():
        raise FileNotFoundError(f"No existe carpeta Chroma: {local_path}")
    verify_chroma(local_path, label="Local")
    n = upload_folder_to_gcs(bucket, local_path, prefix)
    logger.info("Subido a gs://%s/%s/ (%d archivos)", bucket, prefix, n)
    return n


def download_chroma_from_drive(folder_id: str, dest: str | Path) -> int:
    from rag.io.drive import download_folder_from_drive

    dest = Path(dest)
    if dest.exists():
        import shutil

        shutil.rmtree(dest)
    return download_folder_from_drive(folder_id, dest)


def main() -> int:
    parser = argparse.ArgumentParser(description="Verificar / subir ChromaDB a GCS")
    parser.add_argument("--verify-gcs", action="store_true", help="Verificar Chroma en GCS")
    parser.add_argument("--upload-gcs", action="store_true", help="Subir Chroma a GCS")
    parser.add_argument(
        "--path",
        default=CHROMA_PATH,
        help=f"Carpeta Chroma local (default: {CHROMA_PATH})",
    )
    parser.add_argument(
        "--from-drive",
        action="store_true",
        help="Descargar chroma_db desde Drive antes de subir",
    )
    parser.add_argument(
        "--drive-folder-id",
        default=os.getenv("DRIVE_CHROMA_FOLDER_ID", ""),
        help="ID carpeta Drive chroma_db (env: DRIVE_CHROMA_FOLDER_ID)",
    )
    parser.add_argument("--bucket", default=GCP_BUCKET_NAME or "rag-actas-db-bucket")
    parser.add_argument("--gcs-prefix", default=GCS_CHROMA_PREFIX)
    args = parser.parse_args()

    if not args.verify_gcs and not args.upload_gcs:
        parser.error("Indica --verify-gcs y/o --upload-gcs")

    if args.verify_gcs:
        if not args.bucket:
            logger.error("Falta GCP_BUCKET_NAME (o --bucket)")
            return 1
        counts = verify_gcs(args.bucket, args.gcs_prefix)
        main_n = counts.get(COLLECTION_NAME, 0)
        if abs(main_n - TARGET_V2_CHUNKS) <= 50:
            logger.info(
                "OK: GCS alineado con v2 Colab (%r ≈ %d chunks)",
                COLLECTION_NAME,
                TARGET_V2_CHUNKS,
            )
        elif main_n >= MIN_V2_CHUNKS:
            logger.warning(
                "GCS tiene %d chunks (esperado ~%d). Sube Chroma desde Drive.",
                main_n,
                TARGET_V2_CHUNKS,
            )
            return 1
        else:
            logger.error("GCS insuficiente para demo v2 — sube Chroma desde Drive")
            return 1

    if args.upload_gcs:
        if not args.bucket:
            logger.error("Falta GCP_BUCKET_NAME (o --bucket)")
            return 1
        chroma_path = Path(args.path)
        if args.from_drive:
            if not args.drive_folder_id:
                logger.error(
                    "Falta DRIVE_CHROMA_FOLDER_ID (o --drive-folder-id). "
                    "Es el ID de MyDrive/.../chroma_db en Drive."
                )
                return 1
            logger.info("Descargando chroma_db desde Drive...")
            download_chroma_from_drive(args.drive_folder_id, chroma_path)
        upload_to_gcs(chroma_path, args.bucket, args.gcs_prefix)
        logger.info("Verificando GCS tras upload...")
        verify_gcs(args.bucket, args.gcs_prefix)

    return 0


if __name__ == "__main__":
    raise SystemExit(main())

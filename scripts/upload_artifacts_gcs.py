#!/usr/bin/env python3
"""Sube Chroma + BM25 + reranker a GCS (Fase 3 deploy)."""
from __future__ import annotations

import argparse
import logging
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))

from dotenv import load_dotenv

load_dotenv()

from rag.config import (
    CHROMA_PATH,
    COLLECTION_NAME_TRISTAN,
    GCP_BUCKET_NAME,
    RR_MODEL_PATH,
)
from rag.io.gcs import upload_folder_to_gcs

logging.basicConfig(level=logging.INFO, format="%(levelname)s: %(message)s")
logger = logging.getLogger(__name__)

GCS_CHROMA_PREFIX = "chroma_db"
GCS_RR_MODEL_PREFIX = "rr_model"


def verify_local(chroma_path: Path, rr_model_path: Path) -> None:
    import chromadb

    from rag.bm25_index import BM25Log1

    collection = os.getenv("CHROMA_COLLECTION", COLLECTION_NAME_TRISTAN)
    client = chromadb.PersistentClient(path=str(chroma_path))
    counts = {c.name: c.count() for c in client.list_collections()}
    logger.info("Chroma local: %s", counts)
    if counts.get(collection, 0) == 0:
        raise RuntimeError(f"Coleccion {collection!r} vacia o ausente en {chroma_path}")

    bm25_file = os.getenv("BM25_PATH") or str(chroma_path / "bm25.json")
    if not Path(bm25_file).is_file():
        raise FileNotFoundError(f"Falta BM25: {bm25_file}")
    bm25 = BM25Log1.load(bm25_file)
    logger.info("BM25: %d chunks", len(bm25.chunk_ids))

    if not (rr_model_path / "model.pt").is_file():
        raise FileNotFoundError(f"Falta reranker en {rr_model_path}/model.pt")
    logger.info("Reranker OK en %s", rr_model_path)


def main() -> int:
    parser = argparse.ArgumentParser(description="Subir artefactos Tristan a GCS")
    parser.add_argument("--bucket", default=GCP_BUCKET_NAME or "rag-actas-db-bucket")
    parser.add_argument("--chroma-path", default=CHROMA_PATH)
    parser.add_argument("--rr-model-path", default=RR_MODEL_PATH)
    parser.add_argument("--skip-chroma", action="store_true")
    parser.add_argument("--skip-reranker", action="store_true")
    parser.add_argument("--dry-run", action="store_true")
    args = parser.parse_args()

    if not args.bucket:
        logger.error("Falta GCP_BUCKET_NAME")
        return 1

    chroma_path = Path(args.chroma_path)
    rr_model_path = Path(args.rr_model_path)
    verify_local(chroma_path, rr_model_path)

    if args.dry_run:
        logger.info("Dry run — no se sube nada")
        return 0

    if not args.skip_chroma:
        n = upload_folder_to_gcs(args.bucket, chroma_path, GCS_CHROMA_PREFIX)
        logger.info("Chroma subido: %d archivos -> gs://%s/%s/", n, args.bucket, GCS_CHROMA_PREFIX)

    if not args.skip_reranker:
        n = upload_folder_to_gcs(args.bucket, rr_model_path, GCS_RR_MODEL_PREFIX)
        logger.info("Reranker subido: %d archivos -> gs://%s/%s/", n, args.bucket, GCS_RR_MODEL_PREFIX)

    print()
    print("=== GCS listo ===")
    print(f"  gs://{args.bucket}/{GCS_CHROMA_PREFIX}/")
    print(f"  gs://{args.bucket}/{GCS_RR_MODEL_PREFIX}/")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())

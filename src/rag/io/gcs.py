from __future__ import annotations

import logging
import os
from pathlib import Path

from google.cloud import storage

logger = logging.getLogger(__name__)


def download_folder_from_gcs(
    bucket_name: str,
    source_prefix: str,
    destination_folder: str | Path,
) -> int:
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0
    for blob in bucket.list_blobs(prefix=source_prefix):
        if blob.name.endswith("/"):
            continue
        relative_path = os.path.relpath(blob.name, source_prefix)
        local_path = destination_folder / relative_path
        local_path.parent.mkdir(parents=True, exist_ok=True)
        blob.download_to_filename(str(local_path))
        count += 1
    logger.info(
        "Descargados %s archivos de gs://%s/%s -> %s",
        count,
        bucket_name,
        source_prefix,
        destination_folder,
    )
    return count


def upload_folder_to_gcs(
    bucket_name: str,
    source_folder: str | Path,
    destination_prefix: str,
) -> int:
    source_folder = Path(source_folder)
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    count = 0
    for local_path in source_folder.rglob("*"):
        if not local_path.is_file():
            continue
        relative_path = local_path.relative_to(source_folder).as_posix()
        blob_path = f"{destination_prefix.rstrip('/')}/{relative_path}"
        bucket.blob(blob_path).upload_from_filename(str(local_path))
        count += 1
    logger.info(
        "Subidos %s archivos de %s -> gs://%s/%s",
        count,
        source_folder,
        bucket_name,
        destination_prefix,
    )
    return count


def delete_gcs_prefix(bucket_name: str, prefix: str) -> int:
    """Elimina todos los blobs bajo un prefijo (p. ej. chroma_db/)."""
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    blobs = list(bucket.list_blobs(prefix=prefix.rstrip("/") + "/"))
    if not blobs:
        return 0
    bucket.delete_blobs(blobs)
    logger.info(
        "Eliminados %s objetos en gs://%s/%s/",
        len(blobs),
        bucket_name,
        prefix.rstrip("/"),
    )
    return len(blobs)

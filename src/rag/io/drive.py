from __future__ import annotations

import io
import logging
import os
from pathlib import Path

from googleapiclient.discovery import build
from googleapiclient.http import MediaIoBaseDownload
from google.oauth2 import service_account
import google.auth

logger = logging.getLogger(__name__)

DRIVE_SCOPES = ["https://www.googleapis.com/auth/drive.readonly"]
CREDENTIALS_FILE = "credentials.json"


def get_drive_service():
    """Drive API: service account (credentials.json) por defecto; ADC solo si DRIVE_USE_ADC=1."""
    use_adc = os.getenv("DRIVE_USE_ADC", "").lower() in ("1", "true", "yes")
    if use_adc:
        logger.info("Drive API: ADC (cuenta de usuario)")
        creds, _ = google.auth.default(scopes=DRIVE_SCOPES)
    elif os.path.exists(CREDENTIALS_FILE):
        logger.info("Drive API: credentials.json (service account)")
        creds = service_account.Credentials.from_service_account_file(
            CREDENTIALS_FILE, scopes=DRIVE_SCOPES
        )
    else:
        logger.info("Drive API: identidad de Google Cloud (Cloud Run)")
        creds, _ = google.auth.default(scopes=DRIVE_SCOPES)
    return build("drive", "v3", credentials=creds)


def download_file_from_drive(file_id: str, destination: str | Path) -> Path:
    """Descarga un fichero de Drive por ID."""
    destination = Path(destination)
    destination.parent.mkdir(parents=True, exist_ok=True)
    service = get_drive_service()
    request = service.files().get_media(fileId=file_id)
    with io.FileIO(destination, "wb") as handle:
        downloader = MediaIoBaseDownload(handle, request)
        done = False
        while not done:
            _, done = downloader.next_chunk()
    logger.info("Descargado de Drive: %s", destination.name)
    return destination


def find_child_file(
    service,
    parent_id: str,
    name: str,
) -> dict | None:
    """Busca un hijo por nombre exacto en una carpeta Drive."""
    safe_name = name.replace("'", "\\'")
    q = f"'{parent_id}' in parents and name = '{safe_name}' and trashed = false"
    res = (
        service.files()
        .list(
            q=q,
            fields="files(id, name, mimeType, size)",
            pageSize=10,
            supportsAllDrives=True,
            includeItemsFromAllDrives=True,
        )
        .execute()
    )
    files = res.get("files", [])
    return files[0] if files else None


def download_graph_json_from_drive(
    folder_id: str,
    destination_folder: str | Path,
    *,
    name_prefix: str = "graph_actas_e5",
) -> int:
    """Descarga JSON del grafo desde una carpeta de Google Drive."""
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)

    service = get_drive_service()
    query = f"'{folder_id}' in parents and trashed = false"
    results = service.files().list(q=query, fields="files(id, name)").execute()
    items = results.get("files", [])

    count = 0
    for item in items:
        name = item["name"]
        if not name.lower().endswith(".json"):
            continue
        if name_prefix and not name.startswith(name_prefix):
            continue

        dest = destination_folder / name
        request = service.files().get_media(fileId=item["id"])
        with io.FileIO(dest, "wb") as handle:
            downloader = MediaIoBaseDownload(handle, request)
            done = False
            while not done:
                _, done = downloader.next_chunk()
        count += 1
        logger.info("Descargado de Drive: %s", name)

    if count == 0:
        logger.warning(
            "No se encontraron JSON con prefijo '%s' en la carpeta Drive %s",
            name_prefix,
            folder_id,
        )
    else:
        logger.info("Descargados %s JSON del grafo desde Drive", count)
    return count


def download_folder_from_drive(
    folder_id: str,
    destination_folder: str | Path,
) -> int:
    """Descarga recursivamente una carpeta de Drive (p. ej. chroma_db/)."""
    destination_folder = Path(destination_folder)
    destination_folder.mkdir(parents=True, exist_ok=True)
    service = get_drive_service()
    count = 0

    def _walk(parent_id: str, rel_base: Path) -> None:
        nonlocal count
        query = f"'{parent_id}' in parents and trashed = false"
        page_token = None
        while True:
            results = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=200,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in results.get("files", []):
                name = item["name"]
                rel_path = rel_base / name
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    _walk(item["id"], rel_path)
                else:
                    dest = destination_folder / rel_path
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    request = service.files().get_media(fileId=item["id"])
                    with io.FileIO(dest, "wb") as handle:
                        downloader = MediaIoBaseDownload(handle, request)
                        done = False
                        while not done:
                            _, done = downloader.next_chunk()
                    count += 1
                    if count % 10 == 0:
                        logger.info("Descargados %s archivos...", count)
            page_token = results.get("nextPageToken")
            if not page_token:
                break

    _walk(folder_id, Path("."))
    logger.info("Descargados %s archivos de Drive -> %s", count, destination_folder)
    return count


def upload_folder_from_drive_to_gcs(
    folder_id: str,
    bucket_name: str,
    destination_prefix: str,
) -> int:
    """Copia recursiva Drive -> GCS sin escribir en disco local del repo."""
    from google.cloud import storage

    service = get_drive_service()
    client = storage.Client()
    bucket = client.bucket(bucket_name)
    prefix = destination_prefix.rstrip("/")
    count = 0

    def _walk(parent_id: str, rel_base: Path) -> None:
        nonlocal count
        query = f"'{parent_id}' in parents and trashed = false"
        page_token = None
        while True:
            results = (
                service.files()
                .list(
                    q=query,
                    fields="nextPageToken, files(id, name, mimeType)",
                    pageSize=200,
                    pageToken=page_token,
                    supportsAllDrives=True,
                    includeItemsFromAllDrives=True,
                )
                .execute()
            )
            for item in results.get("files", []):
                name = item["name"]
                rel_path = rel_base / name
                if item.get("mimeType") == "application/vnd.google-apps.folder":
                    _walk(item["id"], rel_path)
                else:
                    blob_path = f"{prefix}/{rel_path.as_posix()}"
                    request = service.files().get_media(fileId=item["id"])
                    buffer = io.BytesIO()
                    downloader = MediaIoBaseDownload(buffer, request)
                    done = False
                    while not done:
                        _, done = downloader.next_chunk()
                    buffer.seek(0)
                    bucket.blob(blob_path).upload_from_file(buffer)
                    count += 1
                    if count % 10 == 0:
                        logger.info("Subidos %s archivos a GCS...", count)
            page_token = results.get("nextPageToken")
            if not page_token:
                break

    _walk(folder_id, Path("."))
    logger.info(
        "Subidos %s archivos Drive -> gs://%s/%s/",
        count,
        bucket_name,
        prefix,
    )
    return count

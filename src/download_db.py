import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from rag.config import CHROMA_PATH, RR_MODEL_PATH
from rag.io.gcs import download_folder_from_gcs

GCS_CHROMA_PREFIX = "chroma_db"
GCS_RR_MODEL_PREFIX = "rr_model"


def download_bm25_from_gcs(bucket_name, chroma_path):
    """Descarga bm25.json si existe en el prefijo chroma_db."""
    filename = "bm25.json"
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blob = bucket.blob(f"{GCS_CHROMA_PREFIX}/{filename}")
        if not blob.exists():
            logging.warning("%s no encontrado en gs://%s/%s/", filename, bucket_name, GCS_CHROMA_PREFIX)
            return
        dest = os.path.join(chroma_path, filename)
        os.makedirs(os.path.dirname(dest), exist_ok=True)
        blob.download_to_filename(dest)
        logging.info("✅ %s descargado a %s", filename, dest)
    except Exception as e:
        logging.warning("No se pudo descargar %s: %s", filename, e)


if __name__ == "__main__":
    from dotenv import load_dotenv
    load_dotenv()

    BUCKET_NAME = os.getenv("GCP_BUCKET_NAME")
    if not BUCKET_NAME:
        logging.error("La variable GCP_BUCKET_NAME no está configurada en el archivo .env")
        exit(1)

    chroma_dest = os.getenv("CHROMA_PATH", CHROMA_PATH)
    rr_dest = os.getenv("RR_MODEL_PATH", RR_MODEL_PATH)

    logging.info("Descargando Chroma desde gs://%s/%s ...", BUCKET_NAME, GCS_CHROMA_PREFIX)
    download_folder_from_gcs(BUCKET_NAME, GCS_CHROMA_PREFIX, chroma_dest)
    download_bm25_from_gcs(BUCKET_NAME, chroma_dest)

    logging.info("Descargando reranker desde gs://%s/%s ...", BUCKET_NAME, GCS_RR_MODEL_PREFIX)
    download_folder_from_gcs(BUCKET_NAME, GCS_RR_MODEL_PREFIX, rr_dest)

    logging.info("✅ Artefactos listos en %s y %s", chroma_dest, rr_dest)

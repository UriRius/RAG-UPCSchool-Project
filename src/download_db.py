import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

from google.cloud import storage
import logging

logging.basicConfig(level=logging.INFO, format='%(levelname)s: %(message)s')

from rag.config import bm25_path, CHROMA_PATH

def download_folder_from_gcs(bucket_name, source_folder, destination_folder):
    """Descarga una carpet completa desde un bucket de GCS a un directorio local."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        blobs = bucket.list_blobs(prefix=source_folder)

        count = 0
        for blob in blobs:
            if blob.name.endswith("/"):
                continue

            relative_path = os.path.relpath(blob.name, source_folder)
            local_path = os.path.join(destination_folder, relative_path)

            os.makedirs(os.path.dirname(local_path), exist_ok=True)
            blob.download_to_filename(local_path)
            count += 1

        logging.info(f"✅ Descarga completada: {count} archivos descargados a {destination_folder}.")
    except Exception as e:
        logging.error(f"❌ Error descargando la base de datos de GCS: {e}")
        logging.error("Asegúrate de que estás autenticado con 'gcloud auth application-default login'")

def download_bm25_from_gcs(bucket_name, chroma_path):
    """Descarga bm25.json y bm25_v2.json si existen en el prefijo chroma_db."""
    from rag.config import BM25_FILENAME, BM25_FILENAME_V2

    for filename in (BM25_FILENAME, BM25_FILENAME_V2):
        try:
            storage_client = storage.Client()
            bucket = storage_client.bucket(bucket_name)
            blob = bucket.blob(f"chroma_db/{filename}")
            if not blob.exists():
                logging.warning("%s no encontrado en gs://%s/chroma_db/", filename, bucket_name)
                continue
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

    destination = os.getenv("CHROMA_PATH", CHROMA_PATH)

    logging.info(f"Iniciando descarga de ChromaDB desde gs://{BUCKET_NAME}/chroma_db ...")
    download_folder_from_gcs(BUCKET_NAME, "chroma_db", destination)
    download_bm25_from_gcs(BUCKET_NAME, destination)

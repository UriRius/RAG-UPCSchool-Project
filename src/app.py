import streamlit as st
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag.config import CHROMA_PATH, RAG_MODE, RRF_MIN_SCORE, TOP_N, bm25_path
from rag.bm25_index import load_or_build_bm25
from rag.embeddings import E5Embedder
from rag.retrieval import retrieve

logger = logging.getLogger(__name__)

RAG_MODES = ("v1", "v2", "v2_table")
RAG_MODE_LABELS = {
    "v1": "v1 — baseline (top-5+5, BM25 clásico)",
    "v2": "v2 — BM25Log1 + pool 50+50",
    "v2_table": "v2_table — v2 sobre índice table_hybrid",
}


def collection_for_mode(mode: str) -> str:
    if os.getenv("CHROMA_COLLECTION"):
        return os.getenv("CHROMA_COLLECTION")
    if mode == "v2_table":
        return "cosora_actas_e5_v2"
    return "cosora_actas_e5"


st.set_page_config(page_title="COSORA RAG Demo", page_icon="🏗️", layout="centered")


def check_password():
    def password_entered():
        correct_password = os.getenv("APP_PASSWORD") or st.secrets.get("APP_PASSWORD")
        if st.session_state["password"] == correct_password:
            st.session_state["password_correct"] = True
            del st.session_state["password"]
        else:
            st.session_state["password_correct"] = False

    if st.session_state.get("password_correct", False):
        return True

    st.title("🔒 Acceso Restringido")
    st.text_input(
        "Introduce la contraseña para acceder a la demo:",
        type="password",
        on_change=password_entered,
        key="password",
    )
    if "password_correct" in st.session_state and not st.session_state["password_correct"]:
        st.error("😕 Contraseña incorrecta")
    return False


if not check_password():
    st.stop()

with st.sidebar:
    st.subheader("Modo RAG")
    default_mode = RAG_MODE if RAG_MODE in RAG_MODES else "v2"
    rag_mode = st.selectbox(
        "Pipeline de retrieval",
        RAG_MODES,
        index=RAG_MODES.index(default_mode),
        format_func=lambda m: RAG_MODE_LABELS[m],
        key="rag_mode",
    )
    collection_name = collection_for_mode(rag_mode)
    st.caption(f"Colección: `{collection_name}`")
    st.caption("Override: variable `CHROMA_COLLECTION` o `RAG_MODE` en .env")


@st.cache_resource
def load_resources(_rag_mode: str, _collection_name: str):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(name=_collection_name)

    all_data = collection.get(include=["documents", "metadatas"])
    all_docs = all_data["documents"]
    all_metas = all_data["metadatas"]
    chunk_ids = [m["chunk_id"] for m in all_metas]

    bm25_file = bm25_path(CHROMA_PATH, collection_name=_collection_name)
    bm25_v2 = load_or_build_bm25(
        CHROMA_PATH, all_docs, chunk_ids, bm25_file=bm25_file, logger=logger
    )
    bm25_v1 = BM25Okapi([doc.lower().split() for doc in all_docs])

    model_path = os.getenv("HF_MODEL_PATH", "./hf_model")
    embedder = E5Embedder.from_pretrained(model_path, local_files_only=True)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Falta la variable de entorno OPENAI_API_KEY.")
        st.stop()
    openai_client = OpenAI(api_key=api_key)

    return collection, all_docs, all_metas, bm25_v2, bm25_v1, embedder, openai_client


with st.spinner("Cargando la base de datos de actas y modelos..."):
    try:
        collection, all_docs, all_metas, bm25_v2, bm25_v1, embedder, openai_client = load_resources(
            rag_mode, collection_name
        )
    except Exception as e:
        st.error(f"No se pudo cargar la colección `{collection_name}`: {e}")
        st.stop()


def ask_cosora(query: str, mode: str):
    retrieval_mode = "v1" if mode == "v1" else "v2"
    hits = retrieve(
        query,
        retrieval_mode,
        collection,
        embedder,
        bm25_v2,
        bm25_v1,
        all_docs,
        all_metas,
    )

    if not hits or hits[0]["score"] < RRF_MIN_SCORE:
        return "No he encontrado información relevante en las actas para responder a tu pregunta.", []

    context_blocks = []
    for i, chunk in enumerate(hits, 1):
        doc_id = chunk["meta"]["doc_id"]
        context_blocks.append(f"--- Documento {i} (Origen: {doc_id}) ---\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    system_prompt = (
        "Eres COSORA, un asistente experto en ingeniería civil especializado en actas de obra del Proyecto UPCSchool.\n"
        "Básate ÚNICAMENTE en el siguiente contexto extraído de actas oficiales para responder.\n"
        "Si la información no está en el contexto, di claramente 'No dispongo de esa información en las actas actuales'."
    )

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": f"CONTEXTO DE ACTAS:\n{context_str}\n\nPREGUNTA DEL USUARIO:\n{query}"},
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content, hits


st.title("🏗️ COSORA: Asistente de Actas de Obra")
st.markdown(
    "Hazme cualquier pregunta sobre las actas de la obra Variante de Vallirana "
    "(Drenajes, Estructuras, Expropiaciones, etc.)"
)
st.caption(f"Modo: **{RAG_MODE_LABELS[rag_mode]}** · `{collection_name}`")

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            with st.expander("Ver fuentes documentales"):
                for source in message["sources"]:
                    st.caption(f"**Origen:** `{source['meta']['doc_id']}` (Score RRF: {source['score']:.4f})")
                    st.text(source["text"])

if prompt := st.chat_input("Ej: ¿Qué decisiones se tomaron sobre el talud?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Buscando en las actas..."):
            answer, sources = ask_cosora(prompt, rag_mode)
            st.markdown(answer)
            if sources:
                with st.expander("Ver fuentes documentales"):
                    for source in sources:
                        st.caption(f"**Origen:** `{source['meta']['doc_id']}` (Score RRF: {source['score']:.4f})")
                        st.text(source["text"])

    st.session_state.messages.append({"role": "assistant", "content": answer, "sources": sources})

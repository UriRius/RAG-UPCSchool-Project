import streamlit as st
import os
import sys
import logging
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))

import chromadb
from openai import OpenAI
from rank_bm25 import BM25Okapi

from rag.config import (
    CHROMA_PATH,
    EMBEDDING_STYLE,
    EMBED_MODEL_NAME,
    GRAPH_MODES,
    HF_MODEL_DIR,
    RR_MODEL_PATH,
    bm25_path,
)
from rag.bm25_index import BM25Log1, load_or_build_bm25
from rag.embeddings import E5Embedder
from rag.graph import GraphRetrievalResult, retrieve_graph
from rag.graph.neo4j_client import get_driver, neo4j_configured
from rag.graph.stats import fetch_graph_overview, fetch_ego_network, fetch_batch_hubs
from rag.graph.prompts import build_graph_prompt
from rag.query_rewrite import hybrid_retrieve
from rag.retrieval import normalize_meta, retrieve
from ui import RagSettings, SidebarStatus, collection_for_mode, render_sidebar_config, render_sidebar_status
from ui.labels import RAG_MODE_LABELS
from ui.sources import render_sources

logger = logging.getLogger(__name__)


def neo4j_status() -> str:
    if not neo4j_configured():
        return "No configurado"
    try:
        drv = get_driver()
        drv.verify_connectivity()
        drv.close()
        return "Conectado"
    except Exception:
        return "Error de conexión"


st.set_page_config(page_title="COSORA Graph RAG Demo", page_icon="🏗️", layout="centered")


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
    settings = render_sidebar_config()

collection_name = collection_for_mode(settings.rag_mode)


def _load_collection_rows(collection) -> tuple[list[str], list[dict], dict]:
    """Carga todos los chunks de Chroma (columnar → filas validadas)."""
    all_docs: list[str] = []
    all_metas: list[dict] = []
    offset = 0
    page_size = 5000
    while True:
        batch = collection.get(
            include=["documents", "metadatas"],
            limit=page_size,
            offset=offset,
        )
        docs = batch.get("documents") or []
        metas = batch.get("metadatas") or []
        if not docs and not metas:
            break
        if not isinstance(docs, list) or not isinstance(metas, list):
            raise TypeError(
                f"Chroma devolvió tipos inesperados: documents={type(docs).__name__}, "
                f"metadatas={type(metas).__name__}"
            )
        for doc, meta in zip(docs, metas):
            if doc is None or meta is None:
                continue
            if not isinstance(doc, str):
                raise TypeError(f"documento Chroma no es str: {type(doc).__name__}")
            all_docs.append(doc)
            all_metas.append(normalize_meta(meta))
        if len(docs) < page_size:
            break
        offset += len(docs)
    chunk_by_id = {m["chunk_id"]: (doc, m) for doc, m in zip(all_docs, all_metas)}
    return all_docs, all_metas, chunk_by_id


@st.cache_resource
def load_resources(_collection_name: str):
    chroma_path = Path(CHROMA_PATH)
    if not chroma_path.is_dir():
        raise FileNotFoundError(f"No existe carpeta Chroma: {chroma_path}")

    client = chromadb.PersistentClient(path=str(chroma_path))
    collection = client.get_collection(name=_collection_name)
    all_docs, all_metas, chunk_by_id = _load_collection_rows(collection)
    chunk_ids = [m["chunk_id"] for m in all_metas]

    bm25_file = bm25_path(str(chroma_path), collection_name=_collection_name)
    if (
        Path(bm25_file).is_file()
        and os.getenv("REBUILD_BM25", "0") != "1"
    ):
        bm25_v2 = BM25Log1.load(bm25_file)
        logger.info("BM25 cargado desde %s (%d docs)", bm25_file, len(bm25_v2.chunk_ids))
    elif os.getenv("REBUILD_BM25", "1") == "1":
        from rag.bm25_index import build_bm25_index

        bm25_v2 = build_bm25_index(all_docs, chunk_ids)
        logger.info("BM25 reconstruido desde Chroma (%d docs)", len(all_docs))
    else:
        bm25_v2 = load_or_build_bm25(
            str(chroma_path), all_docs, chunk_ids, bm25_file=bm25_file, logger=logger
        )
    bm25_v1 = BM25Okapi([doc.lower().split() for doc in all_docs])

    model_path = os.getenv("HF_MODEL_PATH", HF_MODEL_DIR)
    try:
        embedder = E5Embedder.from_pretrained(
            model_path, local_files_only=True, style=EMBEDDING_STYLE
        )
    except OSError:
        embedder = E5Embedder.from_pretrained(EMBED_MODEL_NAME, style=EMBEDDING_STYLE)

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        st.error("Falta la variable de entorno OPENAI_API_KEY.")
        st.stop()
    openai_client = OpenAI(api_key=api_key)

    return collection, all_docs, all_metas, chunk_by_id, bm25_v2, bm25_v1, embedder, openai_client


@st.cache_data(ttl=300)
def load_graph_overview() -> dict | None:
    if not neo4j_configured():
        return None
    try:
        driver = get_driver()
        overview = fetch_graph_overview(driver)
        driver.close()
        return overview
    except Exception:
        logger.exception("No se pudieron cargar stats Neo4j")
        return None


@st.cache_data(ttl=120)
def load_ego_triples(entity_query: str) -> list[tuple[str, str, str]]:
    if not neo4j_configured() or not entity_query.strip():
        return []
    try:
        driver = get_driver()
        triples = fetch_ego_network(driver, entity_query)
        driver.close()
        return triples
    except Exception:
        logger.exception("Ego network failed for %r", entity_query)
        return []


@st.cache_data(ttl=300)
def load_batch_hubs(batch: str) -> list[dict]:
    if not neo4j_configured() or not batch:
        return []
    try:
        driver = get_driver()
        hubs = fetch_batch_hubs(driver, batch)
        driver.close()
        return hubs
    except Exception:
        logger.exception("Batch hubs failed for %r", batch)
        return []


@st.cache_resource
def load_reranker():
    path = Path(RR_MODEL_PATH)
    if not (path / "model.pt").is_file():
        logger.warning("Reranker no encontrado en %s", path)
        return None
    from e5_reranker import E5Reranker

    return E5Reranker.load(str(path))


@st.cache_resource
def load_neo4j_driver():
    return get_driver()


with st.spinner("Cargando Chroma, BM25 y modelos..."):
    try:
        (
            collection,
            all_docs,
            all_metas,
            chunk_by_id,
            bm25_v2,
            bm25_v1,
            embedder,
            openai_client,
        ) = load_resources(collection_name)
        reranker = load_reranker()
    except Exception as e:
        st.error(f"No se pudo cargar la colección `{collection_name}`: {e}")
        st.caption(f"Ruta Chroma: `{CHROMA_PATH}`")
        with st.expander("Detalle técnico"):
            import traceback
            st.code(traceback.format_exc())
        st.stop()

with st.sidebar:
    render_sidebar_status(
        SidebarStatus(
            collection_name=collection_name,
            n_chunks=len(all_metas),
            embedding_style=EMBEDDING_STYLE,
            neo4j_status=neo4j_status(),
            graph_stats=load_graph_overview(),
            reranker_loaded=reranker is not None,
            last_debug=st.session_state.get("last_pipeline_debug"),
            fetch_ego=load_ego_triples,
            fetch_batch_hubs=load_batch_hubs,
        )
    )


def _active_reranker(cfg: RagSettings):
    if not cfg.rerank_enabled or reranker is None:
        return None
    return reranker


def _classic_system_prompt() -> str:
    return (
        "Eres COSORA, un asistente experto en ingeniería civil especializado en actas de obra del Proyecto UPCSchool.\n"
        "Básate ÚNICAMENTE en el siguiente contexto extraído de actas oficiales para responder.\n"
        "Si la información no está en el contexto, di claramente 'No dispongo de esa información en las actas actuales'."
    )


def _generate_classic(query: str, hits: list[dict]) -> str:
    context_blocks = []
    for i, chunk in enumerate(hits, 1):
        doc_id = chunk["meta"].get("doc_id") or chunk["meta"].get("docx_id", "?")
        context_blocks.append(f"--- Documento {i} (Origen: {doc_id}) ---\n{chunk['text']}")
    context_str = "\n\n".join(context_blocks)

    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {"role": "system", "content": _classic_system_prompt()},
            {
                "role": "user",
                "content": f"CONTEXTO DE ACTAS:\n{context_str}\n\nPREGUNTA DEL USUARIO:\n{query}",
            },
        ],
        temperature=0.0,
    )
    return response.choices[0].message.content


def _generate_graph(query: str, hits: list[dict]) -> str:
    prompt = build_graph_prompt(query, hits)
    response = openai_client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=800,
    )
    return (response.choices[0].message.content or "").strip()


def _hits_below_threshold(hits: list[dict], min_rrf: float) -> bool:
    if not hits:
        return True
    if hits[0].get("rerank_score") is not None:
        return False
    return hits[0]["score"] < min_rrf


def _record_pipeline_debug(
    hits: list[dict],
    graph_result: GraphRetrievalResult | None,
    cfg: RagSettings,
) -> None:
    dbg: dict = {
        "n_hits": len(hits),
        "rerank_used": cfg.rerank_enabled and reranker is not None,
        "query_rewrite_used": cfg.query_rewrite_enabled,
        "preset": cfg.preset_id,
        "rag_mode": cfg.rag_mode,
    }
    if graph_result and graph_result.debug:
        g = graph_result.debug
        dbg["n_graph_chunks"] = g.get("n_graph_chunks")
        dbg["n_triples"] = g.get("n_triples")
        dbg["bridge"] = g.get("graph_bridge")
        dbg["fallback_baseline"] = g.get("fallback_baseline")
        dbg["cypher_route_used"] = g.get("cypher_route_used")
    st.session_state.last_pipeline_debug = dbg


def ask_cosora(query: str, cfg: RagSettings):
    active_rr = _active_reranker(cfg)
    if cfg.rag_mode in GRAPH_MODES:
        if cfg.rag_mode == "cypher_transversal" and not neo4j_configured():
            return (
                "El modo Graph RAG requiere Neo4j (`NEO4J_URI`, `NEO4J_PASSWORD`).",
                [],
                None,
            )
        try:
            driver = load_neo4j_driver() if cfg.rag_mode == "cypher_transversal" else None
            result: GraphRetrievalResult = retrieve_graph(
                query,
                cfg.rag_mode,
                driver=driver,
                collection=collection,
                embedder=embedder,
                bm25_v2=bm25_v2,
                bm25_v1=bm25_v1,
                all_docs=all_docs,
                all_metas=all_metas,
                chunk_by_id=chunk_by_id,
                top_n=cfg.top_n,
                retrieval_k=cfg.retrieval_k,
                rrf_k=cfg.rrf_k,
                graph_merge_min_hits=cfg.graph_merge_min_hits,
                cypher_route=cfg.cypher_route if cfg.rag_mode == "cypher_transversal" else None,
                openai_client=openai_client,
                reranker=active_rr,
                rerank_pool_n=cfg.rerank_pool_n,
                query_rewrite=cfg.query_rewrite_enabled,
            )
        except Exception as exc:
            logger.exception("Graph RAG retrieval failed")
            return f"No se pudo consultar el grafo: {exc}", [], None

        hits = result.hits
        if _hits_below_threshold(hits, cfg.rrf_min_score):
            _record_pipeline_debug([], result, cfg)
            return (
                "No he encontrado información relevante en las actas para responder a tu pregunta.",
                [],
                result,
            )
        _record_pipeline_debug(hits, result, cfg)
        return _generate_graph(query, hits), hits, result

    retrieval_mode = "v1" if cfg.rag_mode == "v1" else "v2"
    if cfg.rag_mode == "v2" and cfg.query_rewrite_enabled:
        hits, rewrite_dbg = hybrid_retrieve(
            query,
            collection,
            embedder,
            bm25_v2,
            all_docs,
            all_metas,
            openai_client=openai_client,
            query_rewrite=True,
            retrieval_k=cfg.retrieval_k,
            top_n=cfg.top_n,
            rrf_k=cfg.rrf_k,
            reranker=active_rr,
            rerank_pool_n=cfg.rerank_pool_n,
        )
        graph_result = GraphRetrievalResult(hits=hits, debug={**rewrite_dbg, "baseline_hits": hits})
    else:
        hits = retrieve(
            query,
            retrieval_mode,
            collection,
            embedder,
            bm25_v2,
            bm25_v1,
            all_docs,
            all_metas,
            retrieval_k=cfg.retrieval_k,
            top_n=cfg.top_n,
            rrf_k=cfg.rrf_k,
            reranker=active_rr,
            rerank_pool_n=cfg.rerank_pool_n,
        )
        graph_result = None

    if _hits_below_threshold(hits, cfg.rrf_min_score):
        _record_pipeline_debug([], graph_result, cfg)
        return "No he encontrado información relevante en las actas para responder a tu pregunta.", [], graph_result

    _record_pipeline_debug(hits, graph_result, cfg)
    return _generate_classic(query, hits), hits, graph_result


st.title("🏗️ COSORA Graph RAG")
st.markdown(
    "Asistente sobre actas de obra — **Neo4j** (triples + provenance) + **Chroma** (híbrido denso/BM25)."
)
st.caption(
    f"Preset: **{settings.preset_id}** · Modo: **{RAG_MODE_LABELS.get(settings.rag_mode, settings.rag_mode)}** · "
    f"`{collection_name}` · top-{settings.top_n} · pool-{settings.retrieval_k} · "
    f"embed={EMBEDDING_STYLE} · rerank={'on' if settings.rerank_enabled and reranker else 'off'} · "
    f"rewrite={'on' if settings.query_rewrite_enabled else 'off'}"
)

if "messages" not in st.session_state:
    st.session_state.messages = []

for message in st.session_state.messages:
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if "sources" in message:
            render_sources(message["sources"], message.get("graph_result"))

if prompt := st.chat_input("Ej: ¿Qué incidencias hay sobre el talud?"):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    with st.chat_message("assistant"):
        with st.spinner("Consultando grafo y actas..."):
            answer, sources, graph_result = ask_cosora(prompt, settings)
            st.markdown(answer)
            render_sources(sources, graph_result)

    st.session_state.messages.append(
        {
            "role": "assistant",
            "content": answer,
            "sources": sources,
            "graph_result": graph_result,
        }
    )

from __future__ import annotations

from rag.graph.text_utils import strip_doc_prefix


def build_graph_prompt(query: str, chunks: list[dict]) -> str:
    blocks = []
    for index, chunk in enumerate(chunks, 1):
        meta = chunk["meta"]
        doc_id = meta.get("doc_id") or meta.get("docx_id", "")
        chunk_id = chunk["meta"].get("chunk_id", "")
        text = strip_doc_prefix(chunk["text"])
        blocks.append(f"[Fragmento {index} - Fuente: {doc_id} | chunk: {chunk_id}]\n{text}")

    context = "\n".join(blocks)
    return f"""Eres COSORA, un asistente técnico especializado en análisis de actas de obra ferroviaria en España.

REGLAS:
1. Responde ÚNICAMENTE con información del CONTEXTO.
2. Si no está, dilo explícitamente.
3. Cita la fuente entre paréntesis: (Fuente: nombre_del_acta).

VOCABULARIO: DF=Dirección Facultativa, UTE=Unión Temporal de Empresas, DEO=Director de Ejecución de Obra.

=== CONTEXTO ===
{context}

=== PREGUNTA ===
{query}

=== RESPUESTA ==="""

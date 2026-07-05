from __future__ import annotations

import re

from openai import OpenAI

from rag.config import CYPHER_LLM_MODEL, CYPHER_ROUTE
from rag.graph.cypher import cypher_template_route, extract_seeds

CYPHER_SCHEMA_PROMPT = """
SCHEMA Neo4j COSORA v2:
- (:Entity {name: string, norm: string, batch: string})
- (a:Entity)-[r:RELATED {
    predicate: string, batch: string,
    source_doc: string, source_chunk_id: string
  }]->(b:Entity)

REGLAS:
- Solo MATCH, OPTIONAL MATCH, WHERE, RETURN, ORDER BY, LIMIT
- Prohibido: CREATE, MERGE, SET, DELETE, DETACH, DROP, CALL db.*
- LIMIT <= 50
- Buscar entidades: a.norm CONTAINS 'kw' OR b.norm CONTAINS 'kw' (minúsculas)
- Une condiciones con OR; no uses AND entre entidad y predicado
- NO uses r.predicate CONTAINS con palabras de la pregunta
- RETURN siempre: subject, predicate, object, source_doc, source_chunk_id

EJEMPLO BUENO:
P: ¿Qué incidencias hay sobre el talud?
Keywords: talud, incidencia
C:
MATCH (a:Entity)-[r:RELATED]->(b:Entity)
WHERE a.norm CONTAINS 'talud' OR b.norm CONTAINS 'talud'
   OR a.norm CONTAINS 'incidencia' OR b.norm CONTAINS 'incidencia'
RETURN a.name AS subject, r.predicate AS predicate, b.name AS object,
       r.source_doc AS source_doc, r.source_chunk_id AS source_chunk_id
LIMIT 50
"""

FORBIDDEN = re.compile(
    r"\b(CREATE|MERGE|SET|DELETE|DETACH|DROP|REMOVE|FOREACH|LOAD\s+CSV)\b",
    re.IGNORECASE,
)


def validate_cypher(cypher: str) -> tuple[bool, str]:
    text = cypher.strip().rstrip(";")
    if not text.upper().startswith("MATCH") and not text.upper().startswith("OPTIONAL"):
        return False, "Debe empezar por MATCH"
    if "RETURN" not in text.upper():
        return False, "Falta RETURN"
    if FORBIDDEN.search(text):
        return False, "Operación prohibida detectada"
    if "LIMIT" not in text.upper():
        text += "\nLIMIT 50"
    return True, text


def generate_cypher_llm(query: str, *, client: OpenAI) -> str:
    keywords = extract_seeds(query)
    kw_hint = ", ".join(keywords) if keywords else "(extrae palabras clave de la pregunta)"
    prompt = (
        CYPHER_SCHEMA_PROMPT
        + f"\nKeywords detectadas: {kw_hint}\n"
        + "Usa SOLO estas keywords en a.norm CONTAINS o b.norm CONTAINS (minúsculas, unidas con OR).\n"
        + f"\nP: {query}\nC:\n"
    )
    resp = client.chat.completions.create(
        model=CYPHER_LLM_MODEL,
        messages=[{"role": "user", "content": prompt}],
        temperature=0.0,
        max_tokens=400,
    )
    raw = (resp.choices[0].message.content or "").strip()
    raw = re.sub(r"^```(?:cypher)?\s*", "", raw, flags=re.IGNORECASE)
    raw = re.sub(r"\s*```$", "", raw)
    return raw.strip()


def resolve_cypher_query(
    query: str,
    *,
    route: str | None = None,
    openai_client: OpenAI | None = None,
) -> tuple[str, dict, str]:
    """Devuelve (cypher, params, route_used)."""
    route = (route or CYPHER_ROUTE).lower()
    if route == "template":
        cypher, params = cypher_template_route(query)
        return cypher, params, "template"
    if route == "llm":
        if openai_client is None:
            raise ValueError("OpenAI client requerido para ruta llm")
        raw = generate_cypher_llm(query, client=openai_client)
        ok, fixed = validate_cypher(raw)
        if not ok:
            raise ValueError(f"Cypher LLM inválido: {fixed}")
        return fixed, {}, "llm"
    if route == "hybrid":
        if openai_client is not None:
            try:
                raw = generate_cypher_llm(query, client=openai_client)
                ok, fixed = validate_cypher(raw)
                if ok:
                    return fixed, {}, "llm"
            except Exception:
                pass
        cypher, params = cypher_template_route(query)
        return cypher, params, "template"
    raise ValueError(f"CYPHER_ROUTE desconocida: {route}")

from __future__ import annotations

import re

DOC_PREFIX = "passage: "

STOPWORDS_ES = {
    "de", "la", "que", "el", "en", "y", "a", "los", "del", "se", "las", "por", "un",
    "para", "con", "no", "una", "su", "al", "es", "lo", "como", "más", "pero", "sus",
    "le", "ya", "o", "este", "sí", "porque", "esta", "entre", "cuando", "muy", "sin",
    "sobre", "también", "me", "hasta", "hay", "donde", "quien", "desde", "todo", "nos",
    "durante", "todos", "uno", "les", "ni", "contra", "otros", "ese", "eso", "ante",
}

_NORMALIZE_RE = re.compile(r"[^\wáéíóúñü]+", re.IGNORECASE)
_LEGACY_CHUNK_ID_RE = re.compile(r"^(?P<doc>.+)__c(?P<idx>\d+)$", re.IGNORECASE)


def normalize_doc_key(name: str) -> str:
    """Clave comparable entre Neo4j source_doc y Chroma docx_id."""
    if not name:
        return ""
    key = name.strip().lower()
    for ext in (".docx", ".doc", ".pdf"):
        if key.endswith(ext):
            key = key[: -len(ext)]
    key = re.sub(r"\s+\(\d+\)$", "", key)
    key = re.sub(r"\s+", " ", key).strip()
    return key


def parse_legacy_chunk_id(chunk_id: str) -> tuple[str, int] | None:
    """Parsea IDs v2 del grafo: ``acta__c0005`` → (acta, 5)."""
    match = _LEGACY_CHUNK_ID_RE.match((chunk_id or "").strip())
    if not match:
        return None
    return normalize_doc_key(match.group("doc")), int(match.group("idx"))


def strip_doc_prefix(text: str) -> str:
    if text.startswith(DOC_PREFIX):
        return text[len(DOC_PREFIX) :]
    return text


def tokenize_match(text: str) -> set[str]:
    return {
        token
        for token in _NORMALIZE_RE.split(text.lower())
        if token and len(token) > 2 and token not in STOPWORDS_ES
    }

#!/usr/bin/env python3
"""Propone source_chunk_id y valida source_docs en gold v2 (Chroma).

Uso (Colab):
  python scripts/annotate_gold_chunks.py \\
    --chroma-path /content/drive/MyDrive/RAG_UPC_Final_project/chroma_db

  python scripts/annotate_gold_chunks.py --chroma-path ... --write

Requiere: chromadb
"""

from __future__ import annotations

import argparse
import json
import re
import unicodedata
from pathlib import Path


def norm_text(text: str) -> str:
    text = unicodedata.normalize("NFKD", text)
    text = "".join(c for c in text if not unicodedata.combining(c))
    return re.sub(r"\s+", " ", text.lower().strip())


def norm_doc_stem(name: str | None) -> str:
    if not name:
        return ""
    return Path(name).stem.lower()


def load_gold(path: Path) -> tuple[dict, list[dict]]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if isinstance(data, dict) and "queries" in data:
        return data, data["queries"]
    if isinstance(data, list):
        return {"schema_version": 1, "queries": data}, data
    raise ValueError(f"Formato gold no reconocido: {path}")


def extract_terms(*texts: str, min_len: int = 4, max_terms: int = 14) -> list[str]:
    stop = {
        "para", "como", "sobre", "desde", "hasta", "donde", "durante", "toda",
        "todo", "todos", "cada", "esta", "este", "estos", "estas", "con", "sin",
        "por", "que", "del", "las", "los", "una", "uno", "son", "hay", "fue",
        "ser", "sus", "ese", "esa", "the", "and", "conforme", "durante",
    }
    terms: list[str] = []
    for text in texts:
        if not text:
            continue
        for tok in re.findall(r"\w+", norm_text(text)):
            if len(tok) < min_len or tok in stop:
                continue
            if tok not in terms:
                terms.append(tok)
            if len(terms) >= max_terms:
                return terms
    return terms


def score_chunk(text: str, terms: list[str]) -> float:
    body = norm_text(text.removeprefix("passage: "))
    if not terms:
        return 0.0
    hits = sum(1 for t in terms if t in body)
    return hits / len(terms)


def chunk_preview(text: str, limit: int = 120) -> str:
    body = text.removeprefix("passage: ").replace("\n", " ")
    return body[:limit] + ("…" if len(body) > limit else "")


def find_best_chunk(
    collection,
    *,
    doc_stem: str,
    terms: list[str],
) -> tuple[str | None, float, str]:
    res = collection.get(include=["documents", "metadatas"])
    best_score, best_id, best_preview = 0.0, None, ""
    for doc, meta in zip(res["documents"], res["metadatas"]):
        if doc_stem and norm_doc_stem(meta.get("doc_id", "")) != doc_stem:
            continue
        sc = score_chunk(doc, terms)
        if sc > best_score:
            best_score = sc
            best_id = meta.get("chunk_id")
            best_preview = chunk_preview(doc)
    return best_id, best_score, best_preview


def validate_any_doc(collection, row: dict) -> dict:
    docs = row.get("source_docs") or []
    terms = extract_terms(row.get("expected_answer", ""), row.get("query", ""))
    per_doc = []
    for doc in docs:
        cid, score, preview = find_best_chunk(
            collection, doc_stem=norm_doc_stem(doc), terms=terms
        )
        per_doc.append({
            "source_doc": doc,
            "best_chunk_id": cid,
            "score": round(score, 3),
            "preview": preview,
        })
    ok_docs = [d for d in per_doc if d["score"] >= 0.25]
    return {
        "id": row["id"],
        "eval_mode": row.get("eval_mode"),
        "n_source_docs": len(docs),
        "n_docs_with_signal": len(ok_docs),
        "per_doc": per_doc,
        "ok": len(ok_docs) >= 1,
    }


def annotate_single_queries(collection, queries: list[dict], *, min_score: float) -> list[dict]:
    rows = []
    for q in queries:
        qid = q["id"]
        mode = q.get("eval_mode", "single_doc")
        if mode not in ("single_doc", "single_chunk"):
            continue
        if q.get("source_chunk_id"):
            rows.append({
                "id": qid,
                "status": "skip",
                "source_chunk_id": q["source_chunk_id"],
            })
            continue

        doc_stem = norm_doc_stem(q.get("source_doc"))
        terms = extract_terms(q.get("expected_answer", ""), q.get("query", ""))
        best_id, best_score, preview = find_best_chunk(
            collection, doc_stem=doc_stem, terms=terms
        )
        accepted = bool(best_id and best_score >= min_score)
        row = {
            "id": qid,
            "source_doc": q.get("source_doc"),
            "terms": terms[:8],
            "proposed_chunk_id": best_id,
            "score": round(best_score, 3),
            "accepted": accepted,
            "preview": preview,
        }
        rows.append(row)
        mark = "✓" if accepted else "?"
        print(f"{mark} {qid}: {best_id or '—'}  score={best_score:.2f}")
    return rows


def apply_annotations(queries: list[dict], rows: list[dict], *, min_score: float) -> int:
    by_id = {r["id"]: r for r in rows if r.get("proposed_chunk_id")}
    n = 0
    for q in queries:
        row = by_id.get(q["id"])
        if not row or not row.get("accepted"):
            continue
        if row["score"] < min_score:
            continue
        q["source_chunk_id"] = row["proposed_chunk_id"]
        q["eval_mode"] = "single_chunk"
        q["notes"] = f"Anotado auto score={row['score']} — revisar manualmente"
        n += 1
    return n


def main() -> None:
    parser = argparse.ArgumentParser(description="Bloque B: anotar gold v2 desde Chroma")
    parser.add_argument("--gold", type=Path, default=Path("docs/rag_eval_queries.json"))
    parser.add_argument("--chroma-path", type=Path, required=True)
    parser.add_argument("--collection", default="cosora_actas_e5")
    parser.add_argument("--min-score", type=float, default=0.35)
    parser.add_argument("--write", action="store_true", help="Escribir chunk_ids aceptados en gold JSON")
    args = parser.parse_args()

    import chromadb

    wrapper, queries = load_gold(args.gold)
    client = chromadb.PersistentClient(path=str(args.chroma_path))
    collection = client.get_collection(args.collection)
    n_chunks = collection.count()
    print(f"Chroma: {n_chunks} chunks  collection={args.collection}")

    print("\n=== Q1–Q17: proponer source_chunk_id ===")
    single_rows = annotate_single_queries(collection, queries, min_score=args.min_score)

    print("\n=== Q18–Q21: validar source_docs ===")
    multi_rows = []
    for q in queries:
        if q.get("eval_mode") not in ("any_doc", "cross_doc_aggregate"):
            continue
        vr = validate_any_doc(collection, q)
        multi_rows.append(vr)
        mark = "✓" if vr["ok"] else "✗"
        print(
            f"{mark} {vr['id']}: {vr['n_docs_with_signal']}/{vr['n_source_docs']} "
            f"actas con señal (eval_mode={vr['eval_mode']})"
        )

    if args.write:
        n = apply_annotations(queries, single_rows, min_score=args.min_score)
        wrapper["queries"] = queries
        args.gold.write_text(
            json.dumps(wrapper, ensure_ascii=False, indent=2) + "\n",
            encoding="utf-8",
        )
        print(f"\n✅ Gold actualizado ({n} queries → single_chunk) → {args.gold}")
    else:
        print("\n(dry-run — añade --write para guardar)")

    report = {
        "gold_path": str(args.gold),
        "chroma_path": str(args.chroma_path),
        "n_chunks": n_chunks,
        "min_score": args.min_score,
        "single_doc_annotations": single_rows,
        "multi_doc_validation": multi_rows,
    }
    out = args.gold.parent / "gold_chunk_annotation_report.json"
    out.write_text(json.dumps(report, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(f"📄 Reporte → {out}")


if __name__ == "__main__":
    main()

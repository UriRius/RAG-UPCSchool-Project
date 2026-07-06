# Plan Graph RAG v2 — COSORA

Regeneración completa del grafo con provenance (doc + chunk), reindex Chroma alineado,
Neo4j wipe, notebooks v2 para experimentos, y **tres rutas Cypher** (plantilla / LLM / híbrido).

**Principio:** notebooks = experimentos · `src/` = prod (solo lo validado en notebooks).

### Convención Colab + Drive (siempre)

| Recurso | Colab | Local |
|---------|-------|-------|
| `.env` | `/content/drive/MyDrive/variablentorno/.env` | repo `.env` |
| Actas | `MyDrive/RAG_UPC_Final_project/` | `data/raw/` |
| Chroma | `.../RAG_UPC_Final_project/chroma_db` | `data/chroma_db/` |
| Grafos JSON | `.../RAG_UPC_Final_project/graph/` | `data/graph/` |
| Setup | `drive.mount` + `antiword` apt | antiword opcional |

Los notebooks v2 **no asumen repo clonado** en Colab: datos y artefactos viven en Drive.

---

## Decisiones cerradas

| ID | Decisión |
|----|----------|
| D0.1 | Schema relations **objeto** v2 |
| D0.2 | Provenance **source_doc + source_chunk_id** |
| D0.3 | **Regenerar todo** el grafo |
| D0.4 | Dedup `.doc`/`.docx` → un acta, preferir `.docx` |
| D0.5 | Refactor total, **sin legacy** ni fallback wordset |
| D0.6 | **Notebooks v2** aparte; promover a `src/` tras eval |
| D0.7 | Gold en curso: **`doc_ids` obligatorio**, `chunk_ids` opcional |
| D0.8 | GCP + Neo4j listos |
| P1 | KGGen **por chunk Chroma** (mismo chunking que ingest) |
| P2 | **KG_CLUSTER=True** |
| P3 | JSON: `graph_actas_e5_original_batch{N:02d}_v2.json` + `catalog.json` |
| P4 | **Reindex Chroma** (`recursive`, colección prod) |
| P5 | **Wipe Neo4j** + carga solo v2 |
| **CYP** | Cypher libre como **ruta con flag** (ver sección Rutas Cypher) |

---

## Arquitectura objetivo

```text
Actas (dedup) → Chroma reindex (recursive)
                    ↓ chunk_id, doc_id
              kg_ingest_v2 (KGGen/chunk + cluster)
                    ↓ JSON batchNN_v2.json
              Drive → sync_graph.py → Neo4j (wipe)
                    ↓
Query → [CYPHER_ROUTE] → triples + provenance
                    ↓
              lookup chunk_id en Chroma
                    ↓
              RRF con baseline v2 → prompt → gpt-4o-mini
```

**Nombre académico:** Graph RAG híbrido (KG Neo4j + vector Chroma + provenance).

---

## Rutas Cypher (flag)

Tres modos, mismo bridge provenance después de Neo4j:

| Flag | Valor | Descripción |
|------|-------|-------------|
| `CYPHER_ROUTE` | `template` | Plantillas fijas en `cypher.py` (prod baseline) |
| | `llm` | Text-to-Cypher: LLM genera query con schema + few-shots |
| | `hybrid` | LLM primero; si falla validación o 0 filas → `template` |

### Config (notebooks y prod)

```python
# Notebooks v2
CYPHER_ROUTE = "template"  # "template" | "llm" | "hybrid"

# Prod (env)
CYPHER_ROUTE=hybrid  # recomendado tras eval
```

```python
# src/rag/config.py (tras promoción)
CYPHER_ROUTE = os.getenv("CYPHER_ROUTE", "template").lower()
```

### Flujo por ruta

```text
template:
  extract_seeds(query) → CYPHER_FACTUAL / TRANSVERSAL_* → Neo4j

llm:
  prompt(schema + reglas + few-shots + query) → gpt-4o-mini
  → validar Cypher → Neo4j
  (sin fallback)

hybrid:
  llm → validar → Neo4j
  si error o rows vacías → template
```

### Validador Cypher (obligatorio para llm/hybrid)

- Solo `MATCH` / `OPTIONAL MATCH` / `WHERE` / `RETURN` / `ORDER BY` / `LIMIT`
- Prohibido: `CREATE`, `MERGE`, `SET`, `DELETE`, `DETACH`, `DROP`, `CALL db.*`
- Añadir `LIMIT 50` si falta
- Una sola sentencia

### Prompt LLM (schema v2)

El LLM recibe:

1. Schema: `(:Entity {name, norm, batch})`, `[:RELATED {predicate, source_doc, source_chunk_id, batch}]`
2. Reglas: buscar con `norm CONTAINS`, predicado en `r.predicate`, solo lectura
3. 3–5 few-shots del dominio COSORA (talud, UTE, incidencias, agregación)
4. Pregunta del usuario

### Modos app (prod, tras promoción)

| RAG_MODE | Cypher | Grafo |
|----------|--------|-------|
| `v2` | — | — |
| `graph_baseline` | — | — |
| `cypher_transversal` | `CYPHER_ROUTE=template` | Neo4j |
| `cypher_llm` | `CYPHER_ROUTE=llm` | Neo4j |
| `cypher_hybrid` | `CYPHER_ROUTE=hybrid` | Neo4j |

*(Nombres finales a confirmar en eval; puede ser un solo modo `cypher_transversal` + env `CYPHER_ROUTE`.)*

---

## Notebooks v2

| Notebook | Contenido |
|----------|-----------|
| **`kg_ingest_v2.ipynb`** | Dedup, reindex alineado, KGGen/chunk, cluster, schema v2, regen batches |
| **`neo4j_graph_rag_v2.ipynb`** | Wipe Neo4j, carga v2, bridge provenance, demo Cypher + RRF |
| **`graph_rag_eval_v2.ipynb`** | Gold set, métricas, comparativa template / llm / hybrid → `eval_report_v2.json` |

Notebooks originales **no se modifican**.

### Flags por notebook

```python
# kg_ingest_v2.ipynb
FORCE_REGEN = True
BATCH_ID = 1          # o loop ALL_BATCHES
KG_CLUSTER = True
CHUNK_STRATEGY = "recursive"

# neo4j_graph_rag_v2.ipynb
WIPE_NEO4J = True
CYPHER_ROUTE = "hybrid"   # cambiar para experimentos
BRIDGE_MODE = "provenance"  # sin wordset
```

---

## Schema JSON v2

```json
{
  "relations": [
    {
      "subject": "talud",
      "predicate": "estado",
      "object": "en ejecución",
      "source_doc": "244170-DOB-AVO-07-V00-A0",
      "source_chunk_id": "244170-DOB-AVO-07-V00-A0__c0003",
      "source_docs": [],
      "source_chunk_ids": []
    }
  ],
  "meta": {
    "schema_version": 2,
    "chunk_strategy": "recursive",
    "batch_id": 1,
    "kg_cluster": true,
    "sources": ["..."]
  }
}
```

Tras `cluster()`: rellenar arrays `source_docs` / `source_chunk_ids` con unión de orígenes fusionados.

---

## Fases de ejecución

### Fase 0 — Setup (medio día)

- [x] Crear `kg_ingest_v2.ipynb`, `neo4j_graph_rag_v2.ipynb`, `graph_rag_eval_v2.ipynb`
- [x] Generador: `scripts/build_graphrag_v2_notebooks.py` (regenerar notebooks tras cambios)
- [x] Utilidades ingest v2: dedup doc/docx, `chunk_document(recursive)`, schema v2 helpers
- [x] Flags documentados en celdas markdown (`CYPHER_ROUTE`, `FORCE_REGEN`, etc.)
- [x] Inventario corpus + validación Chroma (celdas listas; ejecutar en Colab/local)
- [x] Notebook `graph_rag_eval_v2` (template / llm / hybrid + métricas gold)
- [ ] Ejecutar eval en Colab con Neo4j real (Fase 5)
- [ ] Contar actas totales → N batches (**ejecutar** celda inventario en vuestro entorno)

### Fase 1 — Reindex Chroma (P4)

- [x] Celda §4 en `kg_ingest_v2.ipynb`: extract + E5 + Chroma + BM25 + `chunks_catalog_v2.json`
- [ ] Ejecutar en Colab con `REINDEX_CHROMA=True`
- [ ] Validación §5: 0 actas huérfanas, 0 duplicados

**Entregable:** Chroma en Drive + `graph/chunks_catalog_v2.json`.

### Fase 2 — KGGen por chunk (`kg_ingest_v2` §7–8b)

- [x] Celda §7: KGGen por chunk, aggregate, cluster, provenance, JSON v2
- [x] Celda §8: validación provenance + chunk_ids ∈ Chroma
- [x] Celda §8b: `refresh_catalog_v2` → `catalog.json`
- [ ] **Probar en Colab** (junto con Fase 1): `REINDEX_CHROMA=True`, `RUN_GRAPH_EXTRACT=True`
- [ ] Piloto `BATCH_ID=1` → medir tiempo/coste → `RUN_ALL_BATCHES=True` para regen total

**Entregable:** `batch01_v2.json` (+ resto batches) + `catalog.json` en Drive.

### Fase 3 — Regeneración grafo completa

- [x] `RUN_ALL_BATCHES=True` por defecto
- [x] `SKIP_COMPLETED_BATCHES` + `BATCH_IDS` para reanudar
- [x] `regen_report_v2.json` en Drive (§8b)
- [x] Manifests por batch (`batch_NN_manifest_v2.json`)
- [x] `catalog.json` v2 (§8c)
- [ ] Ejecutar en Colab (Fase 1 + 2 + 3 en una sesión)

**Entregable:** todos los `batchNN_v2.json` + `catalog.json` + `regen_report_v2.json`.

### Fase 4 — Neo4j + bridge (`neo4j_graph_rag_v2`)

- [x] §3 Wipe + carga `catalog.json` v2 con provenance en `RELATED`
- [x] §4 Chroma + BM25 baseline
- [x] §5 Bridge `source_chunk_id` → lookup Chroma + RRF
- [x] §5c Demo query (Cypher + retrieval + respuesta LLM)
- [ ] Ejecutar en Colab tras Fase 1–3 de `kg_ingest_v2`

**Entregable:** Neo4j cargado + demo end-to-end en notebook.

### Fase 5 — Rutas Cypher (`graph_rag_eval_v2`)

- [x] §0–1 Setup + stack Neo4j/Chroma/retrieval (autocontenido Colab+Drive)
- [x] §2 Métricas: `doc_recall@K`, `bridge_doc_hit`, `cypher_nonempty`, `fallback`
- [x] §3 Loop template / llm / hybrid sobre gold (23 queries)
- [x] §4 `eval_report_v2.json` en Drive + tabla resumen
- [ ] Ejecutar en Colab tras Fase 1–4
- [ ] Elegir ruta prod según resultados

**Entregable:** `graph/eval_report_v2.json` + recomendación de ruta Cypher.

### Fase 6 — Promoción a `src/`

- [ ] `schema.py` — load v2 + provenance
- [ ] `cypher.py` — plantillas (existente)
- [ ] `cypher_llm.py` — **nuevo**: generate + validate + few-shots
- [ ] `cypher_router.py` — **nuevo**: `resolve_cypher_route(query, driver)`
- [ ] `retrieval.py` — bridge provenance, sin wordset
- [ ] `config.py` — `CYPHER_ROUTE`, modos app
- [ ] `sync_graph.py` — wipe + sync v2

**Entregable:** PR listo para deploy.

### Fase 7 — Deploy

- [ ] Cloud Run Job sync Drive → Neo4j
- [ ] App con `RAG_MODE=cypher_transversal`, `CYPHER_ROUTE=hybrid` (o modo dedicado)
- [ ] UI: mostrar Cypher usado + ruta (`template`/`llm`)
- [ ] Smoke test 5 queries

### Fase 8 — Eval + memoria (paralelo desde Fase 5)

**Evaluación**

- [ ] Ampliar `docs/rag_eval_queries.json` → ~25 gold (`doc_ids`, `chunk_ids` opcional)
- [ ] Correr gold en `graph_rag_eval_v2.ipynb` por ruta Cypher
- [ ] LLM-judge secundario (correctness, faithfulness, citation)
- [ ] Filtrar ~50 queries de `docs/queries_dataset_training.json` (stress, no ground truth)
- [ ] Opcional: `scripts/eval_rag.py` tras notebook estable

**Memoria / documentación**

- [ ] `docs/ontology/cosora_v1.md` — **paralelo**, semanas 2–4
- [ ] Actualizar arquitectura: Graph RAG híbrido + provenance + rutas Cypher
- [ ] Sección memoria: por qué `hybrid` vs solo LLM vs solo plantillas
- [ ] Error analysis (queries donde falla cada ruta)
- [ ] Mejorar `src/rag/graph/prompts.py` tras retrieval estable

---

## Métricas de éxito

| Métrica | Objetivo orientativo |
|---------|----------------------|
| Provenance coverage | 100% triples con source_chunk_id |
| bridge_hit_rate (gold) | ↑ vs wordset histórico |
| chunk_recall@5 | ≥ baseline Chroma o mejor |
| `cypher_valid_rate` (llm) | ≥ 90% |
| `hybrid` vs `template` en gold | hybrid ≥ template en recall |

---

## Fuera de scope (v2)

| Tema | Motivo |
|------|--------|
| Wordset bridge | Sustituido por provenance |
| JSON v1 / legacy sub10 | Regen total |
| Microsoft GraphRAG / LightRAG | Otra arquitectura |
| Ollama / Gemma prod | No Cloud Run |
| `table_hybrid` en regen v2 | No mezclar variables; experimento posterior |
| KG multi-fuente (chunks/hybrid) | Solo `original` |
| Text-to-Cypher sin validador | Inseguro |
| NDCG LLM-judge como métrica principal | Gold humano primero |

---

## Orden resumido

```text
0. Setup notebooks v2 + flags
1. Reindex Chroma
2. Piloto batch01 grafo v2
3. Regen grafo total
4. Neo4j wipe + bridge provenance
5. Experimentar CYPHER_ROUTE (template / llm / hybrid)
6. Promover a src/
7. Deploy
8. Gold + memoria (paralelo)
```

---

## Referencias en repo

| Recurso | Uso |
|---------|-----|
| `docs/rag_eval_queries.json` | Gold parcial (23 queries) |
| `docs/queries.md` | Inspiración queries transversales |
| `docs/queries_dataset_training.json` | Synthetic secundario |
| `src/rag/graph/cypher.py` | Plantillas template route |
| `src/ingest.py` | Dedup + chunking alineado |

---

*Última actualización: plan v2 con rutas Cypher (template / llm / hybrid).*

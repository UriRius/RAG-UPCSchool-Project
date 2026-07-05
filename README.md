# COSORA — RAG & Knowledge Graph over Construction Site Visit Reports

> **Retrieval-Augmented Generation (RAG)** + **Knowledge Graph (KG)** demo that answers natural-language
> questions about construction site meeting minutes (*"actas de obra"*), backed by the official documents.

**🔗 Repository:** https://github.com/UriRius/RAG-UPCSchool-Project

Developed for **UPCSchool**, this project demonstrates how to integrate AI retrieval and structured
knowledge extraction into engineering and construction workflows. It has two complementary components:

1. A deployed **hybrid RAG demo** (Streamlit on GCP) that retrieves relevant document chunks and generates answers.
2. A **Knowledge Graph pipeline** (Graph-RAG) that extracts entities and relations from the same documents
   and exposes them for semantic and analytical querying.

---

# Table of Contents

1. [About the Project](#1-about-the-project)
2. [System Architecture](#2-system-architecture)
3. [How to Run the Code](#3-how-to-run-the-code)
4. [Experiments](#4-experiments)
   - [Part A — Retrieval](#part-a--retrieval-experiments)
   - [Part B — Knowledge Graph](#part-b--knowledge-graph-experiments)
5. [Overall Conclusions](#5-overall-conclusions)
6. [References](#6-references)
7. [Deliverables](#7-deliverables)

---

# 1. About the Project

Construction projects generate large volumes of unstructured meeting minutes documenting decisions,
incidents, responsibilities and progress. COSORA makes this knowledge queryable in natural language.

The system can answer questions such as:

- *¿Qué se decidió sobre el talud?*
- *¿Cuál es el estado del camino provisional?*
- *¿Qué documentación debe aportar la UTE sobre estabilidad?*

It combines **hybrid retrieval** (dense embeddings + BM25) for text answers with a **Knowledge Graph**
for structured reasoning, ranking and explainability.

# 2. System overview

Retrieval-Augmented Generation (RAG) enhances Large Language Models (LLMs) by retrieving relevant information from external knowledge sources during inference and incorporating it into the prompt. By complementing the model's parametric knowledge with external context, RAG addresses two key limitations of LLMs: their inability to access information beyond their training cutoff and their lack of access to organization-specific knowledge.

In this work, we focus on the latter by designing a RAG-based system capable of answering questions about the information contained in the construction site visit reports produced by a particular company.

Traditional RAG systems rely on chunking documents and retrieving the most relevant passages using either semantic similarity (through vector embeddings) or lexical similarity (through keyword-based retrieval methods). In our approach, this retrieval process is complemented by extracting additional contextual information from a knowledge graph automatically constructed from the same collection of documents. Both the retrieved document chunks and the graph-derived context are then provided to the LLM to support answer generation.


# 3. System Architecture

## 3.1 RAG Retrieval Pipeline

A **serverless architecture on Google Cloud Platform (GCP)** for low cost and high scalability:

| Layer | Technology |
|-------|------------|
| **Frontend** | [Streamlit](https://streamlit.io/) on Google Cloud Run |
| **Vector DB** | [ChromaDB](https://www.trychroma.com/) |
| **Embeddings** | `intfloat/multilingual-e5-base` (runs locally for hybrid search) |
| **Sparse retrieval** | BM25 (`rank_bm25`) |
| **Fusion** | Reciprocal Rank Fusion (RRF) |
| **Generation (LLM)** | OpenAI API (`gpt-4o-mini`) |

**Data pipeline (Cloud Run Job):**

1. Downloads `.doc` / `.docx` files from Google Drive.
2. Extracts text using `antiword` and `python-docx`.
3. Chunks the text, creates embeddings, and builds the ChromaDB database.
4. Uploads the ready-to-use database to Google Cloud Storage.

## 3.2 Knowledge Graph database

A graph-based information-extraction methodology that turns unstructured reports into a structured
knowledge representation, following four stages:

1. **Document ingestion & preprocessing** — `.doc`/`.docx` → plain text (`python-docx`, `textract`).
2. **Entity extraction** — prompt-based extraction with a local LLM (Ollama + Gemma 3).
3. **Relation extraction** — a second prompt enforces strict `(entity1, relation, entity2)` triples.
4. **Graph construction & consolidation** — normalization, deduplication and state prioritization,
   stored both in-memory and as JSON.

The resulting graph is integrated into a **hybrid Graph-RAG retrieval system** (dense + sparse + graph),
where relevant triples are injected into the prompt as structured context, improving reasoning and
explainability. The graph can also be queried directly with **Cypher** (Neo4j), with an LLM translating
results into natural language:

```
User Question → Cypher Query → Graph Results → LLM → Natural-Language Answer
```

---

Graphs represent entities as nodes and the ways in which those entities relate to the world as relationships.

KGs consist of a set of subject-predicate-object triples and have become a fundamental data structure for information retrieval. 

To solve this problem, we propose KGGen, a text-to-knowledge-graph generator that leverages Retrieval Augmented Generatin systems (RAG) and an algorithm for entity and edge resolution to extract high-quality, dense KGs from text. First, KGGen uses an LM-based extractor to read unstructured text and predict subject-predicate-object triples to capture entities and relations; after extracting the triples, it applies a novel, iterative clustering algorithm to refine the raw graph. Inspired by crowd-sourcing strategies for entity resolution, KGGen identifies nodes that refer to the same underlying entities, and consolidates edges that have equivalent meanings.

### Entity and Relation Extraction
The first stage takes unstructured text of construction site reports as input and produces an initial knowledge graph as extracted triples. We use a language model to provide structured output. The first step takes in source text and extracts a list of entities. Given the source text and entities list, the second step outputs a list of subject-predicate-object relations. 

### Aggregation
After extracting triples from each source text, we collect all the unique entities and edges across all source graphs and combine them into a single graph. All entities and edges are normalized to be in lowercase letters only. The aggregation step reduces redundancy in the KG. Aggregation step does not require an LLM.

### Entity and Edge Resolution
After extraction and aggregation, it typically have a raw graph containing duplicate or synonymous entities and possibly redundant edges. The resolution stage merges nodes and edges representing the same real-world entity or concept. The resolution process employs a two-stage approach combining embedding-based clustering with LLMbased de-duplication to efficiently handle large knowledge graphs. 

First, all items in the graph are clustered.

### Methodology
#### 1. Overview

This project follows a graph-based information extraction methodology designed to transform unstructured construction reports into a structured knowledge representation.
The approach consists of four main stages:
- Document ingestion
- Entity extraction
- Relation extraction
- Graph construction and consolidation

Each stage is supported by Language Models (LLMs) and rule-based post-processing to ensure consistency and quality of the resulting knowledge graph.

#### 2. Document Ingestion and Preprocessing
The input data consists of technical construction reports (actas and technical documents) in both .doc and .docx formats.
Documents are processed using text extraction tools:
•	python-docx for .docx
•	textract for .doc
The extracted content is converted into plain text, preserving as much structure as possible while removing formatting artifacts.
To ensure efficiency and compatibility with LLMs, the text is truncated into manageable segments before further processing.

#### 3. Entity Extraction
Entities are extracted using a prompt-based approach with a local Language Model (Ollama + Gemma 3).

A structured prompt is designed to enforce:

- Controlled output format (comma-separated list)
- Domain-specific entity types
- Short and normalized entity names

The LLM identifies relevant entities such as:

- construction elements
- organizations (e.g., UTE, DF)
- technical concepts
- actions and documentation

A post-processing step removes:

- duplicates
- noisy tokens
- invalid entity strings

This results in a clean set of entities representing the key concepts in each document.

#### 4. Relation Extraction
Relationships between entities are extracted using a second LLM prompt.

The prompt enforces a strict triple format:
(entity1, relation, entity2)

Additionally:

- a predefined list of allowed relations is provided
- the number of candidate entities is limited to improve precision
- instructions prevent explanations or free text

The output is parsed using robust pattern matching to extract valid triples while filtering malformed responses.
This step produces the core knowledge representation used to build the graph.

#### 5. Normalization and Deduplication
To improve graph quality, a normalization and deduplication process is applied:

##### Relation normalization
Semantically equivalent relations are unified:

- "realiza" → "ejecuta"
- "comienza" → "inicio"
- "termina" → "finalizado"

##### State prioritization
Special handling is applied to state-related relations (estado):

- conflicting states are resolved using priority rules
- the most advanced state (e.g., "finalizado") is retained

##### Duplicate removal
Triples are deduplicated using normalized keys:
(subject, relation, object)
This ensures that the graph remains consistent and avoids redundancy.

#### 6. Graph Construction
The final graph is constructed as:

- a set of nodes (entities)
- a set of directed edges (relations)

Each triple is converted into:

```source → relation → target```

The resulting structure is stored in two formats:

##### 1.	In-memory representation 

Used for processing and querying

##### 2.	JSON format 

Exported for interoperability and visualization
This graph serves as the foundation for both retrieval and analytical tasks.

#### 7. Integration with Retrieval Pipeline

The constructed graph is integrated into a hybrid retrieval system:

- Dense retrieval (embeddings)
- Sparse retrieval (BM25)
- Graph-based retrieval (LLM over triples)

Graph-based retrieval selects the most relevant triples for a given query and injects them into the prompt as structured context.

This enhances the system by:

- improving reasoning
- adding explainability
- capturing relationships not present in raw text retrieval

#### 8. Summary
The methodology combines:

- unstructured document processing
- LLM-based information extraction
- hybrid retrieval strategies

This pipeline enables the transformation of construction reports into a reusable knowledge representation, supporting both semantic reasoning and analytical exploration.

### Graph-Based Analysis — Content vs Database Usage

#### 1. Graph as Content (Semantic Knowledge)
In this stage, the extracted knowledge graph is used as a semantic content source, where each triple represents a meaningful piece of information derived from construction reports.

The graph is composed of triples of the form:
(subject, relation, object)

These triples encode domain knowledge such as:

- incidents affecting construction elements
- pending actions
- technical issues described in the reports

For example:

(talud, estado, inestable)
(UTE, solicita, documentación técnica)
(acta AVO-03, contiene, incidencia)

To demonstrate the use of the graph as content, the system performs a filtering process over the triples to identify those that contain keywords related to incidents (e.g., “incidencia”, “problema”, “defecto”, “pendiente”, “incumplimiento”).
By aggregating these triples, it is possible to determine which reports (actas) contain the highest number of issues.

This approach does not rely on strict database queries, but instead on the semantic interpretation of triples, making it suitable for integration with Language Models (LLMs) in a Graph-RAG pipeline.

Additionally, the retrieved triples can be directly inspected to provide explainability, showing how specific conclusions are derived from the graph structure.

#### 2. Graph as Database (Structured Querying)
In contrast, the knowledge graph can also be treated as a structured database, where nodes and relationships represent records that can be queried and aggregated.

In this scenario, the graph is used similarly to a relational database:

- entities correspond to records
- relations correspond to links between records
- triples can be processed as rows of structured data

To perform analytical queries, a filtering function is defined to identify which nodes correspond to construction reports (actas), typically based on naming patterns such as “AVO”, “ACTA”, or “DOB”.

Then, incident-related triples are counted per report, producing a structured aggregation:

Acta → Number of incidents

This enables:

- ranking reports by number of issues
- generating summary tables
- performing statistical analysis
- visualizing distributions of incidents

Unlike the Graph as Content approach, this method focuses on quantitative analysis, where the graph is treated as a dataset rather than a semantic knowledge source.

#### 3. Comparison of Both Approaches
The same knowledge graph supports two complementary paradigms:
Aspect	Graph as Content	Graph as Database
Purpose	Semantic reasoning	Quantitative analysis
Usage	Context for LLMs	Data aggregation
Output	Relevant triples	Numerical results
Interpretation	Flexible, contextual	Structured, deterministic
Role in pipeline	RAG enhancement	Analytical queries

This dual use demonstrates the versatility of the constructed knowledge graph:

- As content, it enhances language model reasoning by providing structured context
- As a database, it enables measurable insights and data-driven conclusions

#### 4. Conclusion
This experiment shows that a knowledge graph extracted from construction reports can serve both as:

- A semantic layer, supporting reasoning and contextual understanding in Graph-RAG systems.
- A data layer, supporting structured queries and statistical analysis.

The ability to switch between these two perspectives is a key advantage of graph-based systems, especially in domains where both qualitative interpretation and quantitative analysis are required.

## Querying Graphs: Cypher. A Graph query language.
### Graph Querying and Friendly Question Answering

After constructing the Knowledge Graph from construction reports using LLM-based extraction, an additional querying layer was developed to enable interactive exploration of the graph. 

This layer is based on Cypher, the query language of Neo4j, combined with a language model to generate human-readable answers.

Cypher is designed to be intuitive and expressive, allowing users to describe graph structures using patterns that closely resemble how graphs are visually represented. Instead of using tables and joins, Cypher relies on pattern matching over nodes and relationships. 

The fundamental idea is to represent queries as graph patterns of the form:

```(node)-[relationship]->(node)```

This syntax directly reflects how entities are connected in the graph. For example, a relationship between a construction element and its state can be expressed as:

```(element)-[:ESTADO]->(state)```

Cypher queries are typically composed of two main clauses: MATCH and RETURN. 

The MATCH clause is used to define the graph pattern to search for, while the RETURN clause specifies the data that should be retrieved. 

For instance, a basic query in the construction domain could be:

```
MATCH (e)-[:ESTADO]->(s)
RETURN e.name, s.name
```

This query retrieves all elements and their associated states from the graph. Similarly, more specific queries can be defined using conditions:

```
MATCH (e)-[:ESTADO]->(s)
WHERE s.name CONTAINS "ejecución"
RETURN e.name
```

This allows us to identify elements that are currently in execution. Another example in the domain of construction reports is:

```
MATCH (o)-[:EJECUTA]->(a)
RETURN o.name, a.name
```

which retrieves which organizations execute which activities.

The philosophy behind Cypher follows the idea of “specification by example”, meaning that queries describe concrete patterns to match in the graph rather than abstract rules. This makes the query process highly intuitive and aligned with how humans understand relationships in real-world systems.

However, while Cypher provides structured outputs, these results are not directly suitable for non-technical users. To address this limitation, we integrate a language model that transforms graph query results into natural language.

The complete pipeline follows this structure:

User Question → Cypher Query → Graph Results → LLM → Natural Language Answer

For example, a user might ask:

¿Qué está en ejecución?

The corresponding Cypher query retrieves relevant triples from the graph, such as elements linked to the state “en ejecución”. These structured results are then passed to the language model, which generates a human-readable answer like:
“The slope and the foundation are currently under execution according to the construction reports.”
This approach enables the transformation of the graph into an interactive and user-friendly system. Cypher provides precise access to structured knowledge, while the language model ensures that the output is understandable and useful for end users.

As a result, the system supports both technical exploration and practical usage scenarios. It allows users to query construction knowledge, analyze relationships between elements, and obtain clear explanations without needing to understand graph query languages. The combination of Cypher and LLM-based generation therefore bridges the gap between structured data and human interpretation, making the Knowledge Graph a powerful tool for real-world decision support.


## 4. How to Run the Code

### 4.1 Prerequisites

- Python 3.10+
- An OpenAI API key
- (Optional, for the KG pipeline) [Ollama](https://ollama.com/) with `gemma3` pulled, and optionally Neo4j

### 4.2 Install dependencies

```bash
git clone https://github.com/UriRius/RAG-UPCSchool-Project.git
cd RAG-UPCSchool-Project

python -m venv .venv
# Windows
.venv\Scripts\activate
# macOS/Linux
source .venv/bin/activate

pip install -r Requirements.txt
```

### 4.3 Configure environment variables

Copy `.env.example` to `.env` and fill in the values:

| Variable | Description |
|----------|-------------|
| `OPENAI_API_KEY` | Your OpenAI API key (generation + LLM-as-judge). |
| `APP_PASSWORD` | Password protecting the web UI. |
| `GCP_BUCKET_NAME` | Google Cloud Storage bucket name. |
| `DRIVE_FOLDER_ID` | Google Drive folder ID with the source documents. |

Useful retrieval knobs (see [src/rag/config.py](src/rag/config.py)): `RAG_MODE` (`v1`/`v2`/`v2_table`),
`RETRIEVAL_K`, `TOP_N`, `RRF_K`, `RRF_MIN_SCORE`.

### 4.4 Run the demo app locally

```bash
# (optional) download the prebuilt Chroma DB from GCS
python src/download_db.py

streamlit run src/app.py
```

### 4.5 Run the data ingestion (rebuild the index)

```bash
python src/ingest.py
```

This downloads the source documents, extracts and chunks the text, builds embeddings + BM25, and writes
the ChromaDB database.

### 4.6 Run the experiment notebooks

The experiments live in [notebooks/experiments/](notebooks/experiments/). Open them with Jupyter:

```bash
pip install jupyter
jupyter notebook notebooks/experiments/
```

Key notebooks:

| Experiment | Notebook |
|------------|----------|
| Query rewriting (mrBERT vs E5) | [query_pipeline_with_query_mod.ipynb](notebooks/experiments/query_pipeline_with_query_mod.ipynb) |
| Rewriting + reranking | [query_pipeline_e5_rewriting_reranking.ipynb](notebooks/experiments/query_pipeline_e5_rewriting_reranking.ipynb) |
| Knowledge Graph / Graph-RAG | [graph_rag.ipynb](notebooks/experiments/graph_rag.ipynb), [neo4j_graph_rag.ipynb](notebooks/experiments/neo4j_graph_rag.ipynb) |

### 4.7 Deploy to GCP (optional)

CI/CD uses Google Cloud Build and PowerShell scripts:

- [deploy/scripts/deploy_job.ps1](deploy/scripts/deploy_job.ps1) — deploys the data-ingestion Job.
- [deploy/scripts/deploy.ps1](deploy/scripts/deploy.ps1) — deploys the web app. The ~1 GB HuggingFace
  model is pre-downloaded and baked into the Docker image to guarantee instant cold starts and avoid
  timeout crashes.

---

## 5. Experiments

Each experiment is documented with **Hypothesis**, **Setup**, **Results** and **Conclusions**.

### Part A — Retrieval Experiments

#### Experiment 1 — Choosing the Embedding Model

**Hypothesis.** Two Spanish/multilingual embedding models (`mrBERT` and `e5`) will differ in retrieval
quality on the same query test set; we expect one to retrieve more relevant chunks than the other.

**Setup.**
- Backends: 2 (`mrBERT`, `e5`)
- Queries: 8 (shared test set)
- Top-K: 5
- Metrics: RRF scores, unique docs in top-K, corpus coverage, latency.

**Results.**

| Backend | avg_top1_rrf | avg_mean_rrf | avg_unique_docs@K | corpus_coverage | avg_latency_ms |
|---------|--------------|--------------|-------------------|-----------------|----------------|
| mrBERT  | 0.025854     | 0.022785     | 4.375             | 26              | **2010.87**    |
| e5      | **0.032325** | **0.029888** | 3.375             | 26              | 6483.19        |

Per-query detail:

| # | Query | overlap_chunks@5 | overlap_docs@5 | mrbert_top1 | e5_top1 |
|---|-------|------------------|----------------|-------------|---------|
| 0 | ¿Qué se decidió sobre el talud? | 0.0 | 0.000 | 0.024924 | 0.033333 |
| 1 | ¿Cuál es el estado del camino provisional? | 0.0 | 0.250 | 0.022436 | 0.033333 |
| 2 | ¿Qué incidencias AR-29 aparecen? | 0.0 | 0.125 | 0.024348 | 0.030092 |
| 3 | ¿Cuáles son las incidencias más frecuentes…? | 0.0 | 0.333 | 0.027418 | 0.030366 |
| 4 | ¿Qué responsable está asignado a las acciones…? | 0.0 | 0.167 | 0.021775 | 0.033060 |
| 5 | ¿Qué se acordó sobre hormigonado de zapatas? | 0.2 | 0.200 | 0.027390 | 0.032018 |
| 6 | Estado de las instalaciones de megafonía | 0.4 | 0.400 | 0.029877 | 0.033060 |
| 7 | ¿Qué documentación debe aportar la UTE sobre…? | 0.0 | 0.600 | 0.028665 | 0.033333 |

**Conclusions.** **E5 wins.** `e5` produces higher RRF scores on *every* query (top-1 and mean),
indicating stronger ranking confidence. `mrBERT` is ~3× faster (≈2.0 s vs ≈6.5 s) and surfaces slightly
more unique documents, but at lower relevance. Both cover the same 26 unique chunks, and their results are
largely disjoint (low overlap@5), confirming they rank by different signals. `e5`'s richer semantic
embedding space captures meaning beyond exact keyword matches; its higher latency is an acceptable
trade-off, so it was selected as the production embedding model. *This leads naturally to Experiment 2:
does query rewriting further improve the chosen model?*

---

#### Experiment 2 — Query Rewriting

**Hypothesis.** Generating semantic variants of the user query with an LLM (`gpt-4o-mini`, 3 variants
fused via RRF) will retrieve more relevant chunks than the original query alone. We expect the effect to
depend on the backend.

**Setup.**
- Backends: 2 (`mrBERT`, `e5`)
- Queries: 8 · Top-K: 5
- Rewriting: 3 `gpt-4o-mini` semantic variants per query, fused with the original via second-level RRF.
- Evaluation: **LLM-as-judge** (`gpt-4o-mini`), relevance scale 0–3.
- Metrics: Precision@5, NDCG@5, MRR, MAP.
- Notebook: [query_pipeline_with_query_mod.ipynb](notebooks/experiments/query_pipeline_with_query_mod.ipynb)

**Results.**

Aggregate — Original vs Rewritten per backend:

| Metric | mrBERT orig | mrBERT rew | Δ | e5 orig | e5 rew | Δ |
|--------|-------------|------------|--------|---------|--------|--------|
| Precision@5 | 0.525 | 0.450 | **−0.075** | 0.700 | **0.775** | **+0.075** |
| NDCG@5 | 0.744 | 0.718 | −0.026 | 0.880 | **0.922** | +0.042 |
| MRR | 0.750 | 0.629 | −0.121 | 0.838 | **0.917** | +0.079 |
| MAP | 0.724 | 0.628 | −0.096 | 0.860 | **0.921** | +0.061 |

Per-query detail (**e5**, original → rewritten):

| # | Query | ΔP@5 | ΔNDCG@5 | ΔMRR | ΔMAP |
|---|-------|------|---------|------|------|
| 0 | ¿Qué se decidió sobre el talud? | +0.000 | −0.009 | +0.000 | −0.050 |
| 1 | ¿Cuál es el estado del camino provisional? | +0.000 | −0.035 | +0.000 | +0.000 |
| 2 | ¿Qué incidencias AR-29 aparecen? | +0.000 | +0.000 | +0.000 | +0.000 |
| 3 | ¿Cuáles son las incidencias más frecuentes…? | **+0.200** | **+0.184** | **+0.133** | **+0.217** |
| 4 | ¿Qué responsable está asignado a las acciones…? | +0.000 | +0.000 | +0.000 | +0.000 |
| 5 | ¿Qué se acordó sobre hormigonado de zapatas? | **+0.200** | **+0.239** | **+0.500** | **+0.321** |
| 6 | Estado de las instalaciones de megafonía | +0.000 | −0.044 | +0.000 | +0.000 |
| 7 | ¿Qué documentación debe aportar la UTE…? | **+0.200** | +0.000 | +0.000 | +0.000 |

**Conclusions.** **E5 + Rewriting is the winning configuration**, with consistent gains and no severe
degradations. Two clear patterns emerge:

- **`e5` is robust to rewriting.** When it already works well, rewriting doesn't break it; when it fails,
  rewriting fixes it (e.g. Q5 MRR 0.5 → 1.0). The GPT variants explore neighboring regions of the same
  semantic space, adding useful signal.
- **`mrBERT` is fragile to rewriting.** When it already retrieves perfectly (MRR = 1.0), the extra queries
  inject chunks from other documents that displace the correct chunk in the second-level RRF (e.g. Q6 MRR
  1.0 → 0.2). As a morphological model, it gains no new terminology from rewriting — only dilution.

The most vague query (Q3, *"incidencias más frecuentes"*) is the only one where **both** backends improve,
because the original query lacks enough signal. *This raises a new hypothesis: can a cross-encoder reranker
recover precision after retrieval? → Experiment 3.*

---

#### Experiment 3 — Reranking with a Cross-Encoder

**Hypothesis.** Adding a **cross-encoder reranker** (`BAAI/bge-reranker-v2-m3`) after hybrid retrieval will
improve precision over `E5 + Query Rewriting` alone, because a cross-encoder scores each `(query, chunk)`
pair jointly — a far more precise relevance estimate than vector similarity.

**Setup.**
- Baseline: `E5 + Query Rewriting` (Experiment 2 winner).
- Candidate: `E5 + Query Rewriting + Reranking` (`BAAI/bge-reranker-v2-m3`).
- Flow: `retrieve_multiquery` → fuse candidates → rerank → top-K.
- Evaluation: LLM-as-judge (`gpt-4o-mini`), metrics P@5, AP@5, NDCG@5.
- Notebook: [query_pipeline_e5_rewriting_reranking.ipynb](notebooks/experiments/query_pipeline_e5_rewriting_reranking.ipynb)

**Results.** *Evaluation in progress* — the pipeline is implemented and the comparison harness is ready in
the notebook above; final numbers will be added once the full run completes.

**Conclusions.** *Pending results.* The expectation, based on Experiment 2, is that reranking mainly helps
on the harder/vaguer queries where retrieval order is still suboptimal after fusion.

---

### Part B — Knowledge Graph Experiments

#### Experiment 4 — Entity & Relation Extraction (Prompting Strategy)

**Hypothesis.** A **structured prompt** (controlled output format, domain-specific entity types, allowed
relation list) extracts higher-quality and more consistent entities/relations than a **general prompt**.

**Setup.**
- 60 construction reports processed.
- Local LLM (Ollama + Gemma 3) for prompt-based extraction.
- Strict triple format `(entity1, relation, entity2)` with a predefined relation list.
- Robust pattern matching to parse and filter malformed responses.

**Results.**

| Metric | General Prompt | Structured Prompt (ours) |
|--------|----------------|--------------------------|
| Documents processed | 60 | 60 |
| Total entities (unique) | 80–100 | **150** |
| Entity quality | Medium | **High** |
| Entity consistency | Low | **High** |
| Entity noise | High | Medium |
| Raw relations (LLM output) | 40–80 | **275** |
| Valid parsed relations | 20–40 | **150–200** |
| Final relations (after cleaning) | 20–30 | **50–80** |
| Relation diversity | Low | **High** |
| Graph structure | Weak | **Strong** |

**Conclusions.** Structured prompting transforms the LLM from a *text generator* into a *controlled
knowledge-extraction system*. The improvement is not only in quantity, but in the **structure and
consistency** of the extracted knowledge — producing a far more usable graph.

---

#### Experiment 5 — Graph Processing Pipeline (Normalization & Deduplication)

**Hypothesis.** Parsing, cleaning, normalization and deduplication will reduce the raw triple count while
*increasing* graph quality (less redundancy and noise) — i.e. consolidation, not data loss.

**Setup.** Five-stage pipeline applied to the raw LLM output. Relation normalization unifies synonyms
(`realiza` → `ejecuta`, `comienza` → `inicio`, `termina` → `finalizado`); state prioritization resolves
conflicting `estado` relations (most advanced state wins); duplicate triples removed by normalized
`(subject, relation, object)` key.

**Results.**

| Stage | Relations | Description |
|-------|-----------|-------------|
| Raw LLM output | **295** | All triples, incl. duplicates, noise and variations |
| After parsing | **200** | Only valid `(e1, relation, e2)` triples |
| After cleaning | **150** | Noisy entities / invalid text / low-quality relations removed |
| After normalization | **120** | Similar relations merged |
| After deduplication (final) | **50–80** | Duplicates removed, states consolidated |

**Conclusions.** The relation count drops sharply across stages because the pipeline prioritizes
**quality over quantity**. Deduplication is **knowledge consolidation, not information loss**, yielding a
cleaner and more reliable graph.

---

#### Experiment 6 — Graph QA: Cypher Queries (Content vs Database)

**Hypothesis.** The same Knowledge Graph can serve two complementary roles — as **semantic content**
(context for LLM reasoning) and as a **structured database** (analytical counting/ranking) — and each
role suits different question types.

**Setup.** Cypher (Neo4j) queries over the graph, with an LLM (Gemma via Ollama) translating results into
natural language. Example pipeline:

```cypher
MATCH (n)-[r]->(m)
WHERE type(r) IN ["EJECUTA", "INICIA", "FINALIZA", "INSTALA", "DESMONTA"]
RETURN m.name AS actividad, count(*) AS frecuencia
ORDER BY frecuencia DESC
LIMIT 10
```

The structured result (e.g. `instalación de chapa → 3`, `andamiaje → 3`, `talud → 2`, …) is passed to the
LLM, which generates a human-readable summary of the most frequent activities.

**Results.** Suitability of each paradigm by question type:

| Question Type | Graph as Content (Semantic) | Graph as Database (Analytical) |
|---------------|-----------------------------|--------------------------------|
| Construction elements (most frequent) | Good | Good (counts) |
| Construction process / activities | Good | Good (counts) |
| Work relationships (connections) | Partial | N/A |
| Dependencies (most frequent relations) | Good | Good (counts) |
| Health & safety factors | Good | Limited |
| Safety relations | Good | Limited |
| Safety measures (which & why) | Partial | N/A |
| Occupational risks | Partial | N/A |
| Incidents (problems & connections) | Good | Good (counts) |
| Contractor (UTE) activities | Good | Good (counts) |
| Technical impact of defects | Good | Not measurable |
| Documentation impact (complex) | Weak | Not supported |

**Conclusions.** The two paradigms are **complementary**: semantic queries excel at *understanding
relationships*, analytical queries at *counting and ranking*. Graph-QA performance depends on the question
type (semantic reasoning vs. quantitative analysis), and combining both — plus an LLM to verbalize results
— bridges structured data and human interpretation, enabling non-technical users to query the graph.

---

## 6. Overall Conclusions

- **Retrieval:** `e5` is the better embedding model, and **`e5 + query rewriting`** is the best validated
  configuration — consistent gains with no severe degradation. `mrBERT` is faster but fragile to rewriting.
  Cross-encoder reranking is the next lever under evaluation.
- **Knowledge Graph:** **Structured prompting** is decisive — it turns the LLM into a controlled extractor,
  producing a denser, more consistent graph. The **processing pipeline** consolidates knowledge rather than
  losing it, and the graph supports both **semantic** and **analytical** querying via Cypher + LLM.
- **Together**, the KG transforms the system from a simple text-retrieval tool into a structured reasoning
  system, enabling more precise, explainable and domain-aware question answering. This first graph is small;
  the natural next step is to build a larger graph over the full corpus.

---

## 7. References

**Papers**

- *KGGen: Extracting Knowledge Graphs from Plain Text with Language Models* — Mo et al., arXiv:2502.09956 [cs.CL]
- *Retrieval-Augmented Generation with Graphs (GraphRAG)* — Han et al., 2024, arXiv:2501.00309
- *Unifying Large Language Models and Knowledge Graphs: A Roadmap* — Pan et al., 2023, arXiv:2306.08302
- *HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases* — Lee et al., 2024, arXiv:2412.16311
- *GraphER: A Structure-aware Text-to-Graph Model for Entity and Relation Extraction* — Zaratiana et al., 2024, arXiv:2404.12491
- *G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and QA* — He et al., 2024, arXiv:2402.07630

**Courses & tools**

- DeepLearning.AI — [Knowledge Graphs for RAG](https://www.deeplearning.ai/courses/knowledge-graphs-rag)
- [Stanford Open Information Extraction (OpenIE)](https://nlp.stanford.edu/software/openie.html)
- Microsoft GraphRAG — [project](https://www.microsoft.com/en-us/research/project/graphrag/) · [docs](https://microsoft.github.io/graphrag/) · [GitHub](https://github.com/microsoft/graphrag)
- Visual graph tools — [yEd Live](https://www.yworks.com/yed-live/) · [Kumu](https://kumu.io/) · [Gephi Lite](https://lite.gephi.org/) · [Cosmograph](https://cosmograph.app/)

---

## 78. Deliverables

- **Final Report:** this README (the official Final Report of the project).
- **Slides:** submitted as an official deliverable. The presentation's front slide links to this
  repository: https://github.com/UriRius/RAG-UPCSchool-Project
- **Note:** per the assignment rules, code alone (even well documented) and the slides do **not** count as
  the Final Report — this README does.

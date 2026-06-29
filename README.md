# COSORA Demo - RAG GRAPH UPCSchool Project

## 📌 About the Project
This is a **Retrieval-Augmented Generation (RAG)** demo designed to analyze construction site meeting minutes ("actas de obra"). It allows users to ask natural language questions about construction progress and get answers backed by official documents.

This project was developed for the **UPCSchool**, demonstrating how to integrate AI into engineering and construction workflows.

## 🏗️ Architecture
The system uses a **Serverless Architecture on Google Cloud Platform (GCP)** for low cost and high scalability:

- **Frontend:** [Streamlit](https://streamlit.io/) hosted on Google Cloud Run.
- **Vector Database:** [ChromaDB](https://www.trychroma.com/).
- **AI Models:**
  - **Search (Embeddings):** `intfloat/multilingual-e5-base` running locally for fast hybrid search.
  - **Text Generation (LLM):** OpenAI API (`gpt-4o-mini`).
- **Data Pipeline (Cloud Run Job):**
  1. Downloads `.doc` and `.docx` files from Google Drive.
  2. Extracts text using `antiword` and python-docx.
  3. Chunks the text, creates embeddings, and builds the ChromaDB database.
  4. Uploads the ready-to-use database to Google Cloud Storage.

## 🚀 Deployment
We use Google Cloud Build and PowerShell scripts for CI/CD:
- **`deploy_job.ps1`**: Deploys the Data Ingestion Job.
- **`deploy.ps1`**: Deploys the Web App. The 1GB HuggingFace model is pre-downloaded and baked into the Docker image to ensure instant cold starts and prevent timeout crashes.

## 🔐 Environment Variables Required
To run this project, you need to set up the following secrets in Cloud Run:
- `OPENAI_API_KEY`: Your OpenAI API key.
- `APP_PASSWORD`: A password to protect the web UI.
- `GCP_BUCKET_NAME`: The Google Cloud Storage bucket name.
- `DRIVE_FOLDER_ID`: The Google Drive folder ID containing your source documents.


# Retrieval Experiments

## Experiment 1 — Choosing the Semantic Model

### Hypothesis
The first experiment was choosing the semantic model used to create the embeddings. We evaluated two Spanish/multilingual models (`mrBERT` and `e5`) on the same query test set, expecting one to retrieve more relevant chunks than the other.

### Setup
- **Backends:** 2 (`mrBERT`, `e5`)
- **Queries:** 8 (same test set as Experiment 2)
- **Top-K:** 5
- **Metrics:** RRF scores, unique docs in top-K, corpus coverage, latency.

### Results

| Backend | avg_top1_rrf_score | avg_mean_rrf_score | avg_unique_docs_in_topK | corpus_coverage (unique chunks) | avg_latency_ms |
|---------|--------------------|--------------------|-------------------------|---------------------------------|----------------|
| mrBERT  | 0.025854           | 0.022785           | 4.375                   | 26                              | **2010.87**    |
| e5      | **0.032325**       | **0.029888**       | 3.375                   | 26                              | 6483.19        |

Per-query detail:

| # | Query                                                  | overlap_chunks@5 | overlap_docs@5 | mrbert_top1_score | e5_top1_score | mrbert_unique_docs | e5_unique_docs |
|---|--------------------------------------------------------|------------------|----------------|-------------------|---------------|--------------------|----------------|
| 0 | ¿Qué se decidió sobre el talud?                        | 0.0              | 0.000          | 0.024924          | 0.033333      | 4                  | 3              |
| 1 | ¿Cuál es el estado del camino provisional?             | 0.0              | 0.250          | 0.022436          | 0.033333      | 5                  | 5              |
| 2 | ¿Qué incidencias AR-29 aparecen?                       | 0.0              | 0.125          | 0.024348          | 0.030092      | 4                  | 5              |
| 3 | ¿Cuáles son las incidencias más frecuentes…?           | 0.0              | 0.333          | 0.027418          | 0.030366      | 5                  | 3              |
| 4 | ¿Qué responsable está asignado a las acciones…?        | 0.0              | 0.167          | 0.021775          | 0.033060      | 4                  | 3              |
| 5 | ¿Qué se acordó sobre hormigonado de zapatas?           | 0.2              | 0.200          | 0.027390          | 0.032018      | 4                  | 2              |
| 6 | Estado de las instalaciones de megafonía               | 0.4              | 0.400          | 0.029877          | 0.033060      | 4                  | 3              |
| 7 | ¿Qué documentación debe aportar la UTE sobre…?         | 0.0              | 0.600          | 0.028665          | 0.033333      | 5                  | 3              |

### Observations
- `e5` produces higher RRF scores on every query (both top-1 and mean), indicating stronger ranking confidence.
- `mrBERT` is ~3× faster (≈2.0 s vs ≈6.5 s avg latency) and surfaces slightly more unique documents in the top-K, but at lower relevance.
- Both backends cover the same 26 unique chunks of the corpus.
- The two models retrieve largely disjoint results (low overlap@5), confirming they rank by different signals.

### Conclusions
**E5 is better.**

Multilingual-e5 is better because it is richer in its semantic embedding space, capturing meaning beyond exact keyword matches. `mrBERT` relies on morphological and lexical similarity, which is more effective for exact terms but not for paragraphs. The higher latency of `e5` is an acceptable trade-off for its retrieval quality, so it was selected as the production embedding model.


## Experiment 2 — Query Rewriting

### Hypothesis
We tested query rewriting with both retrieval backends (`mrBERT` and `e5`). `gpt-4o-mini` generated 3 semantic variants per query, under the hypothesis that model-generated queries would retrieve more relevant chunks than the original user query.

### Setup
- **Judge model:** `gpt-4o` (LLM-as-Judge)
- **Queries:** 8
- **Backends:** 2 (`mrBERT`, `e5`)
- **Top-K:** 5
- **Total API calls:** 80
- Rewriting: 3 `gpt-4o-mini` semantic variants per query, fused via RRF.

### Results

| Backend | Precision@5 | MRR    | NDCG@5  | MAP    | Avg relevance |
|---------|-------------|--------|---------|--------|---------------|
| mrBERT  | 0.25        | 0.4375 | 0.4448  | 0.4097 | 0.400         |
| e5      | **0.55**    | **0.6500** | **0.6262** | **0.6500** | **0.875** |


### Conclusions
**E5 + Rewriting is better.**

Rewriting helps `e5` because the semantic variants explore neighboring regions of its embedding space, improving coverage. With `mrBERT`, the extra queries introduce noise and displace already well-positioned chunks. The `gpt-4o-mini` variants add no new morphological signal.


#  Knowledge Graph Generation from Construction Reports

##  Project Overview

This project builds a Knowledge Graph (KG) from construction site reports ("actas de obra") using NLP and LLM-based techniques.
We use a KG, a text-to-kowledge-graph generator with the aim to leverage RAG Retrieval Augmented Generation extraction architecture. KG uses a LM Language Model LM_based extractor to read unstructured text and predict subject-predicate-object triples to capture entities and relationships.
Interest in automated methods to produce structured text dates back to at least 2001 when large volumes of plain text began to flood on the Internet.
Early work, like YAGO (KG extracted from WIKIPEDIA), used hard-coded rules. With the development of modern natural language processing hard-coded rules ceded to more avanced approaches based on Neural Networks. For instance, OpenIE.
As early as 2015, there were believed that extracting KGs would go hand-in-hand with developing better language models. More recently, #transformer-based architectures can identify relationsships between entities leading to transformer-based KG extraction techniques.

The pipeline extracts:
- Entities (organizations, construction elements, installations, safety issues)
- Relations (actions, decisions, technical relations)
- Graph structure (nodes and edges)

##  Repository

https://github.com/UriRius/RAG-UPCSchool-Project

#  How to run the project

## 1. Install dependencies

pip install transformers torch python-docx

2. Load data
Place the construction reports (.docx) inside:
Dataset/

3. Extract entities
entities = get_entities_llm(text)

4. Extract relations
relations = get_relations_llm(text, entities)

5. Build graph
graph = {"nodes": set(), "edges": []}

for text in texts[:2]:  # test small first
    ents = get_entities_llm(text)
    rels = get_relations_llm(text, ents)

    for s, r, o in rels:
        graph["nodes"].add(s)
        graph["nodes"].add(o)
        graph["edges"].append((s, r, o))

6. Visualization
Neo4j (graph database)
yEd / Gephi (graph visualization)

Experiments
Experiment 1 — Entity Extraction
Hypothesis
LLMs can extract relevant entities from construction reports.
Setup

LLM model
Prompt-based extraction
Input: .docx reports

Results

Entities extracted successfully
Example:
['renfe', 'muro', 'andén', 'iluminación']

Conclusions

Works well but noisy
Needs normalization

Experiment 2 — Relation Extraction
Hypothesis
The model can identify relations between entities.
Setup

Prompt-based relation extraction
Expected output: triples

Results

consistent results but they could improve

Example:
[('Constructora', 'ejecuta', 'trabajos')]

Experiment 3 — Graph Construction
Hypothesis
Entities + relations can create a Knowledge Graph.
Setup

Nodes: entities
Edges: relations

What did we expect to learn? and What is the experiment about?

Graphs represent entities as nodes and the ways in which those entities relate to the world as relationships.
KGs consist of a set of subject-predicate-object triples and have become a fundamental data structure for information retrieval. 
To solve this problem, we propose KGGen, a text-to-knowledge-graph generator that leverages Retrieval Augmented Generatin systems (RAG) and an algorithm for entity and edge resolution to extract high-quality, dense KGs from text. First, KGGen uses an LM-based extractor to read unstructured text and predict subject-predicate-object triples to capture entities and relations; after extracting the triples, it applies a novel, iterative clustering algorithm to refine the raw graph. Inspired by crowd-sourcing strategies for entity resolution, KGGen identifies nodes that refer to the same underlying entities, and consolidates edges that have equivalent meanings.
Entity and Relation Extraction
The first stage takes unstructured text of construction site reports as input and produces an initial knowledge graph as extracted triples. We use a language model to provide structured output. The first step takes in source text and extracts a list of entities. Given the source text and entities list, the second step outputs a list of subject-predicate-object relations. 
Aggregation
After extracting triples from each source text, we collect all the unique entities and edges across all source graphs and combine them into a single graph. All entities and edges are normalized to be in lowercase letters only. The aggregation step reduces redundancy in the KG. Aggregation step does not require an LLM.
Entity and Edge Resolution
After extraction and aggregation, it typically have a raw graph containing duplicate or synonymous entities and possibly redundant edges. The resolution stage merges nodes and edges representing the same real-world entity or concept. The resolution process employs a two-stage approach combining embedding-based clustering with LLMbased de-duplication to efficiently handle large knowledge graphs. 
First, all items in the graph are clustered. 
Methodology
1. Overview
This project follows a graph-based information extraction methodology designed to transform unstructured construction reports into a structured knowledge representation.
The approach consists of four main stages:
1.	Document ingestion
2.	Entity extraction
3.	Relation extraction
4.	Graph construction and consolidation
Each stage is supported by Language Models (LLMs) and rule-based post-processing to ensure consistency and quality of the resulting knowledge graph.

2. Document Ingestion and Preprocessing
The input data consists of technical construction reports (actas and technical documents) in both .doc and .docx formats.
Documents are processed using text extraction tools:
•	python-docx for .docx
•	textract for .doc
The extracted content is converted into plain text, preserving as much structure as possible while removing formatting artifacts.
To ensure efficiency and compatibility with LLMs, the text is truncated into manageable segments before further processing.

3. Entity Extraction
Entities are extracted using a prompt-based approach with a local Language Model (Ollama + Gemma 3).
A structured prompt is designed to enforce:
•	Controlled output format (comma-separated list)
•	Domain-specific entity types
•	Short and normalized entity names
The LLM identifies relevant entities such as:
•	construction elements
•	organizations (e.g., UTE, DF)
•	technical concepts
•	actions and documentation
A post-processing step removes:
•	duplicates
•	noisy tokens
•	invalid entity strings
This results in a clean set of entities representing the key concepts in each document.

4. Relation Extraction
Relationships between entities are extracted using a second LLM prompt.
The prompt enforces a strict triple format:
(entity1, relation, entity2)
Additionally:
•	a predefined list of allowed relations is provided
•	the number of candidate entities is limited to improve precision
•	instructions prevent explanations or free text
The output is parsed using robust pattern matching to extract valid triples while filtering malformed responses.
This step produces the core knowledge representation used to build the graph.

5. Normalization and Deduplication
To improve graph quality, a normalization and deduplication process is applied:
Relation normalization
Semantically equivalent relations are unified:
•	"realiza" → "ejecuta"
•	"comienza" → "inicio"
•	"termina" → "finalizado"
State prioritization
Special handling is applied to state-related relations (estado):
•	conflicting states are resolved using priority rules
•	the most advanced state (e.g., "finalizado") is retained
Duplicate removal
Triples are deduplicated using normalized keys:
(subject, relation, object)
This ensures that the graph remains consistent and avoids redundancy.

6. Graph Construction
The final graph is constructed as:
•	a set of nodes (entities)
•	a set of directed edges (relations)
Each triple is converted into:
source → relation → target
The resulting structure is stored in two formats:
1.	In-memory representation 
o	used for processing and querying
2.	JSON format 
o	exported for interoperability and visualization
This graph serves as the foundation for both retrieval and analytical tasks.

7. Integration with Retrieval Pipeline
The constructed graph is integrated into a hybrid retrieval system:
•	Dense retrieval (embeddings)
•	Sparse retrieval (BM25)
•	Graph-based retrieval (LLM over triples)
Graph-based retrieval selects the most relevant triples for a given query and injects them into the prompt as structured context.
This enhances the system by:
•	improving reasoning
•	adding explainability
•	capturing relationships not present in raw text retrieval

8. Summary
The methodology combines:
•	unstructured document processing
•	LLM-based information extraction
•	structured graph construction
•	hybrid retrieval strategies
This pipeline enables the transformation of construction reports into a reusable knowledge representation, supporting both semantic reasoning and analytical exploration.

Graph-Based Analysis — Content vs Database Usage
1. Graph as Content (Semantic Knowledge)
In this stage, the extracted knowledge graph is used as a semantic content source, where each triple represents a meaningful piece of information derived from construction reports.
The graph is composed of triples of the form:
(subject, relation, object)
These triples encode domain knowledge such as:
•	incidents affecting construction elements
•	pending actions
•	technical issues described in the reports
For example:
(talud, estado, inestable)
(UTE, solicita, documentación técnica)
(acta AVO-03, contiene, incidencia)
To demonstrate the use of the graph as content, the system performs a filtering process over the triples to identify those that contain keywords related to incidents (e.g., “incidencia”, “problema”, “defecto”, “pendiente”, “incumplimiento”).
By aggregating these triples, it is possible to determine which reports (actas) contain the highest number of issues.
This approach does not rely on strict database queries, but instead on the semantic interpretation of triples, making it suitable for integration with Language Models (LLMs) in a Graph-RAG pipeline.
Additionally, the retrieved triples can be directly inspected to provide explainability, showing how specific conclusions are derived from the graph structure.
2. Graph as Database (Structured Querying)
In contrast, the knowledge graph can also be treated as a structured database, where nodes and relationships represent records that can be queried and aggregated.
In this scenario, the graph is used similarly to a relational database:
•	entities correspond to records
•	relations correspond to links between records
•	triples can be processed as rows of structured data
To perform analytical queries, a filtering function is defined to identify which nodes correspond to construction reports (actas), typically based on naming patterns such as “AVO”, “ACTA”, or “DOB”.
Then, incident-related triples are counted per report, producing a structured aggregation:
Acta → Number of incidents
This enables:
•	ranking reports by number of issues
•	generating summary tables
•	performing statistical analysis
•	visualizing distributions of incidents
Unlike the Graph as Content approach, this method focuses on quantitative analysis, where the graph is treated as a dataset rather than a semantic knowledge source.

3. Comparison of Both Approaches
The same knowledge graph supports two complementary paradigms:
Aspect	Graph as Content	Graph as Database
Purpose	Semantic reasoning	Quantitative analysis
Usage	Context for LLMs	Data aggregation
Output	Relevant triples	Numerical results
Interpretation	Flexible, contextual	Structured, deterministic
Role in pipeline	RAG enhancement	Analytical queries

This dual use demonstrates the versatility of the constructed knowledge graph:
•	As content, it enhances language model reasoning by providing structured context
•	As a database, it enables measurable insights and data-driven conclusions

4. Conclusion
This experiment shows that a knowledge graph extracted from construction reports can serve both as:
1.	A semantic layer, supporting reasoning and contextual understanding in Graph-RAG systems
2.	A data layer, supporting structured queries and statistical analysis
The ability to switch between these two perspectives is a key advantage of graph-based systems, especially in domains where both qualitative interpretation and quantitative analysis are required.


##  Graph Querying and Friendly Question Answering

After constructing the Knowledge Graph from construction reports using LLM-based extraction, an additional querying layer was developed to enable interactive exploration of the graph. This layer is based on Cypher, the query language of Neo4j, combined with a language model to generate human-readable answers.

Cypher is designed to be intuitive and expressive, allowing users to describe graph structures using patterns that closely resemble how graphs are visually represented. Instead of using tables and joins, Cypher relies on pattern matching over nodes and relationships. The fundamental idea is to represent queries as graph patterns of the form:

(node)-[relationship]->(node)

This syntax directly reflects how entities are connected in the graph. For example, a relationship between a construction element and its state can be expressed as:

(element)-[:ESTADO]->(state)

Cypher queries are typically composed of two main clauses: MATCH and RETURN. The MATCH clause is used to define the graph pattern to search for, while the RETURN clause specifies the data that should be retrieved. For instance, a basic query in the construction domain could be:

MATCH (e)-[:ESTADO]->(s)
RETURN e.name, s.name

This query retrieves all elements and their associated states from the graph. Similarly, more specific queries can be defined using conditions:

MATCH (e)-[:ESTADO]->(s)
WHERE s.name CONTAINS "ejecución"
RETURN e.name

This allows us to identify elements that are currently in execution. Another example in the domain of construction reports is:

MATCH (o)-[:EJECUTA]->(a)
RETURN o.name, a.name

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
        
cypher = """
MATCH (n)-[r]->(m)
WHERE type(r) IN ["EJECUTA", "INICIA", "FINALIZA", "INSTALA", "DESMONTA"]
RETURN m.name AS actividad, count(*) AS frecuencia
ORDER BY frecuencia DESC
LIMIT 10
"""

result = kg.query(cypher)

print("Resultado Cypher:\n")
print(result)

Resultado Cypher:

[{'actividad': 'instalación de chapa', 'frecuencia': 3}, {'actividad': 'impermeabilización con poliurea', 'frecuencia': 3}, {'actividad': 'andamiaje', 'frecuencia': 3}, {'actividad': 'Marquesina', 'frecuencia': 3}, {'actividad': 'marquesina vía 02', 'frecuencia': 2}, {'actividad': 'talud', 'frecuencia': 2}, {'actividad': 'correas', 'frecuencia': 2}, {'actividad': 'hinca de pilotes', 'frecuencia': 2}, {'actividad': 'andamio', 'frecuencia': 2}, {'actividad': 'actuaciones', 'frecuencia': 2}]


graph_text = "\n".join([
    f"{row['actividad']} → {row['frecuencia']}"
    for row in result if row['actividad']
])

print(graph_text)

instalación de chapa → 3
impermeabilización con poliurea → 3
andamiaje → 3
Marquesina → 3
marquesina vía 02 → 2
talud → 2
correas → 2
hinca de pilotes → 2
andamio → 2
actuaciones → 2

question = "¿Cuáles son las actividades más frecuentes en las actas de obra?"

prompt = f"""
Eres un asistente experto en análisis de actas de obra.

Explica estos resultados del grafo en lenguaje natural.

PREGUNTA:
{question}

DATOS:
{graph_text}

RESPUESTA:
"""

answer = generate_llm(prompt)

print("Respuesta final:\n")
print(answer)

Respuesta final:

De acuerdo con el análisis de las actas de obra, las actividades que más se repiten y por lo tanto son las más frecuentes en el proyecto son:

*   **Instalación de chapa:** Aparece en 3 actas.
*   **Impermeabilización con poliurea:** También aparece en 3 actas.
*   **Andamiaje:** Es la actividad que más se registra, apareciendo en 3 actas.
*   **Marquesina:**  Esta actividad y su variante "marquesina vía 02" aparecen en 3 actas cada una.

Las siguientes actividades también son bastante comunes, apareciendo en 2 actas cada una:

*   **Talud**
*   **Correas**
*   **Hinca de pilotes**
*   **Andamio**
*   **Actuaciones**

En resumen, la instalación de chapa, la impermeabilización con poliurea y el andamiaje son las actividades que más se han realizado hasta el momento según los datos de las actas de obra.

# Experiments & Results

Experiment : Entity Extraction and Relation extraction. Graph Construction

##  Entity & Relation Extraction Results

| Metric                          | General Prompt (LLM) | Structured Prompt (Our Approach) |
|--------------------------------|---------------------|----------------------------------|
| Documents processed             | 60                  | 60                               |
| Total entities (unique)         | 80–100              | 150                              |
| Entity quality                  | Medium              | High                             |
| Entity consistency              | Low                 | High                             |
| Entity noise                    | High                | Medium                           |
| Raw relations (LLM output)      | 40–80               | 275                              |
| Valid parsed relations          | 20–40               | 150–200                          |
| Final relations (after cleaning)| 20–30               | 50–80                            |
| Relation diversity              | Low                 | High                             |
| Graph structure                 | Weak                | Strong                           |
| Graph usability                 | Low                 | High                             |


CONCLUSION:

Using a general prompt produces noisy and inconsistent outputs, while a structured prompt significantly improves both entity and relation extraction, resulting in a more usable and higher-quality knowledge graph. The main improvement is not only in quantity, but in structure and consistency of the extracted knowledge.
Structured prompting transforms the LLM from a text generator into a controlled knowledge extraction system.

# Experiment 4: Deduplication. Graph Creation

##  Impact of Graph Processing Pipeline

| Stage                         | Relations Count | Description                                                                                 |
|------------------------------|----------------|----------------------------------------------------------------------------
| **Raw LLM Output**            | **295**        | All triples generated by the LLM, including duplicates, noise, and variations              |
| **After Parsing**             | **200**        | Only valid triples with correct format (entity1, relation, entity2)                        |
| **After Cleaning (Filtering)**| **150**        | Removal of noisy entities, invalid text, and low-quality relations                         |
| **After Normalization**       | **120**        | Similar relations merged (e.g., “realiza” → “ejecuta”)                                     |
| **After Deduplication (Final)**| **50–80**     | Duplicate triples removed and states consolidated into a cleaner graph                     |

Conclusion:
The number of relations decreases significantly across processing stages because the pipeline prioritizes quality, removing duplicates, noise, and inconsistent information to produce a cleaner and more reliable graph. Deduplication is not data loss, but knowledge consolidation.

Experiment. SAPHER QUERIES
##  Graph QA Evaluation (Pipeline Questions)

##  Graph QA Evaluation (Pipeline Questions)

| Question Type              | Pipeline Question                                                                 | Graph as Content (Semantic) | Graph as Database (Analytical) |
|---------------------------|-----------------------------------------------------------------------------------|-----------------------------|--------------------------------|
| **Construction Elements**  | Which construction elements appear most frequently in the graph?                 | Good                        | Good (counts occurrences)      |
| **Construction Process**   | What are the most common construction activities?                                | Good                        | Good (counts occurrences)      |
| **Work Relationships**     | How are different construction activities connected?                             | Partial                     | Not applicable                 |
| **Dependencies**           | Which relations (executes, requires, affects…) appear most frequently?           | Good                        | Good (counts occurrences)      |
| **Health & Safety**        | What factors affect health and safety on site?                                   | Good                        | Limited                        |
| **Safety Relations**       | Which activities require safety measures most frequently?                        | Good                        | Limited                        |
| **Safety Measures**        | Which elements require safety measures and why?                                  | Partial                     | Not applicable                 |
| **Occupational Risks**     | How do activities influence occupational risks?                                  | Partial                     | Not applicable                 |
| **Incidents**              | What problems appear and how are they connected?                                 | Good                        | Good (counting capability)     |
| **Contractor Activities**  | What activities are most frequent for the contractor (UTE)?                      | Good                        | Good (counts occurrences)      |
| **Technical Impact**       | What is the impact of defects (e.g., mold, cracks) in the project?               | Good                        | Not measurable                 |
| **Documentation (Complex)**| How does technical documentation affect project execution?                       | Weak                        | Not supported                  |

conclusions:

This evaluation shows that different query strategies provide complementary capabilities.
Semantic graph queries are effective for understanding relationships, while analytical queries 
enable counting and ranking. 
Graph querying performance depends on the type of question:
semantic reasoning vs. quantitative analysis.

##  Conclusions - Knowledge Graph

The integration of a Knowledge Graph significantly improves the representation and usability of information extracted from construction reports. By structuring data as entities and relationships, the system is able to capture complex dependencies and real-world interactions that are difficult to model using raw text alone, but not all the relations because this firts graph is small. The idea is after the project, build a bigger graph.

The use of structured prompting plays a key role in this process. Compared to a general prompt, the structured approach generates a higher number of entities and relations with greater consistency and lower noise. This results in a more reliable and semantically meaningful graph.

The graph processing pipeline (parsing, cleaning, normalization, and deduplication) further enhances data quality. Although the number of relations decreases throughout this process, the final graph is significantly cleaner and more useful. This reduction reflects knowledge consolidation rather than information loss.

Cypher queries enable efficient and precise retrieval of structured information from the graph. They are particularly effective for:
- counting and ranking elements or activities  
- identifying frequent patterns  
- analyzing relationships between entities  

However, Cypher outputs are inherently structured and not directly user-friendly. To address this limitation, a local language model (Gemma via Ollama) is used to transform query results into natural language explanations. This combination allows the system to provide both accurate and interpretable answers.

The evaluation shows that:
- semantic queries are effective for understanding relationships  
- analytical queries are effective for quantitative insights  
- both approaches are complementary and necessary  

Overall, the Knowledge Graph transforms the system from a simple text retrieval tool into a structured reasoning system. It enables more precise, explainable, and domain-aware question answering, demonstrating the value of combining graph-based representations with LLM-based interpretation.
The combination of structured knowledge graphs and language models provides a powerful framework for transforming unstructured domain-specific data into actionable insights.


# REFERENCES

# KGGen: Extracting Knowledge Graphs from Plain Text with Language Models
Belinda Mo, Kyssen Yu, Joshua Kazdan, Joan Cabezas, Proud Mpala, Lisa Yu, Chris Cundy, Charilaos Kanatsoulis, Sanmi Koyejo
	arXiv:2502.09956 [cs.CL]
  
# Retrieval-Augmented Generation with Graphs (GraphRAG)
Published December 31st, 2024
By Haoyu Han, Yu Wang, Harry Shomer, Kai Guo, Jiayuan Ding, Yongjia Lei, Mahantesh Halappanavar, Ryan A. Rossi, Subhabrata Mukherjee, Xianfeng Tang, Qi He, Zhigang Hua, Bo Long, Tong Zhao, Neil Shah, Amin Javari, Yinglong Xia, Jiliang Tang
arXiv:2501.00309 [ cs.IR, cs.CL, cs.LG ] github:Graph-RAG/GraphRAG/

# Unifying Large Language Models and Knowledge Graphs: A Roadmap
Published June 14th, 2023
By Shirui Pan, Linhao Luo, Yufei Wang, Chen Chen, Jiapu Wang, Xindong Wu
arXiv:2306.08302 [ cs.CL, cs.AI ]

# HybGRAG: Hybrid Retrieval-Augmented Generation on Textual and Relational Knowledge Bases
Published December 20th, 2024
By Meng-Chieh Lee, Qi Zhu, Costas Mavromatis, Zhen Han, Soji Adeshina, Vassilis N. Ioannidis, Huzefa Rangwala, Christos Faloutsos
arXiv:2412.16311 [ cs.LG, cs.AI, cs.IR ]

# GraphER: A Structure-aware Text-to-Graph Model for Entity and Relation Extraction
Published April 18th, 2024
By Urchade Zaratiana, Nadi Tomeh, Niama El Khbir, Pierre Holat, Thierry Charnois
arXiv:2404.12491 [ cs.CL, cs.AI ] github:urchade/GraphER

# G-Retriever: Retrieval-Augmented Generation for Textual Graph Understanding and Question Answering
Published February 12th, 2024
By Xiaoxin He, Yijun Tian, Yifei Sun, Nitesh V. Chawla, Thomas Laurent, Yann LeCun, Xavier Bresson, Bryan Hooi
arXiv:2402.07630 [ cs.LG ] github:XiaoxinHe/G-Retriever
Abstract

Course deeplearning.ai 
# Knowledge Graphs for RAG
https://www.deeplearning.ai/courses/knowledge-graphs-rag 

https://nlp.stanford.edu/software/openie.html
Software > Stanford OpenIE
# Stanford Open Information Extraction

# MICROSOFT GRAPHRAG
https://www.ibm.com/es-es/think/topics/graphrag
https://microsoft.github.io/graphrag/
https://www.microsoft.com/en-us/research/project/graphrag/
https://github.com/microsoft/graphrag


# Tools to build visual graphs 
https://www.yworks.com/yed-live/
https://kumu.io/
https://lite.gephi.org/
https://cosmograph.app/




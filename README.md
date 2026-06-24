# COSORA Demo - RAG UPCSchool Project

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

#  Knowledge Graph Generation from Construction Reports

##  Project Overview

This project builds a Knowledge Graph (KG) from construction site reports ("actas de obra") using NLP and LLM-based techniques.

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
[('renfe', 'ejecuta', 'trabajos')]

Experiment 3 — Graph Construction
Hypothesis
Entities + relations can create a Knowledge Graph.
Setup

Nodes: entities
Edges: relations

Results

Graph generated successfully

Conclusions
Graph depends heavily on relation quality


        








# Cloud Run Job: Drive (JSON grafo) -> Neo4j Aura + backup GCS

$PROJECT_ID = "esoteric-code-489918-v1"
$REGION = "europe-west1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/cosora-graph-sync"
$JOB_NAME = "cosora-graph-sync-job"

# ID de la carpeta graph/ en Google Drive (RAG_UPC_Final_project/graph)
$DRIVE_GRAPH_FOLDER_ID = "REEMPLAZA_CON_TU_FOLDER_ID"

Write-Host "Paso 1: Build imagen graph-sync..." -ForegroundColor Cyan
gcloud builds submit --config deploy/cloudbuild-graph-sync.yaml

Write-Host "Paso 2: Crear/actualizar Cloud Run Job..." -ForegroundColor Cyan
gcloud run jobs update $JOB_NAME `
    --image $IMAGE_NAME `
    --region $REGION `
    --cpu 2 `
    --memory 2Gi `
    --max-retries 1 `
    --task-timeout 15m `
    --set-env-vars="GCP_BUCKET_NAME=rag-actas-db-bucket,GCS_GRAPH_PREFIX=graph,DRIVE_GRAPH_FOLDER_ID=$DRIVE_GRAPH_FOLDER_ID,GRAPH_DIR=/app/data/graph,NEO4J_USER=neo4j,NEO4J_DATABASE=neo4j"

Write-Host "Anade en la consola Cloud Run Job (o en --set-env-vars):" -ForegroundColor Yellow
Write-Host "  NEO4J_URI, NEO4J_PASSWORD" -ForegroundColor Yellow

Write-Host "Listo. Ejecutar sync:" -ForegroundColor Green
Write-Host "gcloud run jobs execute $JOB_NAME --region $REGION --wait" -ForegroundColor Yellow

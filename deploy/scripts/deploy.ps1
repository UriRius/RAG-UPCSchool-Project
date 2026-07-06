# === SCRIPT DE DESPLIEGUE A GOOGLE CLOUD RUN ===

$PROJECT_ID = "esoteric-code-489918-v1"
$REGION = "europe-west1"
$IMAGE_NAME = "gcr.io/$PROJECT_ID/cosora-demo"

# Carga .env del repo (copia Neo4j/OpenAI desde Drive variablentorno/.env)
function Get-DotEnv($path) {
    $vars = @{}
    if (-not (Test-Path $path)) { return $vars }
    Get-Content $path | ForEach-Object {
        if ($_ -match '^\s*([A-Z_][A-Z0-9_]*)=(.*)$') {
            $vars[$matches[1]] = $matches[2].Trim().Trim('"')
        }
    }
    return $vars
}

$envFile = Join-Path (Split-Path (Split-Path $PSScriptRoot -Parent) -Parent) ".env"
$dotenv = Get-DotEnv $envFile

# Runtime COSORA (opcion B) + reranker Tristan + Graph RAG
$runtimeVars = @{
    CHROMA_PATH        = "/app/data/chroma_db"
    CHROMA_COLLECTION  = "cosora_actas_e5"
    BM25_PATH          = "/app/data/chroma_db/bm25.json"
    HF_MODEL_PATH      = "/app/hf_model"
    RR_MODEL_PATH      = "/app/rr_model"
    EMBEDDING_STYLE    = "e5"
    RERANK_ENABLED     = "1"
    RERANK_POOL_N      = "20"
    RAG_MODE           = "cypher_transversal"
    CYPHER_ROUTE       = "hybrid"
    TOP_N              = "10"
    RETRIEVAL_K        = "50"
    RRF_K              = "60"
    REBUILD_BM25       = "0"
    NEO4J_USER         = "neo4j"
    NEO4J_DATABASE     = "neo4j"
}

# Secretos / config desde .env local si existen
foreach ($key in @(
    "OPENAI_API_KEY", "APP_PASSWORD",
    "NEO4J_URI", "NEO4J_PASSWORD"
)) {
    if ($dotenv.ContainsKey($key) -and $dotenv[$key]) {
        $runtimeVars[$key] = $dotenv[$key]
    }
}

$envPairs = ($runtimeVars.GetEnumerator() | ForEach-Object { "$($_.Key)=$($_.Value)" }) -join ","
Write-Host "Env vars a configurar: $($runtimeVars.Keys -join ', ')" -ForegroundColor Cyan
if (-not $runtimeVars.ContainsKey("NEO4J_URI")) {
    Write-Host "AVISO: NEO4J_URI no esta en .env - Graph RAG no conectara hasta anadirlo." -ForegroundColor Yellow
}

Write-Host "Iniciando Cloud Build..." -ForegroundColor Cyan
gcloud builds submit --config deploy/cloudbuild.yaml

Write-Host "Desplegando en Cloud Run..." -ForegroundColor Cyan
gcloud run deploy cosora-demo `
    --image $IMAGE_NAME `
    --platform managed `
    --region $REGION `
    --allow-unauthenticated `
    --memory 8Gi `
    --cpu 4 `
    --update-env-vars $envPairs

Write-Host "Despliegue finalizado!" -ForegroundColor Green
Write-Host "URL: https://cosora-demo-475080291256.europe-west1.run.app" -ForegroundColor Green

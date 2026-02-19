# PIBot Serving

Production-ready **FastAPI** endpoint for the PIBot JointBERT model.  
Serves multi-head intent classification + BIO slot filling for macroeconomic queries in Spanish.

---

## Features

- **Single & batch inference** (`POST /predict`, `POST /predict/batch`)
- **Model from HF Hub or local** – controlled via `MODEL_SOURCE` env var
- **Docker-ready** – multi-stage build (CPU & GPU variants)
- **Prometheus metrics** at `/metrics`
- **Structured JSON logging** (structlog)
- **OpenAPI docs** auto-generated (`/docs`, `/redoc`)
- **Healthcheck** endpoint (`/health`)
- **CORS** configurable
- **Non-root container** user for security

---

## Project structure

```text
pibert_serving/
├── app/
│   ├── __init__.py
│   ├── main.py                # ASGI entrypoint (uvicorn app.main:app)
│   ├── config.py              # pydantic-settings (env vars / .env)
│   ├── logging_config.py      # Structured logging setup
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # /predict, /predict/batch, /health, /labels
│   │   └── schemas.py         # Pydantic request/response models
│   └── model/
│       ├── __init__.py
│       ├── loader.py          # Download from HF or load local, init model
│       └── predictor.py       # Tokenize → forward → decode
├── tests/
│   ├── test_api.py            # HTTP endpoint tests (mocked model)
│   └── test_predictor.py      # Unit tests for BIO extraction
├── scripts/
│   └── healthcheck.py         # Docker HEALTHCHECK script
├── Dockerfile                 # CPU multi-stage build
├── Dockerfile.gpu             # GPU (CUDA 12.1) multi-stage build
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
└── README.md
```

---

## Quick start (local)

### 1) Create virtual environment

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configure

```bash
cp .env.example .env
# Edit .env with your HF repo or local model path
```

Key variables:

| Variable | Description | Default |
|---|---|---|
| `MODEL_SOURCE` | `huggingface` or `local` | `huggingface` |
| `HF_REPO_ID` | HF repo (e.g. `smenaaliaga/pibert`) | `smenaaliaga/pibert` |
| `HF_TOKEN` | HF token (only for private repos) | — |
| `MODEL_LOCAL_DIR` | Absolute path to local model dir | — |
| `MAX_SEQ_LEN` | Max input sequence length | `64` |
| `DEVICE` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `APP_PORT` | Server port | `8000` |
| `LOG_LEVEL` | `debug`, `info`, `warning`, `error` | `info` |
| `ENABLE_METRICS` | Expose `/metrics` (Prometheus) | `true` |
| `CORS_ORIGINS` | JSON array of allowed origins | `["*"]` |

### 3) Run

```bash
uvicorn app.main:app --reload
```

Server starts at `http://localhost:8000`. Interactive docs at `http://localhost:8000/docs`.

---

## Docker

### CPU

```bash
docker compose up --build
```

### GPU

Edit `docker-compose.yml` to uncomment the GPU service, or build directly:

```bash
docker build -f Dockerfile.gpu -t pibert-serving-gpu .
docker run --gpus all -p 8000:8000 --env-file .env pibert-serving-gpu
```

### With local model (no HF download)

Mount the model directory at container startup:

```bash
docker run -p 8000:8000 \
  -e MODEL_SOURCE=local \
  -e MODEL_LOCAL_DIR=/app/model \
  -v /path/to/model_package:/app/model:ro \
  pibert-serving
```

---

## API Reference

### `GET /health`

Returns model status and device.

```json
{
  "status": "ok",
  "model_loaded": true,
  "device": "cuda",
  "model_source": "huggingface"
}
```

### `POST /predict`

**Request:**
```json
{
  "text": "cual fue el ultimo imacec"
}
```

**Response:**
```json
{
  "text": "cual fue el ultimo imacec",
  "words": ["cual", "fue", "el", "ultimo", "imacec"],
  "calc_mode": { "label": "variacion", "confidence": 0.92 },
  "activity":  { "label": "total",     "confidence": 0.88 },
  "region":    { "label": "nacional",  "confidence": 0.95 },
  "investment":{ "label": "total",     "confidence": 0.91 },
  "req_form":  { "label": "valor",     "confidence": 0.97 },
  "slot_tags": ["O", "O", "O", "B-period", "B-indicator"],
  "entities": {
    "period": ["ultimo"],
    "indicator": ["imacec"]
  }
}
```

### `POST /predict/batch`

**Request:**
```json
{
  "texts": [
    "cual fue el ultimo imacec",
    "pib real acumulado 2024"
  ]
}
```

**Response:**
```json
{
  "predictions": [
    { "text": "...", "calc_mode": {...}, ... },
    { "text": "...", "calc_mode": {...}, ... }
  ]
}
```

### `GET /labels`

Returns all label mappings loaded from the model.

### `GET /metrics`

Prometheus metrics (request count, latency histograms, etc).

---

## Usage examples

### cURL

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "cual fue la ultima cifra del imacec"}'
```

Batch:

```bash
curl -X POST http://localhost:8000/predict/batch \
  -H "Content-Type: application/json" \
  -d '{"texts": ["cual fue el ultimo imacec", "pib real acumulado 2024"]}'
```

### PowerShell

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict `
  -Method Post -ContentType "application/json" `
  -Body '{"text": "cual fue la ultima cifra del imacec"}'
```

Batch:

```powershell
Invoke-RestMethod -Uri http://localhost:8000/predict/batch `
  -Method Post -ContentType "application/json" `
  -Body '{"texts": ["cual fue el ultimo imacec", "pib real acumulado 2024"]}'
```

### Python (httpx)

```python
import httpx

response = httpx.post(
    "http://localhost:8000/predict",
    json={"text": "cual fue el ultimo imacec"},
)
print(response.json())
```

### Swagger UI

Open `http://localhost:8000/docs` in your browser to test endpoints interactively.

---

## Tests

```bash
pip install -r requirements-dev.txt
pytest -v
```

Tests use mocked model bundles – no real weights needed.

---

## Deployment patterns

### Azure Container Apps / App Service

1. Build & push image to ACR:
   ```bash
   az acr build --registry <acr-name> --image pibert-serving:latest .
   ```
2. Deploy as Container App with env vars from `.env`.

### Kubernetes

Use the Docker image + a `ConfigMap` / `Secret` for env vars.  
Set `APP_WORKERS=1` when using GPU (model is not fork-safe).

### Azure ML Managed Endpoint

See the `azureml_endpoint_project/` in the parent repo for YAML-based deployment.

---

## Architecture decisions

| Decision | Rationale |
|---|---|
| **FastAPI + uvicorn** | Async ASGI, auto-generated OpenAPI, production-proven |
| **pydantic-settings** | Type-safe config from env vars with validation |
| **Lifespan events** | Model loads once at startup, not per-request |
| **Singleton ModelBundle** | Single model in memory; 1 worker for GPU safety |
| **Multi-stage Docker** | Small image (~1.5 GB CPU); deps cached in builder layer |
| **structlog JSON** | Machine-readable logs for ELK / Loki / CloudWatch |
| **Prometheus instrumentator** | Zero-config request metrics with histograms |
| **Docker HEALTHCHECK** | Container orchestrators (Compose, K8s) detect readiness |
| **Non-root user** | Security best practice for production containers |

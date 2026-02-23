# PIBot Serving

Endpoint **FastAPI** para producción para el modelo JointBERT de PIBot + clasificadores ligeros de routing.  

Entrega clasificación de decisiones de routing e intención multi-head + slot filling BIO para consultas macroeconómicas en español.

---

## Características

- **Inferencia unificada** (`POST /predict`, `POST /predict/batch`)
- **Clasificadores de routing** – sentence embeddings + 3 regresiones logísticas para routing de nodos en LangGraph
- **Interpretación JointBERT** – clasificación de intención de 5 cabezas + slot filling BIO (NER)
- **Normalizador de entidades** – fuzzy matching + reglas de inferencia para `indicator`, `seasonality`, `frequency`, `activity`, `region`, `investment`, `period`
- **Salida dual de entidades** – entidades originales extraídas + entidades normalizadas en la misma respuesta
- **Modelo desde HF Hub o local** – controlado por la variable `MODEL_SOURCE`
- **Listo para Docker** – build multi-stage (CPU y GPU)
- **Métricas Prometheus** en `/metrics`
- **Logging JSON estructurado** (structlog)
- **Documentación OpenAPI** autogenerada (`/docs`, `/redoc`)
- **Endpoint de healthcheck** (`/health`)
- **CORS** configurable
- **Usuario no-root** en contenedor para seguridad

---

## Documentación técnica

- Reglas del normalizador y orden de aplicación: [docs/normalizer_rules.md](docs/normalizer_rules.md)

---

## Estructura del proyecto

```text
bc_pibert_endpoint/
├── app/
│   ├── __init__.py
│   ├── main.py                # Punto de entrada ASGI (uvicorn app.main:app)
│   ├── config.py              # pydantic-settings (variables de entorno / .env)
│   ├── logging_config.py      # Configuración de logging estructurado
│   ├── api/
│   │   ├── __init__.py
│   │   ├── routes.py          # /predict, /predict/batch, /health, /labels, /router/labels
│   │   └── schemas.py         # Modelos Pydantic de request/response
│   └── model/
│       ├── __init__.py
│       ├── loader.py          # Descarga desde HF o carga local, inicializa JointBERT
│       ├── predictor.py       # Tokenize → forward → decode (JointBERT)
│       ├── normalizer.py      # Normalización de entidades (fuzzy + inferencia)
│       └── router.py          # Sentence embeddings + clasificadores sklearn (routing)
├── tests/
│   ├── test_api.py            # Tests HTTP de endpoints (modelos mockeados)
│   └── test_predictor.py      # Tests unitarios de extracción BIO
├── scripts/
│   └── healthcheck.py         # Script Docker HEALTHCHECK
├── Dockerfile                 # Build multi-stage CPU
├── Dockerfile.gpu             # Build multi-stage GPU (CUDA 12.1)
├── docker-compose.yml
├── .dockerignore
├── .env.example
├── .gitignore
├── requirements.txt
├── requirements-dev.txt
├── pyproject.toml
├── INTEGRATION.md             # Guía para conectar modelos reales de routing
└── README.md
```

---

## Inicio rápido (local)

### 1) Crear entorno virtual

```bash
python -m venv .venv

# Windows
.venv\Scripts\activate

# Linux / macOS
source .venv/bin/activate

pip install -r requirements.txt
```

### 2) Configurar

```bash
cp .env.example .env
# Edita .env con tu repo de HF o ruta local del modelo
```

Variables clave:

| Variable | Descripción | Valor por defecto |
|---|---|---|
| `MODEL_SOURCE` | `huggingface` o `local` | `huggingface` |
| `HF_REPO_ID` | Repo de HF (ej. `smenaaliaga/pibert`) | `smenaaliaga/pibert` |
| `HF_TOKEN` | Token de HF (solo repos privados) | — |
| `MODEL_LOCAL_DIR` | Ruta absoluta al directorio local del modelo | — |
| `MAX_SEQ_LEN` | Longitud máxima de secuencia de entrada | `64` |
| `DEVICE` | `auto`, `cpu`, `cuda`, `mps` | `auto` |
| `ROUTER_ENABLED` | Habilita clasificadores de routing | `true` |
| `ROUTER_EMBEDDING_MODEL` | Nombre del modelo sentence-transformers | `paraphrase-multilingual-MiniLM-L12-v2` |
| `ROUTER_HF_REPO_ID` | Repo de HF con clasificadores `.joblib` | `smenaaliaga/pibert-router` |
| `ROUTER_HF_TOKEN` | Token de HF para repo de router (solo privado) | — |
| `APP_PORT` | Puerto del servidor | `8000` |
| `LOG_LEVEL` | `debug`, `info`, `warning`, `error` | `info` |
| `ENABLE_METRICS` | Expone `/metrics` (Prometheus) | `true` |
| `CORS_ORIGINS` | Arreglo JSON de orígenes permitidos | `["*"]` |

### 3) Ejecutar

```bash
uvicorn app.main:app --reload
```

El servidor inicia en `http://localhost:8000`. Documentación interactiva en `http://localhost:8000/docs`.

---

## Docker

### CPU

```bash
docker compose up --build
```

### GPU

Edita `docker-compose.yml` para descomentar el servicio GPU, o construye directamente:

```bash
docker build -f Dockerfile.gpu -t pibert-serving-gpu .
docker run --gpus all -p 8000:8000 --env-file .env pibert-serving-gpu
```

### Con modelo local (sin descarga desde HF)

Monta el directorio del modelo al iniciar el contenedor:

```bash
docker run -p 8000:8000 \
  -e MODEL_SOURCE=local \
  -e MODEL_LOCAL_DIR=/app/model \
  -v /path/to/model_package:/app/model:ro \
  pibert-serving
```

---

## Referencia de API

### `GET /health`

Retorna estado del modelo, estado del router y dispositivo.

```json
{
  "status": "ok",
  "model_loaded": true,
  "router_loaded": true,
  "device": "cuda",
  "model_source": "huggingface"
}
```

### `POST /predict`

**Request:**
```json
{
  "text": "cuanto crecio el sector no minerio entre mayo y agosto del 2025"
}
```

**Response:**
```json
{
  "text": "cuanto crecio el sector no minerio entre mayo y agosto del 2025",
  "routing": {
    "macro": { "label": 1, "confidence": 0.97 },
    "intent": { "label": "value", "confidence": 0.94 },
    "context": { "label": "standalone", "confidence": 0.89 }
  },
  "interpretation": {
    "words": ["cuanto", "crecio", "el", "sector", "no", "minerio", "entre", "mayo", "y", "agosto", "del", "2025"],
    "intents": {
      "calc_mode": { "label": "yoy", "confidence": 0.99 },
      "activity": { "label": "specific", "confidence": 0.99 },
      "region": { "label": "none", "confidence": 0.99 },
      "investment": { "label": "none", "confidence": 0.99 },
      "req_form": { "label": "range", "confidence": 0.99 }
    },
    "slot_tags": ["O", "O", "O", "O", "B-activity", "I-activity", "B-period", "I-period", "I-period", "I-period", "I-period", "I-period"],
    "entities": {
      "activity": ["no minerio"],
      "period": ["entre mayo y agosto del 2025"]
    },
    "entities_normalized": {
      "indicator": ["imacec"],
      "seasonality": ["nsa"],
      "frequency": ["m"],
      "activity": ["no_mineria"],
      "region": [],
      "investment": [],
      "period": ["2025-05-01", "2025-08-31"]
    }
  }
}
```

> **Nota:** El campo `routing` es `null` cuando `ROUTER_ENABLED=false` o cuando falla la carga del router.
> El campo `entities_normalized` es `null` cuando falla la normalización.
> La respuesta incluye ambos campos: `entities` (valores originales extraídos) y `entities_normalized` (mapa normalizado).

### Detalles de `interpretation.entities_normalized`

- Las claves son fijas: `indicator`, `seasonality`, `frequency`, `activity`, `region`, `investment`, `period`.
- Los valores son `list[string]` para todas las claves excepto `period`.
- `period` se normaliza a **YYYY-MM-DD**.
- `period` según `req_form`:
  - `latest`, `point` y `range` → lista de 2 elementos `[fecha_inicio, fecha_fin]`
- Los errores de normalización no rompen la request: la API devuelve predicción y `entities_normalized: null`.

### Valores posibles por campo

#### `routing`

- `routing.macro.label`: `1 | 0`
- `routing.intent.label`: `"value" | "method" | "other"`
- `routing.context.label`: `"standalone" | "followup"`

#### `interpretation.intents` (clasificadores de etiqueta única)

Estos campos retornan **un único valor** por predicción:

- `interpretation.intents.calc_mode.label`: `"original" | "prev_period" | "yoy" | "contribution"`
- `interpretation.intents.activity.label`: `"general" | "specific" | "none"`
- `interpretation.intents.region.label`: `"general" | "specific" | "none"`
- `interpretation.intents.investment.label`: `"general" | "specific" | "none"`
- `interpretation.intents.req_form.label`: `"latest" | "point" | "range"`

#### `entities_normalized`

- `entities_normalized.indicator`: `["imacec"] | ["pib"]`
- `entities_normalized.seasonality`: `["sa"] | ["nsa"]`
- `entities_normalized.frequency`: `["m"] | ["q"] | ["a"]`
- `entities_normalized.activity`: `[]` o alguna clave normalizada:
  - IMACEC: `bienes | mineria | industria | resto_bienes | comercio | servicios | no_mineria | impuestos`
  - PIB: `agropecuario | pesca | industria | electricidad | construccion | comercio | restaurantes | transporte | comunicaciones | servicio_financieros | servicios_empresariales | servicio_viviendas | servicio_personales | admin_publica | impuestos`
- `entities_normalized.region`: `[]` o una clave de región:
  - `arica_parinacota | tarapaca | antofagasta | atacama | coquimbo | valparaiso | metropolitana | ohiggins | maule | nuble | biobio | araucania | los_rios | los_lagos | aysen | magallanes`
- `entities_normalized.investment`: `[]` o una clave normalizada:
  - `demanda_interna | consumo | consumo_gobierno | inversion | inversion_fijo | existencia | exportacion | importacion | ahorro_externo | ahorro_interno`
- `entities_normalized.period`:
  - `latest`, `point` y `range`: lista `["YYYY-MM-DD", "YYYY-MM-DD"]`

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
    { "text": "...", "routing": {...}, "interpretation": {...} },
    { "text": "...", "routing": {...}, "interpretation": {...} }
  ]
}
```

### `GET /labels`

Retorna el mapeo de labels de JointBERT cargado desde el modelo.

### `GET /router/labels`

Retorna el mapeo de labels de los clasificadores de routing.

```json
{
  "macro": [1, 0],
  "intent": ["value", "method", "other"],
  "context": ["standalone", "followup"]
}
```

### `GET /metrics`

Métricas Prometheus (conteo de requests, histogramas de latencia, etc).

---

## Ejemplos de uso

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
data = response.json()

# Decisiones de routing para LangGraph
print(data["routing"]["macro"]["label"])      # 1
print(data["routing"]["intent"]["label"])     # "value"

# Interpretación para consulta de series
print(data["interpretation"]["entities"])            # valores originales extraídos
print(data["interpretation"]["entities_normalized"]) # valores normalizados
```

### Swagger UI

Abre `http://localhost:8000/docs` en tu navegador para probar endpoints de forma interactiva.

### Chat interactivo (`tests/endpoint_chat.py`)

Este script permite enviar múltiples consultas al endpoint `/predict` en modo chat y ver la respuesta formateada.

Uso:

```bash
python tests/endpoint_chat.py
python tests/endpoint_chat.py --url http://localhost:8000
```

Requisito: el servidor debe estar encendido (por ejemplo, `uvicorn app.main:app --reload`).

---

## Integrar modelos reales de routing

El router actualmente retorna **predicciones dummy**. Para conectar modelos
reales de sentence-transformer + clasificadores scikit-learn, sigue la guía
paso a paso en **[INTEGRATION.md](INTEGRATION.md)**.

---

## Pruebas

```bash
pip install -r requirements-dev.txt
pytest -v
```

Las pruebas usan bundles de modelo mockeados: no se requieren pesos reales.

---

## Patrones de despliegue

### Azure Container Apps / App Service

1. Construye y publica la imagen en ACR:
   ```bash
   az acr build --registry <acr-name> --image pibert-serving:latest .
   ```
2. Despliega como Container App con variables de entorno desde `.env`.

### Kubernetes

Usa la imagen Docker + un `ConfigMap` / `Secret` para variables de entorno.  
Define `APP_WORKERS=1` al usar GPU (el modelo no es fork-safe).

### Azure ML Managed Endpoint

Consulta `azureml_endpoint_project/` en el repo padre para despliegue basado en YAML.

---

## Decisiones de arquitectura

| Decisión | Justificación |
|---|---|
| **FastAPI + uvicorn** | ASGI asíncrono, OpenAPI autogenerado, probado en producción |
| **pydantic-settings** | Configuración type-safe desde variables de entorno con validación |
| **Eventos de lifespan** | Los modelos cargan una vez al inicio, no por request |
| **Singleton ModelBundle + RouterBundle** | Modelos únicos en memoria; 1 worker por seguridad en GPU |
| **Carga tolerante del router** | Fallos del router no bloquean JointBERT; retorna `routing: null` |
| **Formato de respuesta unificado** | `routing` + `interpretation` en una sola respuesta para LangGraph |
| **Modo dummy de routing** | El endpoint funciona antes de entrenar clasificadores reales |
| **Docker multi-stage** | Imagen más pequeña (~1.5 GB CPU); dependencias cacheadas en capa builder |
| **structlog JSON** | Logs legibles por máquinas para ELK / Loki / CloudWatch |
| **Instrumentación Prometheus** | Métricas de requests con histogramas sin configuración compleja |
| **Docker HEALTHCHECK** | Orquestadores (Compose, K8s) detectan readiness |
| **Usuario no-root** | Buena práctica de seguridad para contenedores en producción |

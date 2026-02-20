# Router Model Integration Guide

This document explains how to replace the **dummy routing predictions** with real
sentence-transformer embeddings + scikit-learn logistic regression classifiers.

---

## Architecture Overview

```text
                    ┌─────────────────────┐
   text ──────────▶│ SentenceTransformer  │──▶ embedding (384-d)
                    └─────────────────────┘
                              │
              ┌───────────────┼───────────────┐
              ▼               ▼               ▼
        ┌──────────┐   ┌──────────┐   ┌──────────┐
        │ macro    │   │ intent   │   │ context  │
        │ LogReg   │   │ LogReg   │   │ LogReg   │
        └──────────┘   └──────────┘   └──────────┘
              │               │               │
              ▼               ▼               ▼
                 1            value        standalone
        (conf: 0.97)    (conf: 0.94)     (conf: 0.89)
```

The sentence-transformer produces a dense embedding vector for the input text.
Three independent logistic regression classifiers (trained with scikit-learn)
consume that embedding and produce the routing labels used by LangGraph.

---

## Prerequisites

### 1. Train the classifiers

```python
from sentence_transformers import SentenceTransformer
from sklearn.linear_model import LogisticRegression
import joblib

# 1. Encode training corpus
model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
X = model.encode(train_texts)  # shape: (N, 384)

# 2. Train one classifier per head
for head in ("macro", "intent", "context"):
    clf = LogisticRegression(max_iter=1000, multi_class="multinomial")
    clf.fit(X, y_labels[head])
    joblib.dump(clf, f"{head}_clf.joblib")
```

### 2. Upload to Hugging Face Hub

Create a HF repository (e.g., `smenaaliaga/pibert-router`) with these files:

```text
smenaaliaga/pibert-router/
├── macro_clf.joblib
├── intent_clf.joblib
└── context_clf.joblib
```

```python
from huggingface_hub import HfApi

api = HfApi()
for head in ("macro", "intent", "context"):
    api.upload_file(
        path_or_fileobj=f"{head}_clf.joblib",
        path_in_repo=f"{head}_clf.joblib",
        repo_id="smenaaliaga/pibert-router",
    )
```

---

## Integration Steps

### Step 1 – Update `RouterBundle.load()` in `app/model/router.py`

Replace the stub with real loading logic:

```python
def load(self) -> None:
    from sentence_transformers import SentenceTransformer
    from huggingface_hub import hf_hub_download
    import joblib

    # Load embedding model
    self.embedding_model = SentenceTransformer(
        settings.router_embedding_model,
        device=settings.device if settings.device != "auto" else None,
    )
    logger.info(
        "Sentence-transformer loaded: %s", settings.router_embedding_model
    )

    # Download and load sklearn classifiers from HF
    for head in ("macro", "intent", "context"):
        path = hf_hub_download(
            repo_id=settings.router_hf_repo_id,
            filename=f"{head}_clf.joblib",
            token=settings.router_hf_token,
            cache_dir="model_cache",
        )
        self.classifiers[head] = joblib.load(path)
        logger.info("Loaded classifier: %s_clf.joblib", head)

    self._loaded = True
    logger.info("RouterBundle loaded successfully")
```

### Step 2 – Update `route()` in `app/model/router.py`

Replace the dummy branch with real inference:

```python
def route(bundle: RouterBundle, text: str) -> dict:
    embedding = bundle.embedding_model.encode(text)
    result: dict = {}
    for head, clf in bundle.classifiers.items():
        probs = clf.predict_proba([embedding])[0]
        pred_idx = probs.argmax()
        result[head] = bundle.labels[head][pred_idx]
        result[f"{head}_confidence"] = round(float(probs[pred_idx]), 6)
    return result
```

### Step 3 – Update labels (if needed)

If your trained classifiers have different labels than the defaults in
`ROUTER_LABELS`, update the dict in `app/model/router.py`:

```python
ROUTER_LABELS: Dict[str, List[str | int]] = {
  "macro": [1, 0],
  "intent": ["value", "method", "other"],
  "context": ["standalone", "followup"],
}
```

> **Important:** The label order must match `clf.classes_` from your trained
> sklearn model. Verify with `print(clf.classes_)` after loading.

### Step 4 – Configure environment

```bash
# .env
ROUTER_ENABLED=true
ROUTER_EMBEDDING_MODEL=paraphrase-multilingual-MiniLM-L12-v2
ROUTER_HF_REPO_ID=smenaaliaga/pibert-router
# ROUTER_HF_TOKEN=hf_xxx  # only if the repo is private
```

### Step 5 – Test

```bash
# Unit test
pytest -v

# Manual test
uvicorn app.main:app --reload
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"text": "cual fue la ultima cifra del imacec"}'
```

Verify the `routing` block now has real confidence scores (not the hardcoded
0.97 / 0.94 / 0.89).

---

## Environment Variables Reference

| Variable | Description | Default |
|---|---|---|
| `ROUTER_ENABLED` | Enable/disable routing predictions | `true` |
| `ROUTER_EMBEDDING_MODEL` | sentence-transformers model name | `paraphrase-multilingual-MiniLM-L12-v2` |
| `ROUTER_HF_REPO_ID` | HF repo with `.joblib` classifiers | `smenaaliaga/pibert-router` |
| `ROUTER_HF_TOKEN` | HF token (private repos only) | — |

---

## Response Format

When the router is active, `/predict` returns:

```json
{
  "text": "cual fue la ultima cifra del imacec",
  "routing": {
    "macro": { "label": 1, "confidence": 0.97 },
    "intent": { "label": "value", "confidence": 0.94 },
    "context": { "label": "standalone", "confidence": 0.89 }
  },
  "interpretation": {
    "words": ["cual", "fue", "la", "ultima", "cifra", "del", "imacec"],
    "intents": {
      "calc_mode": { "label": "variacion", "confidence": 0.92 },
      "activity": { "label": "total", "confidence": 0.88 },
      "region": { "label": "nacional", "confidence": 0.95 },
      "investment": { "label": "total", "confidence": 0.91 },
      "req_form": { "label": "valor", "confidence": 0.97 }
    },
    "slot_tags": ["O", "O", "O", "B-period", "O", "O", "B-indicator"],
    "entities": {
      "period": ["ultima"],
      "indicator": ["imacec"]
    },
    "entities_normalized": {
      "indicator": ["imacec"],
      "seasonality": [],
      "frequency": [],
      "activity": [],
      "region": [],
      "investment": [],
      "period": "01-06-2025"
    }
  }
}
```

When `ROUTER_ENABLED=false` or the router fails to load:

```json
{
  "text": "...",
  "routing": null,
  "interpretation": { ... }
}
```

---

## Troubleshooting

| Issue | Solution |
|---|---|---|
| `routing` is always `null` | Check `ROUTER_ENABLED=true` and logs for load errors |
| Label mismatch | Ensure `ROUTER_LABELS` order matches `clf.classes_` |
| Slow first request | sentence-transformers downloads model on first use; pre-warm in Docker build |
| OOM on GPU | sentence-transformers + JointBERT may not fit; set `DEVICE=cpu` for the router |

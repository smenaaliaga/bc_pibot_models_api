"""
API routes for the PIBot serving endpoint.

Endpoints:
    POST /predict          – single-text inference
    POST /predict/batch    – batch inference
    GET  /health           – liveness + model status
    GET  /labels           – list label mappings
"""

from __future__ import annotations

import logging
import time

from fastapi import APIRouter, HTTPException

from app.api.schemas import (
    BatchPredictRequest,
    BatchPredictResponse,
    ErrorResponse,
    HealthResponse,
    HeadPrediction,
    PredictRequest,
    PredictResponse,
)
from app.config import settings
from app.model.loader import bundle
from app.model.predictor import predict

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────


def _to_response(raw: dict) -> PredictResponse:
    """Convert raw predictor dict → typed Pydantic response."""
    return PredictResponse(
        text=raw["text"],
        words=raw["words"],
        calc_mode=HeadPrediction(label=raw["calc_mode"], confidence=raw["calc_mode_confidence"]),
        activity=HeadPrediction(label=raw["activity"], confidence=raw["activity_confidence"]),
        region=HeadPrediction(label=raw["region"], confidence=raw["region_confidence"]),
        investment=HeadPrediction(label=raw["investment"], confidence=raw["investment_confidence"]),
        req_form=HeadPrediction(label=raw["req_form"], confidence=raw["req_form_confidence"]),
        slot_tags=raw["slot_tags"],
        entities=raw["entities"],
    )


def _ensure_model_loaded() -> None:
    if not bundle.is_loaded:
        raise HTTPException(status_code=503, detail="Model not loaded yet. Try again shortly.")


# ── Routes ────────────────────────────────────────────────────


@router.get(
    "/health",
    response_model=HealthResponse,
    tags=["health"],
    summary="Health check",
)
async def health():
    return HealthResponse(
        status="ok" if bundle.is_loaded else "loading",
        model_loaded=bundle.is_loaded,
        device=bundle.device if bundle.is_loaded else None,
        model_source=settings.model_source.value,
    )


@router.get(
    "/labels",
    tags=["metadata"],
    summary="List model label mappings",
)
async def get_labels():
    _ensure_model_loaded()
    return {key: labels for key, labels in bundle.labels.items()}


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["inference"],
    summary="Single-text prediction",
)
async def predict_single(req: PredictRequest):
    _ensure_model_loaded()

    start = time.perf_counter()
    raw = predict(bundle, req.text)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info("predict", extra={"text_len": len(req.text), "latency_ms": round(latency_ms, 2)})
    return _to_response(raw)


@router.post(
    "/predict/batch",
    response_model=BatchPredictResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["inference"],
    summary="Batch prediction (up to 64 texts)",
)
async def predict_batch(req: BatchPredictRequest):
    _ensure_model_loaded()

    start = time.perf_counter()
    results = [predict(bundle, text) for text in req.texts]
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict_batch",
        extra={"batch_size": len(req.texts), "latency_ms": round(latency_ms, 2)},
    )
    return BatchPredictResponse(predictions=[_to_response(r) for r in results])

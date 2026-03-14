"""
API routes for the PIBot serving endpoint.

Endpoints:
    POST /predict          – single-text inference (routing + interpretation)
    POST /predict/batch    – batch inference
    GET  /health           – liveness + model status
    GET  /versions         – loaded model/router source version info
    GET  /labels           – list JointBERT label mappings
    GET  /router/labels    – list router label mappings
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
    InterpretationResponse,
    PredictRequest,
    PredictResponse,
    RoutingResponse,
    ModelVersionInfo,
    VersionsResponse,
)
from app.config import settings
from app.model.loader import bundle
from app.model.predictor import predict
from app.model.router import route, router_bundle

logger = logging.getLogger(__name__)

router = APIRouter()


# ── Helpers ───────────────────────────────────────────────────


def _build_routing(raw_route: dict) -> RoutingResponse:
    """Convert flat router dict → typed RoutingResponse."""
    return RoutingResponse(
        macro=HeadPrediction(label=raw_route["macro"], confidence=raw_route["macro_confidence"]),
        intent=HeadPrediction(label=raw_route["intent"], confidence=raw_route["intent_confidence"]),
        context=HeadPrediction(label=raw_route["context"], confidence=raw_route["context_confidence"]),
    )


def _canonicalize_slot_tags(slot_tags: list[str]) -> list[str]:
    """Canonicalize BIO tags to O | B-<entity> | I-<entity> (entity in lowercase)."""
    normalized: list[str] = []
    for tag in slot_tags:
        if not tag or str(tag).upper() == "O":
            normalized.append("O")
            continue

        if "-" not in tag:
            normalized.append(tag)
            continue

        prefix, entity_type = tag.split("-", 1)
        prefix_norm = prefix.upper()
        if prefix_norm not in {"B", "I"}:
            normalized.append(tag)
            continue

        normalized.append(f"{prefix_norm}-{entity_type.lower()}")

    return normalized


def _canonicalize_entities(entities: dict[str, list[str]]) -> dict[str, list[str]]:
    """Lowercase entity keys and merge duplicates while preserving value order."""
    canonical: dict[str, list[str]] = {}
    for key, values in entities.items():
        key_norm = key.lower()
        if key_norm not in canonical:
            canonical[key_norm] = []

        for value in values:
            if value not in canonical[key_norm]:
                canonical[key_norm].append(value)

    return canonical


def _build_interpretation(raw: dict) -> InterpretationResponse:
    """Convert flat predictor dict → typed InterpretationResponse."""
    canonical_slot_tags = _canonicalize_slot_tags(raw["slot_tags"])
    canonical_entities = _canonicalize_entities(raw["entities"])

    return InterpretationResponse(
        words=raw["words"],
        intents={
            "calc_mode": HeadPrediction(label=raw["calc_mode"], confidence=raw["calc_mode_confidence"]),
            "activity": HeadPrediction(label=raw["activity"], confidence=raw["activity_confidence"]),
            "region": HeadPrediction(label=raw["region"], confidence=raw["region_confidence"]),
            "investment": HeadPrediction(label=raw["investment"], confidence=raw["investment_confidence"]),
            "req_form": HeadPrediction(label=raw["req_form"], confidence=raw["req_form_confidence"]),
        },
        slot_tags=canonical_slot_tags,
        entities=canonical_entities,
    )


def _to_response(raw_predict: dict, raw_route: dict | None) -> PredictResponse:
    """Combine routing + interpretation into a unified response."""
    return PredictResponse(
        text=raw_predict["text"],
        routing=_build_routing(raw_route) if raw_route else None,
        interpretation=_build_interpretation(raw_predict),
    )


def _route_or_none(text: str) -> dict | None:
    """Return routing dict, or None only when router is intentionally disabled."""
    if not settings.router_enabled:
        return None
    if not router_bundle.is_loaded:
        raise HTTPException(status_code=503, detail="Router model is enabled but not loaded.")
    return route(router_bundle, text)


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
        router_loaded=router_bundle.is_loaded,
        device=bundle.device if bundle.is_loaded else None,
        model_source=settings.model_source.value,
        model_hf_repo_id=getattr(bundle, "hf_repo_id", None),
        model_hf_revision=getattr(bundle, "hf_revision", None),
        model_hf_commit=getattr(bundle, "hf_commit", None),
        router_hf_repo_id=getattr(router_bundle, "hf_repo_id", None),
        router_hf_revision=getattr(router_bundle, "hf_revision", None),
        router_hf_commit=getattr(router_bundle, "hf_commit", None),
    )


@router.get(
    "/versions",
    response_model=VersionsResponse,
    tags=["metadata"],
    summary="Model source versions (repo, revision, commit)",
)
async def get_versions():
    model_dir = str(bundle.model_dir) if getattr(bundle, "model_dir", None) else None
    model_info = ModelVersionInfo(
        source=settings.model_source.value,
        loaded=bundle.is_loaded,
        repo_id=getattr(bundle, "hf_repo_id", None),
        revision=getattr(bundle, "hf_revision", None),
        commit=getattr(bundle, "hf_commit", None),
        local_dir=model_dir,
    )

    router_source = "huggingface" if settings.router_enabled else "disabled"
    router_info = ModelVersionInfo(
        source=router_source,
        loaded=router_bundle.is_loaded,
        repo_id=getattr(router_bundle, "hf_repo_id", None),
        revision=getattr(router_bundle, "hf_revision", None),
        commit=getattr(router_bundle, "hf_commit", None),
        local_dir="model_cache/router_artifacts" if settings.router_enabled else None,
    )

    return VersionsResponse(model=model_info, router=router_info)


@router.get(
    "/labels",
    tags=["metadata"],
    summary="List JointBERT label mappings",
)
async def get_labels():
    _ensure_model_loaded()
    return {key: labels for key, labels in bundle.labels.items()}


@router.get(
    "/router/labels",
    tags=["metadata"],
    summary="List router label mappings",
)
async def get_router_labels():
    return router_bundle.labels


@router.post(
    "/predict",
    response_model=PredictResponse,
    responses={503: {"model": ErrorResponse}},
    tags=["inference"],
    summary="Single-text prediction (routing + interpretation)",
)
async def predict_single(req: PredictRequest):
    _ensure_model_loaded()

    start = time.perf_counter()
    raw_route = _route_or_none(req.text)
    raw_predict = predict(bundle, req.text)
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info("predict", extra={"text_len": len(req.text), "latency_ms": round(latency_ms, 2)})
    return _to_response(raw_predict, raw_route)


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
    results = []
    for text in req.texts:
        raw_route = _route_or_none(text)
        raw_predict = predict(bundle, text)
        results.append(_to_response(raw_predict, raw_route))
    latency_ms = (time.perf_counter() - start) * 1000

    logger.info(
        "predict_batch",
        extra={"batch_size": len(req.texts), "latency_ms": round(latency_ms, 2)},
    )
    return BatchPredictResponse(predictions=results)

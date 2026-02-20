"""
Pydantic request / response schemas.

Keeping schemas in a dedicated module makes them reusable for
auto-generated OpenAPI docs, client SDKs, and test fixtures.
"""

from __future__ import annotations

from typing import Dict, List, Optional

from pydantic import BaseModel, Field


# ── Request ───────────────────────────────────────────────────


class PredictRequest(BaseModel):
    """Single-text prediction request."""

    text: str = Field(
        ...,
        min_length=1,
        max_length=1024,
        examples=["cual fue el ultimo imacec"],
        description="Natural-language query to classify.",
    )


class BatchPredictRequest(BaseModel):
    """Batch prediction request."""

    texts: List[str] = Field(
        ...,
        min_length=1,
        max_length=64,
        description="List of queries (max 64 per request).",
    )


# ── Response building blocks ─────────────────────────────────


class HeadPrediction(BaseModel):
    """Result for a single classification head (reused by routing & interpretation)."""

    label: str | int
    confidence: float = Field(ge=0, le=1)


class RoutingResponse(BaseModel):
    """Routing decisions from the lightweight classifiers."""

    macro: HeadPrediction
    intent: HeadPrediction
    context: HeadPrediction


class InterpretationResponse(BaseModel):
    """JointBERT interpretation: intents + slot filling + normalised entities."""

    words: List[str]
    intents: Dict[str, HeadPrediction] = Field(
        description="Classification heads: calc_mode, activity, region, investment, req_form",
    )
    slot_tags: List[str] = Field(description="BIO tag per word")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Original entities extracted from BIO tags",
    )
    entities_normalized: Optional[Dict[str, List[str] | None]] = Field(
        None,
        description="Normalised entities map: list values for all keys; period always as 2-item list [start, end]",
    )


# ── Top-level responses ──────────────────────────────────────


class PredictResponse(BaseModel):
    """Unified prediction output: routing + interpretation for one text."""

    text: str
    routing: Optional[RoutingResponse] = Field(
        None,
        description="Routing decisions (null if router is disabled or not loaded)",
    )
    interpretation: InterpretationResponse


class BatchPredictResponse(BaseModel):
    """Batch prediction output."""

    predictions: List[PredictResponse]


# ── Health ────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    router_loaded: bool = False
    device: Optional[str] = None
    model_source: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str

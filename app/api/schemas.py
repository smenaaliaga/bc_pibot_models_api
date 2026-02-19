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


# ── Response ──────────────────────────────────────────────────


class HeadPrediction(BaseModel):
    """Result for a single classification head."""

    label: str
    confidence: float = Field(ge=0, le=1)


class PredictResponse(BaseModel):
    """Full prediction output for one text."""

    text: str
    words: List[str]

    calc_mode: HeadPrediction
    activity: HeadPrediction
    region: HeadPrediction
    investment: HeadPrediction
    req_form: HeadPrediction

    slot_tags: List[str] = Field(description="BIO tag per word")
    entities: Dict[str, List[str]] = Field(
        default_factory=dict,
        description="Grouped entities extracted from BIO tags",
    )


class BatchPredictResponse(BaseModel):
    """Batch prediction output."""

    predictions: List[PredictResponse]


# ── Health ────────────────────────────────────────────────────


class HealthResponse(BaseModel):
    status: str = "ok"
    model_loaded: bool
    device: Optional[str] = None
    model_source: Optional[str] = None


class ErrorResponse(BaseModel):
    detail: str

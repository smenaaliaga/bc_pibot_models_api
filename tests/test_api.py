"""
Tests for the /predict and /health API endpoints.

These tests use a **mocked** model bundle so they run without
downloading real weights (useful for CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Fake bundles for tests ────────────────────────────────────

_FAKE_LABELS = {
    "calc_mode": ["variacion", "nivel", "promedio"],
    "activity": ["total", "mineria", "industria"],
    "region": ["nacional", "rm", "biobio"],
    "investment": ["total", "fija", "inventarios"],
    "req_form": ["valor", "grafico", "tabla"],
    "slot": ["O", "B-indicator", "I-indicator", "B-period", "I-period"],
}

_FAKE_PREDICT_RESULT = {
    "text": "cual fue el ultimo imacec",
    "words": ["cual", "fue", "el", "ultimo", "imacec"],
    "calc_mode": "variacion",
    "calc_mode_confidence": 0.92,
    "activity": "total",
    "activity_confidence": 0.88,
    "region": "nacional",
    "region_confidence": 0.95,
    "investment": "total",
    "investment_confidence": 0.91,
    "req_form": "valor",
    "req_form_confidence": 0.97,
    "slot_tags": ["O", "O", "O", "B-PERIOD", "B-INDICATOR"],
    "entities": {"PERIOD": ["ultimo"], "INDICATOR": ["imacec"]},
}

_FAKE_ROUTE_RESULT = {
    "macro": 1,
    "macro_confidence": 0.97,
    "intent": "value",
    "intent_confidence": 0.94,
    "context": "standalone",
    "context_confidence": 0.89,
}

_FAKE_ROUTER_LABELS = {
    "macro": [1, 0],
    "intent": ["value", "methodology", "other"],
    "context": ["standalone", "followup"],
}


def _make_fake_bundle():
    """Create a mock ModelBundle that looks loaded."""
    b = MagicMock()
    b.is_loaded = True
    b.device = "cpu"
    b.labels = _FAKE_LABELS
    b.hf_repo_id = "BCCh/pibert"
    b.hf_revision = "main"
    b.hf_commit = "abc123def456"
    b.model_dir = "model_cache/snapshots/BCCh--pibert"
    return b


def _make_fake_router_bundle():
    """Create a mock RouterBundle that looks loaded."""
    b = MagicMock()
    b.is_loaded = True
    b.labels = _FAKE_ROUTER_LABELS
    b.hf_repo_id = "BCCh/pibot-intent-router"
    b.hf_revision = "main"
    b.hf_commit = "fedcba654321"
    return b


@pytest.fixture()
def mock_bundle():
    """Patch the global bundles + predict/route functions for HTTP tests."""
    fake = _make_fake_bundle()
    fake_router = _make_fake_router_bundle()
    with (
        patch("app.api.routes.bundle", fake),
        patch("app.api.routes.predict", return_value=_FAKE_PREDICT_RESULT),
        patch("app.api.routes.router_bundle", fake_router),
        patch("app.api.routes.route", return_value=_FAKE_ROUTE_RESULT),
        patch("app.model.loader.bundle", fake),
        patch("app.model.router.router_bundle", fake_router),
    ):
        yield fake


@pytest.fixture()
async def client(mock_bundle):
    """AsyncClient wired to the FastAPI app with mocked model."""
    # Import after patching so lifespan doesn't try to load real model
    from app.main import create_app

    # Create a fresh app that skips real lifespan model loading
    test_app = create_app()

    # Override lifespan to be a no-op for tests
    from contextlib import asynccontextmanager

    @asynccontextmanager
    async def noop_lifespan(app):
        yield

    test_app.router.lifespan_context = noop_lifespan

    transport = ASGITransport(app=test_app)
    async with AsyncClient(transport=transport, base_url="http://test") as ac:
        yield ac


# ── Tests ─────────────────────────────────────────────────────


@pytest.mark.asyncio
async def test_health(client):
    resp = await client.get("/health")
    assert resp.status_code == 200
    body = resp.json()
    assert body["model_loaded"] is True
    assert body["router_loaded"] is True
    assert body["status"] == "ok"
    assert body["model_hf_repo_id"] == "BCCh/pibert"
    assert body["router_hf_repo_id"] == "BCCh/pibot-intent-router"


@pytest.mark.asyncio
async def test_versions(client):
    resp = await client.get("/versions")
    assert resp.status_code == 200
    body = resp.json()

    assert body["model"]["source"] in {"huggingface", "local"}
    assert body["model"]["loaded"] is True
    assert body["model"]["repo_id"] == "BCCh/pibert"
    assert body["model"]["revision"] == "main"
    assert body["model"]["commit"] == "abc123def456"

    assert body["router"]["source"] in {"huggingface", "disabled"}
    assert body["router"]["loaded"] is True
    assert body["router"]["repo_id"] == "BCCh/pibot-intent-router"
    assert body["router"]["revision"] == "main"
    assert body["router"]["commit"] == "fedcba654321"


@pytest.mark.asyncio
async def test_predict_single(client):
    resp = await client.post("/predict", json={"text": "cual fue el ultimo imacec"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "cual fue el ultimo imacec"

    # Routing block
    assert "routing" in body
    routing = body["routing"]
    assert routing["macro"]["label"] == 1
    assert 0 <= routing["macro"]["confidence"] <= 1
    assert routing["intent"]["label"] == "value"
    assert routing["context"]["label"] == "standalone"

    # Interpretation block
    assert "interpretation" in body
    interp = body["interpretation"]
    assert interp["intents"]["calc_mode"]["label"] == "variacion"
    assert 0 <= interp["intents"]["calc_mode"]["confidence"] <= 1
    assert isinstance(interp["entities"], dict)
    assert isinstance(interp["slot_tags"], list)
    assert interp["words"] == ["cual", "fue", "el", "ultimo", "imacec"]
    assert interp["entities"] == {"period": ["ultimo"], "indicator": ["imacec"]}
    assert interp["slot_tags"] == ["O", "O", "O", "B-period", "B-indicator"]


@pytest.mark.asyncio
async def test_predict_batch(client):
    resp = await client.post("/predict/batch", json={"texts": ["hola", "imacec junio"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2
    for pred in body["predictions"]:
        assert "routing" in pred
        assert "interpretation" in pred


@pytest.mark.asyncio
async def test_predict_empty_text(client):
    resp = await client.post("/predict", json={"text": ""})
    assert resp.status_code == 422  # validation error


@pytest.mark.asyncio
async def test_labels(client):
    resp = await client.get("/labels")
    assert resp.status_code == 200
    body = resp.json()
    assert "calc_mode" in body
    assert "slot" in body


@pytest.mark.asyncio
async def test_router_labels(client):
    resp = await client.get("/router/labels")
    assert resp.status_code == 200
    body = resp.json()
    assert "macro" in body
    assert "intent" in body
    assert "context" in body

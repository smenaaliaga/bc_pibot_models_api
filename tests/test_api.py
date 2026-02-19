"""
Tests for the /predict and /health API endpoints.

These tests use a **mocked** model bundle so they run without
downloading real weights (useful for CI).
"""

from __future__ import annotations

from unittest.mock import MagicMock, patch

import pytest
from httpx import ASGITransport, AsyncClient

# ── Fake bundle for tests ────────────────────────────────────

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
    "slot_tags": ["O", "O", "O", "B-period", "B-indicator"],
    "entities": {"period": ["ultimo"], "indicator": ["imacec"]},
}


def _make_fake_bundle():
    """Create a mock ModelBundle that looks loaded."""
    b = MagicMock()
    b.is_loaded = True
    b.device = "cpu"
    b.labels = _FAKE_LABELS
    return b


@pytest.fixture()
def mock_bundle():
    """Patch the global bundle + predict function for HTTP tests."""
    fake = _make_fake_bundle()
    with (
        patch("app.api.routes.bundle", fake),
        patch("app.api.routes.predict", return_value=_FAKE_PREDICT_RESULT),
        patch("app.model.loader.bundle", fake),
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
    assert body["status"] == "ok"


@pytest.mark.asyncio
async def test_predict_single(client):
    resp = await client.post("/predict", json={"text": "cual fue el ultimo imacec"})
    assert resp.status_code == 200
    body = resp.json()
    assert body["text"] == "cual fue el ultimo imacec"
    assert "calc_mode" in body
    assert body["calc_mode"]["label"] == "variacion"
    assert 0 <= body["calc_mode"]["confidence"] <= 1
    assert isinstance(body["entities"], dict)


@pytest.mark.asyncio
async def test_predict_batch(client):
    resp = await client.post("/predict/batch", json={"texts": ["hola", "imacec junio"]})
    assert resp.status_code == 200
    body = resp.json()
    assert len(body["predictions"]) == 2


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

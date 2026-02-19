"""
FastAPI application factory.

This is the ASGI entrypoint: ``uvicorn app.main:app``
"""

from __future__ import annotations

import logging
from contextlib import asynccontextmanager

from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware

from app.api.routes import router
from app.config import settings
from app.logging_config import setup_logging
from app.model.loader import bundle

logger = logging.getLogger(__name__)


# ── Lifespan (startup / shutdown) ────────────────────────────


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Load the model once on startup; release on shutdown."""
    setup_logging(settings.log_level)
    logger.info(
        "Starting PIBot Serving (source=%s, device=%s)",
        settings.model_source.value,
        settings.device,
    )
    bundle.load()
    logger.info("Model ready – accepting requests")
    yield
    logger.info("Shutting down PIBot Serving")


# ── App creation ─────────────────────────────────────────────


def create_app() -> FastAPI:
    app = FastAPI(
        title="PIBot Serving API",
        description=(
            "Inference endpoint for **PIBert** — a Joint BERT model for intent "
            "classification (calc_mode, activity, region, investment, req_form) "
            "and slot filling (NER) on macroeconomic queries in Spanish.\n\n"
            "Built with FastAPI · Powered by HuggingFace Transformers"
        ),
        version="1.0.0",
        lifespan=lifespan,
        docs_url="/docs",
        redoc_url="/redoc",
        openapi_tags=[
            {"name": "inference", "description": "Model inference endpoints"},
            {"name": "health", "description": "Service health and readiness"},
            {"name": "metadata", "description": "Model metadata and labels"},
        ],
    )

    # CORS
    app.add_middleware(
        CORSMiddleware,
        allow_origins=settings.cors_origins,
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )

    # Prometheus metrics
    if settings.enable_metrics:
        try:
            from prometheus_fastapi_instrumentator import Instrumentator  # type: ignore[import-untyped]

            Instrumentator().instrument(app).expose(app, endpoint="/metrics")
            logger.info("Prometheus metrics enabled at /metrics")
        except ImportError:
            logger.warning("prometheus-fastapi-instrumentator not installed – metrics disabled")

    # Routes
    app.include_router(router)

    return app


app = create_app()

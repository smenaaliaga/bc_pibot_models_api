"""
Centralised application settings loaded from environment variables / .env file.

Priority order: env vars > .env file > defaults defined here.
"""

from __future__ import annotations

from enum import Enum
from typing import List

from pydantic import Field
from pydantic_settings import BaseSettings, SettingsConfigDict


class ModelSource(str, Enum):
    huggingface = "huggingface"
    local = "local"


class Settings(BaseSettings):
    """All tuneable knobs for the serving application."""

    model_config = SettingsConfigDict(
        env_file=".env",
        env_file_encoding="utf-8",
        case_sensitive=False,
        extra="ignore",
    )

    # ── Model source ──────────────────────────────────────────
    model_source: ModelSource = ModelSource.huggingface
    hf_repo_id: str = "bcch/pibert"
    hf_token: str | None = None
    model_local_dir: str | None = None  # required when model_source == local

    # ── Inference ─────────────────────────────────────────────
    max_seq_len: int = 64
    device: str = "auto"  # auto | cpu | cuda | mps

    # ── Router model ──────────────────────────────────────────
    router_enabled: bool = True
    router_hf_repo_id: str = "bcch/pibot-intent-router"
    router_hf_token: str | None = None

    # ── Server ────────────────────────────────────────────────
    app_host: str = "0.0.0.0"
    app_port: int = 8000
    app_workers: int = 1
    log_level: str = "info"
    cors_origins: List[str] = Field(default=["*"])

    # ── Observability ─────────────────────────────────────────
    enable_metrics: bool = True


# Singleton – import ``settings`` from anywhere
settings = Settings()

"""
Router model: sentence-embedding + lightweight classifiers for LangGraph routing.

Currently returns **dummy predictions**. See ``INTEGRATION.md`` for instructions
on wiring up the real sentence-transformer + scikit-learn classifiers.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Tuple

from app.config import settings

logger = logging.getLogger(__name__)

# ── Label definitions for the three routing heads ─────────────

ROUTER_LABELS: Dict[str, List[str | int]] = {
    "macro": [1, 0],
    "intent": ["value", "method", "other"],
    "context": ["standalone", "followup"],
}


class RouterBundle:
    """
    Encapsulates the sentence-embedding model + three logistic-regression
    classifiers used for routing decisions in the LangGraph application.

    Lifecycle mirrors ``ModelBundle``: instantiated at module level,
    populated via ``load()`` at startup inside the lifespan handler.
    """

    def __init__(self) -> None:
        self.embedding_model = None          # TODO: SentenceTransformer instance
        self.classifiers: Dict[str, object] = {}  # TODO: {"macro": LogReg, ...}
        self.labels: Dict[str, List[str | int]] = ROUTER_LABELS
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """
        Load sentence-transformer + sklearn classifiers.

        Currently a **stub** that only marks the bundle as loaded so the
        endpoint returns dummy routing predictions.

        See ``INTEGRATION.md`` for the real implementation steps.
        """
        # TODO: Replace with real loading logic (see INTEGRATION.md)
        # ─── Future code ───────────────────────────────────────
        # from sentence_transformers import SentenceTransformer
        # from huggingface_hub import hf_hub_download
        # import joblib
        #
        # self.embedding_model = SentenceTransformer(
        #     settings.router_embedding_model, device=settings.device
        # )
        #
        # for head in ("macro", "intent", "context"):
        #     path = hf_hub_download(
        #         repo_id=settings.router_hf_repo_id,
        #         filename=f"{head}_clf.joblib",
        #         token=settings.router_hf_token,
        #         cache_dir="model_cache",
        #     )
        #     self.classifiers[head] = joblib.load(path)
        # ───────────────────────────────────────────────────────

        logger.warning("RouterBundle loaded in DUMMY mode – returning stub predictions")
        self._loaded = True


# Module-level singleton – populated via ``router_bundle.load()`` at startup.
router_bundle = RouterBundle()


# ── Routing prediction ────────────────────────────────────────


def _dummy_route(text: str) -> Dict[str, object]:
    """Return hard-coded routing predictions (placeholder)."""
    return {
        "macro": 1,
        "macro_confidence": 0.97,
        "intent": "value",
        "intent_confidence": 0.94,
        "context": "standalone",
        "context_confidence": 0.89,
    }


def route(bundle: RouterBundle, text: str) -> dict:
    """
    Compute routing predictions for a single text.

    Returns
    -------
    dict with keys:
        macro, macro_confidence,
        intent, intent_confidence,
        context, context_confidence
    """
    if bundle.embedding_model is None:
        # Dummy mode – no real models loaded yet
        return _dummy_route(text)

    # TODO: Replace with real inference (see INTEGRATION.md)
    # ─── Future code ───────────────────────────────────────
    # embedding = bundle.embedding_model.encode(text)
    # result: dict = {}
    # for head, clf in bundle.classifiers.items():
    #     probs = clf.predict_proba([embedding])[0]
    #     pred_idx = probs.argmax()
    #     result[head] = bundle.labels[head][pred_idx]
    #     result[f"{head}_confidence"] = round(float(probs[pred_idx]), 6)
    # return result
    # ───────────────────────────────────────────────────────

    return _dummy_route(text)  # pragma: no cover

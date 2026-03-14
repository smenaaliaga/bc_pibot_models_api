"""Router model: real multitask router loaded from Hugging Face artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
import json
import re
from typing import Any, Dict, List

try:
    import torch
except ImportError:  # pragma: no cover - import guard for lightweight test envs
    torch = None  # type: ignore[assignment]

try:
    from huggingface_hub import HfApi, snapshot_download
except ImportError:  # pragma: no cover - import guard for lightweight test envs
    snapshot_download = None  # type: ignore[assignment]
    HfApi = None  # type: ignore[assignment]

from app.config import settings

logger = logging.getLogger(__name__)

_CACHE_META_FILENAME = ".hf_cache_meta.json"

_TASKS: tuple[str, ...] = ("macro", "intent", "context")

ROUTER_LABELS: Dict[str, List[str | int]] = {
    "macro": [0, 1],
    "intent": ["methodology", "other", "value"],
    "context": ["followup", "standalone"],
}


def _coerce_macro_label(value: str | int) -> str | int:
    if isinstance(value, int):
        return value
    if isinstance(value, str) and value.isdigit():
        return int(value)
    return value


def _is_device_available(device: str) -> bool:
    if torch is None:
        return device == "cpu"
    if device == "cuda":
        return torch.cuda.is_available()
    if device == "mps":
        return hasattr(torch.backends, "mps") and torch.backends.mps.is_available()
    return True


def _resolve_device(requested_device: str) -> str:
    if torch is None:
        return "cpu"
    candidate = requested_device.lower()
    if candidate == "auto":
        if torch.cuda.is_available():
            return "cuda"
        if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
            return "mps"
        return "cpu"
    if not _is_device_available(candidate):
        logger.warning("Requested router device '%s' is unavailable. Falling back to cpu.", candidate)
        return "cpu"
    return candidate


def _load_json_file(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        return json.load(handle)


def _load_cache_meta(local_dir: Path) -> dict[str, Any]:
    meta_path = local_dir / _CACHE_META_FILENAME
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        logger.warning("Could not read router cache metadata at %s", meta_path, exc_info=True)
        return {}
    return data if isinstance(data, dict) else {}


def _save_cache_meta(local_dir: Path, *, repo_id: str, revision: str, commit: str | None) -> None:
    local_dir.mkdir(parents=True, exist_ok=True)
    meta_path = local_dir / _CACHE_META_FILENAME
    payload = {
        "repo_id": repo_id,
        "revision": revision,
        "commit": commit or "",
    }
    with meta_path.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, ensure_ascii=False, indent=2)


def _read_hf_cache_ref_commit(repo_id: str, revision: str) -> str | None:
    """Read commit from local HF refs cache if available."""
    repo_cache_dir = Path("model_cache") / f"models--{repo_id.replace('/', '--')}" / "refs"
    ref_path = repo_cache_dir / revision
    if not ref_path.exists():
        return None
    try:
        value = ref_path.read_text(encoding="utf-8").strip()
    except Exception:
        logger.warning("Could not read router HF ref commit from %s", ref_path, exc_info=True)
        return None
    return value or None


def _build_labels(id2label_raw: dict[str, Any]) -> Dict[str, List[str | int]]:
    labels: Dict[str, List[str | int]] = {}
    for task in _TASKS:
        task_map_raw = id2label_raw.get(task)
        if task_map_raw is None:
            continue

        task_labels: List[str | int]
        if isinstance(task_map_raw, dict):
            sorted_items = sorted(task_map_raw.items(), key=lambda item: int(item[0]))
            task_labels = [item[1] for item in sorted_items]
        elif isinstance(task_map_raw, list):
            task_labels = list(task_map_raw)
        else:
            continue

        if task == "macro":
            task_labels = [_coerce_macro_label(value) for value in task_labels]

        labels[task] = task_labels

    missing_tasks = set(_TASKS) - set(labels.keys())
    if missing_tasks:
        raise RuntimeError(f"Router id2label.json missing tasks: {sorted(missing_tasks)}")

    return labels


def _extract_head_parameters(raw_state: dict[str, Any]) -> dict[str, list[tuple[Any, Any]]]:
    state = raw_state.get("state_dict", raw_state)
    task_layers: dict[str, list[tuple[Any, Any]]] = {}

    if all(task in state and isinstance(state[task], dict) for task in _TASKS):
        for task in _TASKS:
            task_state = state[task]
            weight = task_state.get("weight")
            bias = task_state.get("bias")
            if (
                torch is not None
                and isinstance(weight, torch.Tensor)
                and isinstance(bias, torch.Tensor)
            ):
                task_layers[task] = [(weight, bias)]

    if set(task_layers.keys()) == set(_TASKS):
        return task_layers

    flat_state: dict[str, Any] = {}
    for key, value in state.items():
        if torch is not None and isinstance(value, torch.Tensor):
            flat_state[key] = value

    for task in _TASKS:
        pattern_w = re.compile(rf"^heads\.{re.escape(task)}\.classifier\.(\d+)\.weight$", re.IGNORECASE)
        pattern_b = re.compile(rf"^heads\.{re.escape(task)}\.classifier\.(\d+)\.bias$", re.IGNORECASE)

        by_idx: dict[int, dict[str, Any]] = {}
        for key, value in flat_state.items():
            match_w = pattern_w.match(key)
            if match_w is not None:
                idx = int(match_w.group(1))
                by_idx.setdefault(idx, {})["weight"] = value
                continue

            match_b = pattern_b.match(key)
            if match_b is not None:
                idx = int(match_b.group(1))
                by_idx.setdefault(idx, {})["bias"] = value

        if not by_idx:
            continue

        layers: list[tuple[Any, Any]] = []
        for idx in sorted(by_idx.keys()):
            weight = by_idx[idx].get("weight")
            bias = by_idx[idx].get("bias")
            if weight is None or bias is None:
                raise RuntimeError(f"Incomplete router classifier layer for task '{task}' at index {idx}")
            layers.append((weight, bias))

        task_layers[task] = layers

    if set(task_layers.keys()) != set(_TASKS):
        raise RuntimeError("Could not extract all multitask classifier layers from heads.pt")

    return task_layers


def _extract_top_k(
    probs: torch.Tensor,
    id2label_task: dict[int, str | int],
    top_k: int,
) -> list[dict[str, float | str | int]]:
    """Return ranked predictions for one task, matching training-time helper behavior."""
    k = min(max(top_k, 1), int(probs.shape[0]))
    top_probs, top_ids = torch.topk(probs, k=k, dim=-1)
    ranked: list[dict[str, float | str | int]] = []
    for prob, class_id in zip(top_probs.tolist(), top_ids.tolist()):
        class_index = int(class_id)
        if class_index not in id2label_task:
            raise RuntimeError(f"Router class index {class_index} missing in label map")
        label = id2label_task[class_index]
        ranked.append({"label": label, "score": float(prob)})
    return ranked


def _disable_encoder_adapters(encoder_dir: Path) -> None:
    """Disable adapter autoload by renaming adapter files if present.

    Some exported encoder artifacts may include adapter files even when runtime
    inference should run with the base encoder only. ``transformers`` auto-detects
    these files and attempts to inject adapter modules, which can fail on CPU-only Windows
    builds. Renaming them keeps the base encoder load path deterministic.
    """
    adapter_files = (
        "adapter_config.json",
        "adapter_model.safetensors",
        "adapter_model.bin",
    )
    for filename in adapter_files:
        path = encoder_dir / filename
        if not path.exists():
            continue

        disabled_path = encoder_dir / f"{filename}.disabled"
        if disabled_path.exists():
            continue

        path.rename(disabled_path)
        logger.warning("Disabled router adapter file for base-encoder inference: %s", path)


def _download_router_artifacts(*, revision: str, expected_commit: str | None) -> Path:
    if snapshot_download is None:
        raise RuntimeError("huggingface-hub is required to load router artifacts")

    local_dir = Path("model_cache") / "router_artifacts"
    local_dir.mkdir(parents=True, exist_ok=True)

    marker_files = [
        local_dir / "id2label.json",
        local_dir / "train_config.json",
        local_dir / "heads.pt",
        local_dir / "encoder",
    ]
    has_valid_cache = all(path.exists() for path in marker_files)
    if has_valid_cache:
        if not settings.hf_sync_on_startup:
            logger.info("Using cached router artifacts: %s", local_dir)
            return local_dir

        if expected_commit is None:
            logger.warning(
                "HF commit resolution failed for router %s@%s. Reusing cache at %s",
                settings.router_hf_repo_id,
                revision,
                local_dir,
            )
            return local_dir

        cached_commit = _load_cache_meta(local_dir).get("commit") or _read_hf_cache_ref_commit(
            settings.router_hf_repo_id,
            revision,
        )
        if cached_commit is None:
            logger.warning(
                "Cached router commit is unknown for %s at %s. Reusing cache without forced refresh.",
                settings.router_hf_repo_id,
                local_dir,
            )
            return local_dir
        if cached_commit == expected_commit:
            logger.info("Using router cache aligned with HF commit %s at %s", expected_commit, local_dir)
            return local_dir

        logger.info(
            "Refreshing router cache due to HF commit change: cached=%s remote=%s",
            cached_commit or "unknown",
            expected_commit,
        )

    snapshot_kwargs: dict[str, Any] = {
        "repo_id": settings.router_hf_repo_id,
        "revision": revision,
        "token": settings.router_hf_token,
        "cache_dir": "model_cache",
        "local_dir": str(local_dir),
        "force_download": bool(has_valid_cache and expected_commit is not None),
    }
    downloaded_dir = Path(snapshot_download(**snapshot_kwargs))
    if settings.hf_sync_on_startup:
        _save_cache_meta(
            downloaded_dir,
            repo_id=settings.router_hf_repo_id,
            revision=revision,
            commit=expected_commit,
        )
    return downloaded_dir


def _resolve_router_hf_commit(repo_id: str, revision: str, token: str | None) -> str | None:
    if HfApi is None:
        return None
    try:
        info = HfApi().model_info(repo_id=repo_id, revision=revision, token=token)
    except Exception:
        logger.warning("Could not resolve HF commit for router %s@%s", repo_id, revision, exc_info=True)
        return None
    return getattr(info, "sha", None)


class RouterBundle:
    """Encapsulates HF encoder + multitask heads used for routing decisions."""

    def __init__(self) -> None:
        self.embedding_model: Any | None = None
        self.head_layers: dict[str, list[tuple[Any, Any]]] = {}
        self.labels: Dict[str, List[str | int]] = ROUTER_LABELS
        self.max_length: int = 24
        self.device: str = "cpu"
        self.hf_repo_id: str | None = None
        self.hf_revision: str | None = None
        self.hf_commit: str | None = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Load router artifacts from Hugging Face Hub."""
        if torch is None:
            raise RuntimeError("torch is required to run router inference")

        try:
            from sentence_transformers import SentenceTransformer
        except ImportError as exc:  # pragma: no cover - dependency guard
            raise RuntimeError("sentence-transformers is required to load the router encoder") from exc

        self.hf_repo_id = settings.router_hf_repo_id
        self.hf_revision = "main"
        self.hf_commit = _resolve_router_hf_commit(
            repo_id=settings.router_hf_repo_id,
            revision=self.hf_revision,
            token=settings.router_hf_token,
        )

        self.device = _resolve_device(settings.device)
        artifact_dir = _download_router_artifacts(revision=self.hf_revision, expected_commit=self.hf_commit)

        id2label_path = artifact_dir / "id2label.json"
        train_config_path = artifact_dir / "train_config.json"
        heads_path = artifact_dir / "heads.pt"
        encoder_dir = artifact_dir / "encoder"

        if not id2label_path.exists() or not train_config_path.exists() or not heads_path.exists() or not encoder_dir.exists():
            raise FileNotFoundError(
                "Missing required router artifact(s): expected encoder/, heads.pt, id2label.json, train_config.json"
            )

        id2label_raw = _load_json_file(id2label_path)
        train_config_raw = _load_json_file(train_config_path)

        self.labels = _build_labels(id2label_raw)
        self.max_length = int(train_config_raw.get("max_length", 24))

        # Ensure encoder loads as plain base model (no adapter autoload).
        _disable_encoder_adapters(encoder_dir)

        # Load encoder (SentenceTransformer)
        self.embedding_model = SentenceTransformer(str(encoder_dir), device=self.device)

        # Load multitask heads
        raw_state = torch.load(heads_path, map_location=self.device, weights_only=True)
        if not isinstance(raw_state, dict):
            raise RuntimeError("Invalid heads.pt format: expected a dictionary state")

        raw_layers = _extract_head_parameters(raw_state)

        normalized_layers: dict[str, list[tuple[Any, Any]]] = {}
        for task in _TASKS:
            task_layers = raw_layers.get(task)
            if not task_layers:
                raise RuntimeError(f"Router heads missing task '{task}'")

            norm_task_layers: list[tuple[Any, Any]] = []
            for weight, bias in task_layers:
                weight_tensor = weight.to(self.device).float().detach()
                bias_tensor = bias.to(self.device).float().detach()
                norm_task_layers.append((weight_tensor, bias_tensor))

            normalized_layers[task] = norm_task_layers

        self.head_layers = normalized_layers
        self._loaded = True
        logger.info(
            "Router source: repo=%s revision=%s commit=%s",
            self.hf_repo_id,
            self.hf_revision,
            self.hf_commit or "unknown",
        )
        logger.info("RouterBundle loaded from HF repo '%s' on device '%s'", settings.router_hf_repo_id, self.device)


# Module-level singleton – populated via ``router_bundle.load()`` at startup.
router_bundle = RouterBundle()


# ── Routing prediction ────────────────────────────────────────


def route(bundle: RouterBundle, text: str) -> dict:
    """Compute routing predictions for a single text (endpoint-compatible format)."""
    detailed = route_detailed(bundle, text, top_k=1)

    output: dict[str, str | int | float] = {}
    for task in _TASKS:
        best = detailed[task]
        label = best["label"]
        if task == "macro":
            label = _coerce_macro_label(label)

        output[task] = label
        output[f"{task}_confidence"] = float(best["score"])

    return output


def route_detailed(bundle: RouterBundle, text: str, top_k: int = 3) -> dict[str, dict[str, object]]:
    """Compute routing predictions with top-k details per task.

    This mirrors the logic used in ``revision/predict.py`` so training-time and
    endpoint-time router inference share the same decision path.
    """
    if bundle.embedding_model is None or not bundle.head_layers:
        raise RuntimeError("Router model is not loaded")

    embedding = bundle.embedding_model.encode(
        [text],
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.to(bundle.device).float()

    results: dict[str, dict[str, object]] = {}
    for task in _TASKS:
        task_layers = bundle.head_layers.get(task)
        if not task_layers:
            raise RuntimeError(f"Router head layers missing task '{task}'")

        hidden = embedding
        for layer_index, (weight, bias) in enumerate(task_layers):
            hidden = torch.matmul(hidden, weight.transpose(0, 1)) + bias
            if layer_index < len(task_layers) - 1:
                hidden = torch.relu(hidden)

        logits = hidden
        probs = torch.softmax(logits.squeeze(0), dim=-1)

        labels = bundle.labels.get(task)
        if labels is None:
            raise RuntimeError(f"Router labels missing task '{task}'")
        if int(probs.shape[0]) != len(labels):
            raise RuntimeError(
                f"Router head/label mismatch for task '{task}': head_dim={int(probs.shape[0])} labels={len(labels)}"
            )
        id2label_task = {index: label for index, label in enumerate(labels)}
        ranked = _extract_top_k(probs=probs, id2label_task=id2label_task, top_k=top_k)
        if not ranked:
            raise RuntimeError(f"No predictions produced for router task '{task}'")

        best = ranked[0]
        results[task] = {
            "label": best["label"],
            "score": float(best["score"]),
            "top_k": ranked,
        }

    return results
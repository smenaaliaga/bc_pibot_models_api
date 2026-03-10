"""Router model: real multitask router loaded from Hugging Face artifacts."""

from __future__ import annotations

import logging
from pathlib import Path
import json
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

# ── Shim: peft ≥0.18 assumes torch.distributed.tensor.DTensor exists when
#    torch ≥ 2.5 and distributed.is_available(), but the submodule is absent on
#    Windows CPU-only builds.  Add a harmless stub so the isinstance() check in
#    peft/tuners/tuners_utils.py returns False instead of crashing.
if torch is not None and not hasattr(getattr(torch, "distributed", None), "tensor"):
    import types as _types
    _dt_mod = _types.ModuleType("torch.distributed.tensor")

    class _FakeDTensor:
        pass

    _dt_mod.DTensor = _FakeDTensor  # type: ignore[attr-defined]
    torch.distributed.tensor = _dt_mod  # type: ignore[attr-defined]

try:
    from peft import PeftModel, LoraConfig, get_peft_model
except ImportError:  # pragma: no cover - import guard for lightweight test envs
    PeftModel = None  # type: ignore[assignment]
    LoraConfig = None  # type: ignore[assignment]
    get_peft_model = None  # type: ignore[assignment]

from app.config import settings

logger = logging.getLogger(__name__)

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

    if set(labels.keys()) != set(_TASKS):
        return ROUTER_LABELS

    return labels


def _extract_head_parameters(raw_state: dict[str, Any]) -> tuple[dict[str, Any], dict[str, Any]]:
    state = raw_state.get("state_dict", raw_state)
    weights: dict[str, Any] = {}
    biases: dict[str, Any] = {}

    if all(task in state and isinstance(state[task], dict) for task in _TASKS):
        for task in _TASKS:
            task_state = state[task]
            weight = task_state.get("weight")
            bias = task_state.get("bias")
            if torch is not None and isinstance(weight, torch.Tensor):
                weights[task] = weight
            if torch is not None and isinstance(bias, torch.Tensor):
                biases[task] = bias

    if set(weights.keys()) == set(_TASKS):
        return weights, biases

    flat_state: dict[str, Any] = {}
    for key, value in state.items():
        if torch is not None and isinstance(value, torch.Tensor):
            flat_state[key] = value

    def find_key(task_name: str, suffix: str) -> str | None:
        task_name_lower = task_name.lower()
        candidates: list[str] = []
        for key in flat_state:
            key_lower = key.lower()
            if not key_lower.endswith(suffix):
                continue
            if (
                f".{task_name_lower}." in key_lower
                or f"_{task_name_lower}_" in key_lower
                or key_lower.startswith(f"{task_name_lower}.")
                or key_lower.startswith(f"heads.{task_name_lower}.")
                or f"{task_name_lower}{suffix}" in key_lower
            ):
                candidates.append(key)
        if not candidates:
            return None
        candidates.sort(key=len)
        return candidates[0]

    for task in _TASKS:
        weight_key = find_key(task, "weight")
        if weight_key is None:
            continue
        weights[task] = flat_state[weight_key]
        bias_key = find_key(task, "bias")
        if bias_key is not None:
            biases[task] = flat_state[bias_key]

    return weights, biases


def _download_router_artifacts() -> Path:
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
    if all(path.exists() for path in marker_files):
        logger.info("Using cached router artifacts: %s", local_dir)
        return local_dir

    snapshot_kwargs: dict[str, Any] = {
        "repo_id": settings.router_hf_repo_id,
        "token": settings.router_hf_token,
        "cache_dir": "model_cache",
        "local_dir": str(local_dir),
    }
    return Path(snapshot_download(**snapshot_kwargs))


def _resolve_router_hf_commit(repo_id: str, revision: str, token: str | None) -> str | None:
    if HfApi is None:
        return None
    try:
        info = HfApi().model_info(repo_id=repo_id, revision=revision, token=token)
    except Exception:
        logger.warning("Could not resolve HF commit for router %s@%s", repo_id, revision, exc_info=True)
        return None
    return getattr(info, "sha", None)


def _apply_lora_to_encoder(
    encoder: Any,
    train_config: dict[str, Any],
    artifact_dir: Path,
    device: str,
) -> Any:
    """Apply LoRA weights to encoder if use_lora=true in train_config.

    Loading strategy:
    1. Use ``PeftModel.from_pretrained()`` on the base transformer – this reads
       ``adapter_config.json`` written by peft during training so the LoRA
       topology (target_modules, r, alpha, …) is reproduced exactly.
    2. If ``from_pretrained`` fails (e.g. missing config), fall back to manually
       building the LoRA config from train_config.json / adapter_config.json and
       loading the state-dict.

    This mirrors the *save* side of ``SharedEncoder.save()`` in the training
    repo, which calls ``PeftModel.save_pretrained(path/lora_weights)`` followed
    by ``SentenceTransformer.save(path)``.
    """
    use_lora = train_config.get("use_lora", False)

    if not use_lora:
        logger.info("LoRA not enabled in train_config. Loading encoder without LoRA.")
        return encoder

    if PeftModel is None or LoraConfig is None or get_peft_model is None:
        logger.warning("peft library not available. Falling back to encoder without LoRA.")
        return encoder

    lora_weights_path = artifact_dir / "encoder" / "lora_weights"

    if not lora_weights_path.exists():
        logger.warning("use_lora=true but encoder/lora_weights directory not found. Loading without LoRA.")
        return encoder

    # Verify adapter weights exist
    has_bin = (lora_weights_path / "adapter_model.bin").exists()
    has_safetensors = (lora_weights_path / "adapter_model.safetensors").exists()
    if not has_bin and not has_safetensors:
        logger.warning("No adapter weight files found in %s. Loading without LoRA.", lora_weights_path)
        return encoder

    try:
        base_model = encoder[0].auto_model

        # --- Strategy 1: PeftModel.from_pretrained (preferred) ---
        # This reads adapter_config.json written by peft during training,
        # so target_modules / r / alpha / task_type are reproduced exactly
        # without us having to guess or duplicate them.
        try:
            peft_model = PeftModel.from_pretrained(
                base_model,
                str(lora_weights_path),
            )
            peft_model.eval()
            encoder[0].auto_model = peft_model
            logger.info(
                "LoRA applied via PeftModel.from_pretrained from %s",
                lora_weights_path,
            )
            return encoder
        except Exception as from_pretrained_err:
            logger.debug(
                "PeftModel.from_pretrained failed (%s), falling back to manual config.",
                from_pretrained_err,
            )

        # --- Strategy 2: manual LoraConfig + load state-dict ---
        # Read adapter_config.json as primary source; fall back to train_config.
        adapter_cfg: dict[str, Any] = {}
        adapter_cfg_path = lora_weights_path / "adapter_config.json"
        if adapter_cfg_path.exists():
            adapter_cfg = _load_json_file(adapter_cfg_path)

        lora_r = int(adapter_cfg.get("r", train_config.get("lora_r", 8)))
        lora_alpha = int(adapter_cfg.get("lora_alpha", train_config.get("lora_alpha", 16)))
        lora_dropout = float(adapter_cfg.get("lora_dropout", train_config.get("lora_dropout", 0.1)))
        # Training hardcodes ["query", "value"] with task_type=None, bias="none"
        target_modules = adapter_cfg.get(
            "target_modules",
            train_config.get("target_modules", ["query", "value"]),
        )
        task_type = adapter_cfg.get(
            "task_type",
            train_config.get("lora_task_type", None),
        )
        bias = adapter_cfg.get("bias", "none")

        logger.info(
            "Applying LoRA manually: r=%d, alpha=%d, dropout=%.2f, "
            "target_modules=%s, task_type=%s, bias=%s",
            lora_r, lora_alpha, lora_dropout, target_modules, task_type, bias,
        )

        lora_config = LoraConfig(
            r=lora_r,
            lora_alpha=lora_alpha,
            lora_dropout=lora_dropout,
            bias=bias,
            task_type=task_type,
            target_modules=target_modules,
        )

        peft_model = get_peft_model(base_model, lora_config)

        # Load weights
        weight_path = (
            lora_weights_path / "adapter_model.bin"
            if has_bin
            else lora_weights_path / "adapter_model.safetensors"
        )
        if has_bin:
            lora_state = torch.load(weight_path, map_location=device)
        else:
            from safetensors.torch import load_file
            lora_state = load_file(str(weight_path))

        peft_model.load_state_dict(lora_state, strict=False)
        peft_model.eval()

        encoder[0].auto_model = peft_model
        logger.info("LoRA applied via manual config + state_dict from %s", lora_weights_path)

    except Exception as exc:
        logger.warning(
            "Failed to apply LoRA weights: %s. Falling back to encoder without LoRA.",
            exc,
            exc_info=True,
        )

    return encoder


class RouterBundle:
    """Encapsulates HF encoder + multitask heads used for routing decisions."""

    def __init__(self) -> None:
        self.embedding_model: Any | None = None
        self.head_weights: dict[str, Any] = {}
        self.head_biases: dict[str, Any] = {}
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
        artifact_dir = _download_router_artifacts()

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

        # Load encoder (SentenceTransformer)
        self.embedding_model = SentenceTransformer(str(encoder_dir), device=self.device)
        
        # Apply LoRA if enabled in train_config
        self.embedding_model = _apply_lora_to_encoder(
            encoder=self.embedding_model,
            train_config=train_config_raw,
            artifact_dir=artifact_dir,
            device=self.device,
        )

        raw_state = torch.load(heads_path, map_location=self.device)
        if not isinstance(raw_state, dict):
            raise RuntimeError("Invalid heads.pt format: expected a dictionary state")

        weights, biases = _extract_head_parameters(raw_state)
        if set(weights.keys()) != set(_TASKS):
            raise RuntimeError("Could not extract all multitask head weights from heads.pt")

        head_weights: dict[str, Any] = {}
        head_biases: dict[str, Any] = {}
        for task in _TASKS:
            weight_tensor = weights[task].to(self.device).float().detach()
            bias_tensor = biases.get(task)
            if bias_tensor is None:
                bias_tensor = torch.zeros(weight_tensor.shape[0], device=self.device)
            else:
                bias_tensor = bias_tensor.to(self.device).float().detach()

            head_weights[task] = weight_tensor
            head_biases[task] = bias_tensor

        self.head_weights = head_weights
        self.head_biases = head_biases
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
    """Compute routing predictions for a single text."""
    if bundle.embedding_model is None or not bundle.head_weights:
        raise RuntimeError("Router model is not loaded")

    embedding = bundle.embedding_model.encode(
        [text],
        convert_to_tensor=True,
        normalize_embeddings=False,
    )
    if embedding.ndim == 1:
        embedding = embedding.unsqueeze(0)
    embedding = embedding.to(bundle.device).float()

    output: dict[str, str | int | float] = {}
    for task in _TASKS:
        weight = bundle.head_weights[task]
        bias = bundle.head_biases[task]
        logits = torch.matmul(embedding, weight.transpose(0, 1)) + bias
        probs = torch.softmax(logits.squeeze(0), dim=-1)
        pred_idx = int(torch.argmax(probs).item())

        labels = bundle.labels.get(task, ROUTER_LABELS[task])
        if pred_idx >= len(labels):
            raise RuntimeError(f"Predicted index out of bounds for router task '{task}'")

        label = labels[pred_idx]
        if task == "macro":
            label = _coerce_macro_label(label)

        output[task] = label
        output[f"{task}_confidence"] = round(float(probs[pred_idx].item()), 6)

    return output

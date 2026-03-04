"""
Model loader: resolves model artefacts from Hugging Face Hub or a local directory
and instantiates the JointBERT model + tokenizer + label maps.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Dict, List

import torch
from huggingface_hub import snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer

from app.config import ModelSource, settings

logger = logging.getLogger(__name__)


def _load_raw_state_dict(model_dir: Path) -> dict:
    """Load raw checkpoint state dict from safetensors or torch bin if available."""
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        return load_file(str(safetensors_path))

    torch_bin_path = model_dir / "pytorch_model.bin"
    if torch_bin_path.exists():
        state = torch.load(torch_bin_path, map_location="cpu")
        if isinstance(state, dict):
            return state

    return {}


def _restore_legacy_crf_weights(model: torch.nn.Module, model_dir: Path) -> None:
    """Map legacy CRF key names from checkpoint to current CRF parameter names."""
    if not hasattr(model, "crf"):
        return

    raw_state = _load_raw_state_dict(model_dir)
    if not raw_state:
        return

    current_state = model.state_dict()
    key_mapping = {
        "crf.start_trans": "crf.start_transitions",
        "crf.end_trans": "crf.end_transitions",
        "crf.trans_matrix": "crf.transitions",
    }

    patched_state: dict[str, torch.Tensor] = {}
    for old_key, new_key in key_mapping.items():
        if old_key not in raw_state or new_key not in current_state:
            continue

        old_tensor = raw_state[old_key]
        if not isinstance(old_tensor, torch.Tensor):
            continue

        if old_tensor.shape != current_state[new_key].shape:
            logger.warning(
                "Skipping legacy CRF weight remap %s -> %s due to shape mismatch (%s != %s)",
                old_key,
                new_key,
                tuple(old_tensor.shape),
                tuple(current_state[new_key].shape),
            )
            continue

        patched_state[new_key] = old_tensor

    if not patched_state:
        return

    model.load_state_dict(patched_state, strict=False)
    logger.info("Restored legacy CRF weights: %s", ", ".join(sorted(patched_state.keys())))

# ── Label helpers ────────────────────────────────────────────


def _read_label_file(path: Path) -> List[str]:
    """Read a *_label.txt file and return the list of labels."""
    return [line.strip() for line in path.read_text(encoding="utf-8").splitlines() if line.strip()]


def load_labels(model_dir: Path) -> Dict[str, List[str]]:
    """
    Load all *_label.txt files from ``model_dir/labels/``.

    Returns a dict like ``{"calc_mode": [...], "slot": [...], ...}``
    """
    labels_dir = model_dir / "labels"
    if not labels_dir.exists():
        raise FileNotFoundError(f"Labels directory not found: {labels_dir}")

    labels: Dict[str, List[str]] = {}
    for f in sorted(labels_dir.glob("*_label.txt")):
        key = f.stem.replace("_label", "")  # e.g. calc_mode_label.txt -> calc_mode
        labels[key] = _read_label_file(f)
        logger.info("Loaded label file %s (%d classes)", f.name, len(labels[key]))

    required = {"calc_mode", "activity", "region", "investment", "req_form", "slot"}
    missing = required - set(labels.keys())
    if missing:
        raise FileNotFoundError(f"Missing required label files: {missing}")

    return labels


# ── Device detection ─────────────────────────────────────────


def resolve_device(requested: str = "auto") -> str:
    if requested != "auto":
        return requested
    if torch.cuda.is_available():
        return "cuda"
    if hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
        return "mps"
    return "cpu"


# ── Model loading ────────────────────────────────────────────


def _resolve_model_dir() -> Path:
    """Obtain the local path to model artefacts (download from HF if needed)."""
    if settings.model_source == ModelSource.local:
        if not settings.model_local_dir:
            raise ValueError("MODEL_LOCAL_DIR must be set when MODEL_SOURCE=local")
        path = Path(settings.model_local_dir)
        if not path.exists():
            raise FileNotFoundError(f"Local model directory not found: {path}")
        return path

    # Hugging Face Hub
    repo_dirname = settings.hf_repo_id.replace("/", "--")
    local_model_dir = Path("model_cache") / "snapshots" / repo_dirname

    marker_files = [
        local_model_dir / "config.json",
        local_model_dir / "training_args.bin",
        local_model_dir / "modeling_jointbert.py",
        local_model_dir / "module.py",
        local_model_dir / "labels",
    ]
    if all(path.exists() for path in marker_files):
        logger.info("Using cached local model directory: %s", local_model_dir)
        return local_model_dir

    logger.info("Downloading model from HF: %s", settings.hf_repo_id)
    local_dir = snapshot_download(
        repo_id=settings.hf_repo_id,
        token=settings.hf_token,
        local_dir=str(local_model_dir),
        # cache inside working dir so Docker layer caching works
        cache_dir="model_cache",
    )
    logger.info("Model downloaded to %s", local_dir)
    return Path(local_dir)


class ModelBundle:
    """Encapsulates model + tokenizer + labels + device ready for inference."""

    def __init__(self) -> None:
        self.model_dir: Path | None = None
        self.device: str = "cpu"
        self.tokenizer = None
        self.model = None
        self.labels: Dict[str, List[str]] = {}
        self.train_args = None
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Heavy init – call once at startup, NOT at import time."""
        self.model_dir = _resolve_model_dir()
        self.device = resolve_device(settings.device)

        # Labels
        self.labels = load_labels(self.model_dir)

        # Training args (needed to know use_crf, dropout_rate, etc.)
        args_path = self.model_dir / "training_args.bin"
        if args_path.exists():
            self.train_args = torch.load(args_path, weights_only=False, map_location="cpu")
            logger.info("Loaded training_args.bin (use_crf=%s)", getattr(self.train_args, "use_crf", False))
        else:
            raise FileNotFoundError(f"training_args.bin not found in {self.model_dir}")

        # Tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(
            str(self.model_dir),
            trust_remote_code=True,
        )
        logger.info("Tokenizer loaded from %s", self.model_dir)

        # Model – import the custom JointBERT class from the snapshot
        # instead of relying on AutoModel (which needs auto_map in config.json).
        import importlib.util
        spec = importlib.util.spec_from_file_location(
            "modeling_jointbert",
            str(self.model_dir / "modeling_jointbert.py"),
            submodule_search_locations=[str(self.model_dir)],
        )
        mod = importlib.util.module_from_spec(spec)
        # Ensure relative imports within the snapshot work (e.g. .module)
        import sys
        mod.__package__ = "modeling_jointbert_pkg"
        sys.modules[mod.__package__] = mod
        # Also load the companion module.py for the classifiers
        dep_spec = importlib.util.spec_from_file_location(
            f"{mod.__package__}.module",
            str(self.model_dir / "module.py"),
        )
        dep_mod = importlib.util.module_from_spec(dep_spec)
        sys.modules[f"{mod.__package__}.module"] = dep_mod
        dep_spec.loader.exec_module(dep_mod)
        spec.loader.exec_module(mod)

        JointBERT = mod.JointBERT
        config = AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=True)
        self.model = JointBERT.from_pretrained(
            str(self.model_dir),
            config=config,
            args=self.train_args,
            calc_mode_label_lst=self.labels["calc_mode"],
            activity_label_lst=self.labels["activity"],
            region_label_lst=self.labels["region"],
            investment_label_lst=self.labels["investment"],
            req_form_label_lst=self.labels["req_form"],
            slot_label_lst=self.labels["slot"],
        )
        _restore_legacy_crf_weights(self.model, self.model_dir)
        self.model.to(self.device)
        self.model.eval()
        self._loaded = True

        logger.info(
            "Model loaded on %s (%s parameters)",
            self.device,
            f"{sum(p.numel() for p in self.model.parameters()):,}",
        )


# Module-level singleton – populated via ``bundle.load()`` at startup.
bundle = ModelBundle()

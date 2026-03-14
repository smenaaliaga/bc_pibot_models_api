"""
Model loader: resolves model artefacts from Hugging Face Hub or a local directory
and instantiates the JointBERT model + tokenizer + label maps.
"""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Dict, List

import torch
from huggingface_hub import HfApi, snapshot_download
from transformers import AutoConfig, AutoModel, AutoTokenizer
from transformers.utils import logging as hf_logging

from app.config import ModelSource, settings

logger = logging.getLogger(__name__)

_CACHE_META_FILENAME = ".hf_cache_meta.json"


def _load_cache_meta(local_dir: Path) -> dict[str, str]:
    meta_path = local_dir / _CACHE_META_FILENAME
    if not meta_path.exists():
        return {}
    try:
        with meta_path.open("r", encoding="utf-8") as handle:
            data = json.load(handle)
    except Exception:
        logger.warning("Could not read cache metadata at %s", meta_path, exc_info=True)
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
        logger.warning("Could not read HF ref commit from %s", ref_path, exc_info=True)
        return None
    return value or None


def _resolve_hf_commit(repo_id: str, revision: str, token: str | None) -> str | None:
    """Resolve exact commit SHA for a Hugging Face repo revision."""
    try:
        info = HfApi().model_info(repo_id=repo_id, revision=revision, token=token)
    except Exception:
        logger.warning("Could not resolve HF commit for %s@%s", repo_id, revision, exc_info=True)
        return None
    return getattr(info, "sha", None)



_CRF_KEY_MAP = {
    "crf.start_trans": "crf.start_transitions",
    "crf.end_trans": "crf.end_transitions",
    "crf.trans_matrix": "crf.transitions",
}


def _normalize_checkpoint_crf_keys(model_dir: Path) -> None:
    """Rename legacy CRF keys in the checkpoint file so ``from_pretrained`` finds them.

    This is idempotent: if the keys are already renamed the function is a no-op.
    """
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file, save_file

        state = load_file(str(safetensors_path))
        if not any(k in state for k in _CRF_KEY_MAP):
            return  # already normalised or no CRF keys
        # Clone tensors so the memory-mapped file can be released before we
        # overwrite it (Windows locks the file while it is mapped).
        new_state = {_CRF_KEY_MAP.get(k, k): v.clone() for k, v in state.items()}
        del state
        save_file(new_state, str(safetensors_path))
        logger.info("Renamed legacy CRF keys in %s", safetensors_path.name)
        return

    torch_bin_path = model_dir / "pytorch_model.bin"
    if torch_bin_path.exists():
        state = torch.load(torch_bin_path, map_location="cpu")
        if isinstance(state, dict) and any(k in state for k in _CRF_KEY_MAP):
            new_state = {_CRF_KEY_MAP.get(k, k): v for k, v in state.items()}
            torch.save(new_state, torch_bin_path)
            logger.info("Renamed legacy CRF keys in %s", torch_bin_path.name)


def _load_checkpoint_state_dict(model_dir: Path) -> Dict[str, torch.Tensor]:
    """Load model weights from local snapshot checkpoint files."""
    safetensors_path = model_dir / "model.safetensors"
    if safetensors_path.exists():
        from safetensors.torch import load_file

        state = load_file(str(safetensors_path))
        return dict(state)

    torch_bin_path = model_dir / "pytorch_model.bin"
    if torch_bin_path.exists():
        state = torch.load(torch_bin_path, map_location="cpu")
        if isinstance(state, dict) and "state_dict" in state and isinstance(state["state_dict"], dict):
            state = state["state_dict"]
        if not isinstance(state, dict):
            raise RuntimeError(f"Invalid checkpoint format in {torch_bin_path}")
        return state

    raise FileNotFoundError(f"No checkpoint file found in {model_dir} (expected model.safetensors or pytorch_model.bin)")

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
    revision = "main"
    repo_dirname = settings.hf_repo_id.replace("/", "--")
    local_model_dir = Path("model_cache") / "snapshots" / repo_dirname

    marker_files = [
        local_model_dir / "config.json",
        local_model_dir / "training_args.bin",
        local_model_dir / "modeling_jointbert.py",
        local_model_dir / "module.py",
        local_model_dir / "labels",
    ]
    # At least one weights file must be present to consider the cache valid.
    weights_files = [
        local_model_dir / "model.safetensors",
        local_model_dir / "pytorch_model.bin",
    ]
    has_weights = any(f.exists() for f in weights_files)
    has_valid_cache = has_weights and all(path.exists() for path in marker_files)

    expected_commit: str | None = None
    if settings.hf_sync_on_startup:
        expected_commit = _resolve_hf_commit(
            repo_id=settings.hf_repo_id,
            revision=revision,
            token=settings.hf_token,
        )

    if has_valid_cache:
        if not settings.hf_sync_on_startup:
            logger.info("Using cached local model directory: %s", local_model_dir)
            return local_model_dir

        if expected_commit is None:
            logger.warning(
                "HF commit resolution failed for %s@%s. Reusing existing cache at %s",
                settings.hf_repo_id,
                revision,
                local_model_dir,
            )
            return local_model_dir

        cached_commit = _load_cache_meta(local_model_dir).get("commit") or _read_hf_cache_ref_commit(
            settings.hf_repo_id,
            revision,
        )
        if cached_commit is None:
            logger.warning(
                "Cached commit is unknown for %s at %s. Reusing cache without forced refresh.",
                settings.hf_repo_id,
                local_model_dir,
            )
            return local_model_dir
        if cached_commit == expected_commit:
            logger.info("Using cache aligned with HF commit %s at %s", expected_commit, local_model_dir)
            return local_model_dir

        logger.info(
            "Refreshing model cache due to HF commit change: cached=%s remote=%s",
            cached_commit or "unknown",
            expected_commit,
        )

    logger.info("Downloading model from HF: %s", settings.hf_repo_id)
    local_dir = snapshot_download(
        repo_id=settings.hf_repo_id,
        revision=revision,
        token=settings.hf_token,
        local_dir=str(local_model_dir),
        # cache inside working dir so Docker layer caching works
        cache_dir="model_cache",
        force_download=bool(has_valid_cache and expected_commit is not None),
    )
    if settings.hf_sync_on_startup:
        _save_cache_meta(
            Path(local_dir),
            repo_id=settings.hf_repo_id,
            revision=revision,
            commit=expected_commit,
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
        self.hf_repo_id: str | None = None
        self.hf_revision: str | None = None
        self.hf_commit: str | None = None
        self.model_source: str = settings.model_source.value
        self._loaded = False

    @property
    def is_loaded(self) -> bool:
        return self._loaded

    def load(self) -> None:
        """Heavy init – call once at startup, NOT at import time."""
        self.model_dir = _resolve_model_dir()
        self.device = resolve_device(settings.device)
        self.model_source = settings.model_source.value

        if settings.model_source == ModelSource.huggingface:
            self.hf_repo_id = settings.hf_repo_id
            self.hf_revision = "main"
            self.hf_commit = _resolve_hf_commit(
                repo_id=settings.hf_repo_id,
                revision=self.hf_revision,
                token=settings.hf_token,
            )
            logger.info(
                "JointBERT source: repo=%s revision=%s commit=%s",
                self.hf_repo_id,
                self.hf_revision,
                self.hf_commit or "unknown",
            )
        else:
            self.hf_repo_id = None
            self.hf_revision = None
            self.hf_commit = None

        # Labels
        self.labels = load_labels(self.model_dir)

        # Training args (needed to know use_crf, dropout_rate, etc.)
        args_path = self.model_dir / "training_args.bin"
        if args_path.exists():
            self.train_args = torch.load(args_path, weights_only=False, map_location="cpu")
            logger.info("Loaded training_args.bin (use_crf=%s)", getattr(self.train_args, "use_crf", False))
        else:
            raise FileNotFoundError(f"training_args.bin not found in {self.model_dir}")

        # Tokenizer: the predictor needs word_ids() which requires a fast
        # tokenizer.  Try local dir first to avoid a network round-trip, and
        # suppress the sentencepiece byte-fallback warning (cosmetic only).
        tokenizer_remote = getattr(self.train_args, "model_name_or_path", None) or str(self.model_dir)
        tokenizer_local = str(self.model_dir)

        candidates: list[tuple[str, bool]] = []
        for source in (tokenizer_local, tokenizer_remote):
            candidates.append((source, True))
            candidates.append((source, False))

        seen: set[tuple[str, bool]] = set()
        errors: list[str] = []
        selected = False
        for source, use_fast in candidates:
            key = (source, use_fast)
            if key in seen:
                continue
            seen.add(key)

            try:
                import warnings
                with warnings.catch_warnings():
                    warnings.filterwarnings(
                        "ignore",
                        message=".*byte fallback.*",
                        category=UserWarning,
                    )
                    self.tokenizer = AutoTokenizer.from_pretrained(
                        source,
                        trust_remote_code=True,
                        use_fast=use_fast,
                    )
                logger.info("Tokenizer loaded from %s (use_fast=%s)", source, use_fast)
                # Predictor requires fast tokenizer for word_ids() alignment.
                if use_fast and bool(getattr(self.tokenizer, "is_fast", False)):
                    selected = True
                    break
                if use_fast:
                    logger.warning("Tokenizer loaded but is not fast; trying fallback.")
            except Exception as exc:
                errors.append(f"{source} (use_fast={use_fast}): {exc}")
                logger.warning("Failed loading tokenizer from %s (use_fast=%s): %s", source, use_fast, exc)

        if not selected:
            raise RuntimeError("Unable to load tokenizer. Attempts: " + " | ".join(errors))

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
        package_name = "modeling_jointbert_pkg"
        mod.__package__ = package_name
        # transformers 5.x may inspect sys.modules[cls.__module__] during model
        # init; register both canonical and package aliases before execution.
        sys.modules["modeling_jointbert"] = mod
        sys.modules[package_name] = mod
        # Also load the companion module.py for the classifiers
        dep_spec = importlib.util.spec_from_file_location(
            f"{package_name}.module",
            str(self.model_dir / "module.py"),
        )
        dep_mod = importlib.util.module_from_spec(dep_spec)
        sys.modules[f"{package_name}.module"] = dep_mod
        dep_spec.loader.exec_module(dep_mod)
        spec.loader.exec_module(mod)

        JointBERT = mod.JointBERT

        # Rename legacy CRF keys *in the checkpoint file* before from_pretrained
        # so that all parameter names match and no warnings are emitted.
        _normalize_checkpoint_crf_keys(self.model_dir)

        config = AutoConfig.from_pretrained(str(self.model_dir), trust_remote_code=True)
        previous_hf_verbosity = hf_logging.get_verbosity()
        hf_logging.set_verbosity_error()
        try:
            self.model = JointBERT(
                config,
                args=self.train_args,
                calc_mode_label_lst=self.labels["calc_mode"],
                activity_label_lst=self.labels["activity"],
                region_label_lst=self.labels["region"],
                investment_label_lst=self.labels["investment"],
                req_form_label_lst=self.labels["req_form"],
                slot_label_lst=self.labels["slot"],
            )
        finally:
            hf_logging.set_verbosity(previous_hf_verbosity)

        state_dict = _load_checkpoint_state_dict(self.model_dir)
        missing_keys, unexpected_keys = self.model.load_state_dict(state_dict, strict=False)
        if missing_keys or unexpected_keys:
            logger.warning(
                "Checkpoint loaded with key mismatch (missing=%d, unexpected=%d)",
                len(missing_keys),
                len(unexpected_keys),
            )

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

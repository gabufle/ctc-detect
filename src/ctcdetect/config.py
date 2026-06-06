"""Configuration and paths for CTC-Detect."""

import os
import sys
from pathlib import Path

# Package version — keep in sync with pyproject.toml
__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
# Maps a short alias to a HuggingFace repo id and local sub-directory name.
# When adding a new model version, extend this dict and add the corresponding
# entry in MODEL_CATALOG below.

MODEL_REGISTRY = {
    "latest": {
        "repo": "ctheodoris/Geneformer-V1-10M",
        "local_dir": "Geneformer-V1-10M",
        "description": "Fine-tuned Geneformer for CTC detection (10M params, LoRA)",
    },
    "v1.0": {
        "repo": "ctheodoris/Geneformer-V1-10M",
        "local_dir": "Geneformer-V1-10M",
        "description": "Geneformer-V1-10M fine-tuned for CTC detection",
    },
    "geneformer-base": {
        "repo": "ctheodoris/Geneformer",
        "local_dir": "Geneformer-V1-10M",
        "description": "Base Geneformer model (not fine-tuned for CTC)",
    },
}

# Cancer-type specific model overrides (future use)
CANCER_MODELS: dict[str, str] = {
    # e.g. "breast": "organization/geneformer-ctc-breast-v1",
}

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Default cache directory for downloaded models
MODEL_CACHE_DIR = Path.home() / ".cache" / "ctc-detect" / "models"

# Default project-level Geneformer directory (for development / manual clones)
GENEFORMER_DIR = PROJECT_ROOT / "Geneformer" / "geneformer"

# Checkpoint directories
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "best_model"
FINETUNED_DIR = PROJECT_ROOT / "Geneformer" / "Geneformer-V1-10M"

# Geneformer data files
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"


def get_model_cache_path(version: str = "latest") -> Path:
    """Return the local cache path for a given model version.

    Creates the parent directory if it does not exist.
    """
    if version not in MODEL_REGISTRY:
        version = "latest"
    local_name = MODEL_REGISTRY[version]["local_dir"]
    path = MODEL_CACHE_DIR / local_name
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_system_info() -> dict:
    """Return a dict of system information for the ``info`` command."""
    import platform
    import torch

    info = {
        "version": __version__,
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": torch.__version__,
        "cuda_available": torch.cuda.is_available(),
    }
    if torch.cuda.is_available():
        info["cuda_version"] = torch.version.cuda or "unknown"
        info["gpu"] = torch.cuda.get_device_name(0)
    else:
        info["cuda_version"] = None
        info["gpu"] = None

    # Check for locally available models
    available_models = []
    for alias, meta in MODEL_REGISTRY.items():
        cache_path = MODEL_CACHE_DIR / meta["local_dir"]
        if cache_path.exists() and any(cache_path.iterdir()):
            available_models.append(alias)
    info["cached_models"] = available_models

    # Check for Geneformer installation
    info["geneformer_installed"] = GENEFORMER_DIR.exists()
    info["geneformer_path"] = str(GENEFORMER_DIR)

    # Check for fine-tuned checkpoint
    has_checkpoint = (
        CHECKPOINT_DIR.exists()
        and (
            (CHECKPOINT_DIR / "pytorch_model.bin").exists()
            or (CHECKPOINT_DIR / "model.safetensors").exists()
            or (CHECKPOINT_DIR / "adapter_model.bin").exists()
            or (CHECKPOINT_DIR / "adapter_model.safetensors").exists()
        )
    )
    info["checkpoint_available"] = has_checkpoint
    info["checkpoint_path"] = str(CHECKPOINT_DIR)

    return info

"""Model registry and version mapping for CTC-Detect."""

from pathlib import Path

# Model registry with metadata
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

# Version alias to HuggingFace repo mapping
VERSION_MAP = {
    "latest": "ctheodoris/Geneformer-V1-10M",
    "v1.0": "ctheodoris/Geneformer-V1-10M",
}

DEFAULT_MODEL = "ctheodoris/Geneformer-V1-10M"

__all__ = [
    "MODEL_REGISTRY",
    "VERSION_MAP",
    "DEFAULT_MODEL",
    "get_version",
    "get_model_cache_path",
]


def get_version(version_str: str) -> str:
    """Resolve a version string to a HuggingFace repo ID.

    Args:
        version_str: Version alias (e.g. 'latest', 'v1.0').

    Returns:
        The HuggingFace repo ID for the requested version.

    Raises:
        ValueError: If the version string is not recognized.
    """
    if version_str in VERSION_MAP:
        return VERSION_MAP[version_str]
    raise ValueError(
        f"Unknown model version '{version_str}'. "
        f"Available versions: {', '.join(sorted(VERSION_MAP))}"
    )


def get_model_cache_path(version: str = "latest") -> Path:
    """Return the local cache path for a given model version.

    Creates the parent directory if it does not exist.
    The cache is organized by version alias, e.g. ``~/.cache/ctc-detect/models/latest/``.
    """
    from ctcdetect.config.paths import MODEL_CACHE_DIR

    if version not in MODEL_REGISTRY:
        version = "latest"
    path = MODEL_CACHE_DIR / version
    path.mkdir(parents=True, exist_ok=True)
    return path

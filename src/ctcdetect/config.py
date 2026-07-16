"""Unified configuration for CTC-Detect.

Combines model registry, paths, and YAML config loading.
"""

import sys
from pathlib import Path
from typing import Any, Optional, Union
import yaml

from ctcdetect.exceptions import ConfigurationError


# ---------------------------------------------------------------------------
# Model registry
# ---------------------------------------------------------------------------
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

VERSION_MAP = {
    "latest": "ctheodoris/Geneformer-V1-10M",
    "v1.0": "ctheodoris/Geneformer-V1-10M",
}

DEFAULT_MODEL = "ctheodoris/Geneformer-V1-10M"

__version__ = "0.1.0"

# ---------------------------------------------------------------------------
# Local paths
# ---------------------------------------------------------------------------
PROJECT_ROOT = Path(__file__).resolve().parents[2]

MODEL_CACHE_DIR = Path.home() / ".cache" / "ctc-detect" / "models"

GENEFORMER_DIR = PROJECT_ROOT / "Geneformer" / "geneformer"

CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "best_model"
FINETUNED_DIR = PROJECT_ROOT / "Geneformer" / "Geneformer-V1-10M"

TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"


# ---------------------------------------------------------------------------
# Config loading
# ---------------------------------------------------------------------------
class Config:
    """Configuration container with attribute-style access."""

    def __init__(self, data: dict):
        self._data = data

    def __getattr__(self, name: str) -> Any:
        if name in self._data:
            value = self._data[name]
            if isinstance(value, dict):
                return Config(value)
            return value
        raise AttributeError(f"Config has no key: {name}")

    def get(self, key: str, default: Any = None) -> Any:
        return self._data.get(key, default)

    def as_dict(self) -> dict:
        return self._data.copy()


def load_config(config_path: Optional[Union[str, Path]] = None) -> Config:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default configs/preprocess.yaml.

    Returns:
        Config object with attribute-style access.

    Raises:
        ConfigurationError: If config file is missing or invalid.
    """
    if config_path is None:
        # Default to package configs/preprocess.yaml (project root / configs)
        package_root = Path(__file__).resolve().parents[2]
        config_path = package_root / "configs" / "preprocess.yaml"

    if config_path is not None:
        config_path = Path(config_path)

    if not config_path.exists():
        raise ConfigurationError(
            f"Configuration file not found: {config_path}",
            hint="Ensure the config file exists or pass a valid path to load_config().",
            details={"config_path": str(config_path)},
        )

    try:
        with open(config_path) as f:
            data = yaml.safe_load(f)
    except yaml.YAMLError as e:
        raise ConfigurationError(
            f"Failed to parse YAML config: {config_path}",
            hint="Check YAML syntax. Common issues: tabs instead of spaces, missing quotes.",
            details={"config_path": str(config_path), "parse_error": str(e)},
        )

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Config file must contain a YAML mapping (dict), got {type(data).__name__}",
            details={"config_path": str(config_path)},
        )

    return Config(data)


# Global config instance (loaded on first access)
_config: Optional[Config] = None


def get_config(config_path: Optional[Union[str, Path]] = None, reload: bool = False) -> Config:
    """Get the global configuration instance.

    Args:
        config_path: Optional path to config file (only used on first load or if reload=True).
        reload: Force reloading the config.

    Returns:
        Config object.
    """
    global _config
    if _config is None or reload:
        if config_path is not None:
            config_path = Path(config_path)
        _config = load_config(config_path)
    return _config


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
    if version not in MODEL_REGISTRY:
        version = "latest"
    path = MODEL_CACHE_DIR / version
    path.mkdir(parents=True, exist_ok=True)
    return path


def get_system_info() -> dict:
    """Return a dict of system information for the ``info`` command."""
    import platform

    info = {
        "version": "0.1.0",
        "python": sys.version.split()[0],
        "platform": platform.platform(),
        "pytorch": "not installed",
        "cuda_available": False,
        "cuda_version": None,
        "gpu": None,
    }

    try:
        from importlib.metadata import version as _pkg_version
        info["version"] = _pkg_version("ctc-detect")
    except Exception:
        pass

    try:
        import torch
        info["pytorch"] = torch.__version__
        info["cuda_available"] = torch.cuda.is_available()
        if torch.cuda.is_available():
            info["cuda_version"] = torch.version.cuda or "unknown"
            info["gpu"] = torch.cuda.get_device_name(0)
    except ImportError:
        pass

    available_models = []
    for alias, meta in MODEL_REGISTRY.items():
        cache_path = MODEL_CACHE_DIR / alias
        if cache_path.exists() and any(cache_path.iterdir()):
            available_models.append(alias)
    info["cached_models"] = available_models

    info["geneformer_installed"] = GENEFORMER_DIR.exists()
    info["geneformer_path"] = str(GENEFORMER_DIR)

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

    try:
        import shutil
        disk = shutil.disk_usage(str(MODEL_CACHE_DIR))
        info["disk_total_gb"] = round(disk.total / (1024 ** 3), 1)
        info["disk_used_gb"] = round(disk.used / (1024 ** 3), 1)
        info["disk_free_gb"] = round(disk.free / (1024 ** 3), 1)
    except OSError:
        info["disk_total_gb"] = None
        info["disk_used_gb"] = None
        info["disk_free_gb"] = None

    return info


# Backwards-compat: allow `from ctcdetect.config import get_version, get_model_cache_path, get_system_info`
__all__ = [
    "MODEL_REGISTRY",
    "VERSION_MAP",
    "DEFAULT_MODEL",
    "PROJECT_ROOT",
    "MODEL_CACHE_DIR",
    "GENEFORMER_DIR",
    "CHECKPOINT_DIR",
    "FINETUNED_DIR",
    "TOKEN_DICT",
    "GENE_MEDIAN",
    "GENE_MAPPING",
    "Config",
    "load_config",
    "get_config",
    "get_version",
    "get_model_cache_path",
    "get_system_info",
    "__version__",
    "get_config_value",
]


def get_config_value(key: str, default=None):
    """Get a nested config value using dot notation (e.g., 'qc.min_genes')."""
    cfg = get_config()
    parts = key.split(".")
    value: Any = cfg.as_dict()
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value
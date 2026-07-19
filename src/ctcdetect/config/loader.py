"""Configuration loading for CTC-Detect."""

from pathlib import Path
from typing import Any

import yaml

from ctcdetect.config.schemas import PreprocessConfig
from ctcdetect.exceptions import ConfigurationError


def load_config(config_path: str | Path | None = None) -> PreprocessConfig:
    """Load configuration from YAML file.

    Args:
        config_path: Path to config file. If None, uses default configs/preprocess.yaml.

    Returns:
        PreprocessConfig object with validation.

    Raises:
        ConfigurationError: If config file is missing or invalid.
    """
    if config_path is None:
        # Default to package configs/preprocess.yaml (project root / configs)
        package_root = Path(__file__).resolve().parents[3]
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
        ) from e

    if not isinstance(data, dict):
        raise ConfigurationError(
            f"Config file must contain a YAML mapping (dict), got {type(data).__name__}",
            details={"config_path": str(config_path)},
        )

    try:
        return PreprocessConfig(**data)
    except Exception as e:
        raise ConfigurationError(
            f"Config validation failed for {config_path}",
            hint="Check that all required fields are present and have valid values.",
            details={"config_path": str(config_path), "validation_error": str(e)},
        ) from e


# Global config instance (loaded on first access)
_config: PreprocessConfig | None = None


def get_config(config_path: str | Path | None = None, reload: bool = False) -> PreprocessConfig:
    """Get the global configuration instance.

    Args:
        config_path: Optional path to config file (only used on first load or if reload=True).
        reload: Force reloading the config.

    Returns:
        PreprocessConfig object.
    """
    global _config
    if _config is None or reload:
        if config_path is not None:
            config_path = Path(config_path)
        _config = load_config(config_path)
    return _config


def get_config_value(key: str, default=None):
    """Get a nested config value using dot notation (e.g., 'qc.min_genes')."""
    cfg = get_config()
    parts = key.split(".")
    value: Any = cfg.model_dump()
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value

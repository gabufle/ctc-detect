"""CTC-Detect: Geneformer-based circulating tumor cell detection from scRNA-seq data."""

from ctcdetect.exceptions import (
    CTCDetectError,
    ConfigurationError,
    DependencyError,
    GeneMappingError,
    InferenceError,
    InputError,
    ValidationError,
    InputValidationError,
    OutputError,
    TokenizationError,
    handle_error,
)

from ctcdetect.config import load_config, get_config

# Submodules are not imported at top level to avoid heavy dependencies (torch, transformers, etc.)
# Users should import them explicitly: from ctcdetect import detect, preprocess, etc.

__all__ = [
    "CTCDetectError",
    "ConfigurationError",
    "DependencyError",
    "GeneMappingError",
    "InferenceError",
    "InputError",
    "ValidationError",
    "InputValidationError",
    "OutputError",
    "TokenizationError",
    "handle_error",
    "load_config",
    "get_config",
]

__version__ = "0.1.0"

"""Configuration package for CTC-Detect.

Exports:
- Model registry (MODEL_REGISTRY, VERSION_MAP, DEFAULT_MODEL, get_version)
- Path constants (PROJECT_ROOT, MODEL_CACHE_DIR, etc.)
- Config loading (Config, load_config, get_config, get_config_value)
- Pydantic schemas (PreprocessConfig, QCConfig, etc.)
- System info (get_system_info)
"""

from ctcdetect.config.loader import get_config, get_config_value, load_config
from ctcdetect.config.paths import (
    CHECKPOINT_DIR,
    FINETUNED_DIR,
    GENE_MAPPING,
    GENE_MEDIAN,
    GENEFORMER_DIR,
    MODEL_CACHE_DIR,
    PROJECT_ROOT,
    TOKEN_DICT,
)
from ctcdetect.config.registry import (
    DEFAULT_MODEL,
    MODEL_REGISTRY,
    VERSION_MAP,
    get_model_cache_path,
    get_version,
)
from ctcdetect.config.schemas import (
    GeneMappingConfig,
    HighlyVariableGenesConfig,
    InferenceConfig,
    NormalizeConfig,
    OutputConfig,
    PreprocessConfig,
    QCConfig,
    TokenizeConfig,
    UMAPConfig,
)
from ctcdetect.config.system import get_system_info

__version__ = "0.1.0"

__all__ = [
    # Registry
    "MODEL_REGISTRY",
    "VERSION_MAP",
    "DEFAULT_MODEL",
    "get_version",
    "get_model_cache_path",
    # Paths
    "PROJECT_ROOT",
    "MODEL_CACHE_DIR",
    "GENEFORMER_DIR",
    "CHECKPOINT_DIR",
    "FINETUNED_DIR",
    "TOKEN_DICT",
    "GENE_MEDIAN",
    "GENE_MAPPING",
    # Config loading
    "load_config",
    "get_config",
    "get_config_value",
    # Schemas
    "PreprocessConfig",
    "QCConfig",
    "NormalizeConfig",
    "HighlyVariableGenesConfig",
    "GeneMappingConfig",
    "TokenizeConfig",
    "UMAPConfig",
    "InferenceConfig",
    "OutputConfig",
    # System info
    "get_system_info",
    "__version__",
]

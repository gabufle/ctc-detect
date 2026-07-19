"""CTC-Detect: Geneformer-based circulating tumor cell detection from scRNA-seq data.

Public API:
- ctcdetect.run_detection: Run CTC detection on single sample
- ctcdetect.core: Core detection pipeline (preprocess, tokenize, detect)
- ctcdetect.evaluation: Metrics, reports, plots
- ctcdetect.config: Configuration loading and validation
- ctcdetect.training: Fine-tuning pipeline
- ctcdetect.data: Data loading and onboarding utilities
- ctcdetect.cli: Command-line interface
- ctcdetect.exceptions: Custom exception hierarchy
- ctcdetect.extensions: Model backend plugin interface
"""

from ctcdetect.core import (
    run_detection,
    load_model,
    check_geneformer_available,
    CTCDetectionPipeline,
    detect_format,
    validate_input,
    load_data,
    run_qc,
    normalize,
    map_genes_to_ensembl,
)
from ctcdetect.evaluation import (
    compute_metrics,
    generate_umap,
    plot_roc_pr,
    plot_score_distribution,
    generate_report,
    generate_html_report,
    generate_eval_report,
    generate_eval_html_report,
)
from ctcdetect.config import (
    # Registry
    MODEL_REGISTRY,
    VERSION_MAP,
    DEFAULT_MODEL,
    get_version,
    get_model_cache_path,
    # Paths
    PROJECT_ROOT,
    MODEL_CACHE_DIR,
    GENEFORMER_DIR,
    CHECKPOINT_DIR,
    FINETUNED_DIR,
    TOKEN_DICT,
    GENE_MEDIAN,
    GENE_MAPPING,
    # Config loading
    load_config,
    get_config,
    get_config_value,
    # Schemas
    PreprocessConfig,
    QCConfig,
    NormalizeConfig,
    HighlyVariableGenesConfig,
    GeneMappingConfig,
    TokenizeConfig,
    UMAPConfig,
    InferenceConfig,
    OutputConfig,
    # System info
    get_system_info,
)
from ctcdetect.training import (
    create_lora_config,
    load_base_model,
    train_model,
)
from ctcdetect.data import (
    merge_per_cell_files,
    prepare_external_dataset,
    combine_training_datasets,
)
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
from ctcdetect.extensions import (
    ModelBackend,
    GeneformerBackend,
    register_backend,
    get_backend,
    list_backends,
)

__version__ = "0.1.0"

__all__ = [
    # Core
    "run_detection",
    "load_model",
    "check_geneformer_available",
    "CTCDetectionPipeline",
    "detect_format",
    "validate_input",
    "load_data",
    "run_qc",
    "normalize",
    "map_genes_to_ensembl",
    # Evaluation
    "compute_metrics",
    "generate_umap",
    "plot_roc_pr",
    "plot_score_distribution",
    "generate_report",
    "generate_html_report",
    "generate_eval_report",
    "generate_eval_html_report",
    # Config
    "MODEL_REGISTRY",
    "VERSION_MAP",
    "DEFAULT_MODEL",
    "get_version",
    "get_model_cache_path",
    "PROJECT_ROOT",
    "MODEL_CACHE_DIR",
    "GENEFORMER_DIR",
    "CHECKPOINT_DIR",
    "FINETUNED_DIR",
    "TOKEN_DICT",
    "GENE_MEDIAN",
    "GENE_MAPPING",
    "load_config",
    "get_config",
    "get_config_value",
    "PreprocessConfig",
    "QCConfig",
    "NormalizeConfig",
    "HighlyVariableGenesConfig",
    "GeneMappingConfig",
    "TokenizeConfig",
    "UMAPConfig",
    "InferenceConfig",
    "OutputConfig",
    "get_system_info",
    # Training
    "create_lora_config",
    "load_base_model",
    "train_model",
    # Data
    "merge_per_cell_files",
    "prepare_external_dataset",
    "combine_training_datasets",
    # Extensions
    "ModelBackend",
    "GeneformerBackend",
    "register_backend",
    "get_backend",
    "list_backends",
    # Exceptions
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
    # Version
    "__version__",
]
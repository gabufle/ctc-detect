"""Core package for CTC-Detect.

Provides the main detection pipeline and preprocessing utilities.
"""

# Import submodules directly to avoid circular imports
from ctcdetect.core import detect, model, pipeline, preprocess

# Re-export public API
from ctcdetect.core.detect import run_detection
from ctcdetect.core.model import check_geneformer_available, load_model
from ctcdetect.core.pipeline import CTCDetectionPipeline
from ctcdetect.core.preprocess import (
    detect_format,
    load_data,
    map_genes_to_ensembl,
    normalize,
    run_qc,
    validate_input,
)

__all__ = [
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
]

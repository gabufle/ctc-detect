"""Core package for CTC-Detect.

Provides the main detection pipeline and preprocessing utilities.
"""

from ctcdetect.core.detect import run_detection
from ctcdetect.core.model import load_model, check_geneformer_available
from ctcdetect.core.pipeline import CTCDetectionPipeline
from ctcdetect.core.preprocess import (
    detect_format,
    validate_input,
    load_data,
    run_qc,
    normalize,
    map_genes_to_ensembl,
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
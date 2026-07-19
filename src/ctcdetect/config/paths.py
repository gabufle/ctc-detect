"""Path constants for CTC-Detect."""

from pathlib import Path

# Project root (ctc-detect/)
PROJECT_ROOT = Path(__file__).resolve().parents[3]

# Model cache directory
MODEL_CACHE_DIR = Path.home() / ".cache" / "ctc-detect" / "models"

# Geneformer directory (cloned from HuggingFace)
GENEFORMER_DIR = PROJECT_ROOT / "Geneformer" / "geneformer"

# Checkpoint directories
CHECKPOINT_DIR = PROJECT_ROOT / "results" / "checkpoints" / "best_model"
FINETUNED_DIR = PROJECT_ROOT / "Geneformer" / "Geneformer-V1-10M"

# Required Geneformer files
TOKEN_DICT = GENEFORMER_DIR / "token_dictionary_gc104M.pkl"
GENE_MEDIAN = GENEFORMER_DIR / "gene_median_dictionary_gc104M.pkl"
GENE_MAPPING = GENEFORMER_DIR / "ensembl_mapping_dict_gc104M.pkl"

__all__ = [
    "PROJECT_ROOT",
    "MODEL_CACHE_DIR",
    "GENEFORMER_DIR",
    "CHECKPOINT_DIR",
    "FINETUNED_DIR",
    "TOKEN_DICT",
    "GENE_MEDIAN",
    "GENE_MAPPING",
]
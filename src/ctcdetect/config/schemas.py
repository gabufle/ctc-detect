"""Pydantic configuration schemas for CTC-Detect.

Provides validated configuration models with IDE autocomplete and runtime validation.
"""

from pathlib import Path
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field, field_validator


class QCConfig(BaseModel):
    """Quality Control filtering thresholds."""

    min_genes: int = Field(ge=0, le=10000, default=200)
    max_genes: int = Field(ge=0, le=50000, default=6000)
    max_pct_mt: float = Field(ge=0, le=100, default=20.0)
    min_counts: int = Field(ge=0, le=100000, default=500)
    max_counts: int = Field(ge=0, le=500000, default=50000)
    min_cells: int = Field(ge=1, le=1000, default=10)
    max_pct_ribo: Optional[float] = Field(default=None, ge=0, le=100)


class NormalizeConfig(BaseModel):
    """Normalization parameters."""

    target_sum: float = Field(gt=0, default=10000)
    log1p: bool = True
    highly_variable_genes: Optional["HighlyVariableGenesConfig"] = None


class HighlyVariableGenesConfig(BaseModel):
    """Highly variable gene selection for UMAP (not tokenization)."""

    n_top_genes: int = Field(ge=100, le=10000, default=2000)
    flavor: Literal["seurat_v3", "seurat", "cell_ranger"] = "seurat_v3"


class GeneMappingConfig(BaseModel):
    """Gene symbol to Ensembl ID mapping parameters."""

    strict: bool = False
    min_mapping_rate: float = Field(ge=0, le=1, default=0.1)
    warn_mapping_rate: float = Field(ge=0, le=1, default=0.5)


class TokenizeConfig(BaseModel):
    """Tokenization parameters (Geneformer TranscriptomeTokenizer)."""

    model_input_size: int = Field(ge=512, le=8192, default=2048)
    special_token: bool = False
    collapse_gene_ids: bool = True
    model_version: Literal["V1", "V2"] = "V1"
    target_sum: float = Field(gt=0, default=10000)
    nproc: int = Field(ge=1, le=64, default=4)
    file_format: Literal["h5ad", "loom", "csv"] = "h5ad"
    keep_uncropped_input_ids: bool = False


class UMAPConfig(BaseModel):
    """UMAP visualization parameters."""

    n_top_genes: int = Field(ge=100, le=10000, default=2000)
    n_neighbors: int = Field(ge=2, le=100, default=15)
    min_dist: float = Field(ge=0, le=1, default=0.5)
    spread: float = Field(gt=0, default=1.0)
    random_state: int = 42
    n_panels: int = Field(ge=1, le=6, default=4)


class InferenceConfig(BaseModel):
    """Model inference parameters."""

    batch_size: int = Field(ge=1, le=256, default=32)
    max_seq_len: int = Field(ge=512, le=8192, default=2048)
    uncertainty_threshold: float = Field(ge=0, le=1, default=0.6)
    classification_threshold: float = Field(ge=0, le=1, default=0.5)


class OutputConfig(BaseModel):
    """Output parameters."""

    save_preprocessed: bool = False
    save_tokenized: bool = False
    save_logits: bool = False
    compression: Optional[Literal["gzip", "bz2", "xz"]] = None


class PreprocessConfig(BaseModel):
    """Complete preprocessing configuration."""

    qc: QCConfig = Field(default_factory=QCConfig)
    normalize: NormalizeConfig = Field(default_factory=NormalizeConfig)
    gene_mapping: GeneMappingConfig = Field(default_factory=GeneMappingConfig)
    tokenize: TokenizeConfig = Field(default_factory=TokenizeConfig)
    umap: UMAPConfig = Field(default_factory=UMAPConfig)
    inference: InferenceConfig = Field(default_factory=InferenceConfig)
    output: OutputConfig = Field(default_factory=OutputConfig)

    @field_validator("normalize", mode="before")
    @classmethod
    def _validate_normalize(cls, v):
        if isinstance(v, dict) and "highly_variable_genes" in v:
            hvg = v["highly_variable_genes"]
            if isinstance(hvg, dict):
                v["highly_variable_genes"] = HighlyVariableGenesConfig(**hvg)
        return v


# Forward reference resolution
NormalizeConfig.model_rebuild()
PreprocessConfig.model_rebuild()
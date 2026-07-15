"""Input format handling for CTC-Detect.

Handles reading and validating various single-cell RNA-seq input formats
produced by Cell Ranger and other pipelines.

Configuration-driven: QC thresholds, normalization parameters, and
tokenization settings are loaded from configs/preprocess.yaml.
"""

from pathlib import Path
from typing import Any

import scanpy as sc
import pandas as pd
from rich.console import Console

from ctcdetect.exceptions import (
    InputError,
    ValidationError,
    GeneMappingError,
    ConfigurationError,
)
from ctcdetect.config import load_config, get_config

console = Console()

# Load configuration at module import
CONFIG = load_config()

# Re-export supported formats for external use
SUPPORTED_FORMATS = {
    "cellranger": "10x Genomics Cell Ranger output (filtered_feature_bc_matrix/)",
    "h5ad": "AnnData HDF5 file (.h5ad)",
    "csv": "Plain CSV/TSV matrix (genes x cells)",
    "tsv": "Tab-separated values matrix (genes x cells)",
    "txt": "Text matrix (genes x cells)",
    "mtx": "Matrix Market Exchange format (.mtx)",
    "loom": "Loom file format (.loom)",
}


def get_config_value(key: str, default=None):
    """Get a nested config value using dot notation (e.g., 'qc.min_genes').

    This wraps config_loader.get_config for backwards compatibility.
    """
    cfg = get_config()
    parts = key.split(".")
    value: Any = cfg.as_dict()
    for part in parts:
        if isinstance(value, dict) and part in value:
            value = value[part]
        else:
            return default
    return value


def _get_int(key: str, default: int) -> int:
    """Get config value as int."""
    val = get_config_value(key, default)
    if val is None:
        return default
    return int(val)


def _get_float(key: str, default: float) -> float:
    """Get config value as float."""
    val = get_config_value(key, default)
    if val is None:
        return default
    return float(val)


def _get_bool(key: str, default: bool) -> bool:
    """Get config value as bool."""
    val = get_config_value(key, default)
    if isinstance(val, bool):
        return val
    if isinstance(val, str):
        return val.lower() in ("true", "yes", "1", "on")
    return bool(val) if val is not None else default


def detect_format(input_path: Path) -> str:
    """Auto-detect the format of the input data.

    Args:
        input_path: Path to the input file or directory.

    Returns:
        Format string key (e.g. 'cellranger', 'h5ad', 'csv').

    Raises:
        InputError: If the format cannot be determined.
    """
    if input_path.is_dir():
        # Check for Cell Ranger directory structure
        if (input_path / "filtered_feature_bc_matrix").exists():
            return "cellranger"
        if (input_path / "matrix.mtx").exists() or (input_path / "matrix.mtx.gz").exists():
            return "cellranger"
    elif input_path.suffix == ".h5ad":
        return "h5ad"
    elif input_path.suffix == ".csv":
        return "csv"
    elif input_path.suffix == ".tsv":
        return "tsv"
    elif input_path.suffix == ".txt":
        return "txt"
    elif input_path.suffix == ".mtx":
        return "mtx"
    elif input_path.suffix == ".loom":
        return "loom"

    raise InputError(
        f"Cannot determine input format for '{input_path}'.",
        input_path=input_path,
        hint="Supported formats:\n" + "\n".join(f"  - {v}" for v in SUPPORTED_FORMATS.values()),
    )


def validate_input(input_path: Path) -> bool:
    """Validate that input data is well-formed and readable.

    Checks:
    1. Format is detectable
    2. Required files exist
    3. Data can be loaded
    4. Data is non-empty
    5. Gene names look reasonable (not all numeric)

    Args:
        input_path: Path to the input file or directory.

    Returns:
        True if validation passes.

    Raises:
        InputError: If validation fails.
    """
    fmt = detect_format(input_path)
    console.print(f"[green]✓[/green] Detected format: {SUPPORTED_FORMATS[fmt]}")

    if fmt == "cellranger":
        _validate_cellranger(input_path)
    elif fmt == "h5ad":
        _validate_h5ad(input_path)
    elif fmt in ("csv", "tsv", "txt"):
        _validate_csv(input_path, fmt)

    console.print("[green]✓[/green] All validation checks passed.")
    return True


def _validate_cellranger(input_path: Path):
    """Validate a Cell Ranger output directory."""
    # Find the matrix directory
    if (input_path / "filtered_feature_bc_matrix").exists():
        mtx_dir = input_path / "filtered_feature_bc_matrix"
    else:
        mtx_dir = input_path

    # Check required files
    matrix_file = None
    for candidate in ["matrix.mtx", "matrix.mtx.gz"]:
        if (mtx_dir / candidate).exists():
            matrix_file = candidate
            break

    if matrix_file is None:
        raise ValidationError(
            f"No matrix.mtx found in {mtx_dir}",
            failed_check="cellranger_matrix_file",
            hint="Cell Ranger output should contain matrix.mtx, barcodes.tsv, and features.tsv",
        )

    console.print(f"  Matrix file: {matrix_file}")

    # Check barcodes
    barcodes_files = list(mtx_dir.glob("barcodes*"))
    if not barcodes_files:
        raise ValidationError(
            f"No barcodes file found in {mtx_dir}",
            failed_check="cellranger_barcodes",
            hint="Expected barcodes.tsv or barcodes.tsv.gz",
        )

    with open(barcodes_files[0]) as f:
        n_barcodes = sum(1 for _ in f)
    console.print(f"  Barcodes: {n_barcodes}")
    if n_barcodes < get_config("qc.min_cells", 10):
        console.print(
            f"[yellow]Warning:[/yellow] Very few barcodes ({n_barcodes}). "
            f"This may be a sparse or filtered dataset."
        )

    # Check features/genes
    features_files = list(mtx_dir.glob("features*")) or list(mtx_dir.glob("genes*"))
    if not features_files:
        raise ValidationError(
            f"No features/genes file found in {mtx_dir}",
            failed_check="cellranger_features",
            hint="Expected features.tsv or genes.tsv",
        )

    with open(features_files[0]) as f:
        n_features = sum(1 for _ in f)
    console.print(f"  Features: {n_features}")
    if n_features < 100:
        console.print(
            f"[yellow]Warning:[/yellow] Very few features ({n_features}). "
            f"This may not be a standard single-cell dataset."
        )

    # Try loading a small sample to verify the file is not corrupted
    try:
        adata = sc.read_10x_mtx(
            str(mtx_dir), var_names="gene_symbols", cache=False, gex_only=True
        )
        console.print(f"  Loaded shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        if adata.shape[0] == 0:
            raise ValidationError(
                "Dataset contains 0 cells after loading.",
                failed_check="cellranger_empty_cells",
            )
        if adata.shape[1] == 0:
            raise ValidationError(
                "Dataset contains 0 genes after loading.",
                failed_check="cellranger_empty_genes",
            )

        # Check gene names look reasonable
        sample_genes = list(adata.var_names[:100])
        alphabetic = sum(1 for g in sample_genes if g and g[0].isalpha())
        if alphabetic < 10 and len(sample_genes) > 0:
            console.print(
                f"[yellow]Warning:[/yellow] Gene names look unusual "
                f"(only {alphabetic}/{len(sample_genes)} start with letters)."
            )
            console.print("  Sample genes:", ", ".join(sample_genes[:10]))
            console.print("  CTC-Detect expects HGNC gene symbols (e.g., TP53, BRCA1, EGFR).")

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Could not load matrix data: {e}",
            failed_check="cellranger_load",
            hint="The matrix files may be corrupted or in an unexpected format.",
        ) from e


def _validate_h5ad(input_path: Path):
    """Validate an .h5ad file."""
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    console.print(f"  File size: {file_size_mb:.1f} MB")

    try:
        adata = sc.read_h5ad(str(input_path), backed="r")
        console.print(f"  Loaded shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        if adata.shape[0] == 0:
            raise ValidationError(
                "Dataset contains 0 cells.",
                failed_check="h5ad_empty_cells",
            )
        if adata.shape[1] == 0:
            raise ValidationError(
                "Dataset contains 0 genes.",
                failed_check="h5ad_empty_genes",
            )

        # Check for gene names
        sample_genes = list(adata.var_names[:100])
        alphabetic = sum(1 for g in sample_genes if g and g[0].isalpha())
        if alphabetic < 10 and len(sample_genes) > 0:
            console.print(
                f"[yellow]Warning:[/yellow] Gene names look unusual "
                f"(only {alphabetic}/{len(sample_genes)} start with letters)."
            )
            console.print("  Sample genes:", ", ".join(sample_genes[:10]))

        # Check for expression data
        if adata.X is None:
            raise ValidationError(
                "No expression matrix (.X) found in h5ad file.",
                failed_check="h5ad_no_expression",
            )

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Could not read h5ad file: {e}",
            failed_check="h5ad_load",
            hint="The file may be corrupted or not a valid AnnData file.",
        ) from e


def _validate_csv(input_path: Path, fmt: str):
    """Validate a CSV/TSV matrix file."""
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    console.print(f"  File size: {file_size_mb:.1f} MB")

    try:
        sep = "\t" if fmt in ("tsv", "txt") else ","
        df = pd.read_csv(input_path, index_col=0, nrows=5, sep=sep)
        console.print(f"  Preview shape: {df.shape[0]} rows x {df.shape[1]} columns")
        console.print(f"  First row names: {list(df.index[:5])}")
        console.print(f"  First column names: {list(df.columns[:5])}")

        if df.shape[0] == 0 or df.shape[1] == 0:
            raise ValidationError(
                "CSV file appears to be empty.",
                failed_check="csv_empty",
            )

    except ValidationError:
        raise
    except Exception as e:
        raise ValidationError(
            f"Could not read CSV file: {e}",
            failed_check="csv_load",
            hint=f"Ensure the file is a valid {fmt.upper()} with genes as rows and cells as columns.",
        ) from e


def load_data(input_path: Path) -> sc.AnnData:
    """Load single-cell data from supported formats into an AnnData object.

    Args:
        input_path: Path to the input file or directory.

    Returns:
        AnnData object containing the data.
    """
    fmt = detect_format(input_path)

    if fmt == "cellranger":
        if (input_path / "filtered_feature_bc_matrix").exists():
            mtx_dir = input_path / "filtered_feature_bc_matrix"
        else:
            mtx_dir = input_path
        adata = sc.read_10x_mtx(
            str(mtx_dir), var_names="gene_symbols", cache=True, make_unique=True
        )

    elif fmt == "h5ad":
        adata = sc.read_h5ad(input_path)

    elif fmt == "csv":
        df = pd.read_csv(input_path, index_col=0)
        adata = sc.AnnData(df.T)
        adata.var_names = df.index.astype(str)
        adata.obs_names = df.columns.astype(str)

    elif fmt == "tsv":
        df = pd.read_csv(input_path, index_col=0, sep="\t")
        adata = sc.AnnData(df.T)
        adata.var_names = df.index.astype(str)
        adata.obs_names = df.columns.astype(str)

    elif fmt == "txt":
        df = pd.read_csv(input_path, index_col=0, sep=r"\s+")
        adata = sc.AnnData(df.T)
        adata.var_names = df.index.astype(str)
        adata.obs_names = df.columns.astype(str)

    elif fmt == "mtx":
        from scipy.io import mmread
        matrix = mmread(input_path)
        adata = sc.AnnData(matrix.T)
        adata.var_names = [f"Gene_{i}" for i in range(adata.shape[1])]
        adata.obs_names = [f"Cell_{i}" for i in range(adata.shape[0])]

    elif fmt == "loom":
        adata = sc.read_loom(input_path)

    else:
        raise ValidationError(
            f"Unsupported format for loading: {fmt}",
            failed_check="load_unsupported_format",
        )

    console.print(f"[green]✓[/green] Loaded data with shape {adata.shape}")
    return adata


def run_qc(adata: sc.AnnData) -> sc.AnnData:
    """Run quality control filtering on AnnData.

    Uses thresholds from configs/preprocess.yaml:
    - min_genes, max_genes, max_pct_mt

    Args:
        adata: AnnData object to filter.

    Returns:
        Filtered AnnData object.
    """
    # Calculate QC metrics
    adata.var["mt"] = adata.var_names.str.startswith("MT-")
    sc.pp.calculate_qc_metrics(
        adata, qc_vars=["mt"], percent_top=None, log1p=False, inplace=True
    )

    n_before = adata.shape[0]

    min_genes = get_config("qc.min_genes", 200)
    max_genes = get_config("qc.max_genes", 6000)
    max_pct_mt = get_config("qc.max_pct_mt", 20)

    adata = adata[
        (adata.obs["n_genes_by_counts"] >= min_genes)
        & (adata.obs["n_genes_by_counts"] <= max_genes)
        & (adata.obs["pct_counts_mt"] <= max_pct_mt),
        :,
    ].copy()

    n_after = adata.shape[0]
    console.print(f"  QC: {n_before} → {n_after} cells ({n_before - n_after} removed)")

    if n_after < get_config("qc.min_cells", 10):
        raise ValidationError(
            f"Too few cells remain after QC ({n_after}). "
            f"Minimum required: {get_config('qc.min_cells', 10)}",
            failed_check="qc_insufficient_cells",
            hint="Consider relaxing QC thresholds in configs/preprocess.yaml",
        )

    return adata


def normalize(adata: sc.AnnData) -> sc.AnnData:
    """Normalize and log-transform the data.

    Uses parameters from configs/preprocess.yaml:
    - target_sum, log1p

    Args:
        adata: AnnData object to normalize.

    Returns:
        Normalized AnnData object.
    """
    target_sum = get_config("normalize.target_sum", 10000)
    log1p = get_config("normalize.log1p", True)

    sc.pp.normalize_total(adata, target_sum=target_sum)
    if log1p:
        sc.pp.log1p(adata)

    return adata


def map_genes_to_ensembl(adata: sc.AnnData, gene_mapping: dict) -> sc.AnnData:
    """Map gene symbols to Ensembl IDs.

    Uses thresholds from configs/preprocess.yaml:
    - min_mapped_fraction, require_mapped, warn_on_low_mapping

    Args:
        adata: AnnData with gene symbols as var_names.
        gene_mapping: Dict mapping gene symbol -> Ensembl ID.

    Returns:
        AnnData with Ensembl IDs as var_names.
    """
    ensembl_ids = []
    mapped = 0
    for gene in adata.var_names:
        if gene in gene_mapping:
            ensembl_ids.append(gene_mapping[gene])
            mapped += 1
        else:
            ensembl_ids.append(None)

    adata.var["ensembl_id"] = ensembl_ids

    total = len(ensembl_ids)
    mapping_rate = mapped / total if total > 0 else 0.0

    min_fraction = get_config("gene_mapping.min_mapped_fraction", 0.1)
    require_mapped = get_config("gene_mapping.require_mapped", True)
    warn_on_low = get_config("gene_mapping.warn_on_low_mapping", True)

    if warn_on_low and mapping_rate < 0.5:
        console.print(
            f"[yellow]Warning:[/yellow] Low gene mapping rate: "
            f"{mapped}/{total} ({mapping_rate:.1%})"
        )
        console.print(
            "  Ensure your data uses standard HGNC gene symbols "
            "(e.g., TP53, BRCA1, EGFR)."
        )

    if require_mapped and mapping_rate < min_fraction:
        raise GeneMappingError(
            f"Insufficient genes mapped to Ensembl IDs: {mapped}/{total} "
            f"({mapping_rate:.1%}), minimum required: {min_fraction:.1%}",
            mapped_count=mapped,
            total_count=total,
            hint="Your input data may use non-standard gene identifiers. "
            "CTC-Detect requires HGNC gene symbols.",
        )

    if mapped == 0:
        raise GeneMappingError(
            "No genes could be mapped to Ensembl IDs.",
            mapped_count=0,
            total_count=total,
            hint="Ensure your data uses standard HGNC gene symbols "
            "(e.g., TP53, BRCA1, EGFR) rather than numeric IDs or custom names.",
        )

    adata = adata[:, adata.var["ensembl_id"].notna()].copy()
    # Use .values to avoid type issues with scanpy's var_names setter
    new_names = adata.var["ensembl_id"].values.astype(str)
    adata.var_names = pd.Index(new_names)
    adata.var_names_make_unique()

    console.print(f"  Gene mapping: {mapped}/{total} genes mapped to Ensembl IDs")
    console.print(f"  After mapping: {adata.shape[0]} cells × {adata.shape[1]} genes")

    if adata.shape[1] == 0:
        raise GeneMappingError(
            "No genes remain after Ensembl ID mapping.",
            mapped_count=mapped,
            total_count=total,
        )

    return adata
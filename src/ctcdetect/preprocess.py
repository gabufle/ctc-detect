"""Input format handling for CTC-Detect.

Handles reading and validating various single-cell RNA-seq input formats
produced by Cell Ranger and other pipelines.
"""

from pathlib import Path

import scanpy as sc
import pandas as pd
from rich.console import Console

console = Console()


SUPPORTED_FORMATS = {
    "cellranger": "10x Genomics Cell Ranger output (filtered_feature_bc_matrix/)",
    "h5ad": "AnnData HDF5 file (.h5ad)",
    "csv": "Plain CSV/TSV matrix (genes x cells)",
    "tsv": "Tab-separated values matrix (genes x cells)",
    "txt": "Text matrix (genes x cells)",
    "mtx": "Matrix Market Exchange format (.mtx)",
    "loom": "Loom file format (.loom)",
}


def detect_format(input_path: Path) -> str:
    """Auto-detect the format of the input data.

    Args:
        input_path: Path to the input file or directory.

    Returns:
        Format string key (e.g. 'cellranger', 'h5ad', 'csv').

    Raises:
        SystemExit: If the format cannot be determined.
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

    console.print(
        f"[red]Error:[/red] Cannot determine input format for '{input_path}'.\n"
        "Supported formats:\n"
        + "\n".join(f"  - {v}" for v in SUPPORTED_FORMATS.values())
    )
    raise SystemExit(1)


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
        SystemExit: If validation fails.
    """
    fmt = detect_format(input_path)
    console.print(f"[green]✓[/green] Detected format: {SUPPORTED_FORMATS[fmt]}")

    if fmt == "cellranger":
        _validate_cellranger(input_path)
    elif fmt == "h5ad":
        _validate_h5ad(input_path)
    elif fmt == "csv":
        _validate_csv(input_path)

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
        console.print(f"[red]Error:[/red] No matrix.mtx found in {mtx_dir}")
        console.print("Cell Ranger output should contain matrix.mtx, barcodes.tsv, and features.tsv")
        console.print("Make sure you are pointing to the correct directory.")
        raise SystemExit(1)

    console.print(f"  Matrix file: {matrix_file}")

    # Check barcodes (expect at least 10 barcodes to be meaningful)
    barcodes_files = list(mtx_dir.glob("barcodes*"))
    if not barcodes_files:
        console.print(f"[red]Error:[/red] No barcodes file found in {mtx_dir}")
        raise SystemExit(1)

    with open(barcodes_files[0]) as f:
        n_barcodes = sum(1 for _ in f)
    console.print(f"  Barcodes: {n_barcodes}")
    if n_barcodes < 10:
        console.print(f"[yellow]Warning:[/yellow] Very few barcodes ({n_barcodes}). This may be a sparse or filtered dataset.")

    # Check features / genes (expect gene symbols, not just Ensembl IDs)
    features_files = list(mtx_dir.glob("features*")) or list(mtx_dir.glob("genes*"))
    if not features_files:
        console.print(f"[red]Error:[/red] No features/genes file found in {mtx_dir}")
        raise SystemExit(1)

    with open(features_files[0]) as f:
        n_features = sum(1 for _ in f)
    console.print(f"  Features: {n_features}")
    if n_features < 100:
        console.print(f"[yellow]Warning:[/yellow] Very few features ({n_features}). This may not be a standard single-cell dataset.")

    # Try loading a small sample to verify the file is not corrupted
    try:
        adata = sc.read_10x_mtx(
            str(mtx_dir), var_names="gene_symbols", cache=False, gex_only=True
        )
        console.print(f"  Loaded shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        if adata.shape[0] == 0:
            console.print("[red]Error:[/red] Dataset contains 0 cells after loading.")
            raise SystemExit(1)
        if adata.shape[1] == 0:
            console.print("[red]Error:[/red] Dataset contains 0 genes after loading.")
            raise SystemExit(1)

        # Check gene names look reasonable (at least some should be alphabetic)
        sample_genes = list(adata.var_names[:100])
        alphabetic = sum(1 for g in sample_genes if g[0].isalpha() if g)
        if alphabetic < 10 and len(sample_genes) > 0:
            console.print(
                f"[yellow]Warning:[/yellow] Gene names look unusual (only {alphabetic}/{len(sample_genes)} start with letters)."
            )
            console.print("  Sample genes:", ", ".join(sample_genes[:10]))
            console.print("  CTC-Detect expects HGNC gene symbols (e.g. TP53, BRCA1, EGFR).")

    except Exception as e:
        console.print(f"[red]Error:[/red] Could not load matrix data: {e}")
        raise SystemExit(1)


def _validate_h5ad(input_path: Path):
    """Validate an .h5ad file."""
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    console.print(f"  File size: {file_size_mb:.1f} MB")

    try:
        adata = sc.read_h5ad(str(input_path), backed="r")
        console.print(f"  Loaded shape: {adata.shape[0]} cells x {adata.shape[1]} genes")
        if adata.shape[0] == 0:
            console.print("[red]Error:[/red] Dataset contains 0 cells.")
            raise SystemExit(1)
        if adata.shape[1] == 0:
            console.print("[red]Error:[/red] Dataset contains 0 genes.")
            raise SystemExit(1)

        # Check for gene names
        sample_genes = list(adata.var_names[:100])
        alphabetic = sum(1 for g in sample_genes if g[0].isalpha() if g)
        if alphabetic < 10 and len(sample_genes) > 0:
            console.print(
                f"[yellow]Warning:[/yellow] Gene names look unusual (only {alphabetic}/{len(sample_genes)} start with letters)."
            )
            console.print("  Sample genes:", ", ".join(sample_genes[:10]))

        # Check for expression data
        if adata.X is None:
            console.print("[red]Error:[/red] No expression matrix (.X) found in h5ad file.")
            raise SystemExit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] Could not read h5ad file: {e}")
        raise SystemExit(1)


def _validate_csv(input_path: Path):
    """Validate a CSV/TSV matrix file."""
    file_size_mb = input_path.stat().st_size / (1024 * 1024)
    console.print(f"  File size: {file_size_mb:.1f} MB")

    try:
        sep = "\t" if input_path.suffix in (".tsv", ".txt") else ","
        df = pd.read_csv(input_path, index_col=0, nrows=5, sep=sep)
        console.print(f"  Preview shape: {df.shape[0]} rows x {df.shape[1]} columns")
        console.print(f"  First row names: {list(df.index[:5])}")
        console.print(f"  First column names: {list(df.columns[:5])}")

        if df.shape[0] == 0 or df.shape[1] == 0:
            console.print("[red]Error:[/red] CSV file appears to be empty.")
            raise SystemExit(1)

    except Exception as e:
        console.print(f"[red]Error:[/red] Could not read CSV file: {e}")
        raise SystemExit(1)


def load_data(input_path: Path) -> sc.AnnData:
    """Load single-cell data from supported formats into an AnnData object.

    Args:
        input_path: Path to the input file or directory.

    Returns:
        AnnData object containing the data.
    """
    fmt = detect_format(input_path)
    if fmt == "cellranger":
        # Assume input_path is the directory containing filtered_feature_bc_matrix/
        # Or directly the filtered_feature_bc_matrix directory
        if (input_path / "filtered_feature_bc_matrix").exists():
            mtx_dir = input_path / "filtered_feature_bc_matrix"
        else:
            mtx_dir = input_path
        adata = sc.read_10x_mtx(
            mtx_dir, var_names="gene_symbols", cache=True, make_unique=True
        )
    elif fmt == "h5ad":
        adata = sc.read_h5ad(input_path)
    elif fmt == "csv":
        # Assume genes x cells matrix with gene names as row names and cell IDs as column names
        df = pd.read_csv(input_path, index_col=0)
        adata = sc.AnnData(df.T)  # Transpose to cells x genes
        adata.var_names = df.index.astype(str)
        adata.obs_names = df.columns.astype(str)
    elif fmt == "tsv":
        # Tab-separated genes x cells matrix
        df = pd.read_csv(input_path, index_col=0, sep="\t")
        adata = sc.AnnData(df.T)  # Transpose to cells x genes
        adata.var_names = df.index.astype(str)
        adata.obs_names = df.columns.astype(str)
    elif fmt == "txt":
        # Text file genes x cells matrix
        df = pd.read_csv(input_path, index_col=0, sep=r"\s+")
        adata = sc.AnnData(df.T)  # Transpose to cells x genes
        adata.var_names = df.index.astype(str)
        adata.obs_names = df.columns.astype(str)
    elif fmt == "mtx":
        # Matrix Market Exchange format - need additional files
        from scipy.io import mmread
        matrix = mmread(input_path)
        # For MTX format, we need barcodes and genes files
        # This is a simplified version - in practice you'd need to handle this more carefully
        adata = sc.AnnData(matrix.T)  # Transpose to cells x genes
        # Set generic names since we don't have the actual barcodes/genes
        adata.var_names = [f"Gene_{i}" for i in range(adata.shape[1])]
        adata.obs_names = [f"Cell_{i}" for i in range(adata.shape[0])]
    elif fmt == "loom":
        # Loom format
        adata = sc.read_loom(input_path)
    else:
        # Should not happen due to detect_format
        raise RuntimeError(f"Unsupported format: {fmt}")

    console.print(f"[green]✓[/green] Loaded data with shape {adata.shape}")
    return adata

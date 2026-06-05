"""Input format handling for CTC-Detect.

Handles reading and validating various single-cell RNA-seq input formats
produced by Cell Ranger and other pipelines.
"""

from pathlib import Path
from typing import Optional

import scanpy as sc
import pandas as pd
from rich.console import Console

console = Console()


SUPPORTED_FORMATS = {
    "cellranger": "10x Genomics Cell Ranger output (filtered_feature_bc_matrix/)",
    "h5ad": "AnnData HDF5 file (.h5ad)",
    "csv": "Plain CSV/TSV matrix (genes x cells)",
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
    elif input_path.suffix in (".csv", ".tsv", ".txt"):
        return "csv"

    console.print(
        f"[red]Error:[/red] Cannot determine input format for '{input_path}'.\n"
        "Supported formats:\n"
        + "\n".join(f"  • {v}" for v in SUPPORTED_FORMATS.values())
    )
    raise SystemExit(1)


def validate_input(input_path: Path) -> bool:
    """Validate that input data is well-formed and readable.

    Args:
        input_path: Path to the input file or directory.

    Returns:
        True if validation passes.
    """
    fmt = detect_format(input_path)
    console.print(f"[green]✓[/green] Detected format: {SUPPORTED_FORMATS[fmt]}")
    console.print("[yellow]Note:[/yellow] Full validation not yet implemented (stub).")
    return True


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
    else:
        # Should not happen due to detect_format
        raise RuntimeError(f"Unsupported format: {fmt}")

    console.print(f"[green]✓[/green] Loaded data with shape {adata.shape}")
    return adata
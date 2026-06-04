"""Input format handling for CTC-Detect.

Handles reading and validating various single-cell RNA-seq input formats
produced by Cell Ranger and other pipelines.
"""

from pathlib import Path
from typing import Optional

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

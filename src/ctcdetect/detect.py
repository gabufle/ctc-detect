"""Core detection logic for CTC-Detect.

This module runs the Geneformer model on preprocessed single-cell
expression data and outputs per-cell CTC probability scores.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

console = Console()


def run_detection(
    input_path: Path,
    output_path: Path,
    cancer_type: Optional[str] = None,
) -> None:
    """Run CTC detection on a single sample.

    Takes Cell Ranger output (filtered feature-barcode matrix) and
    produces a table of per-cell CTC probability scores.

    Args:
        input_path: Path to Cell Ranger output directory or h5ad file.
        output_path: Path to write results (CSV with cell barcodes and scores).
        cancer_type: Optional cancer type hint to select the best model.
    """
    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading Geneformer model...", total=None)
        progress.add_task("Tokenizing expression data...", total=None)
        progress.add_task("Running inference...", total=None)
        progress.add_task("Writing results...", total=None)

    console.print("[yellow]Note:[/yellow] Detection logic not yet implemented (stub).")
    console.print(f"  Input:  {input_path}")
    console.print(f"  Output: {output_path}")
    if cancer_type:
        console.print(f"  Cancer type: {cancer_type}")

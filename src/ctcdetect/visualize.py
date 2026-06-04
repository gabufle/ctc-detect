"""Visualization module for CTC-Detect.

Generates UMAP plots, score distribution histograms, and other
figures for QC and result interpretation.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def generate_umap(
    results_path: Path,
    output_path: Path,
    color_by: str = "ctc_score",
) -> None:
    """Generate a UMAP plot colored by CTC probability score.

    Args:
        results_path: Path to detection results (CSV with scores).
        output_path: Path to write the figure (PNG or PDF).
        color_by: Column name to use for coloring cells.
    """
    console.print("[yellow]Note:[/yellow] UMAP generation not yet implemented (stub).")
    console.print(f"  Results: {results_path}")
    console.print(f"  Output:  {output_path}")
    console.print(f"  Color by: {color_by}")


def generate_score_histogram(
    results_path: Path,
    output_path: Path,
) -> None:
    """Generate a histogram of CTC probability scores.

    Args:
        results_path: Path to detection results (CSV with scores).
        output_path: Path to write the figure (PNG or PDF).
    """
    console.print("[yellow]Note:[/yellow] Score histogram not yet implemented (stub).")
    console.print(f"  Results: {results_path}")
    console.print(f"  Output:  {output_path}")

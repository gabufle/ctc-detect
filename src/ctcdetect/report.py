"""Report generation for CTC-Detect.

Produces clinical summary reports with key statistics,
threshold-based CTC calls, and QC metrics.
"""

from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def generate_report(
    results_path: Path,
    output_path: Path,
    threshold: float = 0.5,
) -> None:
    """Generate a clinical summary report from detection results.

    Args:
        results_path: Path to detection results (CSV with scores).
        output_path: Path to write the report (HTML or PDF).
        threshold: Probability score threshold for calling a cell a CTC.
    """
    console.print("[yellow]Note:[/yellow] Report generation not yet implemented (stub).")
    console.print(f"  Results:   {results_path}")
    console.print(f"  Output:    {output_path}")
    console.print(f"  Threshold: {threshold}")

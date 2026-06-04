"""Shared utilities for CTC-Detect CLI."""

import os
from pathlib import Path
from typing import Optional

from rich.console import Console

console = Console()


def validate_input_path(path: str, description: str = "Input path") -> Path:
    """Validate that an input path exists and is readable.

    Exits with a friendly error message if the path is invalid.
    """
    p = Path(path)
    if not p.exists():
        console.print(f"[red]Error:[/red] {description} '{path}' does not exist.")
        raise SystemExit(1)
    if not os.access(p, os.R_OK):
        console.print(f"[red]Error:[/red] {description} '{path}' is not readable.")
        raise SystemExit(1)
    return p


def validate_output_path(path: str, description: str = "Output path") -> Path:
    """Validate that an output path can be written to.

    Creates parent directories if needed. Exits with a friendly error
    message if the path is not writable.
    """
    p = Path(path)
    parent = p.parent
    if parent and not parent.exists():
        try:
            parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            console.print(
                f"[red]Error:[/red] Cannot create output directory '{parent}': {e}"
            )
            raise SystemExit(1)
    return p


def print_banner() -> None:
    """Print the CTC-Detect welcome banner."""
    console.print(
        "[bold blue]CTC-Detect[/bold blue] — "
        "Circulating Tumor Cell Detection from scRNA-seq"
    )
    console.print("Powered by Geneformer\n")

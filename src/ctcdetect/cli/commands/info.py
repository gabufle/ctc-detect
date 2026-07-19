"""Info command for CTC-Detect CLI."""

import typer
from pathlib import Path
from rich.console import Console
from rich.table import Table

from ctcdetect.cli.utils import print_banner, console
from ctcdetect.config import get_system_info


@typer.Typer(
    help="Show CTC-Detect version and system information.",
)
def info_app():
    pass


@info_app.command()
def info():
    """Show CTC-Detect version and system information.

    Displays the installed version, available models, and
    system configuration. Useful for troubleshooting.
    """
    print_banner()

    info = get_system_info()

    table = Table(show_header=False, box=None, padding=(0, 1))
    table.add_column("Key", style="bold cyan", min_width=22)
    table.add_column("Value")

    table.add_row("Version", info["version"])
    table.add_row("Python", info["python"])
    table.add_row("Platform", info["platform"])
    table.add_row("PyTorch", info["pytorch"])
    table.add_row("CUDA available", "Yes" if info["cuda_available"] else "No")
    if info["cuda_available"]:
        table.add_row("CUDA version", info["cuda_version"])
        table.add_row("GPU", info["gpu"])
    table.add_row("Geneformer installed", "Yes" if info["geneformer_installed"] else "No")
    if info["geneformer_installed"]:
        table.add_row("Geneformer path", info["geneformer_path"])
    table.add_row("Checkpoint available", "Yes" if info["checkpoint_available"] else "No")
    if info["checkpoint_available"]:
        table.add_row("Checkpoint path", info["checkpoint_path"])

    cached = info.get("cached_models", [])
    table.add_row("Cached models", ", ".join(cached) if cached else "None")

    # Disk space
    disk_free = info.get("disk_free_gb")
    disk_total = info.get("disk_total_gb")
    if disk_free is not None and disk_total is not None:
        table.add_row("Disk space (cache)", f"{disk_free} GB free / {disk_total} GB total")
    else:
        table.add_row("Disk space (cache)", "N/A")

    console.print(table)
    console.print()
    console.print("For documentation, visit: https://github.com/gabufle/ctc-detect")


__all__ = ["info_app"]
"""Model management commands for CTC-Detect CLI."""

import typer
from pathlib import Path
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn
from rich.table import Table

from ctcdetect.cli.utils import print_banner, console
from ctcdetect.config import get_model_cache_path, MODEL_REGISTRY, get_version
from ctcdetect.config.system import get_system_info


model_app = typer.Typer(
    help="Manage Geneformer models — download, list, and inspect available models.",
)


@model_app.command("download")
def model_download(
    version: str = typer.Option(
        "latest",
        "--version", "-v",
        help=(
            "Model version to download (e.g. 'v1.0', 'latest').\n"
            "Use 'latest' for the most recent release."
        ),
        rich_help_panel="Model Options",
    ),
):
    """Download a pre-trained Geneformer model.

    Models are cached locally so you only need to download once.
    If you have not downloaded a model yet, CTC-Detect will
    prompt you to run this command.
    """
    print_banner()

    # Resolve version to repo ID
    try:
        repo_id = get_version(version)
    except ValueError as e:
        console.print(f"[red]Error:[/red] {e}")
        raise typer.Exit(1)

    cache_path = get_model_cache_path(version)

    console.print(f"Model:    {version}")
    console.print(f"Repo:     {repo_id}")
    console.print(f"Cache:    {cache_path}")
    console.print()

    # Check if already cached
    if cache_path.exists() and any(cache_path.iterdir()):
        console.print(f"[yellow]Note:[/yellow] Model already exists at {cache_path}")
        overwrite = typer.confirm("Re-download?", default=False)
        if not overwrite:
            console.print("[green]✓[/green] Using cached model.")
            return

    # Download using huggingface_hub
    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        console.print("[red]Error:[/red] huggingface_hub is not installed.")
        console.print("Install it with: pip install huggingface_hub")
        raise typer.Exit(1)

    # Check available disk space (require at least 2 GB free)
    try:
        import shutil
        disk_usage = shutil.disk_usage(str(cache_path))
        free_gb = disk_usage.free / (1024 ** 3)
        if free_gb < 2.0:
            console.print(
                f"[red]Error:[/red] Insufficient disk space "
                f"({free_gb:.1f} GB free, need at least 2 GB)."
            )
            raise typer.Exit(1)
    except OSError:
        pass  # If we can't check, proceed anyway

    try:
        with Progress(
            SpinnerColumn(),
            TextColumn("[progress.description]{task.description}"),
            console=console,
        ) as progress:
            progress.add_task(f"Downloading model '{version}'...", total=None)

            snapshot_download(
                repo_id=repo_id,
                local_dir=str(cache_path),
                local_dir_use_symlinks=False,
            )
    except ConnectionError as e:
        console.print(f"\n[red]Error:[/red] Network failure: {e}")
        console.print("Check your network connection and try again.")
        raise typer.Exit(1)
    except OSError as e:
        if "No space left on device" in str(e):
            console.print("\n[red]Error:[/red] Disk full during download.")
            console.print("Free up space and try again.")
        else:
            console.print(f"\n[red]Error:[/red] OS error: {e}")
        raise typer.Exit(1)
    except Exception as e:
        error_msg = str(e).lower()
        if "401" in error_msg or "unauthorized" in error_msg or "forbidden" in error_msg:
            console.print(f"\n[red]Error:[/red] Access denied for repo '{repo_id}'.")
            console.print("This model may require authentication.")
            console.print("Visit https://huggingface.co/settings/tokens to set up a token.")
        elif "404" in error_msg or "not found" in error_msg:
            console.print(f"\n[red]Error:[/red] Model repo '{repo_id}' not found.")
            console.print("Check the version alias and try again.")
        elif "network" in error_msg or "connection" in error_msg or "timeout" in error_msg:
            console.print(f"\n[red]Error:[/red] Network error: {e}")
            console.print("Check your network connection and try again.")
        else:
            console.print(f"\n[red]Error:[/red] Download failed: {e}")
            console.print("Check your network connection and try again.")
        raise typer.Exit(1)

    console.print(f"\n[green]✓[/green] Model downloaded to {cache_path}")


@model_app.command("list")
def model_list():
    """List available models and their download status."""
    print_banner()

    table = Table(title="Available Models")
    table.add_column("Alias", style="bold cyan")
    table.add_column("Description")
    table.add_column("Repo")
    table.add_column("Status")

    for alias, meta in MODEL_REGISTRY.items():
        cache_path = get_model_cache_path(alias)
        if cache_path.exists() and any(cache_path.iterdir()):
            status = "[green]Downloaded[/green]"
        else:
            status = "[dim]Not downloaded[/dim]"

        table.add_row(alias, meta["description"], meta["repo"], status)

    console.print(table)
    console.print()
    console.print("Download a model with: ctc-detect model download")


__all__ = ["model_app"]
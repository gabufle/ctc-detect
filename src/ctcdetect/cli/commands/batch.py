"""Batch command for CTC-Detect CLI."""


import typer
from rich.progress import Progress, TextColumn
from rich.table import Table

from ctcdetect.cli.utils import console, print_banner, validate_input_path, validate_output_path
from ctcdetect.core.detect import run_detection


@typer.Typer(
    name="batch",
    help=(
        "Run CTC detection on multiple samples at once.\n\n"
        "Point this at a folder containing multiple Cell Ranger outputs\n"
        "and CTC-Detect will process each one sequentially."
    ),
    rich_markup_mode="rich",
)
def batch_app():
    pass


@batch_app.command()
def batch(
    input_dir: str = typer.Option(
        ...,
        "--input-dir",
        help=(
            "Directory containing multiple Cell Ranger output folders.\n"
            "Each subfolder should be one sample."
        ),
        rich_help_panel="Input/Output",
    ),
    output_dir: str = typer.Option(
        ...,
        "--output-dir",
        help=(
            "Directory to write results for all samples.\n"
            "One results CSV will be created per sample."
        ),
        rich_help_panel="Input/Output",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold", "-t",
        help="Probability threshold for CTC calls (default 0.5).",
        rich_help_panel="Model Options",
    ),
    skip_umap: bool = typer.Option(
        False,
        "--skip-umap",
        help="Skip UMAP visualization for faster runs.",
        rich_help_panel="Model Options",
    ),
):
    """Run CTC detection on multiple samples at once."""
    print_banner()
    input_path = validate_input_path(input_dir, "Input directory")
    output_path = validate_output_path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    # Discover sample subdirectories
    samples = sorted([
        d for d in input_path.iterdir()
        if d.is_dir() and not d.name.startswith(".")
    ])

    if not samples:
        console.print(f"[red]Error:[/red] No subdirectories found in {input_path}")
        console.print("Each sample should be in its own subdirectory.")
        raise typer.Exit(1)

    console.print(f"Found {len(samples)} samples to process.\n")

    success_count = 0
    fail_count = 0
    failed_samples: list[str] = []

    with Progress(
        TextColumn("[progress.description]{task.description}"),
        TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
        console=console,
    ) as progress:
        task = progress.add_task("Processing samples...", total=len(samples))

        for sample_dir in samples:
            sample_name = sample_dir.name
            sample_output = output_path / sample_name
            progress.update(task, description=f"[bold]Processing: {sample_name}[/bold]")

            try:
                sample_output.mkdir(parents=True, exist_ok=True)
                run_detection(
                    sample_dir, sample_output,
                    threshold=threshold, skip_umap=skip_umap,
                )
                success_count += 1
            except SystemExit as e:
                fail_count += 1
                failed_samples.append(sample_name)
                console.print(f"[red]✗[/red] {sample_name} failed (exit code {e.code})")
            except Exception as e:
                fail_count += 1
                failed_samples.append(sample_name)
                console.print(f"[red]✗[/red] {sample_name} failed: {e}")

            progress.advance(task)

    # Summary
    console.print()
    console.print("=" * 50)
    summary_table = Table(title="Batch Processing Summary", show_header=False, box=None)
    summary_table.add_column("Key", style="bold cyan", min_width=18)
    summary_table.add_column("Value")
    summary_table.add_row("Total samples", str(len(samples)))
    summary_table.add_row("Successful", f"[green]{success_count}[/green]")
    summary_table.add_row("Failed", f"[red]{fail_count}[/red]" if fail_count > 0 else "0")
    if failed_samples:
        summary_table.add_row("Failed samples", ", ".join(failed_samples))
    console.print(summary_table)

    if fail_count > 0:
        console.print("[yellow]Check individual sample outputs for error details.[/yellow]")
        raise typer.Exit(1)

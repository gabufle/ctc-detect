"""CTC-Detect CLI — Geneformer-based circulating tumor cell detection.

Detect circulating tumor cells (CTCs) from single-cell RNA-seq data
produced by 10x Genomics Cell Ranger. Outputs per-cell CTC probability
scores, UMAP visualizations, and clinical summary reports.
"""

import typer
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, TextColumn

from ctcdetect.utils import validate_input_path, validate_output_path, print_banner

app = typer.Typer(
    name="ctc-detect",
    help=(
        "Detect circulating tumor cells (CTCs) from single-cell RNA-seq data.\n\n"
        "CTC-Detect uses a fine-tuned Geneformer model to score each cell\n"
        "in your Cell Ranger output for likelihood of being a CTC."
    ),
    no_args_is_help=True,
    rich_markup_mode="rich",
)

# Sub-app for model management commands
model_app = typer.Typer(
    help="Manage Geneformer models — download, list, and inspect available models."
)
app.add_typer(model_app, name="model")

console = Console()


@app.command()
def run(
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help=(
            "Path to Cell Ranger output directory or .h5ad file.\n"
            "This should be the filtered_feature_bc_matrix folder\n"
            "from your 10x Genomics run."
        ),
        rich_help_panel="Input/Output",
    ),
    output: str = typer.Option(
        ...,
        "--output", "-o",
        help=(
            "Path to output directory.\n"
            "CTC-Detect will write ctc_probabilities.csv, umap.png,\n"
            "summary.txt, and summary.html to this directory."
        ),
        rich_help_panel="Input/Output",
    ),
    cancer_type: str = typer.Option(
        None,
        "--cancer-type", "-c",
        help=(
            "Cancer type for model selection (e.g. 'breast', 'lung', 'prostate').\n"
            "If not specified, a pan-cancer model will be used."
        ),
        rich_help_panel="Model Options",
    ),
    threshold: float = typer.Option(
        0.5,
        "--threshold", "-t",
        help=(
            "Probability threshold for calling a cell a CTC (0.0–1.0).\n"
            "Cells with CTC probability >= threshold are called CTCs."
        ),
        rich_help_panel="Detection Options",
    ),
    skip_umap: bool = typer.Option(
        False,
        "--skip-umap",
        help=(
            "Skip UMAP visualization for faster runs.\n"
            "CSV scores and summary reports will still be generated."
        ),
        rich_help_panel="Detection Options",
    ),
):
    """Run CTC detection on a single sample.

    Takes your Cell Ranger output and scores each cell for
    circulating tumor cell probability using Geneformer.
    """
    import pandas as pd

    print_banner()
    input_path = validate_input_path(input, "Input path")
    output_path = validate_output_path(output)

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task("Loading model and tokenizing...", total=None)

    from ctcdetect.detect import run_detection
    run_detection(input_path, output_path, cancer_type, threshold, skip_umap)

    # Generate HTML report
    from ctcdetect.report import generate_html_report
    csv_path = output_path / "ctc_probabilities.csv"
    generate_html_report(csv_path, output_path, threshold)

    # Print summary statistics
    results_df = pd.read_csv(csv_path)
    total_cells = len(results_df)
    ctc_count = (results_df["ctc_probability"] >= threshold).sum()
    mean_prob = results_df["ctc_probability"].mean()

    console.print(f"\n[green]✓[/green] Results written to {output_path}")
    console.print(f"\n[bold]Summary Statistics:[/bold]")
    console.print(f"  Total cells analyzed:  {total_cells}")
    console.print(f"  CTCs detected (≥{threshold}): {ctc_count} ({ctc_count/total_cells*100:.1f}%)")
    console.print(f"  Mean CTC probability:  {mean_prob:.4f}")


@app.command()
def validate(
    input: str = typer.Option(
        ...,
        "--input", "-i",
        help=(
            "Path to Cell Ranger output directory or .h5ad file.\n"
            "CTC-Detect will check that the data is valid and readable."
        ),
        rich_help_panel="Input/Output",
    ),
):
    """Validate input data without running detection.

    Use this to check that your Cell Ranger output is complete
    and in the expected format before running the full pipeline.
    """
    print_banner()
    input_path = validate_input_path(input, "Input path")

    from ctcdetect.preprocess import validate_input
    validate_input(input_path)

    console.print(f"\n[green]✓[/green] Input data looks good: {input_path}")


@model_app.command("download")
def model_download(
    version: str = typer.Option(
        "latest",
        "--version", "-v",
        help=(
            "Model version to download (e.g. 'v1.0', 'v2.1').\n"
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

    with Progress(
        SpinnerColumn(),
        TextColumn("[progress.description]{task.description}"),
        console=console,
    ) as progress:
        progress.add_task(f"Downloading model '{version}'...", total=None)

    console.print("[yellow]Note:[/yellow] Model download not yet implemented (stub).")
    console.print(f"  Version: {version}")
    console.print(f"\n[green]✓[/green] Model '{version}' is ready.")


@app.command()
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
):
    """Run CTC detection on multiple samples at once.

    Point this at a folder containing multiple Cell Ranger outputs
    and CTC-Detect will process each one sequentially.
    """
    print_banner()
    input_path = validate_input_path(input_dir, "Input directory")
    output_path = validate_output_path(output_dir)

    console.print(f"  Input directory:  {input_path}")
    console.print(f"  Output directory: {output_path}")
    console.print("[yellow]Note:[/yellow] Batch processing not yet implemented (stub).")


@app.command()
def info():
    """Show CTC-Detect version and system information.

    Displays the installed version, available models, and
    system configuration. Useful for troubleshooting.
    """
    print_banner()
    console.print("Version:  0.1.0")
    console.print("Model:    Geneformer-12L (not downloaded)")
    console.print("Python:   (detected at runtime)")
    console.print("\nFor documentation, visit: https://github.com/ctc-detect/ctc-detect")


if __name__ == "__main__":
    app()
